# main.py
import datetime
import os
import logging
import uuid # For generating unique filenames
import json # For parsing JSON from LLM
import httpx # Import httpx for asynchronous HTTP requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware # Import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import pytesseract
from typing import List, Dict, Union, Optional
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId # Import ObjectId from bson for MongoDB
from pydantic import BaseModel, EmailStr, Field # Import Field for more control over Pydantic models
from datetime import datetime


# Import load_dotenv to load environment variables from .env file
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# MongoDB setup
MONGO_URI = os.getenv('MONGODB_URI')
client = AsyncIOMotorClient(MONGO_URI)
db = client["med_app"]


users_collection = db["users"]
medicine_collection = db['medicine']

class UserResponse(BaseModel):
    userId: str
    email: EmailStr

# Response model for medicine details (as stored in 'medicine' collection)
class MedicineDetailsResponse(BaseModel):
    overview: str
    commonName: str
    commonBrands: List[str]
    commonlyKnownAs: List[str]
    uses: List[str]
    dosesCommonlyGiven: List[str]
    sideEffects: List[str]
    cannonicalName: str
    averagePrice: str
    importantNotes: List[str]
    _id: Optional[str] = None 

# NEW: Pydantic model for the populated medicine details within a scan entry
class PopulatedMedicineDetails(BaseModel):
    overview: str
    commonName: str
    commonBrands: List[str]
    commonlyKnownAs: List[str]
    uses: List[str]
    dosesCommonlyGiven: List[str]
    sideEffects: List[str]
    cannonicalName: str
    averagePrice: str
    importantNotes: List[str]
    id: Optional[str] = Field(None, alias="_id") # The _id from the medicine collection, aliased for Pydantic

    class Config:
        populate_by_name = True # Allow population by field name or alias
        json_encoders = {ObjectId: str} # Ensure ObjectId is converted to string for JSON


# NEW: Pydantic model for a single scan entry with populated medicine details
class PopulatedScanEntry(BaseModel):
    medicine_id: str = Field(..., description="The ID of the medicine document in the medicine collection.")
    scanned_at: datetime = Field(..., description="The UTC datetime when the medicine was scanned.")
    medicine_details: Optional[PopulatedMedicineDetails] = Field(None, description="Detailed information about the scanned medicine, populated from the 'medicine' collection.")

    class Config:
        json_encoders = {datetime: lambda dt: dt.isoformat(), ObjectId: str} # Ensure datetime is ISO formatted


# NEW: Pydantic model for the overall user scan history response
class UserScanHistoryResponse(BaseModel):
    userId: str = Field(..., alias="_id", description="The MongoDB ObjectId of the user, converted to a string.")
    email: EmailStr
    scanned_medicines: List[PopulatedScanEntry] = Field(default_factory=list, description="A list of previous medicine scan entries for the user, with medicine details populated.")

    class Config:
        populate_by_name = True
        json_encoders = {ObjectId: str, datetime: lambda dt: dt.isoformat()}


app = FastAPI(
    title="Medicine Strip Analyzer",
    description="API for analyzing uploaded medicine strip images to extract medicine names and retrieve details.",
    version="1.0.0"
)

# Add CORS middleware

allowed_origins = [
    "http://localhost:3000",
    "http://localhost:8000",  
    "http://127.0.0.1:3000",
    "https://medassistapp.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins, 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)


# Helper function for binarization
def binarize_pixel(pixel_value: int, threshold: int) -> int:
    """
    Transforms a pixel value to 255 (white) if above threshold, else 0 (black).
    """
    return 255 if pixel_value > threshold else 0

# Helper to get a canonical name
def get_canonical_medicine_id(medicine_name: str) -> str:
    """Generates a consistent, URL-safe ID for a medicine name."""
    import re
    return re.sub(r'[^a-z0-9-]+', '', medicine_name.lower().replace(" ", "-")).strip()

@app.get("/")
async def root():
    return JSONResponse(status_code=200, content={"message": "Welcome to the Medicine Strip Analyzer API!"})

@app.get("/get-user-id", response_model=UserResponse)
async def get_or_create_user(email: EmailStr = Query(...)):
    existing_user = await users_collection.find_one({"email": email})

    if existing_user:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "userId": str(existing_user["_id"]),
                "email": existing_user["email"]
            }
        )

    # Create new user if not found
    new_user = {"email": email}
    result = await users_collection.insert_one(new_user)
    logger.info(f"Created new user with ID: {result.inserted_id}")
    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "userId": str(result.inserted_id),
            "email": email
        }
    )

@app.post("/analyze-medicine-strip/")
async def analyze_medicine_strip(
    file: UploadFile = File(...)
) -> Dict[str, Union[str, float]]:
    """
    Uploads a medicine strip image, performs OCR with different preprocessing,
    and then sends all OCR outputs to an LLM to identify the complete medicine name (with power)
    and its confirmation probability.

    Args:
        file (UploadFile): The uploaded image file (e.g., PNG, JPG).

    Returns:
        Dict[str, Union[str, float]]:
        A dictionary containing:
        - 'medicineName': LLM's identified complete medicine name (string, e.g., "Paracetamol 500mg").
        - 'probability': LLM's confirmation probability (float).
    """
    if not isinstance(file.content_type, str) or not file.content_type.startswith("image/"):
        logger.warning(f"Invalid file type uploaded: {file.content_type}")
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image file (e.g., PNG, JPG)."
        )

    # Get the API key from environment variables
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    print('gemini_api_key=>', gemini_api_key)
    if not gemini_api_key :
        logger.error("GEMINI_API_KEY environment variable not set!")
        raise HTTPException(
            status_code=500,
            detail="Server configuration error: GEMINI_API_KEY is not set!"
        )

    try:
        original_image = Image.open(file.file)
        full_ocr_text_combined = ""

        # different binarization thresholds to try
        thresholds = [120, 160, 200]

        for i, threshold in enumerate(thresholds):
            logger.info(f"Performing OCR with binarization threshold: {threshold}...")
            processed_image = original_image.copy().convert('L') # Start with grayscale
            processed_image = processed_image.point(lambda p: binarize_pixel(p, threshold)) # type: ignore

            # Save each processed image for inspection
            # output_dir = "processed_images"
            # os.makedirs(output_dir, exist_ok=True)
            # original_filename_base = os.path.splitext(file.filename if file.filename is not None else "uploaded_file")[0]
            # processed_filename = f"{original_filename_base}_thresh{threshold}_{uuid.uuid4().hex[:8]}.png"
            # processed_filepath = os.path.join(output_dir, processed_filename)
            # processed_image.save(processed_filepath)
            # logger.info(f"Processed image (threshold={threshold}) saved to: {processed_filepath}")

            tesseract_config = '--psm 3' # General purpose PSM
            ocr_data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT, config=tesseract_config)
            full_extracted_text = " ".join([str(text).strip() for text in ocr_data.get('text', []) if str(text).strip()])

            logger.info(f"Full Extracted Text (Threshold {threshold}):\n{full_extracted_text}")
            full_ocr_text_combined += f"OCR Result (Threshold {threshold}):\n{full_extracted_text}\n\n"

        # --- Call LLM for medicine name and probability ---
        logger.info("Sending combined OCR text to LLM for final analysis...")
        llm_prompt = (
            "Analyze the following OCR extracted texts from a medicine strip image, generated using different preprocessing settings. "
            "Your task is to identify the **primary product name or brand name** of the medicine, and its power/strength. "
            "Examples of what to identify: 'Glutaderm plus', 'Paracetamol 500mg', 'Sibelium 10 mg', 'Amoxicillin 250mg'. "
            "Differentiate the main product name from ingredient names (e.g., 'L-glutathione reduced' is likely an ingredient, not the primary product name unless it's the only prominent name). "
            "Prioritize prominently visible names if multiple are present, and aim to extract "
            "the canonical medicine name and its power even if misspelled or partially recognized in OCR. "
            "If no primary medicine name with power can be confidently identified, return 'N/A' for medicineName and 0.0 for probability. "
            "Respond ONLY with a JSON object containing 'medicineName' (string) and 'probability' (float).\n\n"
            f"{full_ocr_text_combined}"
        )

        chat_history = []
        chat_history.append({"role": "user", "parts": [{"text": llm_prompt}]})

        # Define the expected JSON schema for the LLM response
        generation_config = {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "medicineName": {"type": "STRING"},
                    "probability": {"type": "NUMBER"} # Use NUMBER for float
                },
                "propertyOrdering": ["medicineName", "probability"]
            }
        }

        payload = {
            "contents": chat_history,
            "generationConfig": generation_config
        }

        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_api_key}"

        llm_text_response = ""

        # Use httpx.AsyncClient for making asynchronous requests with a timeout
        async with httpx.AsyncClient(timeout=30.0) as client: # Increased timeout here
            llm_response = await client.post(
                api_url,
                headers={"Content-Type": "application/json"},
                json=payload # httpx takes json parameter directly for dict
            )

        llm_response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        llm_result = llm_response.json()

        llm_analysis = {"medicineName": "N/A", "probability": 0.0}

        if llm_result.get('candidates') and len(llm_result['candidates']) > 0 and \
           llm_result['candidates'][0].get('content') and llm_result['candidates'][0]['content'].get('parts') and \
           len(llm_result['candidates'][0]['content']['parts']) > 0:
            try:
                llm_text_response = llm_result['candidates'][0]['content']['parts'][0]['text']
                parsed_llm_json = json.loads(llm_text_response)
                llm_analysis["medicineName"] = parsed_llm_json.get("medicineName", "N/A")
                llm_analysis["probability"] = parsed_llm_json.get("probability", 0.0)
                logger.info(f"LLM Analysis: {llm_analysis}")
            except json.JSONDecodeError as jde:
                logger.error(f"Failed to decode LLM JSON response: {jde}. Raw response: {llm_text_response}")
                llm_analysis["medicineName"] = "Error parsing LLM response"
                llm_analysis["probability"] = 0.0
            except Exception as e:
                # Catch-all for other errors if parts[0].text is present but subsequent ops fail
                logger.error(f"Error processing LLM response: {e}. Raw response: {llm_text_response}")
                llm_analysis["medicineName"] = "Error processing LLM data"
                llm_analysis["probability"] = 0.0
        else:
            logger.warning(f"LLM did not return valid content. Response: {llm_result}")
            llm_analysis["medicineName"] = "No LLM analysis"
            llm_analysis["probability"] = 0.0

        # Return only the LLM analysis results
        return llm_analysis
    except pytesseract.TesseractNotFoundError:
        logger.exception("Tesseract OCR engine not found. Please install it and ensure it's in your system's PATH.")
        raise HTTPException(
            status_code=500,
            detail="Tesseract OCR engine not found. Please install it and ensure it's in your system's PATH."
        )
    except httpx.HTTPStatusError as e:
        logger.exception(f"HTTP error from LLM API: {e.response.status_code} - {e.response.text}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"LLM API error: {e.response.text}"
        )
    except httpx.RequestError as e:
        logger.exception(f"Request error to LLM API: {e}")
        # Changed status code to 504 for read timeouts
        raise HTTPException(
            status_code=504, # Gateway Timeout
            detail=f"Could not connect to LLM API or response timed out: {str(e)}"
        )
    except Exception as e:
        logger.exception(f"An unexpected error occurred during processing for file '{file.filename}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred: {str(e)}"
        )

# Endpoint for Medicine Details (MongoDB cache + LLM fallback)
@app.get("/medicine-details/{medicine_name}", response_model=MedicineDetailsResponse) # Added response_model for clarity
async def get_medicine_details(medicine_name: str, user_id: str = Query(..., description="The ID of the user performing the scan.")) -> MedicineDetailsResponse:
    """
    Provides comprehensive medical information about a medicine by its name,
    fetching from MongoDB cache first, then querying an LLM if not found,
    and storing the LLM response in MongoDB. It also updates the user's scan history.

    Args:
        medicine_name (str): The name of the medicine (e.g., "Paracetamol 500mg").
        user_id (str): The ID of the user performing the scan.

    Returns:
        MedicineDetailsResponse: A dictionary containing medicine details.
        Raises HTTPException if LLM fails or returns invalid data.
    """

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key or gemini_api_key == "YOUR_DUMMY_GEMINI_API_KEY":
        logger.error("GEMINI_API_KEY environment variable not set for /medicine-details.")
        raise HTTPException(
            status_code=500,
            detail="Server configuration error: GEMINI_API_KEY is not set or is a dummy value."
        )

    # Validate user_id upfront
    try:
        user_obj_id = ObjectId(user_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid user ID format: {e}")

    canonical_name = get_canonical_medicine_id(medicine_name)

    # 1. Try to fetch from MongoDB first
    logger.info(f"Attempting to fetch medicine details for {medicine_name} from MongoDB.")
    medicine = await medicine_collection.find_one({"cannonicalName": canonical_name})
    
    if medicine:
        logger.info(f"Found medicine details for {medicine_name} in MongoDB cache.")
        
        # Update user's scan history - ensure medicine_id is ObjectId for consistency
        scan_entry = {
            "medicine_id": medicine["_id"], # This will already be an ObjectId from `find_one`
            "scanned_at": datetime.utcnow()
        }
        try:
            await users_collection.update_one(
                {"_id": user_obj_id},
                {"$push": {"scanned_medicines": scan_entry}}
            )
            logger.info(f"Updated user {user_id} scan history with cached medicine {medicine_name}.")
        except Exception as e:
            logger.error(f"Error updating user scan history with cached medicine {medicine_name}: {e}")
            # Do not fail the request if user history update fails, but log it

        # Convert _id to string for Pydantic model compatibility in the response
        if "_id" in medicine and isinstance(medicine["_id"], ObjectId):
            medicine["_id"] = str(medicine["_id"]) # Convert ObjectId to string for the response model

        return MedicineDetailsResponse(**medicine) # Return using the Pydantic model

    # 2. If not found in MongoDB, query LLM
    logger.info(f"Querying LLM for details of medicine: {medicine_name}")

    llm_prompt = (
        f"Provide comprehensive medical information for the medicine '{medicine_name}' "
        "in a way that is easy for a common person to understand, avoiding complex medical jargon. "
        "Explain each point simply. Include the following details:\n"
        "- **Overview:** A brief 3-4 line summary of what this medicine does and its primary active ingredients or main chemical class. (e.g., 'This medicine helps reduce pain and fever. Its main ingredient is Paracetamol.').\n"
        "- **Common Name:** What is the primary simple name for this medicine? (e.g., 'Paracetamol', 'Ibuprofen')\n"
        "- **Common Brands:** What are some popular brand names this medicine is sold under?\n"
        "- **Also Known As:** Are there other simple names or general classifications for it?\n"
        "- **What it's Used For:** What health problems or conditions does this medicine help with? Explain in simple terms.\n"
        "- **How Much to Take:** What are the usual amounts (doses) of this medicine given to adults? (e.g., 'one tablet when needed', '500mg two times a day'). Be clear about strengths like 500mg or 650mg.\n"
        "- **Possible Side Effects:** What are some common effects you might feel that are not intended? Explain simply.\n"
        "- **Cannonical name:** exact name of the drug and power separated by hyphen in lowercase (e.g., 'paracetamol-500mg').\n"
        "- **Average Price:** What is the estimated price range for this medicine (per strip or pack)? Please specify currency and unit (e.g., '₹X - ₹Y per strip', '$X - $Y per pack of 10').\n"
        "- **Important Things to Know:** Any crucial warnings, advice, or things to be careful about when using this medicine? Explain simply, focusing on patient safety.\n\n"
        "If information is not available for a specific field, indicate 'N/A' or an empty list. "
        "Respond ONLY with a JSON object following this exact structure:\n"
        "{{ "
        '"overview": "string", '
        '"commonName": "string", '
        '"commonBrands": ["string", ...], '
        '"commonlyKnownAs": ["string", ...], '
        '"uses": ["string", ...], '
        '"dosesCommonlyGiven": ["string", ...], '
        '"sideEffects": ["string", ...], '
        '"cannonicalName": "string", '
        '"averagePrice": "string", '
        '"importantNotes": ["string", ...]'
        " }}"
    )

    chat_history = []
    chat_history.append({"role": "user", "parts": [{"text": llm_prompt}]})

    generation_config = {
        "responseMimeType": "application/json",
        "responseSchema": {
            "type": "OBJECT",
            "properties": {
                "overview": {"type": "STRING"},
                "commonName": {"type": "STRING"},
                "commonBrands": {"type": "ARRAY", "items": {"type": "STRING"}},
                "commonlyKnownAs": {"type": "ARRAY", "items": {"type": "STRING"}},
                "uses": {"type": "ARRAY", "items": {"type": "STRING"}},
                "dosesCommonlyGiven": {"type": "ARRAY", "items": {"type": "STRING"}},
                "sideEffects": {"type": "ARRAY", "items": {"type": "STRING"}},
                "cannonicalName": {"type": "STRING"},
                "averagePrice": {"type": "STRING"},
                "importantNotes": {"type": "ARRAY", "items": {"type": "STRING"}}
            },
            "propertyOrdering": [
                "overview",
                "commonName", "commonBrands", "commonlyKnownAs", "uses",
                "dosesCommonlyGiven", "sideEffects", "cannonicalName", "averagePrice", "importantNotes"
            ]
        }
    }

    payload = {
        "contents": chat_history,
        "generationConfig": generation_config
    }

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_api_key}"

    llm_text_response = "" # Initialize to empty string to prevent unbound error
    logger.info(f"Attempting to call LLM API at: {api_url}")
    try:
        # Increased timeout to 30 seconds for LLM API calls
        async with httpx.AsyncClient(timeout=30.0) as client:
            llm_response = await client.post(
                api_url,
                headers={"Content-Type": "application/json"},
                json=payload
            )

        llm_response.raise_for_status()
        llm_result = llm_response.json()

        if llm_result.get('candidates') and len(llm_result['candidates']) > 0 and \
           llm_result['candidates'][0].get('content') and llm_result['candidates'][0]['content'].get('parts') and \
           len(llm_result['candidates'][0]['content']['parts']) > 0:
            llm_text_response = llm_result['candidates'][0]['content']['parts'][0]['text']
            parsed_details = json.loads(llm_text_response)
            logger.info(f"LLM-provided medicine details for {medicine_name}: {parsed_details}")

            # Ensure all values match the expected types in the response model
            final_details = {
                "overview": parsed_details.get("overview", "N/A"),
                "commonName": parsed_details.get("commonName", "N/A"),
                "commonBrands": parsed_details.get("commonBrands", []),
                "commonlyKnownAs": parsed_details.get("commonlyKnownAs", []),
                "uses": parsed_details.get("uses", []),
                "dosesCommonlyGiven": parsed_details.get("dosesCommonlyGiven", []),
                "sideEffects": parsed_details.get("sideEffects", []),
                "cannonicalName": parsed_details.get("cannonicalName", "N/A"),
                "averagePrice": parsed_details.get("averagePrice", "N/A"),
                "importantNotes": parsed_details.get("importantNotes", [])
            }

            # 3. Store in MongoDB after successful LLM query
            try:
                result = await medicine_collection.insert_one(final_details)
                logger.info(f"Stored medicine details for '{medicine_name}' in MongoDB. New ID: {result.inserted_id}")

                scan_entry = {
                    "medicine_id": result.inserted_id, # Use .inserted_id for InsertOneResult object (this is already ObjectId)
                    "scanned_at": datetime.utcnow()
                }
                await users_collection.update_one(
                    {"_id": user_obj_id},
                    {"$push": {"scanned_medicines": scan_entry}}
                )
                logger.info(f"Updated user {user_id} scan history with new medicine {medicine_name}.")
            except Exception as e:
                logger.error(f"Error storing medicine details in MongoDB for {medicine_name} or updating user history: {e}")
                # Don't fail the request if storage/update fails, but log it

            # Add the _id from the newly inserted document for the response, converted to string
            final_details["_id"] = str(result.inserted_id) # type: ignore
            return MedicineDetailsResponse(**final_details) # Return using the Pydantic model
        else:
            logger.warning(f"LLM did not return valid content for medicine details. Response: {llm_result}")
            raise HTTPException(
                status_code=500,
                detail="LLM did not provide valid medicine information."
            )

    except httpx.HTTPStatusError as e:
        logger.exception(f"HTTP error from LLM API for medicine details: {e.response.status_code} - {e.response.text}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"LLM API error fetching details: {e.response.text}"
        )
    except httpx.RequestError as e:
        logger.exception(f"Request error to LLM API for medicine details: {e}")
        raise HTTPException(
            status_code=504, # Gateway Timeout
            detail=f"LLM API response timed out: {str(e)}. The model might be taking too long to generate a response."
        )
    except json.JSONDecodeError as jde:
        logger.error(f"Failed to decode LLM JSON response for medicine details: {jde}. Raw response: {llm_text_response if 'llm_text_response' in locals() else 'N/A'}")
        raise HTTPException(
            status_code=500,
            detail="Error parsing LLM response for medicine details."
        )
    except Exception as e:
        logger.exception(f"An unexpected error occurred while fetching medicine details for '{medicine_name}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred: {str(e)}"
        )

# UPDATED ENDPOINT: Get User Scan History
@app.get("/user-scan-history/{user_id}", response_model=UserScanHistoryResponse)
async def get_user_scan_history(user_id: str) -> UserScanHistoryResponse:
    """
    Retrieves the complete scan history for a given user, populating medicine details
    from the 'medicine' collection for each scanned item.
    """
    try:
        user_obj_id = ObjectId(user_id)
        user_doc = await users_collection.find_one({"_id": user_obj_id})
        if not user_doc:
            raise HTTPException(status_code=404, detail=f"User with ID '{user_id}' not found.")
        if not user_doc.get("scanned_medicines"):
            return UserScanHistoryResponse(
                userId=str(user_doc["_id"]),  # type: ignore
                email=user_doc.get("email", "N/A"),
                scanned_medicines=[]
            )  # type: ignore

        # Aggregation pipeline to populate medicine details
        pipeline = [
            {"$match": {"_id": user_obj_id}},
            {"$unwind": "$scanned_medicines"},
            {"$lookup": {
                "from": "medicine",
                "localField": "scanned_medicines.medicine_id",
                "foreignField": "_id",
                "as": "scanned_medicines.medicine_details"
            }},
            {"$addFields": {
                "scanned_medicines.medicine_details": {
                    "$arrayElemAt": ["$scanned_medicines.medicine_details", 0]
                }
            }},
            {"$group": {
                "_id": "$_id",
                "email": {"$first": "$email"},
                "scanned_medicines": {"$push": "$scanned_medicines"}
            }},
            {"$project": {
                "_id": 0, # Exclude the original _id from the root document
                "userId": "$_id", # Rename the original _id to userId
                "email": "$email",
                "scanned_medicines": "$scanned_medicines"
            }}
        ]

        aggregated_user_data = await users_collection.aggregate(pipeline).to_list(length=1)

        if not aggregated_user_data:
            # This case implies the aggregation didn't find the user's data after unwinding,
            # which could happen if 'scanned_medicines' was initially present but empty
            # or became empty due to subsequent operations.
            return UserScanHistoryResponse(
                userId=str(user_doc["_id"]),  # Use userId directly # type: ignore
                email=user_doc.get("email", "N/A"),
                scanned_medicines=[]
            ) # type: ignore

        user_data = aggregated_user_data[0]
        
        # --- CRITICAL FIX: Ensure ALL ObjectIds are converted to strings before Pydantic validation ---
        # 1. Convert top-level userId to string
        if "userId" in user_data and isinstance(user_data["userId"], ObjectId):
            user_data["userId"] = str(user_data["userId"])

        # 2. Convert ObjectIds within scanned_medicines array
        if "scanned_medicines" in user_data and user_data["scanned_medicines"]:
            for scan_entry in user_data["scanned_medicines"]:
                # Convert medicine_id within the scan_entry
                if "medicine_id" in scan_entry and isinstance(scan_entry["medicine_id"], ObjectId):
                    scan_entry["medicine_id"] = str(scan_entry["medicine_id"])

                # Convert _id within medicine_details and ensure 'id' field is set for Pydantic
                if "medicine_details" in scan_entry and scan_entry["medicine_details"] is not None:
                    if "_id" in scan_entry["medicine_details"] and isinstance(scan_entry["medicine_details"]["_id"], ObjectId):
                        # Convert and set to 'id' as per PopulatedMedicineDetails model alias
                        scan_entry["medicine_details"]["id"] = str(scan_entry["medicine_details"]["_id"])
                        del scan_entry["medicine_details"]["_id"] # Remove the original _id key
                    elif "_id" in scan_entry["medicine_details"]: # If _id is already a string but 'id' isn't set
                         scan_entry["medicine_details"]["id"] = scan_entry["medicine_details"]["_id"]
                         del scan_entry["medicine_details"]["_id"] # Remove the original _id key
        # --- END CRITICAL FIX ---

        return UserScanHistoryResponse(**user_data)

    except Exception as e:
        logger.exception(f"An error occurred while fetching scan history for user {user_id}: {e}")
        if isinstance(e, ValueError) and "could not convert string to ObjectId" in str(e):
             raise HTTPException(
                status_code=400,
                detail=f"Invalid user ID format: '{user_id}'. Must be a 24-character hexadecimal string."
            )
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while fetching scan history: {str(e)}"
        )