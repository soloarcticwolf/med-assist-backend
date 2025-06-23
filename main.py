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
from pydantic import BaseModel, EmailStr
from datetime import datetime


# Import load_dotenv to load environment variables from .env file
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# MongoDB setup
MONGO_URI = os.getenv("MONGO_URI", 'mongodb+srv://python:python1234@cluster0.wvg5e.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
client = AsyncIOMotorClient(MONGO_URI)
db = client["med_app"]

users_collection = db["users"]
medicine_collection = db['medicine'] # Changed from 'med_app' to 'medicine' as per your usage

class UserResponse(BaseModel):
    userId: str
    email: EmailStr

# Response model for medicine details
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


app = FastAPI(
    title="Medicine Strip Analyzer",
    description="API for analyzing uploaded medicine strip images to extract medicine names.",
    version="1.0.0"
)

# Add CORS middleware
# For development, allow all origins. In production, restrict to your frontend's domain(s).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)


# Helper function for binarization
def binarize_pixel(pixel_value: int, threshold: int) -> int:
    """
    Transforms a pixel value to 255 (white) if above threshold, else 0 (black).
    """
    return 255 if pixel_value > threshold else 0

# Helper to get a canonical name (for Firestore document ID)
def get_canonical_medicine_id(medicine_name: str) -> str:
    """Generates a consistent, URL-safe ID for a medicine name."""
    # Ensure all non-alphanumeric characters are handled, not just space, slash, backslash
    import re
    return re.sub(r'[^a-z0-9-]+', '', medicine_name.lower().replace(" ", "-")).strip()

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
    if not gemini_api_key or gemini_api_key == "YOUR_DUMMY_GEMINI_API_KEY":
        logger.error("GEMINI_API_KEY environment variable not set or is a dummy value for /analyze-medicine-strip.")
        raise HTTPException(
            status_code=500,
            detail="Server configuration error: GEMINI_API_KEY is not set or is a dummy value."
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

            # Optional: Save each processed image for inspection
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
        async with httpx.AsyncClient(timeout=30.0) as client:
            llm_response = await client.post(
                api_url,
                headers={"Content-Type": "application/json"},
                json=payload 
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
async def get_medicine_details(medicine_name: str, user_id: str = Query("12345", description="The ID of the user performing the scan.")) -> Dict[str, Union[str, List[str]]]:
    """
    Provides detailed information about a medicine by its name, fetching from MongoDB cache first,
    then querying an LLM if not found, and storing the LLM response in MongoDB.

    Args:
        medicine_name (str): The name of the medicine (e.g., "Paracetamol 500mg").
        user_id (str): The ID of the user performing the scan.

    Returns:
        Dict[str, Union[str, List[str]]]: A dictionary containing medicine details.
        Raises HTTPException if LLM fails or returns invalid data.
    """

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key or gemini_api_key == "YOUR_DUMMY_GEMINI_API_KEY":
        logger.error("GEMINI_API_KEY environment variable not set for /medicine-details.")
        raise HTTPException(
            status_code=500,
            detail="Server configuration error: GEMINI_API_KEY is not set or is a dummy value."
        )

    canonical_name = get_canonical_medicine_id(medicine_name)

    # 1. Try to fetch from MongoDB first
    logger.info(f"Attempting to fetch medicine details for {medicine_name} from MongoDB.")
    medicine = await medicine_collection.find_one({"cannonicalName": canonical_name})
    print('after checking mongodb medicine', medicine)
    if medicine:
        print('FOUND medicine on mongodb', medicine)
        # Convert ObjectId to string before returning to match the return type hint
        if "_id" in medicine and isinstance(medicine["_id"], ObjectId):
            medicine["_id"] = str(medicine["_id"])
        
        logger.info(f"Found medicine details for {medicine_name} in MongoDB cache.")
        
        # This part of the code correctly updates user's scan history when medicine is found in cache
        # The medicine_id is obtained from the cached `medicine` document.
        scan_entry = {
            "medicine_id": str(medicine["_id"]), # This was already correct because 'medicine' is a dict from find_one
            "scanned_at": datetime.utcnow()
        }
        try:
            await users_collection.update_one(
                {"_id": ObjectId(user_id)}, # Ensure user_id is converted to ObjectId
                {"$push": {"scanned_medicines": scan_entry}}
            )
            logger.info(f"Updated user {user_id} scan history with cached medicine {medicine_name}.")
        except Exception as e:
            logger.error(f"Error updating user scan history with cached medicine {medicine_name}: {e}")
            # Do not fail the request if user history update fails, but log it

        return medicine # medicine dictionary with _id as string will be returned

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
        '"cannonicalName": "string", ' # Corrected: should be a single string, not a list
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
                "cannonicalName": {"type": "STRING"}, # Corrected schema for cannonicalName
                "averagePrice": {"type": "STRING"},
                "importantNotes": {"type": "ARRAY", "items": {"type": "STRING"}}
            },
            "propertyOrdering": [
                "overview",
                "commonName", "commonBrands", "commonlyKnownAs", "uses",
                "dosesCommonlyGiven", "sideEffects", "cannonicalName", "averagePrice", "importantNotes" # Added cannonicalName to ordering
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

            # Ensure all values match the expected types in the return annotation
            final_details = {
                "overview": parsed_details.get("overview", "N/A"),
                "commonName": parsed_details.get("commonName", "N/A"),
                "commonBrands": parsed_details.get("commonBrands", []),
                "commonlyKnownAs": parsed_details.get("commonlyKnownAs", []),
                "uses": parsed_details.get("uses", []),
                "dosesCommonlyGiven": parsed_details.get("dosesCommonlyGiven", []),
                "sideEffects": parsed_details.get("sideEffects", []),
                "cannonicalName": parsed_details.get("cannonicalName", "N/A"), # Now expects a string
                "averagePrice": parsed_details.get("averagePrice", "N/A"),
                "importantNotes": parsed_details.get("importantNotes", [])
            }

            # 3. Store in MongoDB after successful LLM query
            try:
                # Store details here. The _id will be generated by MongoDB
                result = await medicine_collection.insert_one(final_details)
                print('###################updated result', result)

                scan_entry = {
                    "medicine_id": str(result.inserted_id), # Corrected: Use .inserted_id for InsertOneResult object
                    "scanned_at": datetime.utcnow()
                }
                await users_collection.update_one(
                    {"_id": ObjectId(user_id)}, # Ensure user_id is converted to ObjectId
                    {"$push": {"scanned_medicines": scan_entry}}
                )
                logger.info(f"Stored medicine details for '{medicine_name}' in MongoDB and updated user scan history.")
            except Exception as e:
                logger.error(f"Error storing medicine details in MongoDB for {medicine_name}: {e}")
                # Don't fail the request if storage fails, but log it
            
            final_details["_id"] = str(result.inserted_id) # type: ignore
            return final_details
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
        # Changed status code to 504 for read timeouts
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


# New Endpoint to record a user's scan in MongoDB (formerly Firestore)
@app.post("/record-medicine-scan/")
async def record_medicine_scan(
    medicine_name_identified: str = Query(..., description="The exact medicine name identified by the LLM."),
    probability: float = Query(..., description="The confidence probability from the LLM analysis."),
    user_id: str = Query(..., description="The ID of the user performing the scan."),
    # image_url: Optional[str] = Query(None, description="Optional URL to the uploaded scanned image.") # Re-add if you store images
) -> Dict[str, str]:
    """
    Records a confirmed medicine scan under a user's history in MongoDB.

    Args:
        medicine_name_identified (str): The exact medicine name and power identified by the LLM.
        probability (float): The confidence probability for the identification.
        user_id (str): The ID of the user.

    Returns:
        Dict[str, str]: A confirmation message.
        Raises HTTPException on error.
    """
    if users_collection is None: # Check for MongoDB collection instead of Firestore db
        raise HTTPException(status_code=500, detail="MongoDB collection for users is not initialized. Check server logs.")

    # Generate canonical ID for referencing the medicine details document
    # You might want to find the medicine document's actual _id if it already exists
    # If the medicine details might not be stored yet, you could potentially fetch or create it here.
    # For now, we'll assume medicine_name_identified is what you want to link.
    # If you intend to link to the *actual* medicine document, you'd need its _id.
    # A more robust approach would be to pass the medicine_id from /medicine-details endpoint here.

    scan_data = {
        "medicineNameIdentified": medicine_name_identified,
        "probability": probability,
        "timestamp": datetime.utcnow(), # Use datetime.utcnow() for MongoDB
        # "imageUrl": image_url # Add if you plan to store image URLs in scan records
    }

    try:
        # Update the user document to push the scan_data into scanned_medicines array
        await users_collection.update_one(
            {"_id": ObjectId(user_id)}, # Convert user_id to ObjectId
            {"$push": {"scanned_medicines": scan_data}}
        )
        logger.info(f"Recorded scan for user {user_id}: {medicine_name_identified}")
        return {"message": "Scan recorded successfully!"}
    except Exception as e:
        logger.error(f"Error recording scan for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to record scan in MongoDB: {str(e)}"
        )