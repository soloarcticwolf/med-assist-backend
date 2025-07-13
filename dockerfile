# Stage 1: Builder Stage - Install dependencies
FROM python:3.11-slim-buster AS builder

# Set environment variables to prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy only the requirements file first to leverage Docker's build cache
COPY requirements.txt .

# Install dependencies (ensure uvicorn is in your requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Production Stage - Copy only necessary files for runtime
FROM python:3.11-slim-buster 

# Create a non-privileged user and group for security
RUN groupadd -g 1000 appuser && useradd -r -u 1000 -g appuser appuser

WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy your application code
COPY . .

# Switch to the non-privileged user
USER appuser

# Expose the port your application will listen on
EXPOSE 8000

# Define the command to run your application using Uvicorn
# Syntax: uvicorn <module_name>:<app_instance_name> --host <host> --port <port>
# Example: If your app is defined as `app` in `main.py`, it's `main:app`
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# If you need to run with workers (similar to gunicorn workers), you might use uvicorn with gunicorn:
# CMD ["gunicorn", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "main:app"]