
import sys
import os
import httpx
import asyncio
import io
from PIL import Image
from fastapi import UploadFile

# A helper function to print colored status messages
def print_check(name, success, message=""):
    # Use simple text markers for broad compatibility
    status = "[✅ PASS]" if success else "[❌ FAIL]"
    print(f"{status:8} {name}")
    if message:
        # Indent messages for readability
        print(f"   -> {message}")
    print("-" * 50)


async def run_diagnostics():
    """Asynchronous function to run all diagnostic checks."""
    print("--- Starting Full System Diagnostic ---")

    # --- 1. Dependency Check ---
    print("1. Checking Python dependencies...")
    try:
        # Attempt to import all required libraries
        import fastapi
        import uvicorn
        import multipart
        import transformers
        import torch
        import torchvision
        import httpx
        import cv2
        from PIL import Image
        print_check("Dependencies Import", True, "All required libraries are installed.")
    except ImportError as e:
        print_check("Dependencies Import", False, f"Missing library: {e.name}. Please run 'pip install -r requirements.txt'")
        sys.exit(1) # Exit early if dependencies are missing

    # --- 2. Hugging Face Model Check ---
    print("2. Checking Hugging Face deepfake model...")
    detector = None
    try:
        from app.services.deepfake_service import DeepfakeDetector
        # This will download and cache the model from Hugging Face on first run
        detector = DeepfakeDetector()
        print_check("HF Model Loading", True, f"Successfully loaded '{detector.model_name}'.")
    except Exception as e:
        print_check("HF Model Loading", False, f"Could not load model. Check internet or HF_TOKEN if needed. Error: {e}")
        # We can continue to run other checks even if this fails
    
    # --- 3. Ollama Server Connection Check ---
    print("3. Checking Ollama server connection...")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Check if the Ollama server is responsive
            response = await client.get("http://localhost:11434")
            response.raise_for_status()
        print_check("Ollama Connection", True, "Successfully connected to Ollama at http://localhost:11434.")
    except (httpx.ConnectError, httpx.TimeoutException):
        print_check("Ollama Connection", False, "Connection failed. Is the Ollama server running locally?")
    except httpx.HTTPStatusError as e:
        print_check("Ollama Connection", False, f"Received an error from Ollama server: {e.response.status_code}")
    
    # --- 4. Core Application Logic Test ---
    print("4. Testing core analysis pipeline...")
    if not detector:
        print_check("Core Logic Test", False, "Skipping test because the deepfake detector failed to load.")
        return

    try:
        from main import analyze_media

        # Create a simple, dummy image in memory for testing
        img = Image.new('RGB', (100, 100), color='red')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # To avoid version conflicts with FastAPI's UploadFile, we create a simple mock
        # object that has the necessary attributes (`filename`, `read`, `content_type`).
        class MockUploadFile:
            def __init__(self, filename, file_bytes, content_type):
                self.filename = filename
                self._file = io.BytesIO(file_bytes)
                self.content_type = content_type

            async def read(self):
                return self._file.read()

        # Get the bytes from the dummy image
        img_bytes = img_byte_arr.getvalue()
        upload_file = MockUploadFile(filename="diagnostic_test.png", file_bytes=img_bytes, content_type="image/png")

        print("   -> Simulating API call to the 'analyze_media' function...")
        # Await the function since it's an async function
        result = await analyze_media(upload_file)

        # Basic validation of the response structure
        assert "is_synthetic" in result
        assert "authenticity_score" in result
        print_check("Core Logic Test", True, "Function executed and returned the expected structure.")
        # Provide a snippet of the result for user verification
        print(f"   -> Sample Result: {{'is_synthetic': {result.get('is_synthetic')}, 'authenticity_score': {result.get('authenticity_score')}, ...}}")

    except Exception as e:
        # Catch any unexpected errors during the logic test
        print_check("Core Logic Test", False, f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Use asyncio.run() to execute the async diagnostic function
    asyncio.run(run_diagnostics())
    print("--- Diagnostic Complete ---")
    print("If all checks passed, the application is likely configured correctly.")

