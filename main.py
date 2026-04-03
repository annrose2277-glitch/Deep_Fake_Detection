import os
import tempfile
import mimetypes
import cv2
from typing import Dict, Any, Optional, List
from pydantic import BaseModel

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.services.deepfake_service import DeepfakeDetector
from app.services.moderation_service import ModerationEngine
from app.services.sentiment_service import SentimentAnalyzer
from app.core.config import settings

# --- Helper Functions ---

def extract_frame_from_video(video_path: str) -> str:
    """Extracts a middle frame from a video and saves it as a temporary image."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not open video file.")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame_idx = frame_count // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
    
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise HTTPException(status_code=400, detail="Could not extract frame from video.")
    
    cap.release()
    
    temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(temp_img.name, frame)
    return temp_img.name

# --- Data Models ---

class AnalysisResponse(BaseModel):
    is_synthetic: bool
    authenticity_score: float
    confidence: float
    detected_label: str
    file_name: str
    content_type: str
    sentiment: Optional[Dict[str, Any]] = None
    moderation: Optional[Dict[str, Any]] = None
    debug_info: Optional[List[Dict[str, Any]]] = None

# --- Application Initialization ---

app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description="An API for detecting synthetic media, evaluating sentiment, and content safety.",
)

# Enable CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load service models into memory on application startup
try:
    detector = DeepfakeDetector()
    sentiment_analyzer = SentimentAnalyzer()
except Exception as e:
    raise RuntimeError(f"Fatal: Could not load ML models. {e}")

moderator = ModerationEngine()

@app.get("/", response_class=FileResponse)
async def serve_frontend():
    """Serves the frontend HTML application."""
    if not os.path.exists("frontend.html"):
        raise HTTPException(status_code=404, detail="Frontend file not found")
    return FileResponse("frontend.html")

@app.post("/api/v1/analyze-media", response_model=AnalysisResponse)
async def analyze_media(file: UploadFile = File(...)) -> AnalysisResponse:
    """
    Accepts a media file and orchestrates the analysis pipeline:
    1. Video Frame Extraction (if applicable)
    2. Authenticity Analysis
    3. Sentiment Analysis
    4. Safety Moderation
    """
    temp_file_path: Optional[str] = None
    analysis_path: Optional[str] = None
    try:
        # Save upload to temporary file
        suffix = os.path.splitext(file.filename)[1].lower() if file.filename else ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Determine content type
        content_type = file.content_type
        if not content_type or content_type == "application/octet-stream":
            content_type, _ = mimetypes.guess_type(file.filename or "")

        # --- Check if it's a video or image ---
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']
        is_video = suffix in video_extensions or (content_type and content_type.startswith("video/"))
        
        # If it's a video, extract a frame for analysis
        if is_video:
            try:
                analysis_path = extract_frame_from_video(temp_file_path)
            except Exception as e:
                print(f"Video extraction error: {e}")
                raise HTTPException(status_code=400, detail=f"Failed to process video: {str(e)}")
        else:
            analysis_path = temp_file_path

        # --- Stage 1: Authenticity Analysis ---
        authenticity_result = await detector.detect(analysis_path)
        if not authenticity_result:
            raise HTTPException(status_code=500, detail="Authenticity analysis failed.")

        real_score = authenticity_result.get("real_score", 0.0)
        fake_score = authenticity_result.get("fake_score", 0.0)
        
        # Determine if it's synthetic based on the fake_score and threshold
        is_synthetic = (fake_score >= settings.CONFIDENCE_THRESHOLD)

        # Labels
        detected_label = "AI GENERATED" if is_synthetic else "REAL ONES"
        
        # Authenticity score is based on the realness
        authenticity_score = real_score
        
        # Confidence in the verdict
        confidence = fake_score if is_synthetic else real_score

        # --- Stage 2: Sentiment Analysis ---
        sentiment_result = sentiment_analyzer.analyze(analysis_path)

        # --- Stage 3: Moderation Engine ---
        # Run moderation for all files
        moderation_result = await moderator.evaluate_safety(analysis_path)

        return AnalysisResponse(
            is_synthetic=is_synthetic,
            authenticity_score=round(authenticity_score, 4),
            confidence=round(confidence, 4),
            detected_label=detected_label,
            file_name=file.filename or "unknown",
            content_type=content_type or ("video/mp4" if is_video else "image/jpeg"),
            sentiment=sentiment_result,
            moderation=moderation_result,
            debug_info=authenticity_result.get("all_predictions")
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    finally:
        # Cleanup temporary files
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except: pass
        if analysis_path and analysis_path != temp_file_path and os.path.exists(analysis_path):
            try:
                os.remove(analysis_path)
            except: pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
