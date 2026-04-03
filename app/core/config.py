import os

class Settings:
    """
    Application settings and configuration.
    """
    APP_TITLE: str = "Multimodal Media Analysis System"
    APP_VERSION: str = "1.1.0"
    
    # Model configuration
    DEEPFAKE_MODEL: str = "prithivMLmods/Deep-Fake-Detector-v2-Model"
    SENTIMENT_MODEL: str = "dima806/facial_emotions_image_detection"
    
    # Ollama configuration
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llava-llama3")
    
    # Business logic thresholds
    CONFIDENCE_THRESHOLD: float = 0.8
    SYNTHETIC_LABELS: set = {"FAKE", "SYNTHETIC", "GENERATED", "AI_GENERATED", "DEEPFAKE", "LABEL_1"}

settings = Settings()
