from transformers import pipeline
from PIL import Image, UnidentifiedImageError
from app.core.config import settings

class SentimentAnalyzer:
    """
    A service to perform visual sentiment/emotion analysis on images.
    """
    def __init__(self):
        """
        Initializes the SentimentAnalyzer by loading the emotion detection
        pipeline from Hugging Face.
        """
        try:
            self.pipe = pipeline("image-classification", model=settings.SENTIMENT_MODEL)
        except Exception as e:
            print(f"Error loading sentiment model: {e}")
            raise

    def analyze(self, file_path: str) -> dict | None:
        """
        Analyzes an image to determine the emotional sentiment.

        Args:
            file_path: The path to the image file.

        Returns:
            A dictionary with the top emotion and all scores.
        """
        try:
            image = Image.open(file_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError) as e:
            print(f"Error opening image for sentiment analysis: {e}")
            return None

        try:
            result = self.pipe(image)
            if not result:
                return None
            
            best_prediction = max(result, key=lambda x: x['score'])
            
            return {
                "sentiment_label": best_prediction['label'],
                "sentiment_score": round(best_prediction['score'], 4),
                "all_sentiments": result
            }
        except Exception as e:
            print(f"Error during sentiment inference: {e}")
            return None
