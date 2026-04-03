from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image, UnidentifiedImageError
import torch
import cv2
import os
from app.core.config import settings

class DeepfakeDetector:
    """
    A service to detect whether an image or video frame is real or a deepfake using a
    pre-trained Hugging Face model. This implementation matches the official usage
    example for the model to ensure accuracy.
    """
    def __init__(self):
        """
        Initializes the DeepfakeDetector by loading the model and its processor
        from Hugging Face.
        """
        self.model_name = settings.DEEPFAKE_MODEL
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    async def detect(self, file_path: str) -> dict | None:
        """
        Asynchronously analyzes an image or video from a given file path to determine
        if it is a deepfake.

        Args:
            file_path: The path to the file to be analyzed.

        Returns:
            A dictionary with 'real_score', 'fake_score', 'best_label', 
            'best_score', and 'all_predictions', or None if the analysis fails.
        """
        image = None
        try:
            image = Image.open(file_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError):
            try:
                cap = cv2.VideoCapture(file_path)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)
                cap.release()
            except Exception as e:
                print(f"Error opening or processing file (image/video): {e}")
                return None

        if image is None:
            print(f"Could not extract any image from file: {file_path}")
            return None

        try:
            # Process the image and get tensors
            inputs = self.processor(images=image, return_tensors="pt")
            
            # Perform inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
            
            # Create a list of all predictions with labels and scores
            all_predictions = []
            for i, prob in enumerate(probabilities):
                label = self.model.config.id2label[i]
                all_predictions.append({
                    "label": label,
                    "score": round(prob.item(), 4)
                })

            # Identify the best overall prediction
            predicted_class_id = logits.argmax(-1).item()
            best_label = self.model.config.id2label[predicted_class_id]
            best_score = probabilities[predicted_class_id].item()

            # The model has two outputs: 'Deepfake' (LABEL_1) and 'Realism' (LABEL_0).
            # We map them to REAL and FAKE for clarity.
            predictions_map = {p['label'].upper(): p['score'] for p in all_predictions}
            
            real_score = predictions_map.get('REALISM', predictions_map.get('LABEL_0', 0.0))
            fake_score = predictions_map.get('DEEPFAKE', predictions_map.get('LABEL_1', 0.0))

            # Fallback if names are different
            if all(k in predictions_map for k in ['REAL', 'FAKE']):
                real_score = predictions_map.get('REAL', 0.0)
                fake_score = predictions_map.get('FAKE', 0.0)

            return {
                "best_label": best_label,
                "best_score": round(best_score, 4),
                "real_score": round(real_score, 4),
                "fake_score": round(fake_score, 4),
                "all_predictions": all_predictions
            }
        except Exception as e:
            print(f"Error during model inference: {e}")
            return None
