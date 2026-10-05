import logging
import torch
from typing import List, Dict, Any
from transformers import pipeline

from app.config import settings

logger = logging.getLogger(__name__)


class SentimentClassifier:
    """Wrapper class for Hugging Face sentiment analysis model pipeline."""

    def __init__(self, model_name: str = settings.MODEL_NAME):
        self.model_name = model_name
        self.device = 0 if torch.cuda.is_available() else -1
        logger.info(
            f"Initializing SentimentClassifier with model '{self.model_name}' on device: "
            f"{'CUDA:0' if self.device == 0 else 'CPU'}"
        )
        self.pipeline = pipeline(
            task="text-classification",
            model=self.model_name,
            top_k=None,
            device=self.device,
        )
        logger.info("SentimentClassifier loaded successfully.")

    @staticmethod
    def _format_single_prediction(raw_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Formats raw pipeline output:
        e.g. [{'label': 'positive', 'score': 0.98}, {'label': 'neutral', 'score': 0.01}, {'label': 'negative', 'score': 0.01}]
        into structured response dictionary.
        """
        # Map labels to standard lowercase positive, negative, neutral
        scores: Dict[str, float] = {}
        for item in raw_scores:
            label = item["label"].lower().strip()
            # Handle potential edge cases in label naming
            if "pos" in label:
                label_key = "positive"
            elif "neg" in label:
                label_key = "negative"
            elif "neu" in label:
                label_key = "neutral"
            else:
                label_key = label
            scores[label_key] = round(float(item["score"]), 4)

        # Ensure all three standard keys exist
        for key in ["positive", "negative", "neutral"]:
            if key not in scores:
                scores[key] = 0.0

        # Renormalize slightly if necessary to preserve precision
        winning_sentiment = max(scores, key=scores.get)
        confidence = scores[winning_sentiment]

        return {
            "sentiment": winning_sentiment,
            "confidence": confidence,
            "scores": {
                "positive": scores["positive"],
                "negative": scores["negative"],
                "neutral": scores["neutral"],
            },
        }

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict sentiment for a list of input texts.
        """
        if not texts:
            return []

        raw_results = self.pipeline(
            texts,
            truncation=True,
            max_length=settings.MAX_TOKEN_LENGTH,
            batch_size=len(texts),
        )

        # When single item is passed with top_k=None, pipeline returns list of list of dicts: [[{...}, {...}]]
        formatted_results = [
            self._format_single_prediction(result) for result in raw_results
        ]
        return formatted_results
