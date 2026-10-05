import sys
from pathlib import Path

# Add task01-sentiment-api to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pytest
from fastapi.testclient import TestClient
from main import app


class MockClassifier:
    """Mock classifier for fast isolated testing."""

    def predict(self, texts):
        results = []
        for text in texts:
            lower = text.lower()
            if "good" in lower or "love" in lower or "great" in lower or "wonderful" in lower:
                sentiment = "positive"
                scores = {"positive": 0.95, "neutral": 0.03, "negative": 0.02}
            elif "bad" in lower or "terrible" in lower or "hate" in lower or "damaged" in lower:
                sentiment = "negative"
                scores = {"positive": 0.02, "neutral": 0.03, "negative": 0.95}
            else:
                sentiment = "neutral"
                scores = {"positive": 0.10, "neutral": 0.80, "negative": 0.10}
            results.append({
                "sentiment": sentiment,
                "confidence": scores[sentiment],
                "scores": scores,
            })
        return results


@pytest.fixture(scope="session")
def client():
    """Test client using mock classifier to ensure rapid test execution."""
    app.state.classifier = MockClassifier()
    with TestClient(app) as test_client:
        yield test_client
