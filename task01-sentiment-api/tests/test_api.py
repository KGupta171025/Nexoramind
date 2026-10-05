import pytest
from app.model import SentimentClassifier
from main import app


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "documentation" in data
    assert data["documentation"] == "/docs"
    assert "endpoints" in data


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model" in data
    assert "version" in data


def test_analyze_positive(client):
    payload = {"text": "I love this product, it is wonderful!"}
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "positive"
    assert data["confidence"] > 0.5
    assert "scores" in data
    assert "positive" in data["scores"]
    assert "negative" in data["scores"]
    assert "neutral" in data["scores"]


def test_analyze_negative(client):
    payload = {"text": "This service is terrible and frustrating."}
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "negative"
    assert data["confidence"] > 0.5


def test_analyze_neutral(client):
    payload = {"text": "The shipment was scheduled for delivery at 2 PM."}
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "neutral"


def test_analyze_empty_string_fails(client):
    payload = {"text": ""}
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_analyze_whitespace_only_fails(client):
    payload = {"text": "     \n\t  "}
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 422


def test_analyze_text_too_long_fails(client):
    payload = {"text": "a" * 2001}
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 422


def test_analyze_missing_field_fails(client):
    payload = {}
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 422


def test_batch_analyze_success(client):
    payload = {
        "texts": [
            "This is a great product!",
            "Horrible experience, never returning.",
            "The document was saved in the folder.",
        ]
    }
    response = client.post("/api/analyze/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 3
    assert data["results"][0]["sentiment"] == "positive"
    assert data["results"][1]["sentiment"] == "negative"
    assert data["results"][2]["sentiment"] == "neutral"


def test_batch_analyze_up_to_10_success(client):
    payload = {"texts": [f"Sample text number {i}" for i in range(10)]}
    response = client.post("/api/analyze/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 10


def test_batch_analyze_exceeds_10_fails(client):
    payload = {"texts": [f"Sample text number {i}" for i in range(11)]}
    response = client.post("/api/analyze/batch", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_batch_analyze_empty_list_fails(client):
    payload = {"texts": []}
    response = client.post("/api/analyze/batch", json=payload)
    assert response.status_code == 422


def test_batch_analyze_with_empty_item_fails(client):
    payload = {"texts": ["Valid text", "", "Another text"]}
    response = client.post("/api/analyze/batch", json=payload)
    assert response.status_code == 422


def test_batch_analyze_with_whitespace_item_fails(client):
    payload = {"texts": ["Valid text", "    ", "Another text"]}
    response = client.post("/api/analyze/batch", json=payload)
    assert response.status_code == 422


@pytest.mark.integration
def test_real_model_inference(client):
    """Integration test using the real Hugging Face pipeline."""
    real_classifier = SentimentClassifier()
    app.state.classifier = real_classifier

    payload = {"text": "I absolutely love this amazing sentiment analysis project!"}
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "positive"
    assert data["confidence"] > 0.8
    assert "scores" in data
    assert "positive" in data["scores"]
    assert "negative" in data["scores"]
    assert "neutral" in data["scores"]
