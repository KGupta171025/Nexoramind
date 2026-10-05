# Task 01: Sentiment Analysis API

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.14-blue.svg?logo=python)](https://python.org)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.40+-yellow.svg)](https://huggingface.co/transformers/)
[![Pytest](https://img.shields.io/badge/Tests-16%20Passed-brightgreen.svg?logo=pytest)](https://pytest.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker)](https://docker.com)

A production-ready **Sentiment Analysis REST API** built with **FastAPI** and **Hugging Face Transformers** that classifies input text as `positive`, `negative`, or `neutral` with confidence scores and full per-class probability distributions.

---

## 🎯 Task Objectives & Features

- **Single Text Classification**: Analyze individual text strings via `POST /api/analyze`.
- **Batch Processing**: Analyze batches of up to 10 text inputs concurrently via `POST /api/analyze/batch`.
- **Granular Output**: Provides winning `sentiment`, `confidence` score, and granular per-class probability scores (`positive`, `negative`, `neutral`).
- **Robust Input Validation & Error Handling**: Comprehensive Pydantic v2 validation against empty inputs, whitespace-only strings, oversized payloads, and batch size violations (>10 items) with clean HTTP 422 error structures.
- **Model Efficiency & Hardware Detection**: Automatically detects and leverages CUDA GPU acceleration if available, falling back to optimized CPU inference.
- **Interactive API Documentation**: Native OpenAPI Swagger UI (`/docs`) and ReDoc (`/redoc`).
- **Containerization**: Production Dockerfile and dockerignore for instant container deployment.

---

## 🧠 Model Architecture & Metrics

### Model Selection
- **Model**: [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- **Architecture**: RoBERTa-base (Robustly Optimized BERT Approach) fine-tuned on ~124M tweets and tailored for multi-domain 3-class sentiment classification (`positive`, `neutral`, `negative`).
- **Max Sequence Length**: 512 tokens with dynamic truncation.

### Benchmark Evaluation Results
The model was evaluated using [`evaluate.py`](evaluate.py) across a diverse, multi-domain benchmark covering reviews, customer support queries, neutral statements, and operational announcements.

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **100.0%** (30/30) |
| **Macro Precision** | **1.0000** |
| **Macro Recall** | **1.0000** |
| **Macro F1-Score** | **1.0000** |
| **Weighted F1-Score**| **1.0000** |
| **Mean Confidence** | **0.9287** (92.87%) |

#### Per-Class Performance
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Positive** | 1.0000 | 1.0000 | 1.0000 | 10 |
| **Neutral** | 1.0000 | 1.0000 | 1.0000 | 10 |
| **Negative** | 1.0000 | 1.0000 | 1.0000 | 10 |

#### Confusion Matrix
```
                 Predicted
              Pos   Neu   Neg
Actual  Pos [ 10,    0,    0 ]
        Neu [  0,   10,    0 ]
        Neg [  0,    0,   10 ]
```

#### Latency & Throughput Benchmark (Standard CPU)
| Benchmark Metric | Measurement |
| :--- | :--- |
| **Single Request Mean Latency** | **44.79 ms** |
| **Single Request P95 Latency** | **54.05 ms** |
| **Batch (10 texts) Total Latency** | **165.06 ms** (~16.51 ms/item) |
| **Batch Processing Throughput** | **~60.58 items/sec** |

*(Benchmarks run on local Intel Core CPU without GPU acceleration; GPU acceleration further yields <10ms latencies)*

---

## 📂 Project Organization

```
task01-sentiment-api/
├── app/
│   ├── __init__.py
│   ├── config.py           # Application settings and hyperparameters
│   ├── model.py            # SentimentClassifier pipeline wrapper (device & batch management)
│   ├── router.py           # API endpoints (/api/analyze, /api/analyze/batch, /health)
│   └── schemas.py          # Pydantic validation models and Swagger JSON schemas
├── tests/
│   ├── __init__.py
│   ├── conftest.py         # Pytest fixtures and mock client
│   └── test_api.py         # 16 unit, validation, and integration tests
├── Dockerfile              # Production container build
├── .dockerignore           # Container ignore rules
├── evaluate.py             # Benchmark evaluation & metrics computation script
├── metrics.json            # Machine-readable evaluation results
├── requirements.txt        # Python dependency manifest
├── main.py                 # FastAPI application entrypoint with lifespan & CORS
└── test_model.py           # Standalone CLI inference verification script
```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.11, 3.12, 3.14)
- Git

### 2. Clone and Setup Environment
```bash
# Clone the repository
git clone https://github.com/KGupta171025/Nexoramind.git
cd Nexoramind/task01-sentiment-api

# Create and activate virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Development Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Once started, access the interactive Swagger documentation at:
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check**: [http://localhost:8000/api/health](http://localhost:8000/api/health)

---

## 📡 API Reference & Examples

### 1. Single Text Analysis
- **Endpoint**: `POST /api/analyze`
- **Headers**: `Content-Type: application/json`

#### Request Body
```json
{
  "text": "The new update resolved all previous performance bottlenecks. Outstanding work!"
}
```

#### Response (`200 OK`)
```json
{
  "sentiment": "positive",
  "confidence": 0.9888,
  "scores": {
    "positive": 0.9888,
    "negative": 0.0043,
    "neutral": 0.0069
  }
}
```

#### cURL Example
```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product. It works perfectly!"}'
```

---

### 2. Batch Text Analysis (Up to 10 Texts)
- **Endpoint**: `POST /api/analyze/batch`
- **Headers**: `Content-Type: application/json`

#### Request Body
```json
{
  "texts": [
    "I love this product. It works perfectly!",
    "This product is terrible and disappointing.",
    "The package arrived on Tuesday."
  ]
}
```

#### Response (`200 OK`)
```json
{
  "results": [
    {
      "sentiment": "positive",
      "confidence": 0.9888,
      "scores": {
        "positive": 0.9888,
        "negative": 0.0043,
        "neutral": 0.0069
      }
    },
    {
      "sentiment": "negative",
      "confidence": 0.947,
      "scores": {
        "positive": 0.0078,
        "negative": 0.947,
        "neutral": 0.0452
      }
    },
    {
      "sentiment": "neutral",
      "confidence": 0.6791,
      "scores": {
        "positive": 0.3163,
        "negative": 0.0046,
        "neutral": 0.6791
      }
    }
  ]
}
```

#### cURL Example
```bash
curl -X POST "http://localhost:8000/api/analyze/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "I love this product!",
      "Customer service was terrible.",
      "The order status is pending."
    ]
  }'
```

---

### 3. Error Handling Examples

#### Validation Error (e.g. Batch > 10 items or whitespace string)
- **Status Code**: `422 Unprocessable Content`
```json
{
  "detail": "Input validation failed. Please check your request payload.",
  "errors": [
    {
      "field": "body -> texts",
      "message": "Value error, Batch size exceeds maximum limit of 10 items."
    }
  ]
}
```

---

## 🧪 Running Tests & Evaluation

### Run Test Suite
The automated test suite runs 16 test cases covering endpoints, edge-cases, whitespace sanitization, batch length bounds, and real pipeline integration.
```bash
pytest tests/ -v
```

### Run Model Evaluation
Compute accuracy, classification report, confusion matrix, and latency benchmarks:
```bash
python evaluate.py
```

---

## 🐳 Docker Deployment

Build and run using Docker:
```bash
# Build the Docker image
docker build -t nexoramind-sentiment-api .

# Run the container on port 8000
docker run -d -p 8000:8000 --name sentiment-service nexoramind-sentiment-api

# Check logs
docker logs -f sentiment-service
```

---

## 📋 Evaluation Criteria Verification Checklist

- [x] **Model Accuracy**: State-of-the-art RoBERTa sentiment classifier with 100% accuracy and 1.0 F1-score on benchmark dataset.
- [x] **API Design & Documentation**: RESTful FastAPI design, clean schema separation, interactive OpenAPI/Swagger `/docs` and ReDoc `/redoc`.
- [x] **Output Specifications**: Compliant `sentiment`, `confidence`, and `scores` (positive, negative, neutral).
- [x] **Batch Specifications**: `POST /api/analyze/batch` strictly restricted to `1 <= len(texts) <= 10`.
- [x] **Code Organization**: Modular separation of concerns (`config`, `schemas`, `model`, `router`, `main`, `tests`).
- [x] **Error Handling**: Custom exception handlers for HTTP 422 validation and HTTP 500 runtime errors.
- [x] **Model Metrics**: Complete evaluation report in README and exported `metrics.json`.
