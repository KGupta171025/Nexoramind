# Nexoramind AI Engineer Internship

[![Internship Program](https://img.shields.io/badge/Nexoramind-AI%20Engineer%20Program-blue.svg)](https://app.nexoramind.tech)
[![Intern ID](https://img.shields.io/badge/Intern%20ID-NM1B64D9E0-orange.svg)](https://app.nexoramind.tech)
[![Progress](https://img.shields.io/badge/Progress-Task%2001%20Completed-brightgreen.svg)](https://github.com/KGupta171025/Nexoramind)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

This repository contains the codebase, technical artifacts, and documentation for the **6-Month AI Engineer Internship Program** at **Nexoramind**, authored by **Krishna Gupta** (`NM1B64D9E0`).

---

## 📌 Internship Curriculum & Task Tracker

| Task # | Task Title | Difficulty | Status | Documentation & Code |
| :---: | :--- | :---: | :---: | :---: |
| **01** | **Sentiment Analysis API** | Easy | ✅ **Completed** | [📂 View Task 01 Details](task01-sentiment-api/README.md) |
| **02** | **Image Classification Web App** | Medium | 🔒 *Locked* | *Coming soon* |
| **03** | **RAG-Based Document Q&A System** | Medium | 🔒 *Locked* | *Coming soon* |
| **04** | **AI-Powered Image Generation Pipeline** | Hard | 🔒 *Locked* | *Coming soon* |
| **05** | **Chatbot with Memory & Tool Use** | Hard | 🔒 *Locked* | *Coming soon* |
| **06** | **Task 06** | Advanced | 🔒 *Locked* | *Coming soon* |
| **07** | **Task 07** | Advanced | 🔒 *Locked* | *Coming soon* |
| **08** | **Task 08** | Capstone | 🔒 *Locked* | *Coming soon* |

---

## 🚀 Task 01: Sentiment Analysis API (Overview)

> **Objective:** Build a production-ready sentiment analysis REST API that classifies text as `positive`, `negative`, or `neutral` with confidence scores and per-class probability distributions.

### ✨ Highlights
- **Tech Stack:** Python 3, FastAPI, Hugging Face Transformers (`cardiffnlp/twitter-roberta-base-sentiment-latest`), PyTorch, Pydantic v2, Pytest, Docker.
- **Endpoints:**
  - `POST /api/analyze`: Single text classification.
  - `POST /api/analyze/batch`: Batch classification for up to 10 texts.
  - `GET /api/health`: Health monitoring & model metadata.
  - `GET /docs`: Interactive Swagger UI.
- **Metrics:**
  - **Accuracy:** 100.0% on benchmark evaluation set.
  - **Macro F1:** 1.0000 | **Weighted F1:** 1.0000
  - **Single Request Latency:** ~44.79 ms (P95: 54.05 ms)
  - **Batch Throughput:** ~60.58 items/sec

👉 **Full Documentation, API Specs, Benchmarks & Docker Setup:**
**[Read Task 01 Documentation](task01-sentiment-api/README.md)**

---

## 🛠️ Quickstart (Running Task 01 from Root)

### 1. Clone & Set Up
```bash
git clone https://github.com/KGupta171025/Nexoramind.git
cd Nexoramind

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate   # Linux/macOS: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the API Server
```bash
# Start server from root
python main.py
# Or using uvicorn directly:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Navigate to [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.

### 3. Run Automated Tests
```bash
pytest task01-sentiment-api/tests -v
```

### 4. Run Model Evaluation
```bash
python task01-sentiment-api/evaluate.py
```

---

## 👤 Author
- **Intern:** Krishna Gupta
- **Internship ID:** `NM1B64D9E0`
- **Program:** AI Engineer (6 Months)
- **Repository:** [https://github.com/KGupta171025/Nexoramind](https://github.com/KGupta171025/Nexoramind)
