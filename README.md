# Multi-Task NLP API with BERT & MLflow

A production-ready NLP system that performs Sentiment Analysis, Named Entity Recognition (NER), and Question Answering (QA) using a single fine-tuned Multi-Task BERT model.

## Features
- **Multi-Task Learning**: Shared DistilBERT encoder with task-specific classification heads.
- **MLOps with MLflow**: Experiment tracking, metrics logging, and artifact management.
- **High-Performance Inference**: Model exported to ONNX format.
- **Production API**: FastAPI serving the ONNX model with robust Pydantic data validation.
- **Monitoring**: Prometheus `/metrics` exposing API latency and request counts.
- **Fully Containerized**: 1-command setup via Docker Compose.

## Project Architecture
1. **Data**: Automatically downloaded and processed via Hugging Face `datasets` (SST-2, CoNLL-2003, SQuAD).
2. **Train**: `train.py` fine-tunes the PyTorch model, evaluates it, converts it to ONNX, and logs it to a local MLflow tracking server.
3. **Serve**: `main.py` (FastAPI) automatically dynamically loads the latest ONNX model from MLflow and exposes REST endpoints.

## Setup Instructions

1. **Clone the repository** (or navigate to this directory)
2. **Review `.env.example`**
   Ensure `.env` matches `.env.example`.
3. **Run with Docker Compose**
   ```bash
   docker-compose up --build
   ```
   *This command will:*
   - Start the MLflow tracking UI at `http://localhost:5000`.
   - Build the NLP API container.
   - Run the data processing script (downloads and processes datasets).
   - Fine-tune the multi-task model, save metrics and artifacts to MLflow.
   - Start the API server at `http://localhost:8000` loading the ONNX model.

*(Note: Training may take a few minutes on startup. You can check the MLflow UI or container logs for progress).*

## API Endpoints

### 1. Health & Metrics
- `GET /health` -> Returns `{"status": "ok"}`
- `GET /metrics` -> Returns Prometheus metrics including `api_requests_total` and `api_request_latency_seconds`.

### 2. Predict Sentiment
**Endpoint**: `POST /predict/sentiment`
**Request:**
```json
{
  "text": "I absolutely love this new design!"
}
```
**Response:**
```json
{
  "text": "I absolutely love this new design!",
  "sentiment": "positive",
  "score": 4.52
}
```

### 3. Predict NER
**Endpoint**: `POST /predict/ner`
**Request:**
```json
{
  "text": "Apple is looking at buying U.K. startup for $1 billion."
}
```
**Response:**
```json
{
  "text": "Apple is looking at buying U.K. startup for $1 billion.",
  "entities": [
    {
      "text": "Apple",
      "type": "B-ORG",
      "start_char": 0,
      "end_char": 5
    }
  ]
}
```

### 4. Predict QA
**Endpoint**: `POST /predict/qa`
**Request:**
```json
{
  "context": "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
  "question": "What is the speed of light?"
}
```
**Response:**
```json
{
  "context": "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
  "question": "What is the speed of light?",
  "answer": {
    "text": "299, 792 kilometers per second",
    "start_char": 48,
    "end_char": 78,
    "score": 12.35
  }
}
```
