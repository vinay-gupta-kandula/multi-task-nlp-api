# Multi-Task NLP API (BERT + MLflow + FastAPI + ONNX)

A production-style Natural Language Processing system that performs **Sentiment Analysis**, **Named Entity Recognition**, and **Question Answering** using a single shared BERT-based model.
The project demonstrates an end-to-end **MLOps workflow**: data → training → experiment tracking → model registry → optimized inference → monitored API.

---

## Problem Statement

Typical NLP projects solve only one task (classification OR NER OR QA).
In real industry systems, multiple NLP capabilities must run together efficiently.

This project implements a **Multi-Task Learning architecture** where a single transformer encoder learns shared language understanding and supports multiple downstream tasks simultaneously.

---

## Key Features

### Machine Learning

* Multi-Task Learning with shared DistilBERT encoder
* Task-specific heads for Sentiment, NER, and QA
* Fine-tuned on real datasets (SST-2, CoNLL-2003, SQuAD)
* Automatic preprocessing pipeline

### MLOps

* MLflow experiment tracking
* Automatic training on container startup
* Artifact storage (model + metrics)
* Model version loading from MLflow

### Production Inference

* Model exported to ONNX (fast inference)
* FastAPI serving REST endpoints
* Pydantic validation
* Health checks
* Prometheus monitoring metrics

### DevOps

* Fully containerized
* One-command execution
* Reproducible environment

---

## System Architecture

```
Datasets → Preprocessing → Multi-Task Training → MLflow Tracking
        → ONNX Export → FastAPI Inference → Monitoring
```

### Workflow

1. Docker starts MLflow tracking server
2. Data automatically downloads & preprocesses
3. Multi-task model trains
4. Metrics + artifacts stored in MLflow
5. ONNX model exported
6. API loads latest model from MLflow
7. Endpoints ready for prediction

---

## Project Structure

```
multi-task-nlp-api/
│
├── src/
│   ├── train.py              # training + logging + ONNX export
│   ├── main.py               # FastAPI inference server
│   ├── model.py              # multi-task architecture
│   ├── data_loader.py        # dataset classes
│   ├── preprocess.py         # dataset processing
│   └── inference.py          # prediction logic
│
├── data/                     # processed datasets (auto-generated)
├── model/                    # saved PyTorch model
├── onnx/                     # exported ONNX model
├── mlruns/                   # MLflow experiments
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Datasets Used

| Task                     | Dataset    |
| ------------------------ | ---------- |
| Sentiment Analysis       | SST-2      |
| Named Entity Recognition | CoNLL-2003 |
| Question Answering       | SQuAD      |

Datasets are downloaded automatically during first run.

---

## How to Run

### Step 1 — Configure Environment

Create `.env` file (already matches `.env.example`)

```
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_EXPERIMENT_NAME=MultiTaskNLP
```

---

### Step 2 — Start the System

```bash
docker-compose up --build
```

This will automatically:

1. Start MLflow UI → [http://localhost:5000](http://localhost:5000)
2. Download datasets
3. Train model
4. Log metrics
5. Export ONNX model
6. Launch API → [http://localhost:8000](http://localhost:8000)

Training runs only on first startup. Later runs reuse artifacts.

---

## API Endpoints

---

### Health Check

```
GET /health
```

Response:

```
{"status": "ok"}
```

---

### Sentiment Analysis

```
POST /predict/sentiment
```

Request:

```json
{
  "text": "I love this project"
}
```

Response:

```json
{
  "text": "I love this project",
  "sentiment": "positive",
  "score": 4.52
}
```

---

### Named Entity Recognition

```
POST /predict/ner
```

Request:

```json
{
  "text": "Apple is buying a startup in London"
}
```

Response:

```json
{
  "entities": [
    {
      "text": "Apple",
      "type": "ORG",
      "start_char": 0,
      "end_char": 5
    }
  ]
}
```

---

### Question Answering

```
POST /predict/qa
```

Request:

```json
{
  "context": "The speed of light is 299,792 km per second.",
  "question": "What is the speed of light?"
}
```

Response:

```json
{
  "answer": {
    "text": "299,792 km per second",
    "start_char": 22,
    "end_char": 43,
    "score": 12.3
  }
}
```

---

## Monitoring

Prometheus metrics available:

```
GET /metrics
```

Includes:

* Request count
* Latency
* Endpoint usage

---

## MLflow Tracking

Open:

```
http://localhost:5000
```

You can view:

* Parameters
* Metrics
* Artifacts
* ONNX model

---

## Model Details

| Component     | Description                                          |
| ------------- | ---------------------------------------------------- |
| Encoder       | DistilBERT                                           |
| Training Type | Multi-Task Learning                                  |
| Tasks         | Classification + Sequence Labeling + Span Prediction |
| Inference     | ONNX Runtime                                         |

---

## What This Project Demonstrates

* Multi-Task Deep Learning
* Model Lifecycle Management
* Experiment Tracking
* Production API Deployment
* Model Optimization
* Observability & Monitoring
* Reproducible ML Systems

---

## Why This Matters

Instead of deploying three separate models, a single shared encoder:

* reduces memory usage
* improves inference speed
* shares language understanding
* simplifies deployment

This mirrors real industry NLP systems (search engines, assistants, chat platforms).

---

## Future Improvements

* GPU training support
* Model registry versioning
* CI/CD pipeline
* Automatic retraining
* Distributed inference

---

✅ **End-to-End ML System: Data → Training → Tracking → Serving → Monitoring**

---


