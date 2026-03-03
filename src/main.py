import os
import time
import mlflow
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import AutoTokenizer
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter, Histogram
from fastapi.responses import Response

# Initialize FastAPI app
app = FastAPI(title="Multi-Task NLP API")

# Requirement 12: Prometheus instrumentation
REQUEST_COUNT = Counter("api_requests_total", "Total Requests", ["path"])
LATENCY = Histogram("api_request_latency_seconds", "Latency", ["path"])

# Requirement 9, 10, 11: Request Schemas
class SentimentRequest(BaseModel):
    text: str

class NERRequest(BaseModel):
    text: str

class QARequest(BaseModel):
    context: str
    question: str

# NER Label Mapping (CoNLL-2003 standard)
ID2LABEL = {
    0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 
    4: "I-ORG", 5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC"
}

# Global variables
ort_session = None
tokenizer = None
MODEL_NAME = "distilbert-base-uncased"

# Requirement 8: Startup and Model Loading with retry logic
@app.on_event("startup")
def load_model():
    global ort_session, tokenizer
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    client = mlflow.tracking.MlflowClient()
    
    for i in range(20):
        try:
            exp = client.get_experiment_by_name("MultiTaskNLP")
            if exp:
                runs = client.search_runs(exp.experiment_id, order_by=["start_time DESC"])
                if runs:
                    path = client.download_artifacts(runs[0].info.run_id, "onnx/model.onnx")
                    ort_session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
                    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                    print("Model loaded successfully!")
                    return
            print(f"Waiting for model in MLflow (Attempt {i+1}/20)...")
            time.sleep(15)
        except Exception as e:
            print(f"Retry loading model: {e}")
            time.sleep(15)
    raise RuntimeError("Model loading failed.")

# Middleware for monitoring
@app.middleware("http")
async def monitor(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    if request.url.path.startswith("/predict"):
        REQUEST_COUNT.labels(path=request.url.path).inc()
        LATENCY.labels(path=request.url.path).observe(time.time() - start)
    return response

# Requirement 8: Health Check
@app.get("/health")
def health():
    return {"status": "ok"} if ort_session else Response(status_code=503)

# Requirement 12: Metrics Endpoint
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# --- SENTIMENT ---
@app.post("/predict/sentiment")
def predict_sentiment(data: SentimentRequest):
    if not ort_session: raise HTTPException(status_code=503, detail="Model not ready")
    inputs = tokenizer(data.text, return_tensors="np", truncation=True, max_length=128)
    outputs = ort_session.run(None, {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64)
    })
    logits = outputs[0][0]
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    label_id = np.argmax(probs)
    return {
        "text": data.text,
        "sentiment": "positive" if label_id == 1 else "negative",
        "score": float(probs[label_id])
    }

# --- NER ---
@app.post("/predict/ner")
def predict_ner(data: NERRequest):
    if not ort_session: raise HTTPException(status_code=503, detail="Model not ready")
    inputs = tokenizer(data.text, return_tensors="np", truncation=True, return_offsets_mapping=True)
    offsets = inputs.pop("offset_mapping")[0]
    outputs = ort_session.run(None, {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64)
    })
    predictions = np.argmax(outputs[1], axis=-1)[0]
    entities = []
    for i, pred in enumerate(predictions):
        label = ID2LABEL.get(int(pred), "O")
        if label != "O":
            start, end = offsets[i]
            if start == end: continue
            entities.append({
                "text": data.text[int(start):int(end)],
                "type": label,
                "start_char": int(start),
                "end_char": int(end)
            })
    return {"text": data.text, "entities": entities}

# --- QA ---
@app.post("/predict/qa")
def predict_qa(data: QARequest):
    if not ort_session: raise HTTPException(status_code=503, detail="Model not ready")
    inputs = tokenizer(data.question, data.context, return_tensors="np", truncation=True, return_offsets_mapping=True)
    offsets = inputs.pop("offset_mapping")[0]
    outputs = ort_session.run(None, {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64)
    })
    start_idx, end_idx = int(outputs[2].argmax()), int(outputs[3].argmax())
    if start_idx < len(offsets) and end_idx < len(offsets):
        ans_text = data.context[offsets[start_idx][0]:offsets[end_idx][1]]
        start_char, end_char = int(offsets[start_idx][0]), int(offsets[end_idx][1])
    else:
        ans_text, start_char, end_char = "", 0, 0
    return {
        "context": data.context,
        "question": data.question,
        "answer": {
            "text": ans_text,
            "start_char": start_char,
            "end_char": end_char,
            "score": float(outputs[2].max())
        }
    }