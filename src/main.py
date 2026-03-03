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

# Global variables for model and tokenizer
ort_session = None
tokenizer = None
MODEL_NAME = "distilbert-base-uncased"

# Requirement 8: Startup and Model Loading with retry logic
@app.on_event("startup")
def load_model():
    global ort_session, tokenizer
    
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    client = mlflow.tracking.MlflowClient()
    
    # Wait loop to ensure the model is ready in MLflow artifacts
    for i in range(20):
        try:
            exp = client.get_experiment_by_name("MultiTaskNLP")
            if exp:
                runs = client.search_runs(exp.experiment_id, order_by=["start_time DESC"])
                if runs:
                    run_id = runs[0].info.run_id
                    path = client.download_artifacts(run_id, "onnx/model.onnx")
                    ort_session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
                    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                    print(f"Model loaded successfully from run: {run_id}")
                    return
            print(f"Waiting for model in MLflow (Attempt {i+1}/20)...")
            time.sleep(15)
        except Exception as e:
            print(f"Retry loading model: {e}")
            time.sleep(15)
    
    raise RuntimeError("Model loading failed after several attempts.")

# Middleware for monitoring
@app.middleware("http")
async def monitor(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    if request.url.path.startswith("/predict"):
        REQUEST_COUNT.labels(path=request.url.path).inc()
        LATENCY.labels(path=request.url.path).observe(time.time() - start_time)
    return response

# Requirement 8: Health Check
@app.get("/health")
def health():
    return {"status": "ok"} if ort_session else Response(status_code=503)

# Requirement 12: Metrics Endpoint
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# --- TASK 1: SENTIMENT ANALYSIS ---
@app.post("/predict/sentiment")
def predict_sentiment(data: SentimentRequest):
    if not ort_session:
        raise HTTPException(status_code=503, detail="Model not ready")
        
    inputs = tokenizer(data.text, return_tensors="np", truncation=True, max_length=128)
    outputs = ort_session.run(None, {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64)
    })
    
    # Softmax normalization for numerical stability and probabilistic scores
    logits = outputs[0][0]
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    label_id = np.argmax(probs)
    
    return {
        "text": data.text,
        "sentiment": "positive" if label_id == 1 else "negative",
        "score": float(probs[label_id])
    }

# --- TASK 2: NAMED ENTITY RECOGNITION ---

@app.post("/predict/ner")
def predict_ner(data: NERRequest):
    if not ort_session:
        raise HTTPException(status_code=503, detail="Model not ready")
        
    inputs = tokenizer(data.text, return_tensors="np", truncation=True, return_offsets_mapping=True)
    offsets = inputs.pop("offset_mapping")[0]
    
    outputs = ort_session.run(None, {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64)
    })
    
    # predictions for token-level classification
    preds = np.argmax(outputs[1], axis=-1)[0]
    entities = []
    
    for p, o in zip(preds, offsets):
        label = ID2LABEL.get(int(p), "O")
        # Filter out 'O' labels and special tokens/empty spans
        if label != "O" and int(o[1]) > int(o[0]):
            entities.append({
                "text": data.text[int(o[0]):int(o[1])],
                "type": label,
                "start_char": int(o[0]),
                "end_char": int(o[1])
            })
            
    return {"text": data.text, "entities": entities}

# --- TASK 3: QUESTION ANSWERING ---
@app.post("/predict/qa")
def predict_qa(data: QARequest):
    if not ort_session:
        raise HTTPException(status_code=503, detail="Model not ready")
        
    inputs = tokenizer(data.question, data.context, return_tensors="np", truncation=True, return_offsets_mapping=True)
    offsets = inputs.pop("offset_mapping")[0]
    
    outputs = ort_session.run(None, {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64)
    })
    
    # QA Head returns start_logits (outputs[2]) and end_logits (outputs[3])
    s_idx, e_idx = int(outputs[2].argmax()), int(outputs[3].argmax())
    
    # Logic to extract answer text using character offsets
    if s_idx < len(offsets) and e_idx < len(offsets):
        ans_text = data.context[offsets[s_idx][0]:offsets[e_idx][1]]
        s_char, e_char = int(offsets[s_idx][0]), int(offsets[e_idx][1])
    else:
        ans_text, s_char, e_char = "", 0, 0
        
    return {
        "context": data.context,
        "question": data.question,
        "answer": {
            "text": ans_text,
            "start_char": s_char,
            "end_char": e_char,
            "score": float(outputs[2].max())
        }
    }