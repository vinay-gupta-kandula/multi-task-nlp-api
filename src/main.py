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

app = FastAPI()

# Prometheus metrics
REQUEST_COUNT = Counter("api_requests_total", "Total API Requests", ["endpoint"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "API Request Latency", ["endpoint"])

ort_session = None
tokenizer = None
MODEL_NAME = "distilbert-base-uncased"


class SentimentRequest(BaseModel):
    text: str


class NERRequest(BaseModel):
    text: str


class QARequest(BaseModel):
    context: str
    question: str


# ---------------- LOAD MODEL FROM MLFLOW ----------------
@app.on_event("startup")
def load_model():
    global ort_session, tokenizer

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    exp_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "MultiTaskNLP")

    print("Connecting to MLflow:", mlflow_uri)
    mlflow.set_tracking_uri(mlflow_uri)

    # wait for experiment
    experiment = None
    for _ in range(12):
        experiment = mlflow.get_experiment_by_name(exp_name)
        if experiment:
            break
        print("Waiting for experiment...")
        time.sleep(5)

    if not experiment:
        print("Experiment not found — API starting without model")
        return

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"]
    )

    if runs.empty:
        print("No MLflow runs found")
        return

    run_id = runs.iloc[0].run_id
    print("Loading run:", run_id)

    try:
        client = mlflow.tracking.MlflowClient()
        local_path = client.download_artifacts(run_id, "onnx/model.onnx")

        ort_session = ort.InferenceSession(
            local_path,
            providers=["CPUExecutionProvider"]
        )

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

        print("Model loaded successfully")

    except Exception as e:
        print("Model loading failed:", e)


# ---------------- METRICS MIDDLEWARE ----------------
@app.middleware("http")
async def add_prometheus_metrics(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time

    path = request.url.path
    if path.startswith("/predict"):
        REQUEST_COUNT.labels(endpoint=path).inc()
        REQUEST_LATENCY.labels(endpoint=path).observe(latency)

    return response


# ---------------- HEALTH ----------------
@app.get("/health")
def health_check():
    return {"status": "ok"}


# ---------------- PROMETHEUS ----------------
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---------------- SENTIMENT ----------------
@app.post("/predict/sentiment")
def predict_sentiment(req: SentimentRequest):
    if ort_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    inputs = tokenizer(req.text, return_tensors="np", truncation=True, max_length=128)

    outputs = ort_session.run(None, {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    })

    logits = outputs[0]
    score = float(logits.max())
    label = int(logits.argmax())
    sentiment = "positive" if label == 1 else "negative"

    return {
        "text": req.text,
        "sentiment": sentiment,
        "score": score
    }


# ---------------- NER ----------------
@app.post("/predict/ner")
def predict_ner(req: NERRequest):
    if ort_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    inputs = tokenizer(req.text, return_tensors="np", return_offsets_mapping=True, truncation=True, max_length=128)
    offsets = inputs.pop("offset_mapping")[0]

    outputs = ort_session.run(None, {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    })

    preds = outputs[1].argmax(axis=-1)[0]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    id2label = {0:"O",1:"B-PER",2:"I-PER",3:"B-ORG",4:"I-ORG",5:"B-LOC",6:"I-LOC",7:"B-MISC",8:"I-MISC"}
    entities = []

    for token, pred, off in zip(tokens, preds, offsets):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        label = id2label.get(int(pred), "O")
        if label != "O":
            entities.append({
                "text": token.replace("##",""),
                "type": label,
                "start_char": int(off[0]),
                "end_char": int(off[1])
            })

    return {
        "text": req.text,
        "entities": entities
    }


# ---------------- QA ----------------
@app.post("/predict/qa")
def predict_qa(req: QARequest):
    if ort_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    inputs = tokenizer(req.question, req.context, return_tensors="np", return_offsets_mapping=True, truncation=True, max_length=384)
    offsets = inputs.pop("offset_mapping")[0]

    outputs = ort_session.run(None, {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    })

    start_logits, end_logits = outputs[2], outputs[3]
    start_idx = int(start_logits.argmax())
    end_idx = int(end_logits.argmax())

    if start_idx <= end_idx < len(offsets):
        answer_ids = inputs["input_ids"][0][start_idx:end_idx+1]
        answer_text = tokenizer.decode(answer_ids)
        start_char = int(offsets[start_idx][0])
        end_char = int(offsets[end_idx][1])
    else:
        answer_text = ""
        start_char = 0
        end_char = 0

    score = float(start_logits[0, start_idx] + end_logits[0, end_idx])

    return {
        "context": req.context,
        "question": req.question,
        "answer": {
            "text": answer_text,
            "start_char": start_char,
            "end_char": end_char,
            "score": score
        }
    }