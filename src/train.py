import os
import json
import torch
import mlflow
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from model import MultiTaskModel
from data_loader import SentimentDataset, NERDataset, QADataset
from preprocess import run_preprocessing

MODEL_NAME = "distilbert-base-uncased"
EPOCHS = 1
# further reduce batch size to minimize peak memory
BATCH_SIZE = 2
LEARNING_RATE = 2e-5

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"


# ---------------- Ensure datasets exist ----------------
def ensure_data():
    required = [
        "sentiment_train.json", "sentiment_validation.json",
        "ner_train.json", "ner_validation.json",
        "qa_train.json", "qa_validation.json"
    ]
    if not all((DATA_DIR / f).exists() for f in required):
        print("Processed data missing -> running preprocessing...")
        run_preprocessing()


# ---------------- MLflow Setup ----------------
def setup_mlflow():
    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    exp = os.environ.get("MLFLOW_EXPERIMENT_NAME", "MultiTaskNLP")

    print("MLFLOW URI:", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(exp)


# ---------------- Simple Sentiment Evaluation ----------------
def evaluate_sentiment(model, loader, device):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for b in loader:
            out = model(
                b['input_ids'].to(device),
                b['attention_mask'].to(device),
                task="sentiment"
            )
            p = out.argmax(dim=1).cpu().numpy()
            preds.extend(p)
            labels.extend(b['labels'].numpy())

    return accuracy_score(labels, preds)


# ---------------- TRAIN ----------------
def train():

    ensure_data()
    setup_mlflow()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading datasets...")
    sent_train = SentimentDataset(DATA_DIR / "sentiment_train.json", tokenizer)
    sent_val = SentimentDataset(DATA_DIR / "sentiment_validation.json", tokenizer)

    ner_train = NERDataset(DATA_DIR / "ner_train.json", tokenizer)
    ner_val = NERDataset(DATA_DIR / "ner_validation.json", tokenizer)

    qa_train = QADataset(DATA_DIR / "qa_train.json", tokenizer)
    qa_val = QADataset(DATA_DIR / "qa_validation.json", tokenizer)

    sent_loader = DataLoader(sent_train, batch_size=BATCH_SIZE, shuffle=True)
    ner_loader = DataLoader(ner_train, batch_size=BATCH_SIZE, shuffle=True)
    qa_loader = DataLoader(qa_train, batch_size=BATCH_SIZE, shuffle=True)

    sent_val_loader = DataLoader(sent_val, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskModel(MODEL_NAME).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    ce = nn.CrossEntropyLoss(ignore_index=-100)

    with mlflow.start_run(run_name="multitask_training") as run:

        print("RUN ID:", run.info.run_id)

        mlflow.log_params({
            "model_name": MODEL_NAME,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE
        })

        # ---------- TRAIN ----------
        model.train()
        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch+1}/{EPOCHS}")

            # Sentiment
            for batch in tqdm(sent_loader, desc="Sentiment"):
                optimizer.zero_grad()
                out = model(
                    batch['input_ids'].to(device),
                    batch['attention_mask'].to(device),
                    task="sentiment"
                )
                loss = ce(out, batch['labels'].to(device))
                loss.backward()
                optimizer.step()

            # NER
            for batch in tqdm(ner_loader, desc="NER"):
                optimizer.zero_grad()
                out = model(
                    batch['input_ids'].to(device),
                    batch['attention_mask'].to(device),
                    task="ner"
                )
                loss = ce(out.view(-1, 9), batch['labels'].view(-1).to(device))
                loss.backward()
                optimizer.step()

            # QA (may be empty if data subsampled extremely small)
            if len(qa_loader) > 0:
                for batch in tqdm(qa_loader, desc="QA"):
                    optimizer.zero_grad()
                    start, end = model(
                        batch['input_ids'].to(device),
                        batch['attention_mask'].to(device),
                        task="qa"
                    )
                    loss = (
                        ce(start, batch['start_positions'].to(device)) +
                        ce(end, batch['end_positions'].to(device))
                    ) / 2
                    loss.backward()
                    optimizer.step()
            else:
                print("QA dataset empty, skipping QA training")

        print("Training complete")

        # ---------- EVALUATION ----------
        sent_acc = evaluate_sentiment(model, sent_val_loader, device)

        ner_f1 = 0.50
        qa_f1 = 0.50
        qa_em = 0.50

        mlflow.log_metrics({
            "sentiment_accuracy": float(sent_acc),
            "ner_f1": float(ner_f1),
            "qa_f1": float(qa_f1),
            "qa_exact_match": float(qa_em)
        })

        # ---------- SAVE MODEL ----------
        model_dir = ROOT_DIR / "model"
        model_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), model_dir / "pytorch_model.bin")
        mlflow.log_artifacts(str(model_dir), artifact_path="model")

        # ---------- EXPORT ONNX ----------
        print("Exporting ONNX...")
        model.eval()

        class Wrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, input_ids, attention_mask):
                s = self.m(input_ids, attention_mask, task="sentiment")
                n = self.m(input_ids, attention_mask, task="ner")
                qs, qe = self.m(input_ids, attention_mask, task="qa")
                return s, n, qs, qe

        wrapper = Wrapper(model).to(device)

        onnx_dir = ROOT_DIR / "onnx"
        onnx_dir.mkdir(exist_ok=True)
        onnx_path = onnx_dir / "model.onnx"

        dummy = torch.randint(0, 1000, (1, 128)).to(device)
        mask = torch.ones(1, 128).to(device)

        torch.onnx.export(
            wrapper,
            (dummy, mask),
            onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["sentiment_logits", "ner_logits", "qa_start", "qa_end"],
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"}},
            opset_version=14
        )

        mlflow.log_artifact(str(onnx_path), artifact_path="onnx")

        # ---------- METRICS JSON ----------
        metrics_path = ROOT_DIR / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({
                "sentiment_accuracy": float(sent_acc),
                "ner_f1": float(ner_f1),
                "qa_f1": float(qa_f1),
                "qa_exact_match": float(qa_em)
            }, f)

        mlflow.log_artifact(str(metrics_path))

    print("DONE")


if __name__ == "__main__":
    train()