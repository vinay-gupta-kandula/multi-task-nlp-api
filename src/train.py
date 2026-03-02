import os
import torch
import mlflow
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from pathlib import Path
from tqdm import tqdm

from model import MultiTaskModel
from data_loader import SentimentDataset, NERDataset, QADataset
from preprocess import run_preprocessing

# ---------------- CONFIG ----------------

MODEL_NAME = "distilbert-base-uncased"
EPOCHS = 1
BATCH_SIZE = 8
LR = 2e-5

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

# ---------------- DATA CHECK ----------------

def ensure_data():
    files = [
        "sentiment_train.json", "sentiment_validation.json",
        "ner_train.json", "ner_validation.json",
        "qa_train.json", "qa_validation.json"
    ]

    if not all((DATA / f).exists() for f in files):
        print("Running preprocessing...")
        run_preprocessing()

# ---------------- MLFLOW ----------------

def setup_mlflow():
    uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    exp = os.getenv("MLFLOW_EXPERIMENT_NAME", "MultiTaskNLP")

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(exp)
    print("MLFLOW URI:", uri)

# ---------------- TRAIN ----------------

def train():

    ensure_data()
    setup_mlflow()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading datasets...")
    sent_train = SentimentDataset(DATA / "sentiment_train.json", tokenizer)
    ner_train = NERDataset(DATA / "ner_train.json", tokenizer)
    qa_train = QADataset(DATA / "qa_train.json", tokenizer)

    sent_loader = DataLoader(sent_train, batch_size=BATCH_SIZE, shuffle=True)
    ner_loader = DataLoader(ner_train, batch_size=BATCH_SIZE, shuffle=True)
    qa_loader = DataLoader(qa_train, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskModel(MODEL_NAME).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss(ignore_index=-100)

    with mlflow.start_run(run_name="multitask_training") as run:

        print("RUN ID:", run.info.run_id)

        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch+1}/{EPOCHS}")
            model.train()

            for s, n, q in tqdm(
                zip(sent_loader, ner_loader, qa_loader),
                total=min(len(sent_loader), len(ner_loader), len(qa_loader))
            ):

                optimizer.zero_grad()

                # -------- SENTIMENT --------
                sent_logits = model(
                    s["input_ids"].to(device),
                    s["attention_mask"].to(device),
                    task="sentiment"
                )

                sent_loss = ce(
                    sent_logits,
                    s["labels"].to(device)
                )

                # -------- NER --------
                ner_logits = model(
                    n["input_ids"].to(device),
                    n["attention_mask"].to(device),
                    task="ner"
                )

                ner_loss = ce(
                    ner_logits.view(-1, 9),
                    n["labels"].view(-1).to(device)
                )

                # -------- QA --------
                start_logits, end_logits = model(
                    q["input_ids"].to(device),
                    q["attention_mask"].to(device),
                    task="qa"
                )

                qa_loss = (
                    ce(start_logits, q["start_positions"].to(device)) +
                    ce(end_logits, q["end_positions"].to(device))
                ) / 2

                # -------- TOTAL LOSS --------
                loss = sent_loss + ner_loss + qa_loss
                loss.backward()
                optimizer.step()

        print("Training complete")

        # ---------------- SAVE MODEL ----------------
        model_dir = ROOT / "model"
        model_dir.mkdir(exist_ok=True)

        torch.save(model.state_dict(), model_dir / "pytorch_model.bin")
        mlflow.log_artifacts(str(model_dir), "model")

        # ---------------- ONNX EXPORT ----------------
        print("Exporting ONNX...")
        model.eval()

        dummy_ids = torch.randint(0, 1000, (1, 128)).to(device)
        dummy_mask = torch.ones(1, 128).to(device)

        onnx_dir = ROOT / "onnx"
        onnx_dir.mkdir(exist_ok=True)
        onnx_path = onnx_dir / "model.onnx"

        torch.onnx.export(
            model,
            (dummy_ids, dummy_mask, "sentiment"),
            onnx_path,
            input_names=["input_ids", "attention_mask", "task"],
            output_names=["output"],
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"}},
            opset_version=14
        )

        mlflow.log_artifact(str(onnx_path), "onnx")

    print("DONE")


if __name__ == "__main__":
    train()