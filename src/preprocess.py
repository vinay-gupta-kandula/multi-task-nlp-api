import json
import os
from datasets import load_dataset
from pathlib import Path

# Trimming dataset sizes to ensure stability in resource-constrained environments
MAX_TRAIN = 100
MAX_VAL = 25

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

def run_preprocessing():
    # Setup paths relative to the project root
    root_dir = Path(__file__).resolve().parent.parent
    processed_dir = root_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- 1. SENTIMENT (Requirement: glue/sst2) ----------------
    print("Processing Sentiment (SST-2)...")
    sst2 = load_dataset("glue", "sst2")

    train_sst2 = sst2["train"].select(range(min(MAX_TRAIN, len(sst2["train"]))))
    val_sst2 = sst2["validation"].select(range(min(MAX_VAL, len(sst2["validation"]))))

    def format_sst2(dataset):
        # Schema: [{"text": "string", "label": "integer"}, ...]
        return [{"text": row["sentence"], "label": int(row["label"])} for row in dataset]

    save_json(processed_dir / "sentiment_train.json", format_sst2(train_sst2))
    save_json(processed_dir / "sentiment_validation.json", format_sst2(val_sst2))

    # ---------------- 2. NER (Requirement: conll2003) ----------------
    print("Processing NER (CoNLL-2003)...")
    # Using eriktks/conll2003 as a reliable mirror for the standard dataset
    conll = load_dataset("eriktks/conll2003")

    train_conll = conll["train"].select(range(min(MAX_TRAIN, len(conll["train"]))))
    val_conll = conll["validation"].select(range(min(MAX_VAL, len(conll["validation"]))))

    def format_ner(dataset):
        # Schema: [{"tokens": ["string"], "tags": ["integer"]}, ...]
        return [{"tokens": row["tokens"], "tags": [int(t) for t in row["ner_tags"]]} for row in dataset]

    save_json(processed_dir / "ner_train.json", format_ner(train_conll))
    save_json(processed_dir / "ner_validation.json", format_ner(val_conll))

    # ---------------- 3. QA (Requirement: squad) ----------------
    print("Processing QA (SQuAD)...")
    squad = load_dataset("squad")

    train_qa = squad["train"].select(range(min(MAX_TRAIN, len(squad["train"]))))
    val_qa = squad["validation"].select(range(min(MAX_VAL, len(squad["validation"]))))

    def format_qa(dataset):
        # Schema: [{"context": "string", "question": "string", "answers": {"text": ["string"], "answer_start": ["integer"]}}, ...]
        formatted = []
        for row in dataset:
            formatted.append({
                "context": row["context"],
                "question": row["question"],
                "answers": {
                    "text": [str(t) for t in row["answers"]["text"]],
                    "answer_start": [int(i) for i in row["answers"]["answer_start"]]
                }
            })
        return formatted

    save_json(processed_dir / "qa_train.json", format_qa(train_qa))
    save_json(processed_dir / "qa_validation.json", format_qa(val_qa))

    print(f"Data processing complete! Files saved to: {processed_dir}")

if __name__ == "__main__":
    run_preprocessing()