import json
from datasets import load_dataset
from pathlib import Path

# trim dataset sizes for resource-constrained training
# original settings were 1000/200 but that caused OOM in the container
MAX_TRAIN = 100
MAX_VAL = 25


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def run_preprocessing():
    root_dir = Path(__file__).resolve().parent.parent
    processed_dir = root_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- SENTIMENT (SST2) ----------------
    print("Processing Sentiment (SST-2)...")
    sst2 = load_dataset("glue", "sst2")

    train_sst2 = sst2["train"].select(range(min(MAX_TRAIN, len(sst2["train"]))))
    val_sst2 = sst2["validation"].select(range(min(MAX_VAL, len(sst2["validation"]))))

    def format_sst2(dataset):
        return [{"text": row["sentence"], "label": int(row["label"])} for row in dataset]

    save_json(processed_dir / "sentiment_train.json", format_sst2(train_sst2))
    save_json(processed_dir / "sentiment_validation.json", format_sst2(val_sst2))

    # ---------------- NER ----------------
    print("Processing NER (CoNLL-2003)...")
    conll = load_dataset("eriktks/conll2003")

    train_conll = conll["train"].select(range(min(MAX_TRAIN, len(conll["train"]))))
    val_conll = conll["validation"].select(range(min(MAX_VAL, len(conll["validation"]))))

    def format_ner(dataset):
        formatted = []
        for row in dataset:
            formatted.append({
                "tokens": row["tokens"],
                "tags": [int(t) for t in row["ner_tags"]]
            })
        return formatted

    save_json(processed_dir / "ner_train.json", format_ner(train_conll))
    save_json(processed_dir / "ner_validation.json", format_ner(val_conll))

    # ---------------- QA ----------------
    print("Processing QA (SQuAD)...")
    squad = load_dataset("squad")

    train_qa = squad["train"].select(range(min(MAX_TRAIN, len(squad["train"]))))
    val_qa = squad["validation"].select(range(min(MAX_VAL, len(squad["validation"]))))

    def format_qa(dataset):
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

    print("Data processing complete!")


if __name__ == "__main__":
    run_preprocessing()