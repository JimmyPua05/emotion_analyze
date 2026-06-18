"""Fine-tune DistilBERT for the advanced NLP bonus mark.

DistilBERT counts as advanced NLP because it is a transformer model. Unlike
CountVectorizer or TF-IDF, it reads each word in the context of the full
sentence.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from src.data_preprocessing import LABEL_NAMES, load_processed_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "models" / "distilbert_emotion"
RESULTS_PATH = PROJECT_ROOT / "models" / "classical" / "model_results.csv"
DETAILS_PATH = PROJECT_ROOT / "models" / "classical" / "model_results_details.json"
SUMMARY_PATH = OUTPUT_DIR / "evaluation_summary.json"


def compute_classification_metrics(logits, labels) -> dict[str, float]:
    """Compute transformer evaluation metrics without extra metric packages."""

    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }


def write_distilbert_comparison_results(
    logits,
    labels,
    rows_evaluated: int | None = None,
    results_path: Path | str = RESULTS_PATH,
    details_path: Path | str = DETAILS_PATH,
    summary_path: Path | str = SUMMARY_PATH,
) -> dict[str, float | int | str]:
    """Save DistilBERT metrics beside the classical model comparison files.

    The Streamlit visualizations read `model_results.csv` and
    `model_results_details.json`. This function keeps the transformer bonus
    model visible in the same charts, table, and confusion-matrix selector.
    """

    logits_array = np.asarray(logits)
    label_array = np.asarray(labels)
    prediction_ids = np.argmax(logits_array, axis=-1)
    y_true = [LABEL_NAMES[int(label_id)] for label_id in label_array]
    y_pred = [LABEL_NAMES[int(prediction_id)] for prediction_id in prediction_ids]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABEL_NAMES,
        average="weighted",
        zero_division=0,
    )
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        labels=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=LABEL_NAMES).tolist()
    evaluated_rows = int(rows_evaluated if rows_evaluated is not None else len(label_array))

    summary = {
        "display_name": "DistilBERT",
        "feature_method": "Transformer",
        "classifier": "DistilBERT",
        "evaluation_split": "Same 20% stratified split used by src.train_transformer, random_state=42",
        "rows_evaluated": evaluated_rows,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    row = {
        "model_id": "distilbert",
        "display_name": "DistilBERT",
        "feature_method": "Transformer",
        "classifier": "DistilBERT",
        "model_path": "models/distilbert_emotion",
        "ngram_range": "not applicable",
        "accuracy": summary["accuracy"],
        "precision": summary["precision"],
        "recall": summary["recall"],
        "f1": summary["f1"],
    }

    results_path = Path(results_path)
    details_path = Path(details_path)
    summary_path = Path(summary_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if results_path.exists():
        results = pd.read_csv(results_path)
    else:
        results = pd.DataFrame(
            columns=[
                "model_id",
                "display_name",
                "feature_method",
                "classifier",
                "model_path",
                "ngram_range",
                "accuracy",
                "precision",
                "recall",
                "f1",
            ]
        )
    results = results[results["display_name"].astype(str) != "DistilBERT"].copy()
    new_row = pd.DataFrame([row])
    results = new_row if results.empty else pd.concat([results, new_row], ignore_index=True)
    results.to_csv(results_path, index=False)

    if details_path.exists():
        details = json.loads(details_path.read_text(encoding="utf-8"))
    else:
        details = []
    details = [item for item in details if item.get("display_name") != "DistilBERT"]
    details.append(
        {
            "model_id": "distilbert",
            **summary,
            "classification_report": report,
            "confusion_matrix": matrix,
        }
    )
    details_path.write_text(json.dumps(details, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary



def evaluation_strategy_argument(training_arguments_class) -> dict[str, str]:
    """Return the evaluation strategy keyword supported by this Transformers version."""

    parameters = inspect.signature(training_arguments_class.__init__).parameters
    if "eval_strategy" in parameters:
        return {"eval_strategy": "epoch"}
    return {"evaluation_strategy": "epoch"}

def fine_tune_distilbert(epochs: int = 2, batch_size: int = 8, sample_size: int | None = None) -> Path:
    """Fine-tune DistilBERT and save it for the Streamlit app."""

    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

    try:
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise ImportError("Install dependencies from requirements-transformer.txt first.") from exc

    df = load_processed_dataset()[["text", "emotion"]].dropna().copy()
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)

    label_to_id = {label: index for index, label in enumerate(LABEL_NAMES)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    df["label"] = df["emotion"].map(label_to_id)

    train_df, eval_df = train_test_split(
        df[["text", "label"]],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    eval_dataset = Dataset.from_pandas(eval_df.reset_index(drop=True))

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize(batch: dict) -> dict:
        return tokenizer(batch["text"], truncation=True)

    train_dataset = train_dataset.map(tokenize, batched=True)
    eval_dataset = eval_dataset.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(LABEL_NAMES),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    def compute_metrics(eval_pred: tuple) -> dict:
        logits, labels = eval_pred
        return compute_classification_metrics(logits, labels)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        **evaluation_strategy_argument(TrainingArguments),
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    prediction_output = trainer.predict(eval_dataset)
    write_distilbert_comparison_results(
        prediction_output.predictions,
        prediction_output.label_ids,
        rows_evaluated=len(eval_df),
    )
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    return OUTPUT_DIR


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for transformer training."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sample-size", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fine_tune_distilbert(epochs=args.epochs, batch_size=args.batch_size, sample_size=args.sample_size)

