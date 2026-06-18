"""Visualization helper functions for the Streamlit app."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pandas as pd

from src.data_preprocessing import STOPWORDS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "models" / "classical" / "model_results.csv"
DETAILS_PATH = PROJECT_ROOT / "models" / "classical" / "model_results_details.json"


def load_model_results(path: Path | str = RESULTS_PATH) -> pd.DataFrame:
    """Load model comparison metrics saved during training."""

    path = Path(path)
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "model_id",
                "display_name",
                "feature_method",
                "classifier",
                "accuracy",
                "precision",
                "recall",
                "f1",
            ]
        )
    return pd.read_csv(path)


def load_model_details(path: Path | str = DETAILS_PATH) -> list[dict]:
    """Load detailed reports such as confusion matrices."""

    path = Path(path)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def add_text_length(df: pd.DataFrame) -> pd.DataFrame:
    """Add a word-count column for text length analysis."""

    output = df.copy()
    output["text_length"] = output["clean_text"].fillna("").map(lambda text: len(str(text).split()))
    return output


def top_words_by_emotion(df: pd.DataFrame, emotion: str, top_n: int = 20) -> pd.DataFrame:
    """Return the most common cleaned words for one emotion label."""

    subset = df[df["emotion"] == emotion]["clean_text"].fillna("")
    counter: Counter[str] = Counter()
    for text in subset:
        counter.update(token for token in str(text).split() if token not in STOPWORDS and len(token) > 2)
    return pd.DataFrame(counter.most_common(top_n), columns=["word", "count"])


def unigram_bigram_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Compare average F1-score for unigram and bigram feature groups."""

    if results.empty or "feature_method" not in results or "f1" not in results:
        return pd.DataFrame(columns=["ngram_type", "f1"])

    ngram_rows = results[
        results["feature_method"].isin(
            ["Count Unigram", "Count Bigram", "TF-IDF Unigram", "TF-IDF Bigram"]
        )
    ].copy()
    if ngram_rows.empty:
        return pd.DataFrame(columns=["ngram_type", "f1"])

    ngram_rows["ngram_type"] = ngram_rows["feature_method"].map(
        lambda value: "Bigram" if "Bigram" in value else "Unigram"
    )
    return (
        ngram_rows.groupby("ngram_type", as_index=False)["f1"]
        .mean()
        .sort_values("ngram_type")
        .reset_index(drop=True)
    )
