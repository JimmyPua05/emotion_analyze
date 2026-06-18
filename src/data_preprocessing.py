"""Data loading and text preprocessing for emotion classification.

The app uses the DAIR Emotion dataset format with these columns: `text`,
`emotion`, and `clean_text`. This module keeps preprocessing simple and
explainable for a class presentation.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "emotion_clean.csv"

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# A small built-in stopword list avoids requiring an NLTK download.
STOPWORDS = {
    "a",
    "am",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "so",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "with",
    "you",
    "your",
}


def lemmatize_token(token: str) -> str:
    """Normalize common English suffixes without external downloads.

    The assignment asks for stemming or lemmatization. A full WordNet lemmatizer
    needs extra downloaded data, so this project uses a small rule-based fallback
    that is easy to explain in a presentation. It handles common social media
    forms such as `feeling` -> `feel`, `loved` -> `love`, and `studies` -> `study`.
    """

    token = token.lower().strip()
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith("ing") and len(token) > 5:
        root = token[:-3]
        if len(root) > 2 and root[-1] == root[-2]:
            root = root[:-1]
        return root
    if token.endswith("ed") and len(token) > 4:
        root = token[:-2]
        if root.endswith(("v", "t")):
            return f"{root}e"
        return root
    if token.endswith("es") and len(token) > 4 and not token.endswith(("ses", "xes")):
        return token[:-2]
    if token.endswith("s") and len(token) > 4 and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def clean_text(text: str) -> str:
    """Return a normalized version of one social media post.

    URLs, mentions, hashtag symbols, numbers, punctuation, stopwords, and extra
    spaces are removed. Remaining words are lightly lemmatized/stemmed so forms
    like `feeling` and `feels` become more consistent for classical models.
    """

    text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [lemmatize_token(token) for token in text.split() if token not in STOPWORDS]
    return " ".join(tokens)


def preprocess_text(text: str) -> str:
    """Clean one text input before prediction."""

    return clean_text(text)


def load_processed_dataset(path: Path | str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load the processed dataset used by the Streamlit app."""

    df = pd.read_csv(path)
    required_columns = {"text", "emotion", "clean_text"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Processed dataset is missing columns: {sorted(missing)}")
    return df


def prepare_dataset(records: Iterable[dict] | None = None) -> pd.DataFrame:
    """Create the processed dataset from records or Hugging Face data."""

    if records is None:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("Install `datasets` to download the Emotion dataset.") from exc

        dataset = load_dataset("dair-ai/emotion")
        label_names = dataset["train"].features["label"].names
        rows = []
        for split in dataset:
            for item in dataset[split]:
                rows.append({"text": item["text"], "emotion": label_names[item["label"]]})
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(records)

    if "text" not in df or "emotion" not in df:
        raise ValueError("Dataset must contain `text` and `emotion` columns.")

    df = df[["text", "emotion"]].dropna().copy()
    df["emotion"] = df["emotion"].astype(str)
    df["clean_text"] = df["text"].map(clean_text)
    return df[df["clean_text"].str.len() > 0].reset_index(drop=True)


def save_processed_dataset(path: Path | str = PROCESSED_DATA_PATH) -> Path:
    """Download, clean, and save the processed dataset CSV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = prepare_dataset()
    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    saved_path = save_processed_dataset()
    print(f"Saved processed dataset to {saved_path}")
