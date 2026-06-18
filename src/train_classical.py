"""Train classical machine learning models for emotion detection.

This script covers the core project requirement: compare multiple feature
extraction methods and classifiers. It intentionally includes unigram and
bigram n-grams so the report can discuss whether phrases like "not happy"
improve emotion prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.data_preprocessing import LABEL_NAMES, load_processed_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "classical"
RESULTS_PATH = MODEL_DIR / "model_results.csv"
DETAILS_PATH = MODEL_DIR / "model_results_details.json"


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for one feature extraction method."""

    model_id_prefix: str
    display_name: str
    vectorizer: object | None
    ngram_range: str


class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    """Convert each document into the average of its Word2Vec word vectors.

    Word2Vec is different from CountVectorizer and TF-IDF because it produces
    dense numeric embeddings that represent word meaning. Naive Bayes is skipped
    for this feature type because MultinomialNB expects non-negative count-like
    features, not dense averaged embeddings.
    """

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        workers: int = 4,
        epochs: int = 20,
        random_state: int = 42,
    ) -> None:
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, X: Iterable[str], y: object | None = None) -> "Word2VecVectorizer":
        """Train a Word2Vec model on tokenized cleaned text."""

        try:
            from gensim.models import Word2Vec
        except ImportError as exc:
            raise ImportError("Install `gensim` to train Word2Vec models.") from exc

        sentences = [str(text).split() for text in X]
        self.model_ = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
            seed=self.random_state,
        )
        return self

    def transform(self, X: Iterable[str]) -> np.ndarray:
        """Average known word vectors for each document."""

        vectors = []
        for text in X:
            token_vectors = [
                self.model_.wv[token]
                for token in str(text).split()
                if token in self.model_.wv
            ]
            if token_vectors:
                vectors.append(np.mean(token_vectors, axis=0))
            else:
                vectors.append(np.zeros(self.vector_size))
        return np.vstack(vectors)


# Backward-compatible aliases help joblib load models if the class had a
# slightly different name during an earlier training run.
MeanWord2VecVectorizer = Word2VecVectorizer
AverageWord2VecVectorizer = Word2VecVectorizer


def get_feature_configs() -> list[FeatureConfig]:
    """Return all feature extraction methods required by the project."""

    return [
        FeatureConfig("count_unigram", "Count Unigram", CountVectorizer(ngram_range=(1, 1), min_df=2), "(1, 1)"),
        FeatureConfig("count_bigram", "Count Bigram", CountVectorizer(ngram_range=(1, 2), min_df=2), "(1, 2)"),
        FeatureConfig("tfidf_unigram", "TF-IDF Unigram", TfidfVectorizer(ngram_range=(1, 1), min_df=2), "(1, 1)"),
        FeatureConfig("tfidf_bigram", "TF-IDF Bigram", TfidfVectorizer(ngram_range=(1, 2), min_df=2), "(1, 2)"),
        FeatureConfig("word2vec", "Word2Vec", Word2VecVectorizer(), "not applicable"),
    ]


def get_classifiers() -> dict[str, object]:
    """Return all classical classifiers used in the experiments."""

    return {
        "naive_bayes": MultinomialNB(),
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "svm": CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }


def display_classifier_name(classifier_id: str) -> str:
    """Convert a classifier id into a reader-friendly name."""

    return {
        "naive_bayes": "Naive Bayes",
        "logistic_regression": "Logistic Regression",
        "svm": "SVM",
        "random_forest": "Random Forest",
    }[classifier_id]


def build_pipeline(feature: FeatureConfig, classifier: object) -> Pipeline:
    """Create a training pipeline from feature extractor plus classifier."""

    return Pipeline([("features", feature.vectorizer), ("classifier", classifier)])


def evaluate_model(model: Pipeline, X_test: pd.Series, y_test: pd.Series) -> tuple[dict, dict]:
    """Evaluate one trained model and return summary plus detailed metrics."""

    predictions = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )
    summary = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    details = {
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=LABEL_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=LABEL_NAMES).tolist(),
    }
    return summary, details


def train_all_models() -> pd.DataFrame:
    """Train all required classical combinations and save artifacts/results."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_processed_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["emotion"],
        test_size=0.2,
        random_state=42,
        stratify=df["emotion"],
    )

    results: list[dict] = []
    details: list[dict] = []
    classifiers = get_classifiers()

    for feature in get_feature_configs():
        for classifier_id, classifier in classifiers.items():
            if feature.model_id_prefix == "word2vec" and classifier_id == "naive_bayes":
                continue

            model_id = f"{feature.model_id_prefix}__{classifier_id}"
            display_name = f"{feature.display_name} + {display_classifier_name(classifier_id)}"
            model_path = MODEL_DIR / f"{model_id}.joblib"

            model = build_pipeline(feature, classifier)
            model.fit(X_train, y_train)
            summary, detail = evaluate_model(model, X_test, y_test)
            joblib.dump(model, model_path)

            row = {
                "model_id": model_id,
                "display_name": display_name,
                "feature_method": feature.display_name,
                "classifier": display_classifier_name(classifier_id),
                "model_path": str(model_path.relative_to(PROJECT_ROOT)),
                "ngram_range": feature.ngram_range,
                **summary,
            }
            results.append(row)
            details.append({**row, **detail})
            print(f"Saved {display_name}: F1={summary['f1']:.3f}")

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    results_df.to_csv(RESULTS_PATH, index=False)
    DETAILS_PATH.write_text(json.dumps(details, indent=2), encoding="utf-8")
    return results_df


if __name__ == "__main__":
    train_all_models()
