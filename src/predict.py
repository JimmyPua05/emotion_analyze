"""Prediction utilities and model registry for the Streamlit app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data_preprocessing import LABEL_NAMES, preprocess_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelInfo:
    """Metadata for one selectable model in the Streamlit app."""

    model_id: str
    display_name: str
    model_type: str
    path: Path
    description: str


def _classical_model(model_id: str, display_name: str, description: str) -> ModelInfo:
    """Build metadata for a saved scikit-learn model."""

    return ModelInfo(
        model_id=model_id,
        display_name=display_name,
        model_type="classical",
        path=PROJECT_ROOT / "models" / "classical" / f"{model_id}.joblib",
        description=description,
    )


def get_model_registry() -> dict[str, ModelInfo]:
    """Return every model option required in the app dropdown."""

    descriptions = {
        "count_unigram": "Bag-of-Words unigram counts individual words.",
        "count_bigram": "Bag-of-Words bigram uses individual words and two-word phrases.",
        "tfidf_unigram": "TF-IDF unigram gives important single words higher weight.",
        "tfidf_bigram": "TF-IDF bigram includes phrase features such as 'not happy'.",
        "word2vec": "Word2Vec averages dense word vectors to represent meaning.",
    }
    classifiers = {
        "naive_bayes": "Naive Bayes",
        "logistic_regression": "Logistic Regression",
        "svm": "SVM",
        "random_forest": "Random Forest",
    }
    features = [
        ("count_unigram", "Count Unigram"),
        ("count_bigram", "Count Bigram"),
        ("tfidf_unigram", "TF-IDF Unigram"),
        ("tfidf_bigram", "TF-IDF Bigram"),
        ("word2vec", "Word2Vec"),
    ]

    registry: dict[str, ModelInfo] = {}
    for feature_id, feature_name in features:
        for classifier_id, classifier_name in classifiers.items():
            if feature_id == "word2vec" and classifier_id == "naive_bayes":
                continue
            model_id = f"{feature_id}__{classifier_id}"
            display_name = f"{feature_name} + {classifier_name}"
            registry[display_name] = _classical_model(
                model_id,
                display_name,
                f"{descriptions[feature_id]} Classifier: {classifier_name}.",
            )

    registry["DistilBERT"] = ModelInfo(
        model_id="distilbert",
        display_name="DistilBERT",
        model_type="transformer",
        path=PROJECT_ROOT / "models" / "distilbert_emotion",
        description=(
            "DistilBERT is the advanced NLP bonus model. It is a transformer "
            "that reads words in context instead of only counting words."
        ),
    )
    return registry


def model_artifact_exists(info: ModelInfo) -> bool:
    """Check whether the selected model artifact is available locally."""

    if info.model_type == "transformer":
        return True
    if not info.path.exists():
        return False
    if info.model_id.startswith("word2vec"):
        try:
            artifact = joblib.load(info.path)
        except Exception:
            return False
        if isinstance(artifact, dict) and "word2vec_path" in artifact:
            embedding_path = Path(str(artifact["word2vec_path"]).replace("\\", "/"))
            if not embedding_path.is_absolute():
                embedding_path = PROJECT_ROOT / embedding_path
            return embedding_path.exists()
    return True


def _extract_classes(model: Any) -> list[str]:
    """Get class labels from a scikit-learn model, pipeline, or saved artifact."""

    if isinstance(model, dict) and "classifier" in model:
        return _extract_classes(model["classifier"])
    if hasattr(model, "classes_"):
        return list(model.classes_)
    if hasattr(model, "named_steps") and "classifier" in model.named_steps:
        classifier = model.named_steps["classifier"]
        if hasattr(classifier, "classes_"):
            return list(classifier.classes_)
    return LABEL_NAMES


def _softmax(scores: Any) -> np.ndarray:
    """Convert decision scores into probability-like values."""

    values = np.asarray(scores, dtype=float)
    if values.ndim > 1:
        values = values[0]
    if values.ndim == 0:
        values = np.array([-float(values), float(values)])
    exp_scores = np.exp(values - np.max(values))
    total = exp_scores.sum()
    if total == 0:
        return np.ones_like(exp_scores) / len(exp_scores)
    return exp_scores / total


def _format_confidence(classes: list[str], values: Any) -> dict[str, float]:
    """Return confidence scores in the fixed project label order."""

    scores = np.asarray(values, dtype=float)
    if scores.ndim > 1:
        scores = scores[0]
    confidence = {str(label): float(value) for label, value in zip(classes, scores)}
    return {label: confidence.get(label, 0.0) for label in LABEL_NAMES}


def _confidence_for_input(model: Any, model_input: Any, classes: list[str]) -> dict[str, float]:
    """Return confidence scores, falling back when probability APIs are incompatible."""

    if hasattr(model, "predict_proba"):
        try:
            return _format_confidence(classes, model.predict_proba(model_input)[0])
        except (AttributeError, TypeError, ValueError):
            # Older saved LogisticRegression objects can fail here when loaded
            # with a different scikit-learn version. Decision scores still work
            # and are enough for the app's confidence chart.
            pass

    if hasattr(model, "decision_function"):
        try:
            return _format_confidence(classes, _softmax(model.decision_function(model_input)))
        except (AttributeError, TypeError, ValueError):
            pass

    prediction = model.predict(model_input)[0]
    values = np.array([1.0 if label == prediction else 0.0 for label in classes])
    return _format_confidence(classes, values)


def _confidence_from_classical(model: Any, cleaned_text: str) -> dict[str, float]:
    """Return probability-like confidence scores for a classical model."""

    return _confidence_for_input(model, [cleaned_text], _extract_classes(model))

def available_model_names(registry: dict[str, ModelInfo] | None = None) -> list[str]:
    """Return model names whose required local artifacts are complete."""

    registry = registry or get_model_registry()
    return [name for name, info in registry.items() if model_artifact_exists(info)]

def _fallback_terms(text: str, reason: str, max_terms: int) -> pd.DataFrame:
    """Create a readable explanation from cleaned tokens when model internals are unavailable."""

    terms = []
    seen = set()
    for token in preprocess_text(text).split():
        if token not in seen:
            seen.add(token)
            terms.append({"term": token, "score": 1.0, "why_it_matters": reason})
        if len(terms) >= max_terms:
            break
    return pd.DataFrame(terms, columns=["term", "score", "why_it_matters"])


def explain_prediction_terms(model_name: str, text: str, max_terms: int = 10) -> pd.DataFrame:
    """Show the words or phrases that influenced the selected prediction.

    For CountVectorizer and TF-IDF models, this function extracts the actual
    unigram/bigram features found in the user's text and ranks them by feature
    value. For Word2Vec and DistilBERT, explanations are token-level because the
    model represents meaning through embeddings/context rather than visible word
    count columns.
    """

    registry = get_model_registry()
    if model_name not in registry:
        raise KeyError(f"Unknown model: {model_name}")

    info = registry[model_name]
    if info.model_type == "transformer":
        return _fallback_terms(
            text,
            "DistilBERT reads this token in sentence context.",
            max_terms,
        )
    if "word2vec" in info.model_id:
        return _fallback_terms(
            text,
            "Word2Vec uses this token as part of the average embedding.",
            max_terms,
        )
    if not info.path.exists():
        return _fallback_terms(text, "Model file is missing, so this is based on cleaned tokens.", max_terms)

    try:
        model = joblib.load(info.path)
        feature_step = None
        if hasattr(model, "named_steps"):
            feature_step = model.named_steps.get("features") or model.named_steps.get("vectorizer")
        if feature_step is None or not hasattr(feature_step, "get_feature_names_out"):
            return _fallback_terms(text, "This model does not expose readable text features.", max_terms)

        cleaned_text = preprocess_text(text)
        matrix = feature_step.transform([cleaned_text]).tocsr()
        feature_names = feature_step.get_feature_names_out()
        row = matrix[0]
        ranked = sorted(
            ((feature_names[index], float(value)) for index, value in zip(row.indices, row.data)),
            key=lambda item: item[1],
            reverse=True,
        )[:max_terms]
    except Exception:
        return _fallback_terms(text, "Readable feature extraction failed, so this is based on cleaned tokens.", max_terms)

    rows = [
        {
            "term": term,
            "score": score,
            "why_it_matters": "Matched unigram/bigram feature used by this model.",
        }
        for term, score in ranked
    ]
    return pd.DataFrame(rows, columns=["term", "score", "why_it_matters"])



def _resolve_artifact_path(path_value: str) -> Path:
    """Resolve saved artifact paths from either Windows or Linux separators."""

    path = Path(str(path_value).replace("\\", "/"))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _average_word2vec_document(cleaned_text: str, word2vec_model: Any) -> np.ndarray:
    """Convert cleaned text into one averaged Word2Vec feature row."""

    word_vectors = word2vec_model.wv
    vectors = [word_vectors[token] for token in cleaned_text.split() if token in word_vectors]
    if vectors:
        return np.asarray([np.mean(vectors, axis=0)])
    return np.zeros((1, word2vec_model.vector_size))


def _predict_word2vec_artifact(artifact: dict[str, Any], cleaned_text: str) -> tuple[str, dict[str, float]]:
    """Predict with older Word2Vec artifacts saved as a dict plus classifier."""

    try:
        from gensim.models import Word2Vec
    except ImportError as exc:
        raise ImportError("Install `gensim` to use Word2Vec models.") from exc

    classifier = artifact.get("classifier")
    if classifier is None:
        raise KeyError("Word2Vec artifact is missing its classifier.")

    embedding_path = _resolve_artifact_path(str(artifact.get("word2vec_path", "")))
    if not embedding_path.exists():
        raise FileNotFoundError(f"Word2Vec embeddings not found: {embedding_path}")

    word2vec_model = Word2Vec.load(str(embedding_path))
    features = _average_word2vec_document(cleaned_text, word2vec_model)
    prediction = str(classifier.predict(features)[0])
    confidence = _confidence_for_input(classifier, features, _extract_classes(classifier))
    return prediction, confidence


def _predict_classical(info: ModelInfo, text: str) -> tuple[str, dict[str, float]]:
    """Predict emotion using a saved scikit-learn pipeline or compatible artifact."""

    if not info.path.exists():
        raise FileNotFoundError(f"Model file not found: {info.path}")

    model = joblib.load(info.path)
    cleaned_text = preprocess_text(text)
    if isinstance(model, dict) and "classifier" in model and "word2vec_path" in model:
        return _predict_word2vec_artifact(model, cleaned_text)

    prediction = str(model.predict([cleaned_text])[0])
    confidence = _confidence_from_classical(model, cleaned_text)
    return prediction, confidence

def _predict_transformer(info: ModelInfo, text: str) -> tuple[str, dict[str, float]]:
    """Predict emotion using a DistilBERT model downloaded dynamically."""

    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError(
            "Install PyTorch and Transformers to use DistilBERT. "
            f"Original error: {exc}"
        ) from exc

    import functools
    @functools.lru_cache(maxsize=1)
    def _get_hf_pipeline():
        return pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=None)
        
    classifier = _get_hf_pipeline()
    
    results = classifier(text)[0]
    
    confidence = {label: 0.0 for label in LABEL_NAMES}
    for result in results:
        label = result["label"].lower()
        if label in confidence:
            confidence[label] = float(result["score"])

    prediction = max(confidence, key=confidence.get)
    return prediction, confidence

def predict_emotion(model_name: str, text: str) -> tuple[str, dict[str, float]]:
    """Predict the emotion and confidence scores for user-entered text."""

    registry = get_model_registry()
    if model_name not in registry:
        raise KeyError(f"Unknown model: {model_name}")
    info = registry[model_name]
    if info.model_type == "transformer":
        return _predict_transformer(info, text)
    return _predict_classical(info, text)


