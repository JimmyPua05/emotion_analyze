"""Lightweight tests for core project behavior."""

from __future__ import annotations

import unittest

import pandas as pd

from src.data_preprocessing import LABEL_NAMES, clean_text, lemmatize_token
from src.debug_tools import run_diagnostics, summarize_diagnostics
from src.predict import _confidence_from_classical, available_model_names, explain_prediction_terms, get_model_registry, model_artifact_exists, predict_emotion
from src.visualization import add_text_length, load_model_results, unigram_bigram_summary


class CoreProjectTests(unittest.TestCase):
    """Tests that protect the important project requirements."""

    def test_clean_text_removes_social_media_noise(self) -> None:
        cleaned = clean_text("I am SO happy!!! Visit https://example.com @friend #Joy123")
        self.assertEqual(cleaned, "happy visit joy")

    def test_light_lemmatization_normalizes_common_suffixes(self) -> None:
        self.assertEqual(lemmatize_token("studies"), "study")
        self.assertEqual(lemmatize_token("feeling"), "feel")
        self.assertEqual(lemmatize_token("loved"), "love")

    def test_registry_contains_required_model_choices(self) -> None:
        registry = get_model_registry()
        self.assertIn("Count Bigram + SVM", registry)
        self.assertIn("TF-IDF Bigram + Logistic Regression", registry)
        self.assertIn("Word2Vec + Random Forest", registry)
        self.assertIn("DistilBERT", registry)
        self.assertEqual(len(registry), 20)


    def test_available_model_names_only_returns_complete_artifacts(self) -> None:
        """The app dropdown should avoid models with missing dependent files."""

        registry = get_model_registry()
        names = available_model_names(registry)

        self.assertTrue(names)
        self.assertTrue(all(model_artifact_exists(registry[name]) for name in names))

    def test_saved_model_comparison_includes_distilbert(self) -> None:
        """Visualizations and Model Info should include the advanced NLP model."""

        results = load_model_results()
        names = results["display_name"].astype(str).tolist()

        self.assertIn("DistilBERT", names)
        row = results[results["display_name"] == "DistilBERT"].iloc[0]
        self.assertEqual(row["feature_method"], "Transformer")
        self.assertGreater(float(row["f1"]), 0)
    def test_explanation_terms_fallback_is_readable(self) -> None:
        explanation = explain_prediction_terms("DistilBERT", "I feel very happy today")
        self.assertFalse(explanation.empty)
        self.assertIn("term", explanation.columns)
        self.assertIn("why_it_matters", explanation.columns)

    def test_diagnostics_return_readable_checks(self) -> None:
        checks = run_diagnostics()
        summary = summarize_diagnostics(checks)
        self.assertGreater(len(checks), 5)
        self.assertIn("PASS", summary)
        self.assertTrue(all("check" in item and "status" in item and "fix" in item for item in checks))

    def test_confidence_falls_back_when_predict_proba_is_incompatible(self) -> None:
        """Old saved LogisticRegression models can fail in predict_proba after sklearn changes."""

        class ProbabilityBrokenModel:
            classes_ = LABEL_NAMES

            def predict_proba(self, texts):
                raise AttributeError("'LogisticRegression' object has no attribute 'multi_class'")

            def decision_function(self, texts):
                return [[0.0, 3.0, 0.0, 0.0, 0.0, 0.0]]

            def predict(self, texts):
                return ["joy"]

        confidence = _confidence_from_classical(ProbabilityBrokenModel(), "happy today")

        self.assertEqual(max(confidence, key=confidence.get), "joy")
        self.assertAlmostEqual(sum(confidence.values()), 1.0, places=6)

    def test_word2vec_artifact_predicts_when_available(self) -> None:
        """Older full-project Word2Vec artifacts are saved as dictionaries."""

        registry = get_model_registry()
        info = registry["Word2Vec + Logistic Regression"]
        if not model_artifact_exists(info):
            self.skipTest("Word2Vec artifact is not included in this project copy.")

        prediction, confidence = predict_emotion("Word2Vec + Logistic Regression", "I am happy today")

        self.assertIn(prediction, LABEL_NAMES)
        self.assertAlmostEqual(sum(confidence.values()), 1.0, places=6)

    def test_distilbert_artifact_predicts_when_available(self) -> None:
        """DistilBERT should run from local artifacts without importing TensorFlow pipeline."""

        registry = get_model_registry()
        info = registry["DistilBERT"]
        if not model_artifact_exists(info):
            self.skipTest("DistilBERT artifact is not included in this project copy.")

        prediction, confidence = predict_emotion("DistilBERT", "I am happy today")

        self.assertIn(prediction, LABEL_NAMES)
        self.assertAlmostEqual(sum(confidence.values()), 1.0, places=4)
    def test_all_emotion_labels_are_present(self) -> None:
        self.assertEqual(LABEL_NAMES, ["sadness", "joy", "love", "anger", "fear", "surprise"])

    def test_text_length_feature(self) -> None:
        df = pd.DataFrame({"clean_text": ["very happy today", "sad"]})
        output = add_text_length(df)
        self.assertEqual(output["text_length"].tolist(), [3, 1])

    def test_unigram_bigram_summary(self) -> None:
        results = pd.DataFrame(
            {
                "feature_method": ["Count Unigram", "Count Bigram", "TF-IDF Unigram", "TF-IDF Bigram"],
                "f1": [0.8, 0.9, 0.7, 0.85],
            }
        )
        summary = unigram_bigram_summary(results)
        values = dict(zip(summary["ngram_type"], summary["f1"]))
        self.assertAlmostEqual(values["Unigram"], 0.75)
        self.assertAlmostEqual(values["Bigram"], 0.875)

    def test_transformer_metrics_use_local_sklearn(self) -> None:
        from src.train_transformer import compute_classification_metrics

        metrics = compute_classification_metrics(
            logits=[[0.1, 0.9], [0.8, 0.2], [0.7, 0.3]],
            labels=[1, 0, 1],
        )

        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["f1"], 2 / 3)

    def test_distilbert_result_writer_adds_comparison_row(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from src.train_transformer import write_distilbert_comparison_results

        with TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            results_path = tmp / "model_results.csv"
            details_path = tmp / "model_results_details.json"
            summary_path = tmp / "evaluation_summary.json"

            metrics = write_distilbert_comparison_results(
                logits=[
                    [5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 5.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
                ],
                labels=[0, 1, 2],
                rows_evaluated=3,
                results_path=results_path,
                details_path=details_path,
                summary_path=summary_path,
            )

            results = pd.read_csv(results_path)
            self.assertIn("DistilBERT", results["display_name"].tolist())
            self.assertEqual(metrics["rows_evaluated"], 3)
            self.assertEqual(results.loc[0, "feature_method"], "Transformer")
            self.assertTrue(details_path.exists())
            self.assertTrue(summary_path.exists())

    def test_transformer_training_args_match_installed_transformers(self) -> None:
        from src.train_transformer import evaluation_strategy_argument
        from transformers import TrainingArguments

        strategy_arg = evaluation_strategy_argument(TrainingArguments)

        self.assertEqual(strategy_arg, {"eval_strategy": "epoch"})


if __name__ == "__main__":
    unittest.main()






