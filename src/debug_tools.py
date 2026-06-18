"""Project diagnostics for easier debugging.

These checks are intentionally lightweight and readable. They help teammates
quickly see whether the dataset, model files, dependencies, and notebook are
available before running the Streamlit app or presenting the project.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import json
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


@dataclass(frozen=True)
class DiagnosticCheck:
    """One diagnostic result shown in the app and terminal script."""

    check: str
    status: str
    details: str
    fix: str


def _exists(path: Path) -> bool:
    """Return True when a path exists, avoiding noisy exceptions."""

    try:
        return path.exists()
    except OSError:
        return False


def _size_mb(path: Path) -> float:
    """Return file size in MB for friendly debug output."""

    return round(path.stat().st_size / (1024 * 1024), 2)


def _check_required_file(root: Path, relative_path: str, label: str) -> DiagnosticCheck:
    """Check that one required project file is present."""

    path = root / relative_path
    if _exists(path):
        details = f"Found {relative_path}"
        if path.is_file():
            details += f" ({_size_mb(path)} MB)"
        return DiagnosticCheck(label, PASS, details, "No action needed.")
    return DiagnosticCheck(label, FAIL, f"Missing {relative_path}", f"Restore or recreate {relative_path}.")


def _check_dependency(import_name: str, package_name: str | None = None, optional: bool = False) -> DiagnosticCheck:
    """Check whether a Python dependency can be imported."""

    package_name = package_name or import_name
    if importlib.util.find_spec(import_name) is not None:
        return DiagnosticCheck(f"Dependency: {package_name}", PASS, f"{package_name} is installed.", "No action needed.")

    status = WARN if optional else FAIL
    install_hint = "requirements-transformer.txt" if package_name in {"transformers", "torch", "evaluate"} else "requirements.txt"
    return DiagnosticCheck(
        f"Dependency: {package_name}",
        status,
        f"{package_name} is not installed in this Python environment.",
        f"Run: pip install -r {install_hint}",
    )


def _check_dataset(root: Path) -> DiagnosticCheck:
    """Check that the processed dataset exists and has the expected columns."""

    path = root / "data" / "processed" / "emotion_clean.csv"
    if not _exists(path):
        return DiagnosticCheck(
            "Processed dataset",
            FAIL,
            "data/processed/emotion_clean.csv is missing.",
            "Download/copy the dataset, or run: python -m src.data_preprocessing",
        )

    try:
        import pandas as pd

        df = pd.read_csv(path)
        missing = {"text", "emotion", "clean_text"}.difference(df.columns)
        if missing:
            return DiagnosticCheck(
                "Processed dataset",
                FAIL,
                f"Dataset exists but is missing columns: {sorted(missing)}",
                "Regenerate the processed CSV with src.data_preprocessing.",
            )
        status = PASS if len(df) >= 1000 else WARN
        return DiagnosticCheck(
            "Processed dataset",
            status,
            f"Found {len(df):,} rows and {df['emotion'].nunique()} emotion labels.",
            "No action needed." if status == PASS else "Use a dataset with at least 1,000 labelled text samples.",
        )
    except Exception as exc:
        return DiagnosticCheck(
            "Processed dataset",
            WARN,
            f"CSV exists ({_size_mb(path)} MB), but detailed validation failed: {exc}",
            "Install pandas or open the CSV manually to inspect it.",
        )


def _check_model_results(root: Path) -> DiagnosticCheck:
    """Check that model comparison results are available."""

    path = root / "models" / "classical" / "model_results.csv"
    if not _exists(path):
        return DiagnosticCheck(
            "Model results table",
            FAIL,
            "models/classical/model_results.csv is missing.",
            "Run: python -m src.train_classical",
        )

    try:
        import pandas as pd

        results = pd.read_csv(path)
        required = {"display_name", "accuracy", "precision", "recall", "f1"}
        missing = required.difference(results.columns)
        if missing:
            return DiagnosticCheck(
                "Model results table",
                FAIL,
                f"Results file is missing columns: {sorted(missing)}",
                "Regenerate results with python -m src.train_classical.",
            )
        return DiagnosticCheck(
            "Model results table",
            PASS,
            f"Found {len(results)} trained model result rows.",
            "No action needed.",
        )
    except Exception as exc:
        return DiagnosticCheck(
            "Model results table",
            WARN,
            f"Results CSV exists, but detailed validation failed: {exc}",
            "Install pandas or inspect the CSV manually.",
        )


def _check_demo_model(root: Path) -> DiagnosticCheck:
    """Check that the small GitHub-safe demo model is available."""

    path = root / "models" / "classical" / "tfidf_bigram__svm.joblib"
    if not _exists(path):
        return DiagnosticCheck(
            "Small demo model",
            FAIL,
            "models/classical/tfidf_bigram__svm.joblib is missing.",
            "Copy it from the full project folder or retrain classical models.",
        )
    size = _size_mb(path)
    status = PASS if size < 50 else WARN
    return DiagnosticCheck(
        "Small demo model",
        status,
        f"Found tfidf_bigram__svm.joblib ({size} MB).",
        "No action needed." if status == PASS else "Use a smaller model for GitHub if possible.",
    )


def _check_distilbert(root: Path) -> DiagnosticCheck:
    """Check whether the optional full DistilBERT artifact is present."""

    path = root / "models" / "distilbert_emotion"
    config = path / "config.json"
    weights = path / "model.safetensors"
    if _exists(config) and _exists(weights):
        return DiagnosticCheck(
            "DistilBERT full model",
            PASS,
            "DistilBERT config and weights are present.",
            "No action needed.",
        )
    return DiagnosticCheck(
        "DistilBERT full model",
        WARN,
        "DistilBERT artifacts are not fully present. This is expected in the GitHub-safe folder.",
        "Download the full models from Google Drive when demoing DistilBERT.",
    )


def _check_notebook(root: Path) -> DiagnosticCheck:
    """Validate the required model-development notebook JSON."""

    path = root / "notebooks" / "model_development.ipynb"
    if not _exists(path):
        return DiagnosticCheck(
            "Model development notebook",
            FAIL,
            "notebooks/model_development.ipynb is missing.",
            "Restore the notebook because it is a final deliverable.",
        )
    try:
        notebook = json.loads(path.read_text(encoding="utf-8-sig"))
        cells = notebook.get("cells", [])
        return DiagnosticCheck(
            "Model development notebook",
            PASS,
            f"Notebook JSON is valid with {len(cells)} cells.",
            "No action needed.",
        )
    except Exception as exc:
        return DiagnosticCheck(
            "Model development notebook",
            FAIL,
            f"Notebook is not valid JSON: {exc}",
            "Open/recreate the notebook in Jupyter and save it again.",
        )


def run_diagnostics(root: Path | str = PROJECT_ROOT) -> list[dict[str, str]]:
    """Run all project diagnostics and return table-ready dictionaries."""

    root = Path(root)
    checks: list[DiagnosticCheck] = []

    checks.extend(
        _check_required_file(root, relative_path, label)
        for relative_path, label in [
            ("app.py", "Streamlit app file"),
            ("README.md", "README file"),
            ("CODE_WALKTHROUGH.md", "Code walkthrough"),
            ("requirements.txt", "Requirements file"),
            ("src/data_preprocessing.py", "Preprocessing source"),
            ("src/predict.py", "Prediction source"),
            ("src/train_classical.py", "Classical training source"),
            ("src/train_transformer.py", "Transformer training source"),
            ("src/visualization.py", "Visualization source"),
        ]
    )

    checks.extend(
        [
            _check_dependency("streamlit"),
            _check_dependency("pandas"),
            _check_dependency("sklearn", "scikit-learn"),
            _check_dependency("plotly"),
            _check_dependency("joblib"),
            _check_dependency("wordcloud", optional=True),
            _check_dependency("gensim", optional=True),
            _check_dependency("transformers", optional=True),
        ]
    )

    checks.extend(
        [
            _check_dataset(root),
            _check_model_results(root),
            _check_demo_model(root),
            _check_distilbert(root),
            _check_notebook(root),
        ]
    )

    return [asdict(check) for check in checks]


def summarize_diagnostics(checks: Iterable[dict[str, str]]) -> dict[str, int]:
    """Count PASS/WARN/FAIL checks for quick display."""

    summary = {PASS: 0, WARN: 0, FAIL: 0}
    for check in checks:
        status = check.get("status", WARN)
        summary[status] = summary.get(status, 0) + 1
    return summary


def format_diagnostics(checks: Iterable[dict[str, str]]) -> str:
    """Format diagnostics as plain text for terminal output."""

    lines = []
    for check in checks:
        lines.append(f"[{check['status']}] {check['check']}: {check['details']}")
        if check["status"] != PASS:
            lines.append(f"    Fix: {check['fix']}")
    return "\n".join(lines)
