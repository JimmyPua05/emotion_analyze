# Debugging Guide

Use this guide when the app does not run, a model is missing, or a teammate cannot reproduce the project.

## Fastest Check

Run this from the project root:

```bash
python scripts/debug_project.py
```

The script prints `PASS`, `WARN`, and `FAIL` checks.

- `PASS`: working
- `WARN`: not fatal, but worth checking
- `FAIL`: must fix before demo/submission

## Streamlit Diagnostics Page

You can also open the app and choose **Diagnostics** from the sidebar:

```bash
streamlit run app.py
```

The Diagnostics page checks:

- required source files
- Python dependencies
- processed dataset
- saved demo model
- model result files
- DistilBERT full model availability
- notebook JSON validity

## Common Problems

### 1. ModuleNotFoundError

Example:

```text
ModuleNotFoundError: No module named 'streamlit'
```

Fix:

```bash
pip install -r requirements.txt
```

For DistilBERT/transformers:

```bash
pip install -r requirements-transformer.txt
```

### 2. Dataset Missing

Example:

```text
data/processed/emotion_clean.csv is missing
```

Fix:

- Copy `emotion_clean.csv` into `data/processed/`, or
- Run preprocessing if dataset download is available:

```bash
python -m src.data_preprocessing
```

### 3. Demo Model Missing

Example:

```text
models/classical/tfidf_bigram__svm.joblib is missing
```

Fix:

- Copy it from the full project folder, or
- Retrain classical models:

```bash
python -m src.train_classical
```

### 4. DistilBERT Warning

In the GitHub-safe folder, this warning is expected:

```text
DistilBERT artifacts are not fully present
```

The DistilBERT model is large, so it should be downloaded from Google Drive for the full demo.

### 5. Port Already In Use

If Streamlit says port `8501` is already used, run another port:

```bash
streamlit run app.py --server.port 8502
```

### 6. GitHub Push Rejects Large Files

If GitHub rejects the push, check for large model files:

```bash
git status
git ls-files | grep -E 'safetensors|optimizer.pt|checkpoint|random_forest.joblib'
```

Large files should be stored in Google Drive, not GitHub.

## Useful Commands

Run app:

```bash
streamlit run app.py
```

Run tests:

```bash
python -m unittest tests.test_core
```

Run diagnostics:

```bash
python scripts/debug_project.py
```

Check Git files before pushing:

```bash
git status
```

## What To Screenshot For Report

Useful app screenshots:

- Home/About page
- Text Analyzer prediction and confidence chart
- Influential words/phrases table
- Data Explorer emotion distribution
- Visualizations page
- Model Info confusion matrix
- Diagnostics page showing project checks
