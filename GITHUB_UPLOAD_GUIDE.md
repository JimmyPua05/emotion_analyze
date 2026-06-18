# GitHub Upload Guide

This project should be shared in two versions:

1. Full version: keep locally or upload to Google Drive.
2. GitHub version: upload code, README, requirements, tests, notebook, sample data, result files, and one small saved model.

GitHub blocks normal Git files over 100 MiB and warns for files over 50 MiB. This project has several model files above that limit, so those files should not be committed to normal GitHub.

## Put These On GitHub

The GitHub version should include:

- `app.py`
- `src/*.py`
- `tests/*.py`
- `notebooks/model_development.ipynb`
- `README.md`
- `PROJECT_REQUIREMENTS_CHECKLIST.md`
- `GITHUB_UPLOAD_GUIDE.md`
- `.gitignore`
- `requirements.txt`
- `requirements-transformer.txt`
- `data/processed/emotion_clean.csv`
- `models/classical/tfidf_bigram__svm.joblib`
- `models/classical/model_results.csv`
- `models/classical/model_results_details.json`
- `.gitkeep` placeholder files

`emotion_clean.csv` is small enough for GitHub and satisfies the sample data requirement. `tfidf_bigram__svm.joblib` is also small enough and gives your friends one saved model to test without downloading everything.

## Put These In Google Drive

Upload these large full-project artifacts to Google Drive:

- `models/distilbert_emotion/model.safetensors`
- `models/distilbert_emotion/checkpoint-1000/`
- `models/distilbert_emotion/checkpoint-2000/`
- `models/classical/*random_forest.joblib`
- Other large `models/classical/*.joblib` files you do not include on GitHub
- `models/classical/word2vec_embeddings.model`

The largest files found in this project are:

- `models/distilbert_emotion/checkpoint-1000/optimizer.pt` - about 510.93 MB
- `models/distilbert_emotion/checkpoint-2000/optimizer.pt` - about 510.93 MB
- `models/distilbert_emotion/model.safetensors` - about 255.44 MB
- `models/distilbert_emotion/checkpoint-1000/model.safetensors` - about 255.44 MB
- `models/distilbert_emotion/checkpoint-2000/model.safetensors` - about 255.44 MB
- `models/classical/count_bigram__random_forest.joblib` - about 185.80 MB
- `models/classical/count_unigram__random_forest.joblib` - about 179.22 MB
- `models/classical/tfidf_bigram__random_forest.joblib` - about 177.99 MB
- `models/classical/tfidf_unigram__random_forest.joblib` - about 166.03 MB
- `models/classical/word2vec__random_forest.joblib` - about 153.51 MB

## Do Not Upload These To GitHub

Do not commit these:

- `models/**/*.safetensors`
- `models/**/*.pt`
- `models/**/*.bin`
- `models/**/checkpoint-*/`
- Large `.joblib` model files over 50 MB
- `__pycache__/`
- `*.pyc`
- `.venv/`, `venv/`, or `env/`
- `.env`, `.streamlit/secrets.toml`, or `kaggle.json`
- `.git/`
- `*.log`

## Clean Git Tracking Without Deleting Local Files

If generated files are already tracked by Git, remove them from Git tracking while keeping them on your computer:

```bash
git rm -r --cached __pycache__ src/__pycache__ tests/__pycache__
git rm -r --cached models data/processed/emotion_clean.csv
git add .
git status
```

Then check that Git includes the small allowed files and does not include huge artifacts.

## Google Drive Link In README

In `README.md`, paste your Drive link in this section:

```markdown
## Full Models And Dataset

Google Drive folder: <paste link here>
```
