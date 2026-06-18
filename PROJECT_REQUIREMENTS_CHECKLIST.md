# Project Requirements Checklist

> If you need a beginner explanation of how these requirements are implemented, read [SUPER_BEGINNER_GUIDE.md](SUPER_BEGINNER_GUIDE.md).


Use this file to show your lecturer or teammates where each requirement is
implemented in the code.

## Core Application

- Text input for user posts: `app.py`, `show_text_analyzer()`
- Emotion prediction result: `app.py`, `predict_emotion()`
- Confidence score chart: `app.py`, `show_text_analyzer()`
- Words or phrases influencing prediction: `app.py`, `explain_prediction_terms()`
- Model selector dropdown: `app.py`, `show_text_analyzer()`
- Home/About page with project description, usage steps, and team names: `app.py`, `show_home()`

## Dataset

- Processed dataset: `data/processed/emotion_clean.csv`
- Dataset loader: `src/data_preprocessing.py`
- Text cleaning and preprocessing: `src/data_preprocessing.py`
- Dataset has labels for six emotions: sadness, joy, love, anger, fear, surprise

## Text Processing And NLP

- Lowercasing, URL removal, mention removal, punctuation/number removal: `src/data_preprocessing.py`
- Tokenization and stopword removal: `src/data_preprocessing.py`
- Lightweight lemmatization/stemming: `src/data_preprocessing.py`, `lemmatize_token()`

## Feature Extraction

- Bag-of-Words unigram: `src/train_classical.py`
- Bag-of-Words bigram / n-grams: `src/train_classical.py`
- TF-IDF unigram: `src/train_classical.py`
- TF-IDF bigram / n-grams: `src/train_classical.py`
- Word2Vec average embeddings: `src/train_classical.py`
- DistilBERT transformer: `src/train_transformer.py`

## Machine Learning Models

- Naive Bayes: `src/train_classical.py`
- Logistic Regression: `src/train_classical.py`
- SVM with calibrated probabilities: `src/train_classical.py`
- Random Forest: `src/train_classical.py`
- DistilBERT fine-tuning: `src/train_transformer.py`

## Evaluation

- Accuracy, precision, recall, F1-score: `src/train_classical.py`
- Model comparison table: `models/classical/model_results.csv`
- Detailed reports and confusion matrices: `models/classical/model_results_details.json`
- App model comparison page: `app.py`, `show_model_info()`

## Visualizations And Insights

- Emotion distribution: `app.py`, `show_data_explorer()` and `show_visualizations()`
- Top words per emotion: `src/visualization.py`, `top_words_by_emotion()`
- Word cloud by emotion: `app.py`, `show_visualizations()`
- Text length distribution: `app.py`, `show_visualizations()`
- Unigram vs bigram insight: `src/visualization.py`, `unigram_bigram_summary()`
- Model comparison chart: `app.py`, `create_model_comparison_figure()`
- Confusion matrix: `app.py`, `show_model_info()`

## Deliverables

- Streamlit app: `app.py`
- Requirements file: `requirements.txt`
- README: `README.md`
- Jupyter notebook: `notebooks/model_development.ipynb`
- Sample dataset for testing: `data/processed/emotion_clean.csv`
- Small saved model for testing: `models/classical/tfidf_bigram__svm.joblib`

## Bonus Marks

- Exceptional visualizations or insights: emotion distribution, word cloud,
  top words, text length distribution, unigram vs bigram comparison, confusion
  matrix, and model comparison chart.
- Advanced NLP: DistilBERT transformer fine-tuning and prediction.

## GitHub Safety

- `.gitignore` keeps huge model artifacts, logs, cache files, and secrets out.
- `GITHUB_UPLOAD_GUIDE.md` explains what goes to GitHub and what goes to Google
  Drive.

