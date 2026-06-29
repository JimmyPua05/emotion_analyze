# Code Walkthrough For Team Members

> If you are new to the project, read [SUPER_BEGINNER_GUIDE.md](SUPER_BEGINNER_GUIDE.md) first. Then use this file to follow the code file by file.


This guide explains the project code in simple language so every team member can understand the app, report, and presentation.

If you are completely new, read [BEGINNER_GUIDE.md](BEGINNER_GUIDE.md) before this walkthrough.

## Big Picture

The project is a Social Media Emotion Analyzer. A user types a social-media-style sentence, chooses a trained model, and the app predicts one emotion:

- sadness
- joy
- love
- anger
- fear
- surprise

The project has five main parts:

1. Data preprocessing
2. Feature extraction
3. Model training and evaluation
4. Streamlit web app
5. Visualizations and explanation

## How The App Flows

```text
User text
  -> clean/preprocess text
  -> selected model predicts emotion
  -> app shows predicted emotion
  -> app shows confidence scores
  -> app shows influential words or phrases
```

For example:

```text
Input: I am not happy today
Cleaned text: not happy today
Useful feature: not happy
Predicted emotion: model output
```

The phrase `not happy` is important because bigram models can learn two-word phrases instead of only single words.

## Folder Structure

```text
app.py                              Main Streamlit app
src/data_preprocessing.py           Text cleaning and dataset loading
src/train_classical.py              Classical ML training code
src/train_transformer.py            DistilBERT training code
src/predict.py                      Prediction, confidence, and explanation logic
src/visualization.py                Helper functions for charts
notebooks/model_development.ipynb   Notebook showing development process
tests/test_core.py                  Simple tests for core logic
README.md                           Project overview and run instructions
PROJECT_REQUIREMENTS_CHECKLIST.md   Where each assignment requirement is covered
GITHUB_UPLOAD_GUIDE.md              What goes to GitHub vs Google Drive
```

## app.py

`app.py` is the main Streamlit file. This is what runs when we type:

```bash
streamlit run app.py
```

Important parts:

### Cached Data Functions

```python
cached_dataset()
cached_results()
cached_model_details()
```

These functions load the dataset and model results once, then reuse them. This makes the app faster because Streamlit does not reload the CSV every time the page refreshes.

### Chart Helper Functions

```python
create_horizontal_bar_figure()
show_horizontal_bar_chart()
create_model_comparison_figure()
```

These functions create clean Plotly charts. We use them so emotion names and model names are readable instead of overlapping.

### Home/About Page

```python
show_home()
```

This page explains:

- project title
- team members
- problem and objective
- how to use the app
- emotion labels
- bonus work

### Text Analyzer Page

```python
show_text_analyzer()
```

This is the most important demo page. It lets the user:

- choose a model
- enter text
- see the cleaned text
- click Analyze Emotion
- see predicted emotion
- see confidence scores
- see influential words or phrases

### Data Explorer Page

```python
show_data_explorer()
```

This page shows:

- number of rows
- number of emotion classes
- average text length
- sample dataset rows
- emotion distribution chart

### Visualizations Page

```python
show_visualizations()
```

This page shows charts required by the project:

- class distribution
- text length distribution
- top words per emotion
- word cloud
- unigram vs bigram insight
- model comparison

### Model Info Page

```python
show_model_info()
```

This page explains the model results:

- model comparison table
- best model
- confusion matrix
- explanation of CountVectorizer, TF-IDF, Word2Vec, and DistilBERT

### main()

```python
main()
```

This function controls the sidebar navigation and decides which page to show.

## src/data_preprocessing.py

This file prepares text before training or prediction.

### LABEL_NAMES

```python
LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
```

This keeps all emotion labels in one place so the app, charts, and models use the same order.

### STOPWORDS

`STOPWORDS` contains common words like `the`, `is`, and `you`. These words are removed because they usually do not help emotion classification.

### lemmatize_token()

This function performs WordNet Lemmatization. It normalizes common word endings:

```text
feeling -> feel
loved -> love
studies -> study
```

This satisfies the preprocessing requirement for stemming or lemmatization without needing extra NLTK downloads.

### clean_text()

This is the main cleaning function. It:

- converts text to lowercase
- removes URLs
- removes @mentions
- removes hashtag symbols
- removes punctuation and numbers
- removes stopwords
- applies WordNet Lemmatization

### preprocess_text()

This is a simple wrapper around `clean_text()`. The prediction code calls this before sending user text into a model.

### load_processed_dataset()

Loads:

```text
data/processed/emotion_clean.csv
```

It also checks that the required columns exist.

### prepare_dataset()

This can create the processed dataset from Hugging Face data or from provided records.

## src/train_classical.py

This file trains all classical machine learning models.

### FeatureConfig

`FeatureConfig` stores one feature extraction setup, such as Count Bigram or TF-IDF Bigram.

### Word2VecVectorizer

This class converts text into Word2Vec average embeddings.

Word2Vec is different from CountVectorizer and TF-IDF because it represents words as dense meaning-based vectors.

### get_feature_configs()

This function defines all feature extraction methods:

- Count Unigram
- Count Bigram
- TF-IDF Unigram
- TF-IDF Bigram
- Word2Vec

Unigram means one word at a time. Bigram means one word plus two-word phrases.

### get_classifiers()

This function defines the classical models:

- Naive Bayes
- Logistic Regression
- SVM
- Random Forest

SVM is calibrated so it can provide probability/confidence scores in the app.

### train_all_models()

This is the main training function. It:

1. Loads the cleaned dataset
2. Splits the data into training and testing sets
3. Trains all suitable model combinations
4. Saves each model using joblib
5. Saves performance results to CSV and JSON

Word2Vec + Naive Bayes is skipped because Naive Bayes is not suitable for dense Word2Vec embeddings.

## src/predict.py

This file handles model selection and prediction.

### ModelInfo

`ModelInfo` stores information about one model:

- model ID
- display name
- model type
- file path
- description

### get_model_registry()

This function creates the dropdown model list used by the Streamlit app. It includes all classical models and DistilBERT.

### model_artifact_exists()

Checks whether a saved model file exists locally.

### predict_emotion()

This is the main prediction function. It:

1. Finds the selected model
2. Loads the saved model
3. Cleans the user text
4. Predicts the emotion
5. Returns confidence scores

### explain_prediction_terms()

This function explains the prediction in a simple way.

For CountVectorizer and TF-IDF models, it shows the actual words or phrases found in the text, such as:

```text
not happy
happy
today
```

For Word2Vec and DistilBERT, it shows cleaned tokens because those models use embeddings or transformer context instead of visible word-count columns.

## src/visualization.py

This file contains helper functions for charts.

### load_model_results()

Loads the model comparison table from:

```text
models/classical/model_results.csv
```

### load_model_details()

Loads detailed evaluation output such as confusion matrices from:

```text
models/classical/model_results_details.json
```

### add_text_length()

Adds a `text_length` column so the app can visualize how long posts are for each emotion.

### top_words_by_emotion()

Counts the most common words for one emotion class.

### unigram_bigram_summary()

Compares average F1-score between unigram and bigram feature methods.

This helps us explain whether phrases like `not happy` improve prediction.

## src/train_transformer.py

This file is for the DistilBERT bonus model.

DistilBERT is an advanced NLP transformer. It reads words in context, so it can understand sentences better than simple word-count models.

### fine_tune_distilbert()

This function:

1. Loads the processed dataset
2. Converts emotion labels into numbers
3. Tokenizes the text using DistilBERT tokenizer
4. Fine-tunes DistilBERT
5. Saves the model into `models/distilbert_emotion/`

This is for the `Use advanced NLP (BERT, transformers)` bonus mark.

## notebooks/model_development.ipynb

This notebook shows the development process required by the project:

- load dataset
- inspect label distribution
- explain preprocessing
- show feature extraction methods
- explain model training
- load model results
- compare feature methods
- describe DistilBERT bonus model

## tests/test_core.py

These tests check the important logic:

- text cleaning removes social media noise
- lemmatization/stemming works
- model dropdown contains all expected models
- explanation terms are readable
- emotion labels are correct
- text length feature works
- unigram vs bigram summary works

Run tests with:

```bash
python -m unittest tests.test_core
```

## What Each Team Member Can Explain

### Member 1: App And Demo

Explain:

- `app.py`
- Text Analyzer page
- confidence chart
- influential words/phrases
- how to run Streamlit

### Member 2: Dataset And Preprocessing

Explain:

- `data/processed/emotion_clean.csv`
- `src/data_preprocessing.py`
- stopword removal
- lemmatization/stemming
- emotion labels

### Member 3: Models And Evaluation

Explain:

- `src/train_classical.py`
- feature extraction methods
- Naive Bayes, Logistic Regression, SVM, Random Forest
- model comparison metrics
- confusion matrix
- DistilBERT bonus model

## Short Presentation Script

Use this simple flow during presentation:

1. Our project detects emotions in social media posts.
2. We use six emotion labels: sadness, joy, love, anger, fear, and surprise.
3. We preprocess text by cleaning, tokenizing, removing stopwords, and applying WordNet Lemmatization.
4. We compare CountVectorizer, TF-IDF, Word2Vec, and DistilBERT.
5. We train several models and compare them using accuracy, precision, recall, F1-score, and confusion matrix.
6. In the app, users type text, choose a model, and get prediction confidence scores.
7. The app also shows influential words or phrases, visualizations, and model comparison results.
8. DistilBERT is included as an advanced NLP bonus model.

## Common Questions

### Why use bigrams?

Bigrams capture two-word phrases. For example, `happy` and `not happy` have different meanings. A bigram model can detect `not happy` as a phrase.

### Why skip Word2Vec + Naive Bayes?

Naive Bayes works best with count-like features. Word2Vec creates dense numeric embeddings, so Logistic Regression, SVM, and Random Forest are more suitable.

### Why use DistilBERT?

DistilBERT is a transformer model. It understands word context better than classical word-count models and helps target the advanced NLP bonus mark.

### Why put some models in Google Drive?

Some model files are larger than normal GitHub limits. GitHub keeps the code, sample data, and one small demo model. Google Drive stores the full large model artifacts.

## Debugging Helpers

The project includes two debugging tools:

1. Streamlit Diagnostics page in `app.py`
2. Terminal script: `scripts/debug_project.py`

Run this command from the project root:

```bash
python scripts/debug_project.py
```

This checks dependencies, dataset files, model files, result files, notebook validity, and optional DistilBERT artifacts. If something fails, the script prints a suggested fix.

For more detail, read `DEBUGGING.md`. For possible presentation questions and answers, read `TEAM_FAQ_AND_PRESENTATION_QA.md`.



