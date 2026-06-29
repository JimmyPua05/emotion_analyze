# Team FAQ And Presentation Q&A

> For the most beginner-friendly full explanation, read [SUPER_BEGINNER_GUIDE.md](SUPER_BEGINNER_GUIDE.md) first.


This file is written for team members who are new to the project. It explains the project using the same questions we discussed while building it, plus possible questions from a lecturer, teammate, or visitor during the showcase.

If you need the slowest beginner explanation, read [BEGINNER_GUIDE.md](BEGINNER_GUIDE.md) first.

## 1. Project Basics

### What is our project?

Our project is a **Social Media Emotion Analyzer**.

It detects the emotion in social-media-style text and predicts one of these labels:

- sadness
- joy
- love
- anger
- fear
- surprise

Example:

```text
Input: I am not happy today
Output: predicted emotion + confidence scores
```

### Which theme did we choose?

We chose **Theme 5: Social Media Emotion Analyzer**.

### Why did we choose Theme 5?

Theme 5 is a good choice because:

- it is clear and easy to explain
- there are available labelled emotion datasets
- it fits NLP classification very well
- it supports many useful visualizations
- it can use both classical machine learning and advanced NLP
- the live demo is simple: user enters text and the app predicts emotion

### Why not Theme 6?

Theme 6, App Review Classifier, is still possible, but it can be less convenient because:

- app review datasets may need extra cleaning
- some datasets may focus only on star ratings, not clear emotion labels
- visualizing trends over time depends on whether date columns exist
- the demo may feel similar to normal sentiment analysis

Theme 5 is more direct for emotion classification and easier to present.

### What problem are we solving?

People post many short emotional messages on social media. Our app helps classify the emotion behind a post automatically.

This can be useful for:

- understanding public mood
- customer feedback analysis
- mental health signal monitoring
- social media insight dashboards
- content moderation research

## 2. Dataset Questions

### What dataset are we using?

We use a labelled emotion text dataset saved as:

```text
data/processed/emotion_clean.csv
```

It contains text samples with emotion labels.

### How many samples does the dataset have?

The current processed dataset has about **20,000 rows** and **6 emotion labels**.

This satisfies the requirement of at least 1,000 labelled text samples.

### What columns are in the dataset?

The important columns are:

```text
text        original sentence
emotion     emotion label
clean_text  cleaned/preprocessed text
```

### Where did the data come from?

The project supports the DAIR Emotion dataset style, which is available through Hugging Face. It is relevant to social media/text emotion classification.

### Why do we preprocess the dataset?

Raw social media text has noise such as punctuation, links, mentions, hashtags, and repeated words. Preprocessing makes text cleaner before training models.

## 3. Preprocessing Questions

### What preprocessing steps do we use?

We use:

- lowercasing
- URL removal
- mention removal, such as `@friend`
- hashtag symbol removal
- punctuation and number removal
- tokenization
- stopword removal
- WordNet Lemmatization

### What is tokenization?

Tokenization means splitting a sentence into words.

Example:

```text
I feel happy today
```

becomes:

```text
I, feel, happy, today
```

### What are stopwords?

Stopwords are common words that usually do not help prediction, such as:

```text
the, is, am, you, and, to
```

We remove them to reduce noise.

### What is lemmatization or stemming?

It means converting different word forms into a simpler base form.

Examples:

```text
feeling -> feel
loved -> love
studies -> study
```

This helps the model treat similar word forms as related.

### Which file handles preprocessing?

```text
src/data_preprocessing.py
```

Important functions:

```text
clean_text()
preprocess_text()
lemmatize_token()
load_processed_dataset()
```

## 4. Feature Extraction Questions

### What is feature extraction?

Machine learning models cannot directly understand text. Feature extraction converts text into numbers.

### What feature extraction methods do we use?

We use:

- CountVectorizer unigram
- CountVectorizer bigram
- TF-IDF unigram
- TF-IDF bigram
- Word2Vec average embeddings
- DistilBERT transformer embeddings/model

### What is unigram?

Unigram means using one word at a time.

Example:

```text
not happy today
```

Unigrams:

```text
not, happy, today
```

### What is bigram?

Bigram means using two-word phrases.

Example:

```text
not happy today
```

Bigrams:

```text
not happy, happy today
```

### Why are n-grams useful?

N-grams help capture phrases. For example:

```text
happy
```

may suggest joy, but:

```text
not happy
```

suggests the opposite. Bigram features can detect `not happy` as a phrase.

### Where are n-grams in the code?

In:

```text
src/train_classical.py
```

Look for:

```python
CountVectorizer(ngram_range=(1, 1))
CountVectorizer(ngram_range=(1, 2))
TfidfVectorizer(ngram_range=(1, 1))
TfidfVectorizer(ngram_range=(1, 2))
```

### What is TF-IDF?

TF-IDF gives higher importance to useful words and lower importance to overly common words.

### What is Word2Vec?

Word2Vec converts words into dense numeric vectors that represent meaning. Similar words should have similar vector representations.

### Why is Word2Vec different from CountVectorizer and TF-IDF?

CountVectorizer and TF-IDF are based on word counts. Word2Vec is based on word meaning/embedding vectors.

## 5. Model Questions

### What models do we use?

Classical models:

- Naive Bayes
- Logistic Regression
- SVM
- Random Forest

Advanced model:

- DistilBERT

### Where are classical models trained?

```text
src/train_classical.py
```

### Where is DistilBERT trained?

```text
src/train_transformer.py
```

### Why do we use multiple models?

The requirement asks us to train and compare at least two classification models. We trained more than two so we can compare performance more strongly.

### Why skip Word2Vec + Naive Bayes?

Naive Bayes is designed for count-like features. Word2Vec creates dense numeric embeddings, so Logistic Regression, SVM, and Random Forest are more suitable.

### Why use SVM?

SVM often works well for text classification. In this project, SVM is also calibrated so the app can show confidence scores.

### Why use DistilBERT?

DistilBERT is a transformer model. It understands context better than simple word-count models and helps target the advanced NLP bonus mark.

### Does DistilBERT count for bonus marks?

Yes. The requirement says advanced NLP such as BERT/transformers can earn bonus marks. DistilBERT is a transformer model.

## 6. Evaluation Questions

### What evaluation metrics do we use?

We use:

- accuracy
- precision
- recall
- F1-score
- confusion matrix

### Where are model results stored?

```text
models/classical/model_results.csv
models/classical/model_results_details.json
```

### What is accuracy?

Accuracy is the percentage of predictions that are correct.

### What is precision?

Precision measures how many predictions for a class were actually correct.

### What is recall?

Recall measures how many actual examples of a class the model found.

### What is F1-score?

F1-score balances precision and recall. It is useful when class distribution is not perfectly balanced.

### What is a confusion matrix?

A confusion matrix shows which emotions the model predicts correctly and which emotions it confuses.

## 7. Streamlit App Questions

### What is Streamlit?

Streamlit is a Python tool for building simple web apps quickly.

### How do we run the app?

```bash
streamlit run app.py
```

### What are the app pages?

- Home/About
- Text Analyzer
- Data Explorer
- Visualizations
- Model Info
- Diagnostics

### What does Text Analyzer do?

It lets the user:

1. choose a model
2. enter text
3. click Analyze Emotion
4. see predicted emotion
5. see confidence scores
6. see influential words or phrases

### What does Data Explorer do?

It shows:

- sample dataset rows
- number of rows
- number of emotion classes
- average text length
- emotion distribution

### What does Visualizations do?

It shows charts such as:

- class distribution
- text length distribution
- top words per emotion
- word cloud
- unigram vs bigram comparison
- model comparison

### What does Model Info do?

It shows:

- model comparison table
- best model
- confusion matrix
- explanation of feature extraction methods

### What does Diagnostics do?

It checks if the project is ready to run. It checks:

- required files
- dependencies
- dataset
- model files
- model result files
- notebook validity
- DistilBERT files

## 8. Visualization Questions

### What visualizations do we have?

The project includes more than five visualizations:

- emotion/class distribution
- text length distribution
- top words per emotion
- word cloud
- unigram vs bigram comparison
- model comparison chart
- confusion matrix heatmap
- prediction confidence chart

### Why do we need visualizations?

The requirement asks for at least five different visualizations. Visualizations also help explain dataset patterns and model performance.

### What is a word cloud?

A word cloud shows common words with larger words appearing more frequently.

### What is top words per emotion?

It shows which words appear most often for a selected emotion.

### What is unigram vs bigram insight?

It compares whether single words or two-word phrases perform better on average.

## 9. GitHub And Google Drive Questions

### Why do we use both GitHub and Google Drive?

GitHub is for code and small files. Google Drive is for very large model files.

### What goes to GitHub?

GitHub should include:

```text
app.py
src/
tests/
notebooks/
README.md
CODE_WALKTHROUGH.md
TEAM_FAQ_AND_PRESENTATION_QA.md
DEBUGGING.md
requirements.txt
requirements-transformer.txt
data/processed/emotion_clean.csv
models/classical/tfidf_bigram__svm.joblib
models/classical/model_results.csv
models/classical/model_results_details.json
```

### What goes to Google Drive?

Google Drive should include large full-model files:

```text
models/distilbert_emotion/
large Random Forest .joblib files
other large .joblib files
word2vec_embeddings.model
checkpoint folders
```

### Why not upload everything to GitHub?

GitHub rejects very large files and may warn for files above 50 MB. Some trained model files are hundreds of MB.

### Will privacy or environment get leaked?

Not if we avoid uploading:

```text
.env
kaggle.json
.streamlit/secrets.toml
.venv/
__pycache__/
.git/
```

The `.gitignore` helps prevent these from being uploaded.

## 10. Debugging Questions

### What command should we run if something breaks?

Run:

```bash
python scripts/debug_project.py
```

### What if Streamlit is not installed?

Run:

```bash
pip install -r requirements.txt
```

### What if the model is missing?

If this file is missing:

```text
models/classical/tfidf_bigram__svm.joblib
```

copy it from the full project folder or retrain models:

```bash
python -m src.train_classical
```

### What if DistilBERT is missing?

In the GitHub-safe folder, this is expected because DistilBERT is too large. Download it from Google Drive for the full demo.

### What if port 8501 is busy?

Run Streamlit on another port:

```bash
streamlit run app.py --server.port 8502
```

### What if GitHub push is rejected?

The remote repo may already have files. Try:

```bash
git pull origin main --allow-unrelated-histories --no-edit
git push
```

If the push is rejected because of large files, check `git status` and remove large files from Git tracking.

## 11. Questions Your Lecturer Might Ask

### Why did you choose this topic?

We chose it because emotion analysis is a clear NLP classification problem with available labelled data, strong visualization opportunities, and a simple live demo.

### What is your input and output?

Input is a text post. Output is the predicted emotion label and confidence scores.

### What is your best model?

Check the Model Info page or `models/classical/model_results.csv`. The best classical model is selected based on weighted F1-score.

### Why use F1-score instead of only accuracy?

F1-score balances precision and recall, which is helpful when classes are not perfectly balanced.

### How do you know the model is working?

We evaluate on a test set using accuracy, precision, recall, F1-score, and confusion matrix. The app also allows live testing.

### What is the difference between TF-IDF and Word2Vec?

TF-IDF measures word importance based on frequency. Word2Vec represents words using meaning-based vectors.

### What is the difference between classical ML and DistilBERT?

Classical ML uses engineered features like word counts or TF-IDF. DistilBERT is a transformer that learns context from the sentence.

### What is the limitation of your project?

Possible limitations:

- the model only predicts six emotion classes
- sarcasm may be hard to detect
- short text can be ambiguous
- social media slang may confuse the model
- DistilBERT model files are large and not stored directly on GitHub

### How can this project be improved?

Possible improvements:

- add multilingual support
- collect more social media data
- improve emoji handling
- deploy publicly on Streamlit Cloud
- add explainability methods like SHAP or LIME
- tune DistilBERT with more epochs or a GPU

## 12. Questions Your Teammates Might Ask

### Which file should I read first?

Read in this order:

1. `README.md`
2. `CODE_WALKTHROUGH.md`
3. `TEAM_FAQ_AND_PRESENTATION_QA.md`
4. `DEBUGGING.md`
5. `app.py`

### I only need to present one part. What should I study?

- App/demo member: `app.py`
- Data/preprocessing member: `src/data_preprocessing.py`
- Model/results member: `src/train_classical.py`, `src/predict.py`, `models/classical/model_results.csv`

### How do I run the app?

```bash
streamlit run app.py
```

### How do I run the debug check?

```bash
python scripts/debug_project.py
```

### How do I run tests?

```bash
python -m unittest tests.test_core
```

### How do I explain the full pipeline simply?

Say:

> The user enters text. We clean the text, convert it into numeric features, send it into a trained model, predict the emotion, and show confidence scores plus visual explanations.

## 13. Short Presentation Answer Template

Use this if you get nervous:

> Our project is a Social Media Emotion Analyzer. It predicts emotions from text using NLP and machine learning. We preprocess the text, extract features using CountVectorizer, TF-IDF, Word2Vec, and DistilBERT, then compare models such as Naive Bayes, Logistic Regression, SVM, and Random Forest. The Streamlit app lets users type text and see the predicted emotion, confidence scores, influential words, dataset visualizations, model comparison, and confusion matrix.

## 14. Responsibility Split Suggestion

### Member 1: Streamlit App And Demo

Explain:

- app pages
- text analyzer
- prediction confidence
- diagnostics page

### Member 2: Dataset And Preprocessing

Explain:

- dataset source and labels
- cleaning steps
- stopword removal
- lemmatization/stemming

### Member 3: Models And Evaluation

Explain:

- feature extraction
- model training
- model comparison
- confusion matrix
- DistilBERT bonus

## 15. Final Checklist Before Presentation

Run these commands:

```bash
python scripts/debug_project.py
python -m unittest tests.test_core
streamlit run app.py
```

Check these pages in the app:

- Home/About
- Text Analyzer
- Data Explorer
- Visualizations
- Model Info
- Diagnostics

Make sure to replace placeholder team names in:

```text
README.md
app.py
```


