<<<<<<< HEAD
﻿# Social Media Emotion Analyzer

This project detects emotions in social-media-style text using classical machine
learning models, n-gram feature extraction, Word2Vec embeddings, and a bonus
DistilBERT transformer model.

## Team Members

- Jimmy
- Haziq
- Arif

Replace the placeholder names with your actual group members before final submission.

## Project Criteria Covered

- User enters text and the app predicts an emotion category.
- The app shows confidence scores for each emotion.
- The app shows words or phrases that influenced the prediction.
- The app visualizes emotion distribution in the dataset.
- The app shows common words associated with each emotion.
- The app compares classical ML models using accuracy, precision, recall, and F1-score.
- The app compares unigram vs bigram n-gram performance.
- The project includes a DistilBERT transformer model for the advanced NLP bonus.
- The repository includes a Jupyter notebook showing the model development process.

## Emotion Labels

- sadness
- joy
- love
- anger
- fear
- surprise

## Dataset

The processed dataset is stored at:

```text
data/processed/emotion_clean.csv
```

It contains more than 1,000 labelled text samples and is relevant to Theme 5:
Social Media Emotion Analyzer.

## Preprocessing

Text preprocessing includes:

- Lowercasing
- URL and mention removal
- Hashtag symbol removal
- Punctuation and number removal
- Tokenization
- Stopword removal
- Lightweight lemmatization/stemming

## Feature Extraction Methods

- Bag-of-Words Unigram: `CountVectorizer(ngram_range=(1, 1))`
- Bag-of-Words Bigram: `CountVectorizer(ngram_range=(1, 2))`
- TF-IDF Unigram: `TfidfVectorizer(ngram_range=(1, 1))`
- TF-IDF Bigram: `TfidfVectorizer(ngram_range=(1, 2))`
- Word2Vec average embeddings
- DistilBERT transformer

N-grams are useful because they can capture short phrases. For example,
`happy` alone may suggest joy, but `not happy` can change the meaning.

## Classical Models

- Naive Bayes
- Logistic Regression
- SVM with probability calibration
- Random Forest

Word2Vec + Naive Bayes is skipped because Naive Bayes expects count-like
non-negative features, while Word2Vec creates dense averaged embeddings.

## How To Run

Install the classical ML dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

Train classical models:

```bash
python -m src.train_classical
```

Fine-tune DistilBERT:

```bash
pip install -r requirements-transformer.txt
python -m src.train_transformer --epochs 2 --batch-size 8
```

## Notebook

The model development notebook is here:

```text
notebooks/model_development.ipynb
```

## GitHub And Google Drive

GitHub should contain the source code, README, requirements, tests, notebook,
sample dataset, model results, and one small saved model for local testing.

Large trained artifacts such as DistilBERT checkpoints and large Random Forest
models should be stored in Google Drive instead of normal GitHub. See
`GITHUB_UPLOAD_GUIDE.md`.

## Full Models And Dataset

=======
﻿# Social Media Emotion Analyzer

> New to this project? Start with [SUPER_BEGINNER_GUIDE.md](SUPER_BEGINNER_GUIDE.md). It explains every folder, code part, model, chart, command, GitHub upload step, and presentation question in beginner language.


This project detects emotions in social-media-style text using classical machine
learning models, n-gram feature extraction, Word2Vec embeddings, and a bonus
DistilBERT transformer model.

## Start Here For Team Members

If you are new to the project, start with [BEGINNER_GUIDE.md](BEGINNER_GUIDE.md). It explains every part slowly from the beginning.

Then read [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md). It explains every file, the main functions, and what each team member can explain during presentation.

If something does not run, read [DEBUGGING.md](DEBUGGING.md) or open the **Diagnostics** page in the Streamlit app.

For common lecturer/member questions, read [TEAM_FAQ_AND_PRESENTATION_QA.md](TEAM_FAQ_AND_PRESENTATION_QA.md).

## Team Members

- Jimmy
- Team Member 2
- Team Member 3

Replace the placeholder names with your actual group members before final submission.

## Project Criteria Covered

- User enters text and the app predicts an emotion category.
- The app shows confidence scores for each emotion.
- The app shows words or phrases that influenced the prediction.
- The app visualizes emotion distribution in the dataset.
- The app shows common words associated with each emotion.
- The app compares classical ML models using accuracy, precision, recall, and F1-score.
- The app compares unigram vs bigram n-gram performance.
- The project includes a DistilBERT transformer model for the advanced NLP bonus.
- The repository includes a Jupyter notebook showing the model development process.

## Emotion Labels

- sadness
- joy
- love
- anger
- fear
- surprise

## Dataset

The processed dataset is stored at:

```text
data/processed/emotion_clean.csv
```

It contains more than 1,000 labelled text samples and is relevant to Theme 5:
Social Media Emotion Analyzer.

## Preprocessing

Text preprocessing includes:

- Lowercasing
- URL and mention removal
- Hashtag symbol removal
- Punctuation and number removal
- Tokenization
- Stopword removal
- Lightweight lemmatization/stemming

## Feature Extraction Methods

- Bag-of-Words Unigram: `CountVectorizer(ngram_range=(1, 1))`
- Bag-of-Words Bigram: `CountVectorizer(ngram_range=(1, 2))`
- TF-IDF Unigram: `TfidfVectorizer(ngram_range=(1, 1))`
- TF-IDF Bigram: `TfidfVectorizer(ngram_range=(1, 2))`
- Word2Vec average embeddings
- DistilBERT transformer

N-grams are useful because they can capture short phrases. For example,
`happy` alone may suggest joy, but `not happy` can change the meaning.

## Classical Models

- Naive Bayes
- Logistic Regression
- SVM with probability calibration
- Random Forest

Word2Vec + Naive Bayes is skipped because Naive Bayes expects count-like
non-negative features, while Word2Vec creates dense averaged embeddings.

## How To Run

Install the classical ML dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

Train classical models:

```bash
python -m src.train_classical
```

Fine-tune DistilBERT:

```bash
pip install -r requirements-transformer.txt
python -m src.train_transformer --epochs 2 --batch-size 8
```

## Notebook

The model development notebook is here:

```text
notebooks/model_development.ipynb
```

## GitHub And Google Drive

GitHub should contain the source code, README, requirements, tests, notebook,
sample dataset, model results, and one small saved model for local testing.

Large trained artifacts such as DistilBERT checkpoints and large Random Forest
models should be stored in Google Drive instead of normal GitHub. See
`GITHUB_UPLOAD_GUIDE.md`.

## Full Models And Dataset

>>>>>>> a88cabc (Add beginner documentation and debugging support)
Google Drive folder: paste your link here.





