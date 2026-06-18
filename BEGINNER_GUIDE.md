# Beginner Guide: Social Media Emotion Analyzer

> Need the slowest explanation first? Read [SUPER_BEGINNER_GUIDE.md](SUPER_BEGINNER_GUIDE.md) before this file.


This guide explains the whole project slowly and clearly. It is written for someone who is new to the project, new to machine learning, or new to Streamlit.

If you only read one file before helping with the project, read this one.

## 1. What This Project Is

This project is a **Social Media Emotion Analyzer**.

That means the app reads a short text, like a tweet or social media post, and predicts the emotion in that text.

Example input:

```text
I am not happy today
```

Possible output:

```text
Predicted emotion: sadness or joy, depending on the trained model
Confidence scores: sadness 0.40, joy 0.35, anger 0.10, etc.
```

The app predicts one of these six emotions:

```text
sadness
joy
love
anger
fear
surprise
```

## 2. Why This Project Is NLP

NLP means **Natural Language Processing**.

Natural language means human language, such as English text. Processing means using code to clean, understand, classify, or generate text.

This project is NLP because it takes human-written text and uses machine learning to classify the emotion.

## 3. The Simple Project Flow

The project works like this:

```text
User types text
    -> app cleans the text
    -> app converts text into numbers
    -> trained model predicts emotion
    -> app shows result and confidence
    -> app shows charts and explanation
```

Machine learning models cannot understand raw words directly. So we first convert words into numbers. This step is called **feature extraction**.

## 4. Important Folders And Files

Here is the project structure in simple words.

```text
app.py
```

This is the main Streamlit application. When you run the app, Python starts from this file.

```text
src/
```

This folder contains the main Python code used by the app.

```text
src/data_preprocessing.py
```

This file cleans text and loads the dataset.

```text
src/train_classical.py
```

This file trains the classical machine learning models.

```text
src/train_transformer.py
```

This file trains the DistilBERT transformer model for bonus marks.

```text
src/predict.py
```

This file loads saved models and makes predictions.

```text
src/visualization.py
```

This file contains helper functions for charts and visual insights.

```text
src/debug_tools.py
```

This file checks whether important files, dependencies, models, and data are available.

```text
tests/test_core.py
```

This file contains tests to check that important logic still works.

```text
scripts/debug_project.py
```

This is a terminal script that runs all debugging checks.

```text
notebooks/model_development.ipynb
```

This Jupyter notebook shows the model development process.

```text
data/processed/emotion_clean.csv
```

This is the cleaned dataset used by the app.

```text
models/classical/tfidf_bigram__svm.joblib
```

This is the small saved demo model included in GitHub so the app can make predictions without downloading every large model.

```text
models/classical/model_results.csv
```

This stores the model comparison results.

```text
models/classical/model_results_details.json
```

This stores more detailed evaluation results, such as confusion matrices.

```text
README.md
```

This is the main project overview on GitHub.

```text
CODE_WALKTHROUGH.md
```

This explains the code file by file.

```text
TEAM_FAQ_AND_PRESENTATION_QA.md
```

This contains common questions and answers for teammates and presentation.

```text
DEBUGGING.md
```

This explains how to fix common errors.

## 5. How To Run The Project

Open terminal in the project folder and run:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

If Streamlit is not installed, run:

```bash
pip install -r requirements.txt
```

Then try again:

```bash
streamlit run app.py
```

## 6. How To Debug The Project

If something breaks, run:

```bash
python scripts/debug_project.py
```

This checks:

- required files
- Python packages
- dataset file
- model file
- model results
- notebook validity
- optional DistilBERT files

Example output:

```text
PASS: 21 | WARN: 1 | FAIL: 0
```

What the statuses mean:

```text
PASS = working
WARN = not fatal, but check it
FAIL = must fix
```

A warning for DistilBERT in the GitHub folder is expected because the full DistilBERT model is too large and belongs in Google Drive.

## 7. The Dataset

The dataset is stored here:

```text
data/processed/emotion_clean.csv
```

It has three important columns:

```text
text        original sentence
emotion     emotion label
clean_text  cleaned version of the sentence
```

Example:

```text
text:       I am feeling happy today!
emotion:    joy
clean_text: feel happy today
```

The original text is what the dataset started with. The clean text is what the model uses after preprocessing.

## 8. What Is Preprocessing?

Preprocessing means cleaning raw text before giving it to a model.

Raw social media text can be messy:

```text
OMG!!! I am feeling SOOO happy today!!! @friend https://example.com #blessed123
```

After preprocessing, it becomes simpler:

```text
omg feel sooo happy today blessed
```

The project preprocessing does these things:

1. Lowercase text
2. Remove URLs
3. Remove mentions like `@friend`
4. Remove hashtag symbols
5. Remove punctuation and numbers
6. Split text into words
7. Remove stopwords
8. Apply lightweight lemmatization/stemming

The code is in:

```text
src/data_preprocessing.py
```

## 9. Lowercasing

Lowercasing means changing all letters to lowercase.

Example:

```text
HAPPY -> happy
Angry -> angry
```

This helps because the model should treat `Happy`, `HAPPY`, and `happy` as the same word.

## 10. URL Removal

Social media text may contain links:

```text
I love this https://example.com
```

The URL usually does not help emotion prediction, so the app removes it.

## 11. Mention Removal

Mentions look like this:

```text
@friend
@user123
```

They usually identify people, not emotions, so the app removes them.

## 12. Hashtag Cleaning

Hashtags can contain useful words:

```text
#happy
```

The app removes the `#` symbol but keeps the word:

```text
happy
```

## 13. Stopword Removal

Stopwords are common words like:

```text
the, is, am, to, and, you
```

They appear often but usually do not help emotion prediction.

Example:

```text
I am very happy today
```

After removing stopwords:

```text
very happy today
```

## 14. Lemmatization And Stemming

Lemmatization/stemming means simplifying word forms.

Examples:

```text
feeling -> feel
loved -> love
studies -> study
```

Why this helps:

The model can treat similar words as related instead of learning each form separately.

The function is:

```python
lemmatize_token()
```

It is inside:

```text
src/data_preprocessing.py
```

## 15. What Is Feature Extraction?

Machine learning models need numbers, not raw words.

Feature extraction means converting text into numeric features.

Example text:

```text
not happy today
```

A feature extractor may convert it into a table like:

```text
happy: 1
not: 1
not happy: 1
today: 1
```

The model uses these numbers to predict the emotion.

## 16. CountVectorizer

CountVectorizer counts words or phrases.

Example:

```text
happy happy today
```

CountVectorizer may create:

```text
happy = 2
today = 1
```

In our project, CountVectorizer is used in two ways:

```python
CountVectorizer(ngram_range=(1, 1))
CountVectorizer(ngram_range=(1, 2))
```

## 17. TF-IDF

TF-IDF means **Term Frequency - Inverse Document Frequency**.

Simple explanation:

TF-IDF gives higher value to words that are important in a text but not too common everywhere.

Example:

A word like `the` appears everywhere, so it is not very useful.

A word like `terrified` may be more useful for detecting fear.

In our project, TF-IDF is used in two ways:

```python
TfidfVectorizer(ngram_range=(1, 1))
TfidfVectorizer(ngram_range=(1, 2))
```

## 18. What Are Unigrams?

Unigrams are single words.

Example text:

```text
not happy today
```

Unigrams:

```text
not
happy
today
```

## 19. What Are Bigrams?

Bigrams are two-word phrases.

Example text:

```text
not happy today
```

Bigrams:

```text
not happy
happy today
```

## 20. Why Bigrams Matter

The word `happy` alone usually suggests joy.

But the phrase `not happy` may suggest sadness or anger.

A unigram model sees:

```text
not
happy
```

A bigram model can also see:

```text
not happy
```

This is why bigrams can improve emotion detection.

## 21. Word2Vec

Word2Vec converts words into meaning-based number vectors.

Instead of just counting words, it tries to represent meaning.

For example, words like:

```text
happy, joyful, excited
```

may have similar vector meanings.

In this project, each sentence is represented by averaging the Word2Vec vectors of the words in that sentence.

## 22. Why Word2Vec + Naive Bayes Is Skipped

Naive Bayes works best with count-like features.

Word2Vec creates dense numeric vectors that can include many continuous values.

So we skip:

```text
Word2Vec + Naive Bayes
```

and use Word2Vec with:

```text
Logistic Regression
SVM
Random Forest
```

## 23. DistilBERT

DistilBERT is a transformer model.

A transformer reads words in context.

Example:

```text
I am not happy
```

DistilBERT can learn that `not` changes the meaning of `happy`.

DistilBERT is included because the assignment gives bonus marks for advanced NLP such as BERT or transformers.

The code is in:

```text
src/train_transformer.py
```

## 24. Classical Models Used

The project trains these classical machine learning models:

```text
Naive Bayes
Logistic Regression
SVM
Random Forest
```

The code is in:

```text
src/train_classical.py
```

## 25. Naive Bayes

Naive Bayes is a simple and fast classification model.

It often works well for text classification because text can be represented as word counts.

## 26. Logistic Regression

Logistic Regression is a classification model that learns weights for features.

If a word strongly suggests joy, the model can learn a high weight for joy.

## 27. SVM

SVM means Support Vector Machine.

It is commonly strong for text classification.

In this project, SVM is calibrated so it can show confidence scores.

## 28. Random Forest

Random Forest is a group of many decision trees.

It can work well, but the saved model files can become very large. That is why large Random Forest files go to Google Drive instead of GitHub.

## 29. Model Training

Training means teaching the model using labelled examples.

Example training row:

```text
text: I feel amazing today
label: joy
```

The model learns patterns between text features and emotion labels.

Training code is in:

```text
src/train_classical.py
```

Run classical training with:

```bash
python -m src.train_classical
```

## 30. Model Evaluation

Evaluation means testing how well the model works on data it did not train on.

The project uses:

```text
accuracy
precision
recall
F1-score
confusion matrix
```

## 31. Accuracy

Accuracy means:

```text
correct predictions / total predictions
```

If the model predicts 90 out of 100 correctly, accuracy is 90%.

## 32. Precision

Precision answers:

```text
When the model predicts this emotion, how often is it correct?
```

## 33. Recall

Recall answers:

```text
Out of all real examples of this emotion, how many did the model find?
```

## 34. F1-Score

F1-score balances precision and recall.

It is useful because accuracy alone can be misleading when classes are not perfectly balanced.

## 35. Confusion Matrix

A confusion matrix shows correct and incorrect predictions by class.

It helps answer questions like:

```text
Does the model confuse sadness with fear?
Does the model confuse joy with love?
```

## 36. Saved Model Files

A trained model is saved as a file so we do not need to retrain every time.

Example:

```text
models/classical/tfidf_bigram__svm.joblib
```

This is the small model included in GitHub.

## 37. Why Some Model Files Are In Google Drive

Some model files are very large.

GitHub is not good for very large files, so we put large models in Google Drive.

GitHub has:

```text
code
small dataset
small demo model
README and docs
```

Google Drive has:

```text
large DistilBERT model
large checkpoint folders
large Random Forest models
other big model files
```

## 38. app.py Explained

`app.py` is the main web application.

It uses Streamlit to create pages.

The pages are:

```text
Home/About
Text Analyzer
Data Explorer
Visualizations
Model Info
Diagnostics
```

## 39. Home/About Page

This page explains:

- project title
- team members
- problem and objective
- how to use the app
- emotion labels
- bonus work

Function:

```python
show_home()
```

## 40. Text Analyzer Page

This is the main demo page.

It lets users:

1. choose a model
2. enter text
3. click Analyze Emotion
4. see predicted emotion
5. see confidence scores
6. see influential words or phrases

Function:

```python
show_text_analyzer()
```

## 41. Data Explorer Page

This page shows information about the dataset:

- number of rows
- number of emotion classes
- average text length
- sample rows
- emotion distribution

Function:

```python
show_data_explorer()
```

## 42. Visualizations Page

This page shows charts:

- class distribution
- text length distribution
- top words per emotion
- word cloud
- unigram vs bigram insight
- model comparison

Function:

```python
show_visualizations()
```

## 43. Model Info Page

This page shows:

- model comparison table
- best model
- confusion matrix
- explanation of CountVectorizer, TF-IDF, Word2Vec, and DistilBERT

Function:

```python
show_model_info()
```

## 44. Diagnostics Page

This page helps debug the project.

It checks whether required files, dependencies, dataset, models, and notebook are available.

Function:

```python
show_diagnostics()
```

## 45. src/predict.py Explained

This file handles prediction.

Main functions:

```python
get_model_registry()
predict_emotion()
explain_prediction_terms()
```

## 46. get_model_registry()

This function creates the list of models shown in the app dropdown.

It includes:

- Count Unigram + models
- Count Bigram + models
- TF-IDF Unigram + models
- TF-IDF Bigram + models
- Word2Vec + models
- DistilBERT

## 47. predict_emotion()

This function:

1. checks which model the user selected
2. loads the saved model
3. cleans the user text
4. predicts emotion
5. returns confidence scores

## 48. explain_prediction_terms()

This function explains which words or phrases influenced the prediction.

For TF-IDF and CountVectorizer models, it can show features like:

```text
not happy
happy
today
```

For Word2Vec and DistilBERT, it shows cleaned tokens because those models use embeddings/context instead of simple visible word columns.

## 49. src/visualization.py Explained

This file has helper functions for charts.

Important functions:

```python
load_model_results()
load_model_details()
add_text_length()
top_words_by_emotion()
unigram_bigram_summary()
```

## 50. top_words_by_emotion()

This finds common words for one emotion.

Example:

If the selected emotion is joy, common words may include:

```text
happy, love, excited, good
```

## 51. unigram_bigram_summary()

This compares average F1-score for unigram and bigram models.

This helps us explain whether two-word phrases improve performance.

## 52. src/debug_tools.py Explained

This file makes debugging easier.

It checks:

- important files exist
- important Python libraries are installed
- dataset exists and has correct columns
- model results exist
- demo model exists
- DistilBERT files exist or are missing only because it is GitHub-safe folder
- notebook JSON is valid

## 53. scripts/debug_project.py Explained

This script runs the diagnostics in terminal.

Run:

```bash
python scripts/debug_project.py
```

If everything is okay, you should see no failures.

## 54. tests/test_core.py Explained

Tests are small checks that make sure important code still works.

Run tests with:

```bash
python -m unittest tests.test_core
```

The tests check:

- text cleaning
- lemmatization/stemming
- model dropdown list
- explanation terms
- diagnostics
- emotion labels
- text length
- unigram vs bigram summary

## 55. The Notebook

The notebook is:

```text
notebooks/model_development.ipynb
```

It shows the model development process for the assignment.

It includes:

- dataset loading
- label distribution
- preprocessing explanation
- feature extraction
- training explanation
- evaluation results
- DistilBERT bonus explanation

## 56. GitHub Upload Version

The GitHub version should contain code and small files only.

It includes:

```text
app.py
src/
tests/
scripts/
notebooks/
README.md
BEGINNER_GUIDE.md
CODE_WALKTHROUGH.md
TEAM_FAQ_AND_PRESENTATION_QA.md
DEBUGGING.md
data/processed/emotion_clean.csv
models/classical/tfidf_bigram__svm.joblib
models/classical/model_results.csv
models/classical/model_results_details.json
```

## 57. Google Drive Full Version

Google Drive should contain the large files:

```text
models/distilbert_emotion/
large .joblib model files
word2vec_embeddings.model
checkpoint folders
```

## 58. What Each Team Member Can Focus On

For three people, a good split is:

### Member 1: App And Demo

Study:

```text
app.py
src/predict.py
DEBUGGING.md
```

Explain:

- how the app works
- how user input becomes prediction
- confidence scores
- diagnostics page

### Member 2: Dataset And Preprocessing

Study:

```text
src/data_preprocessing.py
data/processed/emotion_clean.csv
```

Explain:

- dataset columns
- emotion labels
- text cleaning
- stopword removal
- lemmatization/stemming

### Member 3: Models And Results

Study:

```text
src/train_classical.py
src/train_transformer.py
models/classical/model_results.csv
```

Explain:

- feature extraction
- classical models
- model comparison
- confusion matrix
- DistilBERT bonus

## 59. Very Short Explanation For Presentation

Say this:

> Our project detects emotions in social media text. The user enters a sentence, we clean the text, convert it into numeric features, use a trained model to predict the emotion, and show confidence scores plus visualizations. We compare classical models and also include DistilBERT as an advanced NLP bonus.

## 60. Common Beginner Mistakes

### Mistake 1: Running Streamlit from the wrong folder

Always run from the project folder:

```bash
cd "/home/jimmy_linux/anaconda_projects/nlp/Social media emotion analyzer GitHub"
streamlit run app.py
```

### Mistake 2: Missing dependencies

Fix:

```bash
pip install -r requirements.txt
```

### Mistake 3: Missing model file

Fix:

Make sure this exists:

```text
models/classical/tfidf_bigram__svm.joblib
```

### Mistake 4: Expecting DistilBERT to be fully in GitHub

DistilBERT is large, so it is stored in Google Drive.

### Mistake 5: Uploading huge files to GitHub

Do not upload:

```text
.safetensors
optimizer.pt
checkpoint folders
large random_forest.joblib files
```

## 61. Commands To Remember

Run app:

```bash
streamlit run app.py
```

Run debug checks:

```bash
python scripts/debug_project.py
```

Run tests:

```bash
python -m unittest tests.test_core
```

Train classical models:

```bash
python -m src.train_classical
```

Train DistilBERT:

```bash
python -m src.train_transformer --epochs 2 --batch-size 8
```

## 62. Final Advice For The Team

Do not memorize every line of code.

Understand these main ideas:

1. The app classifies emotion from text.
2. Text must be cleaned before prediction.
3. Text must be converted into numbers before ML models can use it.
4. We compare several feature extraction methods and models.
5. We evaluate using accuracy, precision, recall, F1-score, and confusion matrix.
6. The Streamlit app demonstrates prediction, confidence, visualizations, and diagnostics.
7. GitHub stores code and small files; Google Drive stores large models.

