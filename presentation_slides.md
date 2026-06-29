# Presentation Color Theme (To match your Canva Poster)

To ensure your presentation looks like it belongs with your poster, use this color palette in Canva:
* **Background Color:** Deep Royal Blue (Hex: `#191970` or `#1E3A8A`) - *Use this for the main background of your slides.*
* **Accent & Header Backgrounds:** Coral / Warm Red (Hex: `#E05246` or `#E85A4F`) - *Use this for title bars, important highlights, and the footer.*
* **Content Box Backgrounds:** Pale Lavender / Off-White (Hex: `#F0F4F8` or `#EBEBF2`) - *Use this for the boxes where your main text goes, just like your poster.*
* **Main Text Color:** Dark Navy Blue (Hex: `#0F172A`) - *For high contrast inside the content boxes.*
* **Highlight Color:** Golden Yellow (Hex: `#FBBF24`) - *For icons or small emphasis text.*

---

# Slide Deck Outline (9 Slides)

*Since all team members must speak, I have divided the presentation into three parts (Speaker 1, Speaker 2, and Speaker 3).*

### Slide 1: Title Slide (Speaker 1)
* **Visuals:** Large text "Social Media Emotion Analyzer", Team names, Course Name, and a relevant graphic (like the social media icons on your poster).
* **Talking Points (1 min):** 
  * "Good morning/afternoon everyone. Welcome to our presentation for Natural Language Processing."
  * "Our project is the Social Media Emotion Analyzer."
  * Introduce team members: Haziq, Jimmy, and Arif.

### Slide 2: Problem Statement & Objectives (Speaker 1)
* **Visuals:** Two content boxes. Box 1: "The Problem" (Informal text, noise, URLs). Box 2: "Our Objective" (Build a reliable NLP classifier).
* **Talking Points (1 min):**
  * "Humans can easily read a tweet and know if the person is angry or happy, but machines struggle."
  * "Social media is messy: it has hashtags, URLs, and slang."
  * "Our objective was to design a robust NLP model that cleans this noise and accurately classifies text into six core emotions."

### Slide 3: Dataset Overview (Speaker 1)
* **Visuals:** The Class Distribution Bar Chart (Diagram 5) showing the perfectly balanced ~5,000 samples per class.
* **Talking Points (1 min):**
  * "We combined Google's Reddit 'GoEmotions' dataset with the 'DAIR Emotion' Twitter dataset."
  * "We used rigorous downsampling and oversampling to create a perfectly balanced dataset of exactly 29,997 records across six emotions: Anger, Sadness, Surprise, Fear, Love, and Joy."

### Slide 4: NLP Pipeline (Speaker 2)
* **Visuals:** A simple flowchart: Raw Text ➔ Regex Cleaning (URLs/Punctuation) ➔ Stopword Removal ➔ WordNet Lemmatization ➔ Vectorization.
* **Talking Points (1 min):**
  * "Before training, we built a comprehensive text preprocessing pipeline."
  * "We used Python's regular expressions to remove noisy elements like URLs and user mentions."
  * "We removed standard stopwords and applied a WordNet Lemmatizer to convert words like 'feeling' back to their root form 'feel'."

### Slide 5: Model Comparison & Results (Speaker 2)
* **Visuals:** The Model Comparison Table (Diagram 2) or a horizontal bar chart of the F1-Scores.
* **Talking Points (1.5 mins):**
  * "We compared 5 different models, testing both Bag-of-Words and TF-IDF approaches."
  * "Our Count Bigram + SVM model performed incredibly well for a classical model, achieving an F1-Score of 0.9182."
  * "However, our advanced model, a fine-tuned DistilBERT Transformer, performed the best. It achieved an accuracy and F1-score of 0.9335."

### Slide 6: LIVE DEMO (Speaker 3)
* **Visuals:** A transition slide that says "Live Demonstration". (Switch screen to Streamlit App).
* **Talking Points (3.5 mins):**
  * *Open the Streamlit App.*
  * "Here is our interactive dashboard. As you can see on the left, we have Navigation."
  * *Go to Text Analyzer.* "Let's test our best model, DistilBERT. I'll type in a social media sentence: 'I am so terrified of the upcoming exam!'"
  * *Click Analyze.* "As you can see, the model correctly predicts 'Fear' and shows the confidence scores."
  * *Go to Model Info / Visualizations.* Briefly show the interactive charts to prove the app has multiple features.

### Slide 7: Key Visualizations & Insights (Speaker 3)
* **Visuals:** The Word Cloud (Diagram 4) and the Top 20 Words chart (Diagram 6).
* **Talking Points (1 min):**
  * "Looking at our data insights, the word 'feel' dominates the entire dataset with nearly 18,000 occurrences."
  * "This proves that our social media data relies heavily on first-person narratives and direct emotional expressions."
  * "We also found that removing punctuation splits contractions—which is why the single letter 's' appears as a frequent token."

### Slide 8: Challenges Faced (Speaker 1 or 2)
* **Visuals:** Bullet points: 1. Messy Data (Solved via regex pipeline). 2. Memory Constraints (Solved via Count Bigram). 3. Complex Language (Sarcasm/Negation).
* **Talking Points (1 min):**
  * "Our biggest challenge was the high dimensionality of the feature space. Our initial vocabulary exceeded 20,000 terms, causing memory crashes."
  * "We solved this by optimizing our preprocessing and finding that raw Bigram counts were more computationally efficient and accurate than heavy TF-IDF matrices."

### Slide 9: Conclusion & Future Work (Speaker 3)
* **Visuals:** Summary box and a "Future Improvements" box.
* **Talking Points (1 min):**
  * "To conclude, our DistilBERT pipeline successfully mitigates social media noise for highly accurate emotion detection."
  * "In the future, we would like to implement larger models like RoBERTa to better handle complex sarcasm."
  * "Thank you for listening. We are now open for Q&A, and you can view our full poster at our booth!"
