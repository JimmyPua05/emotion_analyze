import json
from pathlib import Path

notebook_path = Path("/home/jimmy_linux/anaconda_projects/nlp/Social media emotion analyzer GitHub/notebooks/model_development.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

new_source = [
    "import re\n",
    "import nltk\n",
    "from nltk.corpus import wordnet\n",
    "from nltk.stem import WordNetLemmatizer\n",
    "\n",
    "try:\n",
    "    nltk.data.find(\"corpora/wordnet\")\n",
    "except LookupError:\n",
    "    nltk.download(\"wordnet\", quiet=True)\n",
    "try:\n",
    "    nltk.data.find(\"corpora/omw-1.4\")\n",
    "except LookupError:\n",
    "    nltk.download(\"omw-1.4\", quiet=True)\n",
    "\n",
    "LEMMATIZER = WordNetLemmatizer()\n",
    "STOPWORDS = {\"a\", \"am\", \"an\", \"and\", \"are\", \"as\", \"at\", \"be\", \"but\", \"by\", \"for\", \"from\", \"has\", \"have\", \"he\", \"i\", \"in\", \"is\", \"it\", \"its\", \"me\", \"my\", \"of\", \"on\", \"or\", \"our\", \"she\", \"so\", \"that\", \"the\", \"their\", \"them\", \"they\", \"this\", \"to\", \"was\", \"we\", \"were\", \"with\", \"you\", \"your\"}\n",
    "\n",
    "def clean_text(text: str) -> str:\n",
    "    text = \"\" if text is None else str(text).lower()\n",
    "    text = re.sub(r\"http\\S+|www\\.\\S+\", \" \", text)\n",
    "    text = re.sub(r\"@\\w+\", \" \", text)\n",
    "    text = re.sub(r\"#\", \"\", text)\n",
    "    text = re.sub(r\"[^a-z\\s]\", \" \", text)\n",
    "    tokens = [t for t in text.split() if t not in STOPWORDS]\n",
    "    lemmatized = [LEMMATIZER.lemmatize(LEMMATIZER.lemmatize(t, wordnet.VERB), wordnet.NOUN) for t in tokens]\n",
    "    return \" \".join(lemmatized)\n",
    "\n",
    "sample = 'I am feeling so happy today!!! @friend #Joy'\n",
    "print(\"Original:\", sample)\n",
    "print(\"Cleaned:\", clean_text(sample))\n",
    "\n",
    "# Apply to dataset\n",
    "df['clean_text_notebook'] = df['text'].apply(clean_text)\n"
]

for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "def clean_text" in "".join(cell.get("source", [])):
        cell["source"] = new_source
        break

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
