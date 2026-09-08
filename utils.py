import os
import re
import nltk

# Use user home directory or /tmp directory to avoid permission errors on Render
NLTK_DIR = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(NLTK_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DIR)

# -----------------------------
# Download NLTK resources
# -----------------------------
resources = [
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
]

for path, package in resources:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(package, download_dir=NLTK_DIR, quiet=True)

# -----------------------------
# Imports AFTER download
# -----------------------------
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# -----------------------------
# Cleaning Functions
# -----------------------------
def remove_html(text):
    return re.sub(r"<.*?>", "", text)

def remove_url(text):
    return re.sub(r"http[s]?://\S+|www\.\S+", "", text)

def remove_punctuation(text):
    return re.sub(r"[^\w\s]", "", text)

def preprocess_text(text):
    text = text.lower()
    text = remove_html(text)
    text = remove_url(text)
    text = remove_punctuation(text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)