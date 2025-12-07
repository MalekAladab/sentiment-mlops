import pandas as pd
import re
import nltk
from nltk.corpus import stopwords, words
from unidecode import unidecode

nltk.download("stopwords")
nltk.download("words")

STOPWORDS = set(stopwords.words("english"))
STOPWORDS.update(stopwords.words("french"))
STOPWORDS.update(stopwords.words("spanish"))
ENGLISH_WORDS = set(words.words())

def remove_emojis(text):
    return re.sub(r"[\U00010000-\U0010ffff]|[\U0001F600-\U0001F64F]|[\U0001F300-\U0001F5FF]", "", text)

def normalize_text(text):
    text = unidecode(text.lower())
    text = re.sub(r"\d+:\d+|\d+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def clean_text(text):
    text = normalize_text(text)
    text = remove_emojis(text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    words_list = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    words_list = [w for w in words_list if w in ENGLISH_WORDS or len(w) > 3]
    return " ".join(words_list)

def preprocess(input_path="data/raw/youtube_comments_v1.csv", output_path="data/processed/youtube_cleaned_v1.csv"):
    df = pd.read_csv(input_path, encoding="utf-8")  # <--- lire CSV au lieu de JSON
    df["clean_text"] = df["comment"].astype(str).apply(clean_text)
    df = df[df["clean_text"].str.split().apply(len) >= 2]
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f" Nettoyage V1 terminé → {output_path}")

if __name__ == "__main__":
    preprocess()
