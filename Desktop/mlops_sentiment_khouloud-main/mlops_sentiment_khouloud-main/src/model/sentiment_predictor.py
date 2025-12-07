import os
import pandas as pd
import torch
import numpy as np
from transformers import XLMRobertaTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# -------------------------------------------------------------
# Modèle HuggingFace
# -------------------------------------------------------------
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

# Charger tokenizer (slow) + modèle
tokenizer = XLMRobertaTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False   # ⚠️ obligatoire pour éviter les bugs Windows
)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# GPU si disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Labels
LABELS = ["negative", "neutral", "positive"]


# -------------------------------------------------------------
# 🔥 1. Prédiction sur un seul texte
# -------------------------------------------------------------
def predict_sentiment(text: str) -> str:
    if not isinstance(text, str) or len(text.strip()) == 0:
        return "neutral"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    return LABELS[np.argmax(probs)]


# -------------------------------------------------------------
# 🔥 2. Prédiction en batch
# -------------------------------------------------------------
def predict_sentiment_batch(texts, batch_size=32):
    sentiments = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Prediction"):
        batch_texts = texts[i: i + batch_size]

        encodings = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)

        with torch.no_grad():
            logits = model(**encodings).logits

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        batch_sentiments = [LABELS[np.argmax(p)] for p in probs]

        sentiments.extend(batch_sentiments)

    return sentiments


# -------------------------------------------------------------
# 🔥 3. Ajout de la colonne sentiment à un CSV
# -------------------------------------------------------------
def add_sentiment_column(input_csv: str, output_csv: str):
    print(f"[INFO] Chargement du dataset : {input_csv}")
    df = pd.read_csv(input_csv, encoding="utf-8")

    if "clean_text" not in df.columns:
        raise ValueError("❌ La colonne 'clean_text' est absente du dataset !")

    texts = df["clean_text"].astype(str).tolist()

    print("[INFO] Prédiction des sentiments...")
    df["sentiment"] = predict_sentiment_batch(texts, batch_size=64)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] Fichier généré : {output_csv}")

    return output_csv


# -------------------------------------------------------------
# 🔥 4. Exécution directe
# -------------------------------------------------------------
if __name__ == "__main__":
    INPUT = "data/processed/youtube_cleaned_v2.csv"
    OUTPUT = "data/processed/comments_with_sentiment.csv"

    print("[INFO] Lancement de l'analyse des sentiments...")
    add_sentiment_column(INPUT, OUTPUT)
    print("[INFO] Terminé !")


#✔️ Chargement modèle + tokenizer Hugging Face
#✔️ Prédiction batch optimisée
#✔️ Gestion GPU/CPU
#✔️ Nettoyage minimal pour éviter erreurs
#✔️ Ajout colonne sentiment dans ton CSV
#✔️ Code structuré pour DVC + MLflow

