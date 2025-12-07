# src/model/convert_sentiment_to_label.py
import pandas as pd
import os

# Fichiers d'entrée / sortie
INPUT = "data/processed/comments_with_sentiment.csv"
OUTPUT = "data/processed/comments_with_labels.csv"

# Mappage sentiment → label numérique
mapping = {"negative": 0, "neutral": 1, "positive": 2}

def convert():
    print(f"[INFO] Chargement du dataset : {INPUT}")
    df = pd.read_csv(INPUT, encoding="utf-8")

    if "sentiment" not in df.columns:
        raise ValueError("❌ La colonne 'sentiment' est absente du dataset !")

    print("[INFO] Conversion des sentiments en labels numériques...")
    df["label"] = df["sentiment"].map(mapping)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    print(f"[OK] Nouveau fichier sauvegardé : {OUTPUT}")
    return OUTPUT


if __name__ == "__main__":
    convert()
