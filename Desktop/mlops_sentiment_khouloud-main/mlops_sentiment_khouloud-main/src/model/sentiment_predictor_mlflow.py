import os
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import mlflow
import mlflow.pytorch

# -------------------------------------------------------------
# Configuration modèle Hugging Face
# -------------------------------------------------------------
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

# Tokenizer + modèle
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

LABELS = ["negative", "neutral", "positive"]

# -------------------------------------------------------------
# Prédiction batch
# -------------------------------------------------------------
def predict_sentiment_batch(texts, batch_size=32):
    sentiments = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Prediction"):
        batch_texts = texts[i:i + batch_size]

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
        sentiments.extend([LABELS[np.argmax(p)] for p in probs])

    return sentiments

# -------------------------------------------------------------
# Pipeline MLflow
# -------------------------------------------------------------
def run_sentiment_pipeline(input_csv, output_csv):

    with mlflow.start_run(run_name="youtube_sentiment_analysis") as run:

        print(f"[INFO] Chargement dataset : {input_csv}")
        df = pd.read_csv(input_csv, encoding="utf-8")

        if "clean_text" not in df.columns:
            raise ValueError("❌ La colonne 'clean_text' est absente du dataset !")

        texts = df["clean_text"].astype(str).tolist()

        print("[INFO] Prédiction des sentiments...")
        df["sentiment"] = predict_sentiment_batch(texts, batch_size=64)

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] Fichier généré → {output_csv}")

        # ---------- Log artefact ----------
        mlflow.log_artifact(output_csv, artifact_path="predictions")

        # ---------- Log metrics ----------
        sentiment_counts = df["sentiment"].value_counts().to_dict()
        for label in LABELS:
            mlflow.log_metric(f"num_{label}", sentiment_counts.get(label, 0))

        # ---------- Log modèle ----------
        mlflow.pytorch.log_model(model, artifact_path="sentiment_model")

        print(f"[INFO] Run MLflow terminé. Run ID : {run.info.run_id}")

    return df

# -------------------------------------------------------------
# Exécution script
# -------------------------------------------------------------
if __name__ == "__main__":
    INPUT = "data/processed/youtube_cleaned_v2.csv"
    OUTPUT = "data/processed/comments_with_sentiment.csv"
    run_sentiment_pipeline(INPUT, OUTPUT)
