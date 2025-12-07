# src/model/serve_fastapi.py
import os
from typing import List
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(title="Sentiment API", description="API de prédiction de sentiment", version="1.0")

# ---------------------------
# Config modèle
# ---------------------------
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
LABELS = ["negative", "neutral", "positive"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = int(os.environ.get("PRED_BATCH_SIZE", 4))
MAX_LENGTH = int(os.environ.get("PRED_MAX_LENGTH", 32))

# ---------------------------
# Chargement modèle/tokenizer depuis HF Hub
# ---------------------------
print(f"[INFO] Chargement du modèle depuis Hugging Face Hub : {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
model.gradient_checkpointing_enable()  # réduit la mémoire utilisée
model.to(DEVICE)
model.eval()
print("[INFO] Modèle chargé avec succès !")

# ---------------------------
# Input Pydantic
# ---------------------------
class TextIn(BaseModel):
    texts: List[str]

# ---------------------------
# Endpoint racine pour test rapide
# ---------------------------
@app.get("/")
def root():
    return {"message": "Bienvenue sur Sentiment API ! Utilisez /docs pour tester les endpoints."}

# ---------------------------
# Prédiction par batch
# ---------------------------
def batch_predict(texts, batch_size=BATCH_SIZE):
    results = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(DEVICE)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1).tolist()
        for j, text in enumerate(batch_texts):
            results.append({
                "text": text,
                "pred_label": LABELS[preds[j]],
                "scores": probs[j].tolist()
            })
    return results

# ---------------------------
# Endpoint /predict
# ---------------------------
@app.post("/predict")
def predict(payload: TextIn):
    texts = payload.texts
    if len(texts) == 0:
        return {"error": "No texts provided."}
    results = batch_predict(texts)
    return {"predictions": results}

# ---------------------------
# Endpoint /health
# ---------------------------
@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}

# ---------------------------
# Run serveur
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("src.model.serve_fastapi:app", host="127.0.0.1", port=port, reload=True)
