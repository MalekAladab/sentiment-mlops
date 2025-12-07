# src/model/train_finetune_cpu.py

import os
import argparse
import numpy as np
import pandas as pd
import torch
import mlflow
import mlflow.pytorch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from sklearn.metrics import accuracy_score, f1_score

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
LABELS = ["negative", "neutral", "positive"]

# -------------------------------------------------------
# Métriques
# -------------------------------------------------------
def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1}

# -------------------------------------------------------
# Charger dataset CSV
# -------------------------------------------------------
def load_csv_dataset(path_csv, text_col="clean_text", label_col="label", sample_size=None):
    df = pd.read_csv(path_csv, encoding="utf-8")
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError("❌ Le CSV doit contenir les colonnes : clean_text et label")
    
    if sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    ds = Dataset.from_pandas(
        df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    )
    return ds

# -------------------------------------------------------
# Tokenization
# -------------------------------------------------------
def tokenize_fn(examples, tokenizer, max_length=64):
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=max_length
    )

# -------------------------------------------------------
# Entraînement / Fine-tuning
# -------------------------------------------------------
def train(args):
    mlflow.set_experiment(args.mlflow_experiment)

    print("[INFO] Chargement dataset...")
    ds = load_csv_dataset(args.input_csv, sample_size=args.sample_size)

    # Train/test split
    ds = ds.train_test_split(test_size=args.test_size, seed=42)
    train_ds = ds["train"]
    eval_ds = ds["test"]

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    train_ds = train_ds.map(lambda x: tokenize_fn(x, tokenizer, args.max_length), batched=True)
    eval_ds = eval_ds.map(lambda x: tokenize_fn(x, tokenizer, args.max_length), batched=True)

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    eval_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    model.gradient_checkpointing_enable()  # réduit la mémoire utilisée pendant le backprop

    # CPU seulement
    device = torch.device("cpu")
    model.to(device)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        save_total_limit=2,
        report_to="none",
        fp16=False,  # désactiver FP16 sur CPU
        gradient_accumulation_steps=2  # simule un batch plus grand sans consommer trop de RAM
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics((p.predictions, p.label_ids)),
    )

    # MLflow Run
    with mlflow.start_run(run_name="finetune_roberta_cpu") as run:
        print("[INFO] Entraînement en cours...")
        trainer.train()

        print("[INFO] Évaluation...")
        metrics = trainer.evaluate()
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))

        # Sauvegarde modèle local
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Log model MLflow
        mlflow.pytorch.log_model(pytorch_model=model, artifact_path="sentiment_roberta_cpu")

        print("[OK] Fine-tuning terminé !")
        return run.info.run_id

# -------------------------------------------------------
# Main
# -------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_csv", type=str,
                        default="data/processed/comments_with_labels.csv")
    parser.add_argument("--output_dir", type=str, default="models/sentiment_ft_cpu")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)  # batch très petit pour CPU
    parser.add_argument("--max_length", type=int, default=32) # réduit pour CPU
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--mlflow_experiment", type=str, default="Sentiment-Finetune-CPU")
    parser.add_argument("--sample_size", type=int, default=2000)  # sous-échantillon pour test CPU

    args = parser.parse_args()
    train(args)
