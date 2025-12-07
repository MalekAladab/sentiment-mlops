from model.sentiment_predictor import add_sentiment_column, predict_sentiment, model
import mlflow
import mlflow.pytorch

# -----------------------------
# CONFIGURATION MLflow OBLIGATOIRE
# -----------------------------
mlflow.set_tracking_uri("file:./mlruns")    # <--- important !
mlflow.set_experiment("Sentiment-Analysis-MLOps")


if __name__ == "__main__":

    with mlflow.start_run(run_name="sentiment-pytorch"):

        # -----------------------------
        # Exemple de test rapide
        # -----------------------------
        test_phrases = [
            "I love this!",
            "This is bad 😢",
            "Neutral comment"
        ]

        for p in test_phrases:
            print(f"{p} -> {predict_sentiment(p)}")

        # -----------------------------
        # Ajouter la colonne sentiment + log CSV
        # -----------------------------
        input_csv = "data/processed/youtube_cleaned_v2.csv"
        output_csv = "data/processed/youtube_with_sentiment.csv"

        add_sentiment_column(input_csv, output_csv)

        # -----------------------------
        # Log modèle dans MLflow
        # -----------------------------
        mlflow.pytorch.log_model(model, artifact_path="sentiment_model")

        # Log paramètres
        mlflow.log_param("model_name", "cardiffnlp/twitter-xlm-roberta-base-sentiment")
        mlflow.log_param("batch_size", 64)

    print("Modèle et dataset avec sentiments enregistrés avec succès dans MLflow")
