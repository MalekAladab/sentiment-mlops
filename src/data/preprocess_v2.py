import re
import argparse
import pandas as pd
import emoji
from unidecode import unidecode
from sentence_transformers import SentenceTransformer, util
import torch

# ------------------------------------------------------------
# Regex globales
# ------------------------------------------------------------

RE_CONTROL = re.compile(r'[\u200B-\u200F\u202A-\u202E\u2060-\u206F]')
RE_REPEAT = re.compile(r'(.)\1{2,}')
RE_GIBBERISH = re.compile(r'[A-Za-z0-9]{30,}|[^\w\s]{5,}')
RE_URL = re.compile(r'http\S+|www\.\S+')
RE_MENTION = re.compile(r'@\w+')
RE_HASHTAG = re.compile(r'#\w+')
RE_NON_TEXT = re.compile(r'[^\w\s\u0600-\u06FF\U0001F300-\U0001FAFF]+', flags=re.UNICODE)


# ------------------------------------------------------------
# 🔥 Nettoyage avancé
# ------------------------------------------------------------
def clean_comment(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # invisibles
    text = RE_CONTROL.sub("", text)

    # URLs, mentions, hashtags
    text = RE_URL.sub(" ", text)
    text = RE_MENTION.sub(" ", text)
    text = RE_HASHTAG.sub(" ", text)

    # Correction des textes corrompus de type "Ù†Ù†Ø²ØªØª…"
    try:
        text = text.encode("latin1", "ignore").decode("utf-8", "ignore")
    except:
        pass

    text = re.sub(r'\s+', ' ', text).strip()

    # répétitions excessives
    text = RE_REPEAT.sub(r'\1\1', text)

    # spam / incohérent
    if RE_GIBBERISH.search(text):
        return ""

    # supprimer caractères non pertinents
    text = RE_NON_TEXT.sub(" ", text)

    # minuscules
    text = text.lower()

    # nettoyage final
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ------------------------------------------------------------
# 🔥 Filtrage embeddings : suppression des commentaires bizarres/outliers
# ------------------------------------------------------------
def filter_embeddings(df, threshold=0.25):
    print("[INFO] Chargement du modèle embeddings...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    print("[INFO] Calcul des embeddings...")
    texts = df["clean_text"].tolist()
    embeddings = model.encode(texts, convert_to_tensor=True)

    print("[INFO] Calcul du centroïde...")
    centroid = embeddings.mean(dim=0, keepdim=True)

    print("[INFO] Similarité cosinus...")
    scores = util.cos_sim(embeddings, centroid).squeeze()

    print(f"[INFO] Suppression des commentaires incohérents (sim<{threshold})...")
    mask = scores >= threshold

    return df[mask.cpu().numpy()]


# ------------------------------------------------------------
# Pipeline principal
# ------------------------------------------------------------
def process_csv(input_path: str, output_path: str):
    print(f"[INFO] Chargement CSV : {input_path}")
    df = pd.read_csv(input_path)

    text_col = df.columns[0] if "text" not in df.columns else "text"

    print(f"[INFO] Nettoyage de la colonne : {text_col}")
    df["clean_text"] = df[text_col].astype(str).apply(clean_comment)

    df = df[df["clean_text"].str.strip() != ""]

    print("[INFO] Filtrage via embeddings...")
    df = filter_embeddings(df)

    print(f"[INFO] Sauvegarde : {output_path}")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("[OK] Nettoyage + embeddings terminé ✔️")


# ------------------------------------------------------------
# Execution script
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing Comments v2 with Embeddings")
    parser.add_argument("--input", required=True, help="Chemin vers CSV brut")
    parser.add_argument("--output", required=True, help="Chemin vers CSV nettoyé + embeddings")
    args = parser.parse_args()

    process_csv(args.input, args.output)
