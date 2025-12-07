import os
import argparse
import json
from googleapiclient.discovery import build
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm


def load_api_key():
    """Load API key from .env"""
    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise ValueError("❌ ERROR: YOUTUBE_API_KEY introuvable dans .env")
    return api_key


def get_comments(video_id, api_key, max_results=100):
    """Scrap les commentaires YouTube d’une vidéo en paginant"""

    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results,
        textFormat="plainText"
    )

    print(f" Scraping commentaires pour video: {video_id}")

    while request:
        response = request.execute()

        for item in response["items"]:
            snippet = item["snippet"]["topLevelComment"]["snippet"]

            comments.append({
                "author": snippet.get("authorDisplayName"),
                "comment": snippet.get("textDisplay"),
                "like_count": snippet.get("likeCount"),
                "published_at": snippet.get("publishedAt"),
            })

        request = youtube.commentThreads().list_next(request, response)

    print(f"✅ {len(comments)} commentaires récupérés.")
    return comments


def save_to_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"💾 JSON sauvegardé --> {path}")


def save_to_csv(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"💾 CSV sauvegardé --> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_id", required=True, help="ID de la vidéo YouTube")
    parser.add_argument("--out_json", default="data/raw/youtube_comments.json")
    parser.add_argument("--out_csv", default="data/raw/youtube_comments.csv")
    args = parser.parse_args()

    api_key = load_api_key()
    comments = get_comments(args.video_id, api_key)

    save_to_json(comments, args.out_json)
    save_to_csv(comments, args.out_csv)


if __name__ == "__main__":
    main()
