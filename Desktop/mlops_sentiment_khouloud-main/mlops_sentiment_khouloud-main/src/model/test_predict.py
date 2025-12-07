# test_predict.py
import requests

URL = "http://127.0.0.1:8000/predict"  # endpoint FastAPI

# Texte à prédire
data = {
    "texts": [
        "Je suis très content",
        "C’est vraiment nul",
        "Le service était correct"
    ]
}

response = requests.post(URL, json=data)

if response.status_code == 200:
    print("✅ Réponse du serveur :")
    print(response.json())
else:
    print(f"❌ Erreur {response.status_code} : {response.text}")
