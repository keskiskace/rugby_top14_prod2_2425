import os
import sqlite3
import pandas as pd
import requests

# Chemins
DB_FILE = "top14_prod2_players.db"
TABLE = "players"
IMG_DIR = "images"

# Création du dossier images s'il n'existe pas
os.makedirs(IMG_DIR, exist_ok=True)

# Connexion à la DB
with sqlite3.connect(DB_FILE) as con:
    df = pd.read_sql(f"SELECT player_id, photo FROM {TABLE}", con)

# Boucle de téléchargement
for _, row in df.iterrows():
    player_id = row["player_id"]
    url = row["photo"]

    if not url or pd.isna(url):
        continue  # pas de photo dans la DB

    img_path = os.path.join(IMG_DIR, f"photo_{player_id}.jpg")

    # Ne pas retélécharger si déjà présent
    if os.path.exists(img_path):
        continue

    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        with open(img_path, "wb") as f:
            f.write(r.content)
        print(f"[OK] Photo téléchargée pour joueur {player_id}")
    except Exception as e:
        print(f"[ERREUR] Joueur {player_id} ({url}) : {e}")
