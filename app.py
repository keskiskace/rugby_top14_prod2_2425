import os
import sqlite3
import base64
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

DB_FILE = "top14_prod2_players.db"
TABLE = "players"

# -----------------------------------------------------------------
# IMAGE MANAGEMENT
# -----------------------------------------------------------------
def get_image_base64(path, fallback="images/no_player.webp"):
    try:
        with open(path, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        with open(fallback, "rb") as f:
            data = f.read()
    return base64.b64encode(data).decode()


def download_missing_photos(df, img_dir="images"):
    os.makedirs(img_dir, exist_ok=True)
    missing_count = 0
    for _, row in df.iterrows():
        player_id = row["player_id"]
        url = row["photo"]
        if not url or pd.isna(url):
            continue
        img_path = os.path.join(img_dir, f"photo_{player_id}.jpg")
        if not os.path.exists(img_path):
            try:
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                with open(img_path, "wb") as f:
                    f.write(r.content)
                missing_count += 1
            except Exception as e:
                print(f"[ERREUR] {row['nom']} ({url}) : {e}")
    return missing_count


# -----------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------
@st.cache_data
def load_players():
    with sqlite3.connect(DB_FILE) as con:
        df = pd.read_sql(f"SELECT * FROM {TABLE}", con)

    # Conversion en numérique
    df['poids_kg'] = pd.to_numeric(df['poids_kg'], errors='coerce')
    df['taille_cm'] = pd.to_numeric(df['taille_cm'], errors='coerce')
    df['courses'] = pd.to_numeric(df['courses'], errors='coerce')
    df['metres_parcourus'] = pd.to_numeric(df['metres_parcourus'], errors='coerce')
    df['temps_jeu_min'] = pd.to_numeric(df['temps_jeu_min'], errors='coerce')
    df['nombre_matchs_joues'] = pd.to_numeric(df['nombre_matchs_joues'], errors='coerce')

    # Ratios
    df['ratio_poids_taille'] = (df['poids_kg'] / df['taille_cm']).replace([np.inf, -np.inf], np.nan).round(2)
    df['ratio_metres_courses'] = (df['metres_parcourus'] / df['courses']).replace([np.inf, -np.inf], np.nan).round(2)
    df['ratio_min_matchs'] = (df['temps_jeu_min'] / df['nombre_matchs_joues']).replace([np.inf, -np.inf], np.nan).round(2)

    return df


# -----------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------
st.set_page_config(page_title="Rugby Top14/Prod2 Players", layout="wide")
st.title("🏉 Rugby Top14/Prod2 Players")

# Charger les joueurs
df = load_players()

# Téléchargement auto des photos manquantes
with st.spinner("Vérification des photos manquantes..."):
    n_new = download_missing_photos(df)
    if n_new > 0:
        st.success(f"{n_new} photo(s) téléchargée(s) automatiquement ✅")
    else:
        st.info("Toutes les photos sont déjà présentes 👍")

# Sélection d'un joueur
selected_name = st.selectbox("Choisir un joueur", df['nom'].sort_values().unique())
joueur = df[df['nom'] == selected_name].iloc[0]

# Affichage de la photo
img_path = f"images/photo_{joueur['player_id']}.jpg"
img_base64 = get_image_base64(img_path)

st.subheader(joueur['nom'])
st.image(img_path, caption=joueur['club'], width=200)

st.write("Détails du joueur :")
st.json({
    "Club": joueur['club'],
    "Poste": joueur['poste'],
    "Âge": joueur['age'],
    "Taille (cm)": joueur['taille_cm'],
    "Poids (kg)": joueur['poids_kg'],
    "Ratio poids/taille": joueur['ratio_poids_taille']
})

# Radar chart des stats principales
radar_stats = {
    "Matchs joués": joueur['nombre_matchs_joues'],
    "Temps de jeu (min)": joueur['temps_jeu_min'],
    "Courses": joueur['courses'],
    "Mètres parcourus": joueur['metres_parcourus'],
    "Plaquages réussis": joueur['plaquages_reussis'],
}

radar_df = pd.DataFrame({
    "Stat": list(radar_stats.keys()),
    "Valeur": list(radar_stats.values())
})

fig = px.line_polar(radar_df, r="Valeur", theta="Stat", line_close=True)
fig.update_traces(fill="toself")
st.plotly_chart(fig, use_container_width=True)
