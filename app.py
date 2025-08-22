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
def get_image_safe(player):
    fallback = "images/no_player.webp"
    img_path = f"images/photo_{player['player_id']}.jpg"

    # 1. Vérifie si image locale existe
    if os.path.exists(img_path):
        return img_path

    # 2. Sinon, essaye l'URL depuis la DB
    if isinstance(player['photo'], str) and player['photo'].startswith("http"):
        try:
            r = requests.get(player['photo'], timeout=5)
            r.raise_for_status()
            return player['photo']
        except Exception:
            pass

    # 3. Fallback générique
    return fallback


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

# Créer joueurs types (moyennes hors joueurs à 0 match)
df_nonzero = df[df['nombre_matchs_joues'] > 0]
top14_avg = df_nonzero[df_nonzero['club'].str.contains("Top14", case=False)].mean(numeric_only=True)
prod2_avg = df_nonzero[df_nonzero['club'].str.contains("Prod2", case=False)].mean(numeric_only=True)

joueur_type_top14 = {"nom": "Joueur type Top14", "club": "Top14", **top14_avg.to_dict()}
joueur_type_prod2 = {"nom": "Joueur type ProD2", "club": "ProD2", **prod2_avg.to_dict()}

# Fusionner dans df pour pouvoir sélectionner
extra_df = pd.DataFrame([joueur_type_top14, joueur_type_prod2])
df_extended = pd.concat([df, extra_df], ignore_index=True)

# Téléchargement auto des photos manquantes
with st.spinner("Vérification des photos manquantes..."):
    n_new = download_missing_photos(df)
    if n_new > 0:
        st.success(f"{n_new} photo(s) téléchargée(s) automatiquement ✅")
    else:
        st.info("Toutes les photos sont déjà présentes 👍")

# Sélection d'un ou plusieurs joueurs
selected_names = st.multiselect("Choisir un ou plusieurs joueurs", df_extended['nom'].sort_values().unique(), default=[df_extended['nom'].sort_values().iloc[0]])
selected_players = df_extended[df_extended['nom'].isin(selected_names)]

# Affichage des photos et infos (uniquement pour vrais joueurs)
for _, joueur in selected_players.iterrows():
    if "Joueur type" not in joueur['nom']:
        st.subheader(joueur['nom'])
        photo_to_show = get_image_safe(joueur)
        st.image(photo_to_show, caption=joueur['club'], width=150)

        st.json({
            "Club": joueur['club'],
            "Poste": joueur['poste'],
            "Âge": joueur['age'],
            "Taille (cm)": joueur['taille_cm'],
            "Poids (kg)": joueur['poids_kg'],
            "Ratio poids/taille": joueur['ratio_poids_taille']
        })
    else:
        st.subheader(joueur['nom'])
        st.info("📊 Joueur type basé sur la moyenne des stats.")

# Radar chart des stats principales
base_stats = {
    "Matchs joués": "nombre_matchs_joues",
    "Temps de jeu (min)": "temps_jeu_min",
    "Courses": "courses",
    "Mètres parcourus": "metres_parcourus",
    "Plaquages réussis": "plaquages_reussis",
}

# Sélecteur de stats à afficher
selected_stats = st.multiselect(
    "Choisir les statistiques à afficher dans le radar",
    options=list(base_stats.keys()),
    default=list(base_stats.keys())
)

if selected_stats and not selected_players.empty:
    radar_data = []
    for _, joueur in selected_players.iterrows():
        radar_data.append({
            "Joueur": joueur['nom'],
            **{stat: joueur.get(base_stats[stat], np.nan) for stat in selected_stats}
        })

    radar_df = pd.DataFrame(radar_data)
    radar_df_melt = radar_df.melt(id_vars="Joueur", value_vars=selected_stats, var_name="Stat", value_name="Valeur")

    fig = px.line_polar(radar_df_melt, r="Valeur", theta="Stat", color="Joueur", line_close=True)
    fig.update_traces(fill="toself")
    st.plotly_chart(fig, use_container_width=True)

    # Tableau comparatif chiffré
    st.subheader("📊 Tableau comparatif des joueurs")
    table_df = radar_df.set_index("Joueur").T
    st.dataframe(table_df)

    # Export CSV / Excel
    st.download_button("⬇️ Télécharger en CSV", table_df.to_csv().encode("utf-8"), file_name="comparatif_joueurs.csv", mime="text/csv")
    st.download_button("⬇️ Télécharger en Excel", table_df.to_excel("comparatif_joueurs.xlsx"), file_name="comparatif_joueurs.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

else:
    st.warning("Veuillez sélectionner au moins une statistique et un joueur pour afficher le radar.")
