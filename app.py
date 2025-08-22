import os
import sqlite3
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from io import BytesIO

DB_FILE = "top14_prod2_players.db"
TABLE = "players"

# -----------------------------------------------------------------
# IMAGE MANAGEMENT
# -----------------------------------------------------------------
def get_image_safe(player):
    fallback = "images/no_player.webp"
    img_path = f"images/photo_{player['player_id']}.jpg"

    if os.path.exists(img_path):
        return img_path

    if isinstance(player['photo'], str) and player['photo'].startswith("http"):
        try:
            r = requests.get(player['photo'], timeout=5)
            r.raise_for_status()
            return player['photo']
        except Exception:
            pass

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

    # Conversion en numérique (si possible)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    # Ratios
    if "poids_kg" in df.columns and "taille_cm" in df.columns:
        df['ratio_poids_taille'] = (pd.to_numeric(df['poids_kg'], errors='coerce') / pd.to_numeric(df['taille_cm'], errors='coerce')).replace([np.inf, -np.inf], np.nan).round(2)
    if "metres_parcourus" in df.columns and "courses" in df.columns:
        df['ratio_metres_courses'] = (pd.to_numeric(df['metres_parcourus'], errors='coerce') / pd.to_numeric(df['courses'], errors='coerce')).replace([np.inf, -np.inf], np.nan).round(2)
    if "temps_jeu_min" in df.columns and "nombre_matchs_joues" in df.columns:
        df['ratio_min_matchs'] = (pd.to_numeric(df['temps_jeu_min'], errors='coerce') / pd.to_numeric(df['nombre_matchs_joues'], errors='coerce')).replace([np.inf, -np.inf], np.nan).round(2)

    return df


# -----------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------
st.set_page_config(page_title="Rugby Top14/Prod2 Players", layout="wide")
st.title("🏉 Rugby Top14/Prod2 Players")

# Charger les joueurs
df = load_players()

# Créer joueurs types (moyennes hors joueurs à 0 match)
df_nonzero = df[df.get('nombre_matchs_joues', 0) > 0]
extra_players = []

if not df_nonzero.empty:
    # Joueurs types Top14 et ProD2
    top14_avg = df_nonzero[df_nonzero['club'].str.contains("Top14", case=False, na=False)].mean(numeric_only=True)
    prod2_avg = df_nonzero[df_nonzero['club'].str.contains("Prod2", case=False, na=False)].mean(numeric_only=True)
    extra_players.append({"nom": "Joueur type Top14", "club": "Top14", **top14_avg.to_dict()})
    extra_players.append({"nom": "Joueur type ProD2", "club": "ProD2", **prod2_avg.to_dict()})

    # Joueurs types par poste
    postes_groupes = {
        "Joueur type Avant": ["Pilier gauche", "Pilier droit", "Talonner", "1ère ligne"],
        "Joueur type 2eme ligne": ["2eme ligne gauche", "2eme ligne droit", "2ème ligne"],
        "Joueur type 3eme ligne": ["3eme ligne", "3eme ligne centre", "3ème ligne"],
        "Joueur type demi de melee": ["Demi de mêlée"],
        "Joueur type demi d'ouverture": ["Demi d'ouverture"],
        "Joueur type ailier": ["Ailier"],
        "Joueur type centre": ["Centre"],
        "Joueur type arriere": ["Arrière"]
    }

    for nom_type, postes in postes_groupes.items():
        subset = df_nonzero[df_nonzero['poste'].isin(postes)]
        if not subset.empty:
            avg_stats = subset.mean(numeric_only=True)
            extra_players.append({"nom": nom_type, "club": "Poste moyen", **avg_stats.to_dict()})

extra_df = pd.DataFrame(extra_players) if extra_players else pd.DataFrame()

df_extended = pd.concat([df, extra_df], ignore_index=True) if not extra_df.empty else df.copy()

# Téléchargement auto des photos manquantes
with st.spinner("Vérification des photos manquantes..."):
    n_new = download_missing_photos(df)
    if n_new > 0:
        st.success(f"{n_new} photo(s) téléchargée(s) automatiquement ✅")
    else:
        st.info("Toutes les photos sont déjà présentes 👍")

# Sélection des vrais joueurs
selected_names = st.multiselect("Choisir un ou plusieurs joueurs", df['nom'].sort_values().unique(), default=[df['nom'].sort_values().iloc[0]])
selected_players = df[df['nom'].isin(selected_names)]

# Sélection des joueurs types
selected_types = st.multiselect("Choisir un ou plusieurs joueurs types", extra_df['nom'].sort_values().unique() if not extra_df.empty else [])
selected_type_players = extra_df[extra_df['nom'].isin(selected_types)] if not extra_df.empty else pd.DataFrame()

# Concaténer la sélection
selected_players = pd.concat([selected_players, selected_type_players], ignore_index=True)

# Affichage des infos
for _, joueur in selected_players.iterrows():
    if "Joueur type" not in joueur['nom']:
        st.subheader(joueur['nom'])
        photo_to_show = get_image_safe(joueur)
        st.image(photo_to_show, caption=joueur['club'], width=150)

        st.json({
            "Club": joueur['club'],
            "Poste": joueur.get('poste', 'N/A'),
            "Âge": joueur.get('age', 'N/A'),
            "Taille (cm)": joueur.get('taille_cm', 'N/A'),
            "Poids (kg)": joueur.get('poids_kg', 'N/A'),
            "Ratio poids/taille": joueur.get('ratio_poids_taille', 'N/A')
        })
    else:
        st.subheader(joueur['nom'])
        st.info("📊 Joueur type basé sur la moyenne des stats.")

# Colonnes numériques disponibles
numeric_cols = df_extended.select_dtypes(include=[np.number]).columns.tolist()
exclude_cols = ["player_id"]
stat_cols = [c for c in numeric_cols if c not in exclude_cols]

# Sélecteur de stats dynamiques
selected_stats = st.multiselect(
    "Choisir les statistiques à afficher dans le radar",
    options=stat_cols,
    default=stat_cols[:5] if len(stat_cols) > 5 else stat_cols
)

if selected_stats and not selected_players.empty:
    radar_data = []
    for _, joueur in selected_players.iterrows():
        radar_data.append({
            "Joueur": joueur['nom'],
            **{stat: joueur.get(stat, np.nan) for stat in selected_stats}
        })

    radar_df = pd.DataFrame(radar_data)
    radar_df_melt = radar_df.melt(id_vars="Joueur", value_vars=selected_stats, var_name="Stat", value_name="Valeur")

    # Palette de couleurs distinctes
    color_sequence = px.colors.qualitative.Set1 + px.colors.qualitative.Set2 + px.colors.qualitative.Set3

    fig = px.line_polar(
        radar_df_melt,
        r="Valeur",
        theta="Stat",
        color="Joueur",
        line_close=True,
        color_discrete_sequence=color_sequence,
        hover_name="Stat",
        hover_data={"Valeur": ":.2f", "Joueur": True},
        markers=True
    )
    fig.update_traces(fill="toself", mode="lines+markers")
    fig.update_layout(hovermode="closest", width=900, height=700)
    st.plotly_chart(fig, use_container_width=True)

    # Tableau comparatif chiffré
    st.subheader("📊 Tableau comparatif des joueurs")
    table_df = radar_df.set_index("Joueur").T
    st.dataframe(table_df)

    # Export CSV uniquement
    st.download_button("⬇️ Télécharger en CSV", table_df.to_csv().encode("utf-8"), file_name="comparatif_joueurs.csv", mime="text/csv")

else:
    st.warning("Veuillez sélectionner au moins une statistique et un joueur pour afficher le radar.")



