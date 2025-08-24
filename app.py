import os
import sqlite3
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from io import BytesIO

DB_FILE = "top14_prod2_24_25_players_clubs.db"
TABLE = "players"

# -----------------------------------------------------------------
# IMAGE MANAGEMENT
# -----------------------------------------------------------------

def get_image_safe(player):
    fallback = "images/no_player.webp"
    img_path = f"images/photo_{player['player_id']}.jpg"

    if os.path.exists(img_path):
        return img_path

    if isinstance(player.get('photo', ''), str) and player.get('photo', '').startswith("http"):
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
        player_id = row.get("player_id")
        url = row.get("photo")
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
                print(f"[ERREUR] {row.get('nom','?')} ({url}) : {e}")
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

    # Ratios utiles
    if "poids_kg" in df.columns and "taille_cm" in df.columns:
        df['ratio_poids_taille'] = (
            pd.to_numeric(df['poids_kg'], errors='coerce') / pd.to_numeric(df['taille_cm'], errors='coerce')
        ).replace([np.inf, -np.inf], np.nan).round(2)
    if "metres_parcourus" in df.columns and "courses" in df.columns:
        df['ratio_metres_courses'] = (
            pd.to_numeric(df['metres_parcourus'], errors='coerce') / pd.to_numeric(df['courses'], errors='coerce')
        ).replace([np.inf, -np.inf], np.nan).round(2)
    if "temps_jeu_min" in df.columns and "nombre_matchs_joues" in df.columns:
        df['ratio_min_matchs'] = (
            pd.to_numeric(df['temps_jeu_min'], errors='coerce') / pd.to_numeric(df['nombre_matchs_joues'], errors='coerce')
        ).replace([np.inf, -np.inf], np.nan).round(2)

    return df


# -----------------------------------------------------------------
# LEAGUE COLUMN
# -----------------------------------------------------------------

def infer_league(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "division" in df.columns:
        df["__league__"] = df["division"].astype(str)
    else:
        df["__league__"] = None
    return df


# -----------------------------------------------------------------
# CUSTOM RADAR SCATTER WITH GRID
# -----------------------------------------------------------------

def make_scatter_radar(radar_df, selected_stats):
    fig = go.Figure()

    n_stats = len(selected_stats)
    angles = np.linspace(0, 2*np.pi, n_stats, endpoint=False)

    # Déterminer le max global pour la grille
    max_val = radar_df[selected_stats].apply(pd.to_numeric, errors="coerce").max().max()
    if not np.isfinite(max_val) or max_val <= 0:
        max_val = 1.0
    n_circles = 5
    step = max_val / n_circles

    # Palette de couleurs distinctes
    color_palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf", "#393b79", "#637939",
        "#8c6d31", "#843c39", "#7b4173"
    ]

    # Ajouter cercles concentriques + valeurs sur un axe (x positif)
    for i in range(1, n_circles+1):
        r = step * i
        circle_x = [r*np.cos(t) for t in np.linspace(0, 2*np.pi, 200)]
        circle_y = [r*np.sin(t) for t in np.linspace(0, 2*np.pi, 200)]
        fig.add_trace(go.Scatter(
            x=circle_x,
            y=circle_y,
            mode="lines",
            line=dict(color="lightgrey", dash="dot"),
            showlegend=False,
            hoverinfo="skip"
        ))
        fig.add_annotation(
            x=r,
            y=0,
            text=str(round(r, 1)),
            showarrow=False,
            font=dict(size=10, color="grey")
        )

    # Ajouter axes radiaux et labels
    for angle, stat in zip(angles, selected_stats):
        x_axis = [0, max_val*np.cos(angle)]
        y_axis = [0, max_val*np.sin(angle)]
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=y_axis,
            mode="lines",
            line=dict(color="lightgrey", dash="dot"),
            showlegend=False,
            hoverinfo="skip"
        ))
        fig.add_annotation(
            x=max_val*1.05*np.cos(angle),
            y=max_val*1.05*np.sin(angle),
            text=stat,
            showarrow=False,
            font=dict(size=12, color="black")
        )

    # Tracer chaque série
    for idx, (_, row) in enumerate(radar_df.iterrows()):
        r_values = [row.get(stat, np.nan) for stat in selected_stats]
        r_values = [np.nan if (not pd.notna(v)) else float(v) for v in r_values]
        r_values += [r_values[0]]
        theta = np.append(angles, angles[0])

        x = [0 if (v is np.nan or not np.isfinite(v)) else v*np.cos(t) for v, t in zip(r_values, theta)]
        y = [0 if (v is np.nan or not np.isfinite(v)) else v*np.sin(t) for v, t in zip(r_values, theta)]

        hover_texts = [f"{stat}: {'' if (val is np.nan or not np.isfinite(val)) else round(val, 1)}" for stat, val in zip(selected_stats, r_values[:-1])]
        hover_texts.append(hover_texts[0])

        color = color_palette[idx % len(color_palette)]

        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            name=row.get("Joueur", ""),
            fill="toself",
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
            line=dict(color=color),
            marker=dict(color=color)
        ))

    fig.update_layout(
        width=800,
        height=600,
        hovermode="closest",
        dragmode="pan",
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False)
    )

    return fig


# -----------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------

st.set_page_config(page_title="Rugby Top14/Prod2 Players", layout="wide")
st.title("🏉 Rugby Top14/Prod2 Players")

# Charger les joueurs
df = load_players()

# Filtrer les joueurs avec au moins 1 match (si la colonne existe)
if 'nombre_matchs_joues' in df.columns:
    df_nonzero = df[pd.to_numeric(df['nombre_matchs_joues'], errors='coerce').fillna(0) > 0].copy()
else:
    df_nonzero = df.copy()

# Déterminer la ligue avec la colonne division
df_labeled = infer_league(df_nonzero)

# Construire les joueurs types
extra_players = []

if not df_labeled.empty and '__league__' in df_labeled.columns:
    top14_mask = df_labeled['__league__'].str.contains("Top14", case=False, na=False)
    prod2_mask = df_labeled['__league__'].str.contains("ProD2", case=False, na=False)

    if top14_mask.any():
        top14_avg = df_labeled[top14_mask].mean(numeric_only=True).round(1)
        extra_players.append({"nom": "Joueur type Top14", "club": "Top14", **top14_avg.to_dict()})

    if prod2_mask.any():
        prod2_avg = df_labeled[prod2_mask].mean(numeric_only=True).round(1)
        extra_players.append({"nom": "Joueur type ProD2", "club": "ProD2", **prod2_avg.to_dict()})

# Joueurs types par poste
postes_groupes = {
    "Joueur type Avant": ["Pilier gauche", "Pilier droit", "Talonneur", "Talonner", "1ère ligne"],
    "Joueur type 2eme ligne": ["2eme ligne gauche", "2eme ligne droit", "2ème ligne gauche", "2ème ligne droit", "Deuxième ligne"],
    "Joueur type 3eme ligne": ["3eme ligne", "3ème ligne", "3eme ligne centre", "3ème ligne centre"],
    "Joueur type demi de melee": ["Demi de mêlée", "Demi de melee"],
    "Joueur type demi d'ouverture": ["Demi d'ouverture", "Ouverture"],
    "Joueur type ailier": ["Ailier", "Ailiers"],
    "Joueur type centre": ["Centre", "Centres"],
    "Joueur type arriere": ["Arrière", "Arriere"]
}

for nom_type, postes in postes_groupes.items():
    if 'poste' in df_nonzero.columns:
        subset = df_nonzero[df_nonzero['poste'].isin(postes)]
        if not subset.empty:
            avg_stats = subset.mean(numeric_only=True).round(1)
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

# ----------------------
# SECTION JOUEURS
# ----------------------

# Sélection des vrais joueurs
selected_names = st.multiselect(
    "Choisir un ou plusieurs joueurs",
    df['nom'].sort_values().unique(),
    default=[df['nom'].sort_values().iloc[0]] if len(df) else []
)
selected_players = df[df['nom'].isin(selected_names)]

# Sélection des joueurs types
selected_types = st.multiselect(
    "Choisir un ou plusieurs joueurs types",
    extra_df['nom'].sort_values().unique() if not extra_df.empty else []
)
selected_type_players = extra_df[extra_df['nom'].isin(selected_types)] if not extra_df.empty else pd.DataFrame()

# Concaténer la sélection
selected_players = pd.concat([selected_players, selected_type_players], ignore_index=True)

# Affichage des infos
for _, joueur in selected_players.iterrows():
    if "Joueur type" not in str(joueur.get('nom', '')):
        st.subheader(joueur.get('nom', ''))
        photo_to_show = get_image_safe(joueur)
        st.image(photo_to_show, caption=joueur.get('club', ''), width=150)

        st.json({
            "Club": joueur.get('club', 'N/A'),
            "Poste": joueur.get('poste', 'N/A'),
            "Âge": joueur.get('age', 'N/A'),
            "Taille (cm)": joueur.get('taille_cm', 'N/A'),
            "Poids (kg)": joueur.get('poids_kg', 'N/A'),
            "Ratio poids/taille": joueur.get('ratio_poids_taille', 'N/A')
        })
    else:
        st.subheader(joueur.get('nom', ''))
        st.info("📊 Joueur type basé sur la moyenne des stats.")

# Colonnes numériques disponibles (joueurs)
numeric_cols = df_extended.select_dtypes(include=[np.number]).columns.tolist()
exclude_cols = ["player_id"]
stat_cols = [c for c in numeric_cols if c not in exclude_cols]

# Sélecteur de stats dynamiques (joueurs)
selected_stats = st.multiselect(
    "Choisir les statistiques à afficher dans le radar",
    options=stat_cols,
    default=stat_cols[:5] if len(stat_cols) > 5 else stat_cols
)

if selected_stats and not selected_players.empty:
    radar_data = []
    for _, joueur in selected_players.iterrows():
        radar_data.append({
            "Joueur": joueur.get('nom', ''),
            **{stat: joueur.get(stat, np.nan) for stat in selected_stats}
        })

    radar_df = pd.DataFrame(radar_data)

    fig = make_scatter_radar(radar_df, selected_stats)

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "doubleClick": "reset"
        }
    )

    # Tableau comparatif chifré
    st.subheader("📊 Tableau comparatif des joueurs")
    table_df = radar_df.set_index("Joueur").T
    st.dataframe(table_df)

    # Export CSV uniquement
    st.download_button(
        "⬇️ Télécharger en CSV",
        table_df.to_csv().encode("utf-8"),
        file_name="comparatif_joueurs.csv",
        mime="text/csv"
    )
else:
    st.warning("Veuillez sélectionner au moins une statistique et un joueur pour afficher le radar.")


# ----------------------
# SECTION CLUBS
# ----------------------

st.header("📊 Comparaison des Clubs")

# Charger les données clubs
with sqlite3.connect(DB_FILE) as con:
    clubs_df = pd.read_sql("SELECT * FROM clubs", con)

# Sélection des clubs
selected_clubs = st.multiselect(
    "Choisir un ou plusieurs clubs",
    clubs_df['club'].sort_values().unique(),
    default=[clubs_df['club'].sort_values().iloc[0]] if len(clubs_df) else []
)

selected_clubs_df = clubs_df[clubs_df['club'].isin(selected_clubs)]

# Colonnes numériques disponibles (clubs)
numeric_club_cols = clubs_df.select_dtypes(include=[np.number]).columns.tolist()
stat_club_cols = [c for c in numeric_club_cols if c not in ["classement"]]

# Sélection dynamique des stats (clubs)
selected_club_stats = st.multiselect(
    "Choisir les statistiques à afficher dans le radar (clubs)",
    options=stat_club_cols,
    default=stat_club_cols[:5] if len(stat_club_cols) > 5 else stat_club_cols
)

if not selected_clubs_df.empty and selected_club_stats:
    radar_club_data = []
    for _, club in selected_clubs_df.iterrows():
        radar_club_data.append({
            "Joueur": club.get('club', ''),  # on réutilise la clé "Joueur" pour la fonction radar
            **{stat: club.get(stat, np.nan) for stat in selected_club_stats}
        })

    radar_clubs_df = pd.DataFrame(radar_club_data)

    fig_clubs = make_scatter_radar(radar_clubs_df, selected_club_stats)

    st.plotly_chart(
        fig_clubs,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "doubleClick": "reset"
        }
    )

    # Tableau comparatif chiffré (clubs)
    st.subheader("📊 Tableau comparatif des clubs")
    table_clubs_df = radar_clubs_df.set_index("Joueur").T
    st.dataframe(table_clubs_df)

    # Export CSV (clubs)
    st.download_button(
        "⬇️ Télécharger en CSV (clubs)",
        table_clubs_df.to_csv().encode("utf-8"),
        file_name="comparatif_clubs.csv",
        mime="text/csv"
    )
else:
    st.warning("Veuillez sélectionner au moins un club et une statistique pour afficher le radar.")
