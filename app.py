import os
import sqlite3
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

DB_FILE = "top14_prod2_24_25_players_clubs.db"
TABLE = "players"

# ---------------------------------------------------------------
# IMAGE MANAGEMENT (players)
# ---------------------------------------------------------------

def get_image_safe(player):
    fallback = "images/no_player.webp"
    pid = player.get("player_id")
    img_path = f"images/photo_{pid}.jpg" if pid is not None else None

    if img_path and os.path.exists(img_path):
        return img_path

    url = str(player.get("photo", ""))
    if url.startswith("http"):
        try:
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            return url
        except Exception:
            pass

    return fallback


def download_missing_photos(df, img_dir="images"):
    os.makedirs(img_dir, exist_ok=True)
    missing_count = 0
    for _, row in df.iterrows():
        pid = row.get("player_id")
        url = row.get("photo")
        if not url or pd.isna(url) or not str(url).startswith("http") or pid is None:
            continue
        img_path = os.path.join(img_dir, f"photo_{pid}.jpg")
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


# ---------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------
@st.cache_data
def load_players():
    with sqlite3.connect(DB_FILE) as con:
        df = pd.read_sql(f"SELECT * FROM {TABLE}", con)

    # conversions numériques quand c'est possible
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


# ---------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------

def infer_league(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "division" in df.columns:
        df["__league__"] = df["division"].astype(str)
    else:
        df["__league__"] = None
    return df


# ---------------------------------------------------------------
# RADAR (Scatterpolar) avec points survolables + zoom/pan
# ---------------------------------------------------------------

def make_scatter_radar(radar_df: pd.DataFrame, selected_stats: list[str]) -> go.Figure:
    fig = go.Figure()

    # borne max pour l'axe radial
    max_val = pd.to_numeric(radar_df[selected_stats], errors='coerce').max().max()
    if pd.isna(max_val) or float(max_val) <= 0:
        max_val = 1.0

    # palette bien contrastée
    color_palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf", "#393b79", "#637939",
        "#8c6d31", "#843c39", "#7b4173", "#3182bd",
        "#fd8d3c", "#31a354", "#e6550d"
    ]

    # une trace par ligne
    for i, (_, row) in enumerate(radar_df.iterrows()):
        values = [row.get(stat, np.nan) for stat in selected_stats]
        values = [np.nan if not pd.notna(v) else float(v) for v in values]
        values += [values[0]]
        stats = list(selected_stats) + [selected_stats[0]]

        label = row.get("Joueur", row.get("Club", "Inconnu"))
        color = color_palette[i % len(color_palette)]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=stats,
            mode="markers+lines",
            fill="toself",
            name=label,
            line=dict(color=color),
            marker=dict(color=color, size=7),
            hovertemplate=(
                "<b>%{text}</b><br>"  # nom joueur/club
                "%{theta}: %{r}<extra></extra>"
            ),
            text=[label] * len(stats)
        ))

    # apparence + grille + zoom/pan
    fig.update_layout(
        width=800, height=600,
        showlegend=True,
        hovermode="closest",
        dragmode="pan",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max_val],
                gridcolor="lightgrey",
                gridwidth=1,
                tickfont=dict(size=11),
            ),
            angularaxis=dict(
                gridcolor="lightgrey",
                gridwidth=1,
                tickfont=dict(size=12)
            )
        )
    )

    return fig


# ---------------------------------------------------------------
# APP
# ---------------------------------------------------------------

st.set_page_config(page_title="Rugby Top14/ProD2 – Joueurs & Clubs", layout="wide")
st.title("🏉 Rugby Top14/ProD2 – Joueurs & Clubs")

# --- joueurs ---
df = load_players()

# joueurs avec au moins 1 match si la colonne existe
if 'nombre_matchs_joues' in df.columns:
    df_nonzero = df[pd.to_numeric(df['nombre_matchs_joues'], errors='coerce').fillna(0) > 0].copy()
else:
    df_nonzero = df.copy()

# etiquetage ligue
df_labeled = infer_league(df_nonzero)

# joueurs types: top14/proD2
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

# joueurs types par postes
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
if 'poste' in df_nonzero.columns:
    for nom_type, postes in postes_groupes.items():
        subset = df_nonzero[df_nonzero['poste'].isin(postes)]
        if not subset.empty:
            avg_stats = subset.mean(numeric_only=True).round(1)
            extra_players.append({"nom": nom_type, "club": "Poste moyen", **avg_stats.to_dict()})

extra_df = pd.DataFrame(extra_players) if extra_players else pd.DataFrame()
df_extended = pd.concat([df, extra_df], ignore_index=True) if not extra_df.empty else df.copy()

# téléchargement auto des photos (optionnel, garde mais silencieux)
with st.spinner("Vérification des photos manquantes (joueurs)..."):
    try:
        n_new = download_missing_photos(df)
        if n_new > 0:
            st.success(f"{n_new} photo(s) téléchargée(s) automatiquement ✅")
        else:
            st.info("Toutes les photos sont déjà présentes 👍")
    except Exception:
        st.info("Téléchargement automatique des photos ignoré.")

# sélection joueurs réels
selected_names = st.multiselect(
    "👤 Choisir un ou plusieurs joueurs",
    df['nom'].sort_values().unique(),
    default=[df['nom'].sort_values().iloc[0]] if len(df) else []
)
selected_real = df[df['nom'].isin(selected_names)]

# sélection joueurs types
selected_types = st.multiselect(
    "🧪 Choisir un ou plusieurs joueurs types",
    extra_df['nom'].sort_values().unique() if not extra_df.empty else []
)
selected_types_df = extra_df[extra_df['nom'].isin(selected_types)] if not extra_df.empty else pd.DataFrame()

selected_players = pd.concat([selected_real, selected_types_df], ignore_index=True)

# affichage infos + photo
for _, joueur in selected_players.iterrows():
    if "Joueur type" not in str(joueur.get('nom', '')):
        st.subheader(joueur.get('nom', ''))
        st.image(get_image_safe(joueur), caption=joueur.get('club', ''), width=150)
    else:
        st.subheader(joueur.get('nom', ''))
        st.info("📊 Joueur type basé sur la moyenne des stats.")

# colonnes numériques
numeric_cols = df_extended.select_dtypes(include=[np.number]).columns.tolist()
stat_cols = [c for c in numeric_cols if c not in ["player_id"]]

# sélecteur de stats
selected_stats = st.multiselect(
    "📈 Choisir les statistiques à afficher dans le radar (joueurs)",
    options=stat_cols,
    default=stat_cols[:5] if len(stat_cols) > 5 else stat_cols
)

if selected_stats and not selected_players.empty:
    radar_data = []
    for _, r in selected_players.iterrows():
        radar_data.append({"Joueur": r.get('nom', ''), **{s: r.get(s, np.nan) for s in selected_stats}})
    radar_df = pd.DataFrame(radar_data)

    fig = make_scatter_radar(radar_df, selected_stats)
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displaylogo": False, "doubleClick": "reset"})

    st.subheader("📊 Tableau comparatif des joueurs")
    table_df = radar_df.set_index("Joueur").T
    st.dataframe(table_df)

    st.download_button("⬇️ Télécharger en CSV (joueurs)", table_df.to_csv().encode("utf-8"), file_name="comparatif_joueurs.csv", mime="text/csv")
else:
    st.warning("Veuillez sélectionner au moins une statistique et un joueur.")

# --- clubs ---
st.header("📊 Comparaison des Clubs")
with sqlite3.connect(DB_FILE) as con:
    clubs_df = pd.read_sql("SELECT * FROM clubs", con)

# clubs qui ont joué (points_marqués > 0)
points_col = 'points_marqués' if 'points_marqués' in clubs_df.columns else 'points_marques'
clubs_nonzero = clubs_df[pd.to_numeric(clubs_df[points_col], errors='coerce').fillna(0) > 0].copy()

# clubs type Top14 / ProD2 (exclut 0 point)
extra_clubs = []
if not clubs_nonzero.empty and 'division' in clubs_nonzero.columns:
    if (clubs_nonzero['division'].str.contains("Top14", case=False, na=False)).any():
        avg_top14 = clubs_nonzero[clubs_nonzero['division'].str.contains("Top14", case=False, na=False)].mean(numeric_only=True).round(1)
        extra_clubs.append({"club": "Club type Top14", **avg_top14.to_dict(), "logo": None})
    if (clubs_nonzero['division'].str.contains("ProD2", case=False, na=False)).any():
        avg_prod2 = clubs_nonzero[clubs_nonzero['division'].str.contains("ProD2", case=False, na=False)].mean(numeric_only=True).round(1)
        extra_clubs.append({"club": "Club type ProD2", **avg_prod2.to_dict(), "logo": None})

extra_clubs_df = pd.DataFrame(extra_clubs) if extra_clubs else pd.DataFrame()
clubs_extended = pd.concat([clubs_df, extra_clubs_df], ignore_index=True) if not extra_clubs_df.empty else clubs_df.copy()

# sélection clubs
selected_clubs = st.multiselect(
    "🏳️ Choisir un ou plusieurs clubs",
    clubs_extended['club'].sort_values().unique(),
    default=[clubs_extended['club'].sort_values().iloc[0]] if len(clubs_extended) else []
)
selected_clubs_df = clubs_extended[clubs_extended['club'].isin(selected_clubs)]

# logos en ligne "logo1 VS logo2 VS ..."
if not selected_clubs_df.empty:
    logos = []
    for _, club in selected_clubs_df.iterrows():
        logo_url = str(club.get("logo", ""))
        if logo_url.startswith("http"):
            logos.append(f"<img src='{logo_url}' width='60' style='vertical-align:middle;margin:0 8px'>")
        else:
            logos.append(f"<span style='margin:0 8px'>{club.get('club','')}</span>")
    logos_html = " <b>VS</b> ".join(logos)
    st.markdown(f"<div style='text-align:center'>{logos_html}</div>", unsafe_allow_html=True)

# stats numériques clubs
numeric_cols_clubs = clubs_extended.select_dtypes(include=[np.number]).columns.tolist()
stat_cols_clubs = [c for c in numeric_cols_clubs if c not in ["classement"]]

selected_stats_clubs = st.multiselect(
    "📈 Choisir les statistiques à afficher dans le radar (clubs)",
    options=stat_cols_clubs,
    default=stat_cols_clubs[:5] if len(stat_cols_clubs) > 5 else stat_cols_clubs
)

if selected_stats_clubs and not selected_clubs_df.empty:
    radar_club_rows = []
    for _, r in selected_clubs_df.iterrows():
        radar_club_rows.append({"Club": r.get('club', ''), **{s: r.get(s, np.nan) for s in selected_stats_clubs}})
    radar_clubs_df = pd.DataFrame(radar_club_rows)

    figc = make_scatter_radar(radar_clubs_df.rename(columns={"Club": "Joueur"}), selected_stats_clubs)
    st.plotly_chart(figc, use_container_width=True, config={"scrollZoom": True, "displaylogo": False, "doubleClick": "reset"})

    st.subheader("📊 Tableau comparatif des clubs")
    tablec = radar_clubs_df.set_index("Club").T
    st.dataframe(tablec)

    st.download_button("⬇️ Télécharger en CSV (clubs)", tablec.to_csv().encode("utf-8"), file_name="comparatif_clubs.csv", mime="text/csv")
else:
    st.warning("Veuillez sélectionner au moins une statistique et un club.")
