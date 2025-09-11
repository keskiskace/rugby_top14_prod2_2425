import os
import sqlite3
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from io import BytesIO
import matplotlib.pyplot as plt

# -----------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------
DB_FILE = "top14_prod2_24_25_players_clubs.db"  # adapte le nom si besoin
IMAGES_DIR = "images"
FALLBACK_IMAGE = os.path.join(IMAGES_DIR, "no_player.webp")

st.set_page_config(page_title="Rugby Top14/Prod2 Players", layout="wide")
st.title("🏉 Rugby Top14/Prod2 Players")


# -----------------------------------------------------------------
# UTILITAIRES IMAGE / FICHIER
# -----------------------------------------------------------------
def get_image_safe(player):
    """
    Retourne soit un path local (images/photo_{player_id}.jpg),
    soit l'URL si fournie, sinon l'image fallback.
    """
    fallback = FALLBACK_IMAGE
    player_id = player.get("player_id") or player.get("id") or ""
    img_path = os.path.join(IMAGES_DIR, f"photo_{player_id}.jpg")

    if os.path.exists(img_path):
        return img_path

    photo_field = player.get("photo", "")
    if isinstance(photo_field, str) and photo_field.startswith("http"):
        return photo_field

    return fallback


def download_missing_photos(df, img_dir=IMAGES_DIR, timeout=5):
    """
    Télécharge les photos présentes dans la colonne 'photo' si elles n'existent pas localement.
    Retourne le nombre de photos téléchargées.
    """
    os.makedirs(img_dir, exist_ok=True)
    missing_count = 0
    for _, row in df.iterrows():
        player_id = row.get("player_id") or row.get("id")
        url = row.get("photo")
        if not url or pd.isna(url):
            continue

        img_path = os.path.join(img_dir, f"photo_{player_id}.jpg")
        if os.path.exists(img_path):
            continue

        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            with open(img_path, "wb") as f:
                f.write(r.content)
            missing_count += 1
        except Exception as e:
            # on affiche en console pour debug; évite de casser l'app
            print(f"[ERREUR] {row.get('nom','?')} ({url}) : {e}")

    return missing_count


def dataframe_to_image(df, filename="table.png"):
    """
    Convertit un DataFrame en image via matplotlib.table et renvoie le nom de fichier.
    """
    # taille raisonnable
    n_rows, n_cols = max(1, len(df)), max(1, len(df.columns))
    fig, ax = plt.subplots(figsize=(max(6, n_cols * 1.2), max(2, n_rows * 0.5)))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        cellLoc="center",
        loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.2, 1.2)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return filename


# -----------------------------------------------------------------
# CHARGEMENT DES DONNÉES
# -----------------------------------------------------------------
@st.cache_data
def load_players():
    with sqlite3.connect(DB_FILE) as con:
        df = pd.read_sql("SELECT * FROM players", con)

    # conversion prudente des colonnes numériques si possible
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        except Exception:
            pass

    # quelques ratios pratiques si les colonnes existent
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


@st.cache_data
def load_clubs():
    with sqlite3.connect(DB_FILE) as con:
        clubs_df = pd.read_sql("SELECT * FROM clubs", con)

    for col in clubs_df.columns:
        try:
            clubs_df[col] = pd.to_numeric(clubs_df[col], errors='ignore')
        except Exception:
            pass

    return clubs_df


# -----------------------------------------------------------------
# INFÉRENCE LIGUE (petit helper)
# -----------------------------------------------------------------
def infer_league(df: pd.DataFrame) -> pd.DataFrame:
    # conserve la colonne "division" si elle existe, sinon créé __league__ vide
    df = df.copy()
    if "division" in df.columns:
        df["__league__"] = df["division"].astype(str)
    else:
        df["__league__"] = None
    return df


# -----------------------------------------------------------------
# RADAR CUSTOM
# -----------------------------------------------------------------
def make_scatter_radar(radar_df, selected_stats):
    fig = go.Figure()

    n_stats = len(selected_stats)
    angles = np.linspace(0, 2 * np.pi, n_stats, endpoint=False)

    # max pour l'échelle
    max_val = radar_df[selected_stats].apply(pd.to_numeric, errors="coerce").max().max()
    if not np.isfinite(max_val) or max_val <= 0:
        max_val = 1.0
    n_circles = 5
    step = max_val / n_circles

    color_palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf", "#393b79", "#637939",
        "#8c6d31", "#843c39", "#7b4173"
    ]

    # cercles concentriques
    for i in range(1, n_circles + 1):
        r = step * i
        circle_x = [r * np.cos(t) for t in np.linspace(0, 2 * np.pi, 200)]
        circle_y = [r * np.sin(t) for t in np.linspace(0, 2 * np.pi, 200)]
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

    # axes
    for angle, stat in zip(angles, selected_stats):
        x_axis = [0, max_val * np.cos(angle)]
        y_axis = [0, max_val * np.sin(angle)]
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=y_axis,
            mode="lines",
            line=dict(color="lightgrey", dash="dot"),
            showlegend=False,
            hoverinfo="skip"
        ))
        fig.add_annotation(
            x=max_val * 1.05 * np.cos(angle),
            y=max_val * 1.05 * np.sin(angle),
            text=stat,
            showarrow=False,
            font=dict(size=12, color="black")
        )

    # polygones des joueurs/clubs
    for idx, (_, row) in enumerate(radar_df.iterrows()):
        r_values = [row.get(stat, np.nan) for stat in selected_stats]
        r_values = [np.nan if (not pd.notna(v)) else float(v) for v in r_values]
        # ferme le polygone
        r_values += [r_values[0]]
        theta = np.append(angles, angles[0])

        x = [0 if (v is np.nan or not np.isfinite(v)) else v * np.cos(t) for v, t in zip(r_values, theta)]
        y = [0 if (v is np.nan or not np.isfinite(v)) else v * np.sin(t) for v, t in zip(r_values, theta)]

        color = color_palette[idx % len(color_palette)]
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=row.get("Joueur", ""),
            fill="toself",
            line=dict(color=color),
            fillcolor=color,
            opacity=0.25,
            hoverinfo="skip",
            showlegend=False
        ))

        hover_texts = [
            f"{row.get('Joueur', '')}<br>{stat}: {'' if (val is np.nan or not np.isfinite(val)) else round(val, 1)}"
            for stat, val in zip(selected_stats, r_values[:-1])
        ]
        hover_texts.append(hover_texts[0])

        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="markers+lines",
            name=row.get("Joueur", ""),
            line=dict(color=color),
            marker=dict(color=color, size=8),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>"
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
# Chargement complet (toutes saisons)
df = load_players()

if df is None or df.empty:
    st.warning("La table 'players' est vide ou introuvable dans la DB.")
    st.stop()

# --- Saisons (joueurs) : multiselect ---
saisons_disponibles = sorted(df['saison'].dropna().unique(), reverse=True)
if len(saisons_disponibles) == 0:
    st.warning("Aucune saison trouvée dans la table players.")
    st.stop()

selected_saisons = st.multiselect(
    "Choisir une ou plusieurs saisons (joueurs)",
    saisons_disponibles,
    default=[saisons_disponibles[0]]
)

df_filtered = df[df['saison'].isin(selected_saisons)].copy()

# Ajout d'une colonne d'affichage dépendant du choix de saison(s)
if len(selected_saisons) > 1:
    df_filtered['display_name'] = df_filtered['nom'].astype(str) + " (" + df_filtered['saison'].astype(str) + ")"
else:
    df_filtered['display_name'] = df_filtered['nom'].astype(str)

# filtres de base
if 'nombre_matchs_joues' in df_filtered.columns:
    df_nonzero = df_filtered[pd.to_numeric(df_filtered['nombre_matchs_joues'], errors='coerce').fillna(0) > 0].copy()
else:
    df_nonzero = df_filtered.copy()

df_labeled = infer_league(df_nonzero)

# calcul des "joueurs types" (moyennes par division / poste)
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

# dataset "étendu" pour stats (joueurs + joueurs types)
df_extended = pd.concat([df_filtered, extra_df], ignore_index=True) if not extra_df.empty else df_filtered.copy()

# --- Vérification / téléchargement photos (optionnel via bouton / case) ---
col1, col2 = st.columns([1, 3])
with col1:
    auto_check = st.checkbox("Télécharger les photos manquantes au chargement", value=False)
with col2:
    manual_btn = st.button("📥 Vérifier/ Télécharger maintenant")

if auto_check or manual_btn:
    with st.spinner("Vérification des photos manquantes..."):
        try:
            n_new = download_missing_photos(df_filtered)
            if n_new > 0:
                st.success(f"{n_new} photo(s) téléchargée(s) automatiquement ✅")
            else:
                st.info("Toutes les photos sont déjà présentes 👍")
        except Exception as e:
            st.error(f"Erreur lors du téléchargement des photos : {e}")
else:
    st.info("Photos non vérifiées (clique 'Télécharger maintenant' ou coche la case pour le faire automatiquement).")


# ----------------------
# SECTION JOUEURS
# ----------------------
st.header("🔎 Joueurs — Comparateur")

# Sélection joueurs (display_name)
player_options = df_filtered['display_name'].sort_values().unique().tolist()
if player_options:
    default_sel = [player_options[0]]
else:
    default_sel = []

selected_names = st.multiselect(
    "Choisir un ou plusieurs joueurs",
    player_options,
    default=default_sel
)
selected_players = df_filtered[df_filtered['display_name'].isin(selected_names)].copy()

# Sélection joueurs types (si présents)
selected_types = []
if not extra_df.empty:
    types_opts = extra_df['nom'].sort_values().unique().tolist()
    selected_types = st.multiselect("Choisir un ou plusieurs joueurs types", types_opts, default=[])
    selected_type_players = extra_df[extra_df['nom'].isin(selected_types)].copy()
else:
    selected_type_players = pd.DataFrame()

# Concat sélection réelle
if not selected_type_players.empty and not selected_players.empty:
    selected_players = pd.concat([selected_players, selected_type_players], ignore_index=True)
elif not selected_type_players.empty and selected_players.empty:
    selected_players = selected_type_players.copy()

# Affichage des joueurs sélectionnés (photo + infos)
for _, joueur in selected_players.iterrows():
    nom_aff = joueur.get('nom', '')
    st.subheader(nom_aff)
    if "Joueur type" not in str(nom_aff):
        # joueur réel -> image et json info
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
        st.info("📊 Joueur type (moyenne des stats).")

# préparation colonnes statistiques (depuis df_extended)
numeric_cols = df_extended.select_dtypes(include=[np.number]).columns.tolist()
exclude_cols = ["player_id", "id"]
stat_cols = [c for c in numeric_cols if c not in exclude_cols]

selected_stats = st.multiselect(
    "Choisir les statistiques à afficher dans le radar",
    options=stat_cols,
    default=stat_cols[:5] if len(stat_cols) > 5 else stat_cols
)

if selected_stats and not selected_players.empty:
    radar_data = []
    for _, joueur in selected_players.iterrows():
        entry = {"Joueur": joueur.get('nom', '')}
        for stat in selected_stats:
            entry[stat] = joueur.get(stat, np.nan)
        radar_data.append(entry)

    radar_df = pd.DataFrame(radar_data)
    fig = make_scatter_radar(radar_df, selected_stats)
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displaylogo": False, "doubleClick": "reset"})

    st.subheader("📊 Tableau comparatif des joueurs")
    table_df = radar_df.set_index("Joueur").T
    st.dataframe(table_df)

    st.download_button("⬇️ Télécharger en CSV", table_df.to_csv().encode("utf-8"),
                       file_name="comparatif_joueurs.csv", mime="text/csv")

    img_file = dataframe_to_image(table_df, "comparatif_joueurs.png")
    with open(img_file, "rb") as f:
        st.download_button("⬇️ Télécharger en PNG", f, file_name="comparatif_joueurs.png", mime="image/png")
else:
    st.warning("Veuillez sélectionner au moins une statistique et un joueur pour afficher le radar.")


# ----------------------
# SECTION TOP / FLOP TEN
# ----------------------
st.header("🏆 Top / Flop 10 Joueurs")

choice_type = st.radio("Choisir le type", ["Top 10", "Flop 10"])
choice_division = st.selectbox("Choisir une division", ["Toutes", "Top14", "ProD2"])
club_choices = ["Aucun"] + sorted(df_filtered['club'].dropna().unique().tolist())
choice_club = st.selectbox("Choisir un club (optionnel)", club_choices)

choice_stat = st.selectbox("Choisir une statistique", stat_cols) if stat_cols else None

filtered_for_top = df_nonzero.copy()

if choice_division != "Toutes" and "division" in filtered_for_top.columns:
    filtered_for_top = filtered_for_top[filtered_for_top['division'].str.contains(choice_division, case=False, na=False)]

if choice_club != "Aucun":
    filtered_for_top = filtered_for_top[filtered_for_top['club'] == choice_club]

if not filtered_for_top.empty and choice_stat:
    if choice_type == "Top 10":
        result = filtered_for_top.nlargest(10, choice_stat)[["nom", "club", "division", choice_stat]]
    else:
        result = filtered_for_top.nsmallest(10, choice_stat)[["nom", "club", "division", choice_stat]]

    st.subheader(f"{choice_type} sur {choice_stat}")
    st.dataframe(result)

    # Export CSV
    st.download_button(
        "⬇️ Télécharger en CSV",
        result.to_csv(index=False).encode("utf-8"),
        file_name=f"{choice_type.lower().replace(' ', '_')}_{choice_stat}.csv",
        mime="text/csv"
    )

    # Export PNG
    img_file = dataframe_to_image(result, f"{choice_type.lower()}_{choice_stat}.png")
    with open(img_file, "rb") as f:
        st.download_button("⬇️ Télécharger en PNG", f, file_name=f"{choice_type.lower()}_{choice_stat}.png", mime="image/png")
else:
    st.info("Top/Flop indisponible (vérifie les filtres et la statistique choisie).")


# ----------------------
# SECTION CLUBS
# ----------------------
st.header("📊 Comparaison des Clubs")

clubs_df = load_clubs()
if clubs_df is None or clubs_df.empty:
    st.warning("La table 'clubs' est vide ou introuvable dans la DB.")
else:
    # choix des saisons (clubs)
    saisons_disponibles_clubs = sorted(clubs_df['saison'].dropna().unique(), reverse=True)
    if len(saisons_disponibles_clubs) == 0:
        st.warning("Aucune saison trouvée dans la table clubs.")
    else:
        selected_saisons_clubs = st.multiselect(
            "Choisir une ou plusieurs saisons (clubs)",
            saisons_disponibles_clubs,
            default=[saisons_disponibles_clubs[0]]
        )

        clubs_filtered = clubs_df[clubs_df['saison'].isin(selected_saisons_clubs)].copy()
        if len(selected_saisons_clubs) > 1:
            clubs_filtered['display_name'] = clubs_filtered['club'].astype(str) + " (" + clubs_filtered['saison'].astype(str) + ")"
        else:
            clubs_filtered['display_name'] = clubs_filtered['club'].astype(str)

        # clubs types (moyennes)
        clubs_nonzero = clubs_filtered.copy()
        extra_clubs = []
        if not clubs_nonzero.empty:
            if (clubs_nonzero['division'].str.contains("Top14", case=False, na=False)).any():
                avg_top14 = clubs_nonzero[clubs_nonzero['division'].str.contains("Top14", case=False, na=False)].mean(numeric_only=True).round(1)
                extra_clubs.append({"club": "Club type Top14", **avg_top14.to_dict()})
            if (clubs_nonzero['division'].str.contains("ProD2", case=False, na=False)).any():
                avg_prod2 = clubs_nonzero[clubs_nonzero['division'].str.contains("ProD2", case=False, na=False)].mean(numeric_only=True).round(1)
                extra_clubs.append({"club": "Club type ProD2", **avg_prod2.to_dict()})

        extra_clubs_df = pd.DataFrame(extra_clubs) if extra_clubs else pd.DataFrame()
        clubs_extended = pd.concat([clubs_filtered, extra_clubs_df], ignore_index=True) if not extra_clubs_df.empty else clubs_filtered.copy()

        # sélection club
        club_options = clubs_filtered['display_name'].sort_values().unique().tolist()
        selected_clubs = st.multiselect(
            "Choisir un ou plusieurs clubs",
            club_options,
            default=[club_options[0]] if club_options else []
        )
        selected_clubs_df = clubs_filtered[clubs_filtered['display_name'].isin(selected_clubs)].copy()

        if not selected_clubs_df.empty:
            # affichage logos (si urls) ou noms
            logos = []
            for _, club in selected_clubs_df.iterrows():
                logo_field = club.get("logo", "")
                if isinstance(logo_field, str) and logo_field.startswith("http"):
                    logos.append(f"<img src='{logo_field}' width='60'>")
                else:
                    logos.append(club.get("club", ""))
            logos_html = " <b>VS</b> ".join(logos)
            st.markdown(f"<div style='text-align:center;'>{logos_html}</div>", unsafe_allow_html=True)

            numeric_cols_clubs = clubs_extended.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols_clubs = ["classement"]
            stat_cols_clubs = [c for c in numeric_cols_clubs if c not in exclude_cols_clubs]

            selected_stats_clubs = st.multiselect(
                "Choisir les statistiques à afficher dans le radar (clubs)",
                options=stat_cols_clubs,
                default=stat_cols_clubs[:5] if len(stat_cols_clubs) > 5 else stat_cols_clubs
            )

            if selected_stats_clubs and not selected_clubs_df.empty:
                radar_data = []
                for _, club in selected_clubs_df.iterrows():
                    entry = {"Joueur": club.get("club", "")}
                    for stat in selected_stats_clubs:
                        entry[stat] = club.get(stat, np.nan)
                    radar_data.append(entry)

                radar_df = pd.DataFrame(radar_data)
                fig = make_scatter_radar(radar_df, selected_stats_clubs)
                st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displaylogo": False, "doubleClick": "reset"})

                st.subheader("📊 Tableau comparatif des clubs")
                table_df = radar_df.set_index("Joueur").T
                st.dataframe(table_df)

                st.download_button("⬇️ Télécharger en CSV", table_df.to_csv().encode("utf-8"),
                                   file_name="comparatif_clubs.csv", mime="text/csv")

                img_file = dataframe_to_image(table_df, "comparatif_clubs.png")
                with open(img_file, "rb") as f:
                    st.download_button("⬇️ Télécharger en PNG", f, file_name="comparatif_clubs.png", mime="image/png")
            else:
                st.warning("Veuillez sélectionner au moins une statistique et un club pour afficher le radar.")
        else:
            st.info("Aucun club sélectionné pour cette saison.")
