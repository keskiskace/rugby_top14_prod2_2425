# --------------------------------------------------------------
# app.py   (lance:  streamlit run app.py)
# --------------------------------------------------------------
import sqlite3, pickle, os
import numpy as np, pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import QuantileTransformer, StandardScaler
import plotly.graph_objects as go


from matplotlib import cm

import base64

def get_image_base64(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


import streamlit as st
st.set_page_config(page_title="Rugby Stats", layout="wide")


# -----------------------------------------------------------------
# CONSTANTES – à adapter
# -----------------------------------------------------------------
DB_FILE  = "top14_players.db"
TABLE    = "players"

POSTES_LAYOUT = [
        ['Pilier gauche', 'Talonneur', 'Pilier droit'],                         # 1-2-3
        ['2ème ligne gauche', '2ème ligne droit'],                             # 4-5
        ['3ème ligne aile fermée', '3ème ligne centre', '3ème ligne aile ouverte'],  # 6-8-7
        ['Demi de mêlée', "Demi d'ouverture"],                                                 # 9-10
        ['Ailier gauche', 'Premier Centre', 'Deuxième Centre', 'Ailier droit'],    # 11-12-13-14
        ['Arrière']                                                             # 15
    ]

POSTES_ORDER = [p for row in POSTES_LAYOUT for p in row]   # linéaire

POSTES_BDD_NAME = {
        'Pilier gauche' : '1ère ligne', 'Talonneur' : '1ère ligne', 'Pilier droit' : '1ère ligne',                         # 1-2-3
        '2ème ligne gauche' : '2ème ligne', '2ème ligne droit' : '2ème ligne',                             # 4-5
        '3ème ligne aile fermée' : '3ème ligne', '3ème ligne centre' : '3ème ligne', '3ème ligne aile ouverte' : '3ème ligne',  # 6-8-7
        'Demi de mêlée' :'Demi de mêlée', "Demi d'ouverture" :"Demi d'ouverture",                                                 # 9-10
        'Ailier gauche' : 'Ailier', 'Premier Centre' : 'Centre', 'Deuxième Centre' : 'Centre', 'Ailier droit' : 'Ailier',    # 11-12-13-14
        'Arrière' : 'Arrière'
    }


REF_TEAM = {           # ids de joueurs « équipe de référence »
    "Pilier gauche": 2451, "Talonneur": 2445, "Pilier droit": 2448,
    "2ème ligne gauche": 2405, "2ème ligne droit": 2498,
    "3ème ligne aile fermée": 2453, "3ème ligne centre": 2462, "3ème ligne aile ouverte": 2425,
    "Demi de mêlée": 2449, "Demi d'ouverture": 2436,
    "Ailier gauche": 12335, "Premier Centre": 2493, "Deuxième Centre": 2503, "Ailier droit": 2483,
    "Arrière": 2450
}

# -----------------------------------------------------------------
# LOAD DATA 
# -----------------------------------------------------------------
@st.cache_data
def load_players():
    with sqlite3.connect(DB_FILE) as con:
        df = pd.read_sql(f"SELECT * FROM {TABLE}", con)

    # Création du ratio (cm / kg)
    df['ratio_poids_taille'] =  round(df['poids_kg'] / df['taille_cm'], 2)

    # Création du ratio (metres / courses)
    df['ratio_metres_courses'] =  round(df['metres_parcourus'] / df['courses'], 2)

    # Création du ratio (min / matchs)
    df['ratio_min_matchs'] =  round(df['temps_jeu_min'] / df['nombre_matchs_joues'], 2)

    return df

df = load_players()
exclude_exact = {
    'temps_jeu_min', 'taille_cm', 'poids_kg', 'age', 'nombre_matchs_joues', 'ratio_min_matchs',
    'player_id', 'url', 'nom', 'club', 'poste', 'pays', 'scraped_at', 'ratio_poids_taille', 'ratio_metres_courses'
}
exclude_pct = [c for c in df.columns if c.endswith('_pct')]

for col in df.select_dtypes('number'):
    if col in exclude_exact or col in exclude_pct:
        continue
    df[col] = round(df[col] * 80 / df['temps_jeu_min'], 2)

# -----------------------------------------------------------------
# OUTILS COMMUNS
# -----------------------------------------------------------------

import numpy as np, pandas as pd, plotly.graph_objects as go


# ────────────────────────────────────────────────────────────────────
# 0)  Liste complète des variables qu’on peut afficher
# ────────────────────────────────────────────────────────────────────
ALL_STATS = ['age', 'taille_cm', 'poids_kg', 'ratio_metres_courses' , 'ratio_poids_taille' , 'ratio_min_matchs',
             'nombre_matchs_joues', 'nombre_matchs_commences', 'temps_jeu_min', 'points_marques', 
             'essais', 'penales', 'penales_pct', 'transformations', 'transformations_pct', 'drops_pct', 
             'ballon_joues_pied', 'metres_pied', 'courses', 'metres_parcourus', 'passes', 'franchissements', 
             'offloads', 'plaquages_casses', 'plaquages_reussis', 'plaquages_reussis_pct', 'ballon_grattes', 
             'interceptions', 'penales_concedees', 'carton_jaune', 'carton_rouge']

DEFAULT_STATS = [
    'courses','metres_parcourus','franchissements','offloads', 'metres_pied', 'points_marques' ,
    'plaquages_reussis','ballon_grattes', 'taille_cm' , 'ratio_poids_taille', 'plaquages_casses'
]


ALL_POSTES = sorted(df['poste'].dropna().unique())

if "pos_filter" not in st.session_state:
    st.session_state.pos_filter = []           # [] = aucun filtre → tous postes

if "stats_po_sel" not in st.session_state:
    st.session_state.stats_po_sel = DEFAULT_STATS.copy() 

if "stats_sel" not in st.session_state:
    st.session_state.stats_sel = DEFAULT_STATS.copy()     # ① état initial

# ────────────────────────────────────────────────────────────────────
# 1)  Initialiser la sélection en session (une seule fois)
# ────────────────────────────────────────────────────────────────────
if "stats_sel" not in st.session_state:
    st.session_state.stats_sel = DEFAULT_STATS.copy()



# ------------------------------------------------------------------
# 0)  Initialisation : 1er lancement de l’app
# ------------------------------------------------------------------
if "ref_player" not in st.session_state:
    first_nom = df.sort_values("nom")['nom'].iloc[0]
    st.session_state.ref_player = first_nom            # valeur par défaut

# ------------------------------------------------------------------
# 1)  Callback – quand l’utilisateur change le selectbox
# ------------------------------------------------------------------
def update_ref_player():
    st.session_state.ref_player = st.session_state.ref_player_sel

# ─────────── callback appelée quand le multiselect change ───
def update_stats():
    sel = st.session_state.multiselect_stats
    if 3 <= len(sel) <= 15:                 # borne de sécurité
        st.session_state.stats_sel = sel

def update_post_filter():
    st.session_state.pos_filter = st.session_state.multiselect_postes


def update_po_stats():
    sel = st.session_state.multiselect_po_stats             # nouvelle liste
    st.session_state.stats_po_sel = sel

    # ajoute un poids par défaut pour les nouvelles stats
    for v in sel:
        st.session_state.weights.setdefault(v, 1.0)

    # (optionnel) supprimer les poids des stats désélectionnées
    for v in list(st.session_state.weights):
        if v not in sel:
            st.session_state.weights.pop(v)

# la liste active à utiliser partout :
features = st.session_state.stats_sel


def radar_player_vs_median(df, player_row, features, compare_group='Poste équivalent',
                           levels=(.25, .5, .75, 1.0),         # <─ 1ᵉ val = r0
                           col_player="#e6194b", col_median="#2ca02c",
                           size=720):

    # -------- 1. sous-ensemble poste + bornes -------------------------
    poste = player_row['poste']

    if compare_group == "Poste équivalent":
        sub = df[(df["poste"] == poste) & (df["temps_jeu_min"] > 200)]
        if player_row["temps_jeu_min"]<=200 :
            sub = pd.concat(
                [sub, player_row[features].to_frame().T],   # ← Series ➜ DataFrame(1 ligne)
                ignore_index=True
            )
            sub = sub.apply(pd.to_numeric, errors="coerce")      # NaN si valeur non numérique

        poste_reel = poste
        mins, maxs = sub[features].min(), sub[features].max()
    else:
        # On garde les postes qui correspondent au même groupe
        poste_reel = compare_group
        sub = df[(df["poste"] == poste_reel) & (df["temps_jeu_min"] > 200)]

        # ►  Poste du joueur
        sub_poste = df[(df["poste"] == poste) & (df["temps_jeu_min"] > 200)]
        if player_row["temps_jeu_min"]<=200 :
            sub_poste = pd.concat(
                [sub_poste, player_row[features].to_frame().T],   # ← Series ➜ DataFrame(1 ligne)
                ignore_index=True
            )
            sub_poste = sub_poste.apply(pd.to_numeric, errors="coerce")

        # ►  Union des deux sous-ensembles
        sub_union = pd.concat([sub[features], sub_poste[features]])

        # ►  Bornes globales feature par feature
        mins = sub_union.min()
        maxs = sub_union.max()


    
    span       = (maxs - mins).replace(0, 1)

    median = round(sub[features].median() , 2)

    # -------- 2. échelle : min → r0  ,  max → 1 -----------------------
    r0     = levels[0]                    # min sur le 1ᵉʳ cercle
    r_span = 1 - r0
    scale  = lambda s: r0 + (s-mins)/span * r_span

    p_norm = scale(player_row[features])
    m_norm = scale(median[features])

    # -------- 3. polygones fermés ------------------------------------
    theta = np.linspace(0, 2*np.pi, len(features), endpoint=False)
    theta = np.append(theta, theta[0])
    r_p   = np.append(p_norm, p_norm[0])
    r_m   = np.append(m_norm, m_norm[0])

    # valeurs réelles pour hover
    vals_p = pd.to_numeric(player_row[features], errors="coerce").tolist()
    vals_m = pd.to_numeric(median[features],     errors="coerce").tolist()
    hover_p = vals_p + [vals_p[0]]
    hover_m = vals_m + [vals_m[0]]

    fig = go.Figure(layout=dict(width=size, height=size))

    fig.add_trace(go.Scatterpolar(
        r=r_p, theta=np.degrees(theta), name=player_row['nom'],
        fill='toself', opacity=.35, line=dict(color=col_player, width=2),
        customdata=hover_p,
        hovertemplate="%{theta}: %{customdata}<extra>%{fullData.name}</extra>"
    ))
    fig.add_trace(go.Scatterpolar(
        r=r_m, theta=np.degrees(theta), name=f"Médian {poste_reel}",
        fill='toself', opacity=.25, line=dict(color=col_median, width=2),
        customdata=hover_m,
        hovertemplate="%{theta}: %{customdata}<extra>%{fullData.name}</extra>"
    ))


    # -------- 3bis) masque centre (cache lignes < r0) -------------------
    fig.add_trace(go.Scatterpolar(
        r=[r0*0.85]*len(theta),               # un cercle de rayon r0
        theta=np.degrees(theta),
        mode='lines',
        fill='toself',
        fillcolor='white',               # même couleur que le fond
        line=dict(color='white', width=0),
        hoverinfo='skip',
        showlegend=False,
    ))

    # -------- 4. cercles pointillés (r0 et suivants) ------------------
    for lev in levels:
        fig.add_trace(go.Scatterpolar(
            r=[lev]*len(theta), theta=np.degrees(theta),
            mode='lines', line=dict(color='lightgrey', width=.6, dash='dot'),
            hoverinfo='skip', showlegend=False))
    


    # -------- 5. annotations valeurs réelles -------------------------
    ann_r, ann_theta, ann_text = [], [], []
    for ang_deg, var in zip(np.degrees(theta[:-1]), features):
        for lev in levels:
            real = mins[var] + (lev - r0)/r_span * span[var]
            ann_r.append(lev)
            ann_theta.append(ang_deg)
            ann_text.append(f"{real:.2f}")

    fig.add_trace(go.Scatterpolar(
        r=ann_r, theta=ann_theta, mode='text',
        text=[f"<b>{txt}</b>" for txt in ann_text], textfont=dict(size=12),
        hoverinfo='skip', showlegend=False))

    # -------- 6. mise en forme générale ------------------------------
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0,1], showgrid=False),
            angularaxis=dict(
                tickmode='array',
                tickvals=np.degrees(theta[:-1]),
                ticktext=features,
                tickfont=dict(size=15),
                showline=False,           # retire la ligne d’axe 0 °
            )
        ),
        title=dict(text=f"Radar /80min — {player_row['nom']} vs médian ({poste_reel} > 200 min)",
                   y=0.98, pad=dict(t=100)),
        legend=dict(orientation="h", yanchor="bottom", y=1.25,
                    xanchor="center", x=.5),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    


    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return fig


def radar_figure(
    stats_dict: dict,           # totaux d’équipe {feature: valeur}
    label: str,                 # nom dans la légende
    color: str,                 # couleur du polygone
    features: list,             # ordre des variables
    posts_in_team: list,        # ex. ['Pilier gauche','Talonneur', ...]
    *,
    levels=(.25,.5,.75,1.0),
    size=700,
    include_grid=True,
    df_source=None              # DataFrame global, défaut = df
):
    """Radar pour des stats SOMMÉES, avec min/max agrégés poste par poste."""

    df_src = df if df_source is None else df_source

    # --- 1. bornes poste par poste (>200 min) -----------------------------
    mins_tot = pd.Series(0.0, index=features)
    maxs_tot = pd.Series(0.0, index=features)

    for poste in posts_in_team:
        poste_players = df_src[
            (df_src["poste"] == poste) & (df_src["temps_jeu_min"] > 200)
        ][features]

        # si aucun joueur valide à ce poste : on saute
        if poste_players.empty:
            continue

        mins_tot += poste_players.min()
        maxs_tot += poste_players.max()

    span = (maxs_tot - mins_tot).replace(0, 1)   # évite /0

    # --- 2. normalisation des totaux d’équipe -----------------------------
    r0, r_span = levels[0], 1 - levels[0]
    vals   = pd.Series(stats_dict)[features].fillna(0)
    r_scal = r0 + (vals - mins_tot) / span * r_span

    theta_deg = np.degrees(
        np.linspace(0, 2*np.pi, len(features), endpoint=False)
    )
    r_poly  = np.append(r_scal, r_scal.iloc[0])
    theta_p = np.append(theta_deg, theta_deg[0])
    hover   = vals.tolist() + [vals.iloc[0]]

    fig = go.Figure(layout=dict(width=size, height=size))

    # Polygone de l’équipe
    fig.add_trace(
        go.Scatterpolar(
            r=r_poly,
            theta=theta_p,
            name=label,
            fill="toself",
            opacity=.35,
            line=dict(color=color, width=2),
            customdata=hover,
            hovertemplate="%{theta}: %{customdata}<extra>%{fullData.name}</extra>",
        )
    )

    # --- 3. grille / annotations : dessinées 1 fois -----------------------
    

    if include_grid:
        # cercles
        for lev in levels:
            fig.add_trace(
                go.Scatterpolar(
                    r=[lev]*len(theta_p),
                    theta=theta_p,
                    mode="lines",
                    line=dict(color="lightgrey", width=.7, dash="dot"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )


        # graduations réelles
        ann_r, ann_theta, ann_text = [], [], []
        for ang, var in zip(theta_deg, features):
            for lev in levels:
                real = mins_tot[var] + (lev - r0)/r_span * span[var]
                ann_r.append(lev)
                ann_theta.append(ang)
                ann_text.append(f"{real:.0f}")

        fig.add_trace(
            go.Scatterpolar(
                r=ann_r,
                theta=ann_theta,
                mode="text",
                text=[f"<b>{t}</b>" for t in ann_text],
                textfont=dict(size=11),
                hoverinfo="skip",
                showlegend=False,
            )
        )


        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=False, range=[0,1]),
                angularaxis=dict(
                    tickmode="array",
                    tickvals=theta_deg,
                    ticktext=features,
                    tickfont=dict(size=13),
                    showline=False,
                ),
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.15,
                        xanchor="center", x=0.5),
            margin=dict(l=40, r=40, t=40, b=40),
        )


    # -------- 3bis) masque centre (cache lignes < r0) -------------------
    theta = np.linspace(0, 2*np.pi, len(features), endpoint=False)
    fig.add_trace(go.Scatterpolar(
            r=[r0*0.85]*len(theta),               # un cercle de rayon r0
            theta=np.degrees(theta),
            mode='lines',
            fill='toself',
            fillcolor='white',               # même couleur que le fond
            line=dict(color='white', width=0),
            hoverinfo='skip',
            showlegend=False,
    ))

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig







def player_row(player_id):
    return df.loc[df['player_id'] == player_id].iloc[0]

# -----------------------------------------------------------------
# SESSION STATE INIT  (pour la feuille de match)
# -----------------------------------------------------------------
if "team" not in st.session_state:
    st.session_state.team = {pos: None for pos in POSTES_ORDER}

if "weights" not in st.session_state:
    st.session_state.weights = {var: 1.0 for var in ALL_STATS}   # ALL_STATS = liste complète


# -----------------------------------------------------------------
# MENU PRINCIPAL
# -----------------------------------------------------------------
st.sidebar.title("Rugby Stats Explorer")
# (au tout début du script – avant le gros `if/elif` des pages)
PAGES = ["Visualisation joueur", "Composer mon XV", "Recherche similarité"]
if "current_page" not in st.session_state:
    st.session_state.current_page = PAGES[0]

page = st.sidebar.radio("Menu", PAGES, index=PAGES.index(st.session_state.current_page))
st.session_state.current_page = page      # mémorise le choix



# ================================================================
# PAGE 1 : VISU JOUEUR
# ================================================================
if page == "Visualisation joueur":


    st.header("Visualisation individuelle")
    # ------------------------------------------------------------------------------
    # 0) session pour le joueur sélectionné
    # ------------------------------------------------------------------------------
    if "current_player" not in st.session_state:
        st.session_state.current_player = None          # rien de sélectionné

    # ------------------------------------------------------------------------------
    # 1)  FILTRES LATERAUX  (zéro filtre par défaut)
    # ------------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### Statistiques à afficher")
        st.multiselect(
            "Choisir les variables (3–15 max)",
            options=ALL_STATS,
            default=st.session_state.stats_sel,
            key="multiselect_stats",
            on_change=update_stats          # ← déclenche la callback
        )


    mask = pd.Series(True, index=df.index)      # tout passe au départ

    with st.sidebar:
        st.markdown("### Filtres (facultatifs)")

        # --- Poste --------------------------------------------------------
        postes = sorted(df['poste'].dropna().unique())
        sel_postes = st.multiselect("Poste", postes)        # ← défaut = []
        if sel_postes:                                      # seulement si on a coché
            mask &= df['poste'].isin(sel_postes)

        # --- Club ---------------------------------------------------------
        clubs = sorted(df['club'].dropna().unique())
        sel_clubs = st.multiselect("Club", clubs)           # défaut = []
        if sel_clubs:
            mask &= df['club'].isin(sel_clubs)

        # --- Temps de jeu (double-slider plage complète en défaut) --------
        min_min, max_min = int(df['temps_jeu_min'].min()), int(df['temps_jeu_min'].max())
        min_val, max_val = st.slider("Intervalle minutes jouées",
                                    min_min, max_min, (min_min, max_min), step=10)
        if (min_val, max_val) != (min_min, max_min):        # l’utilisateur a bougé
            mask &= df['temps_jeu_min'].between(min_val, max_val)

        # --- Âge max (slider inversé) ------------------------------------
        min_age, max_age = int(df['age'].min()), int(df['age'].max())
        min_val_a, max_val_a = st.slider("Intervalle d'âge",
                                    min_age, max_age, (min_age, max_age), step=1)
        if (min_val_a, max_val_a) != (min_age, max_age):        # l’utilisateur a bougé
            mask &= df['age'].between(min_val_a, max_val_a)

    df_filt = df[mask]
    st.sidebar.markdown(f"**{len(df_filt)} joueur(s)** correspondant(s)")



    # ------------------------------------------------------------------------------
    # 2)  RECHERCHE + SÉLECTEUR  (protégé contre filtres changeants)
    # ------------------------------------------------------------------------------
    candidats = df_filt['nom'].dropna().sort_values().unique()

    # Sécurité si le joueur courant n'est plus dans les filtres
    cur = st.session_state.get("current_player", None)
    if cur and cur not in candidats:
        candidats = np.insert(candidats, 0, cur)

    if len(candidats) == 0:
        st.info("Aucun joueur ne correspond aux critères.")
        st.stop()

    # Selectbox (recherche intégrée)
    idx_default = candidats.tolist().index(cur) if cur in candidats else 0
    joueur_nom = st.selectbox("Choisis un joueur", candidats, index=idx_default)
    st.session_state.current_player = joueur_nom


    # 1)  REMISE À ZÉRO DU GROUPE DE COMPARAISON (avant le selectbox !)
    if (
        "last_joueur" not in st.session_state            # première exécution
        or st.session_state.last_joueur != joueur_nom    # l’utilisateur a changé de joueur
    ):
        st.session_state.last_joueur   = joueur_nom
        st.session_state.compare_group = "Poste équivalent"
        
    # Liste des groupes possibles
    groupes_comparaison = ['Poste équivalent'] + sorted(set(POSTES_BDD_NAME.values()))

    compare_group = st.sidebar.selectbox(
    "Comparer à :",
    groupes_comparaison,
    index=groupes_comparaison.index(st.session_state.get("compare_group", 'Poste équivalent')),
    key="compare_group"
    )


    # ------------------------------------------------------------------------------
    # 3)  LIGNE DU JOUEUR  (toujours trouvée dans le df complet)
    # ------------------------------------------------------------------------------
    joueur = df[df['nom'] == joueur_nom].iloc[0]

    # Radar
    # Radar et infos joueur
    st.markdown("### Profil du joueur")

    col_1, col_2 = st.columns([3, 2])  # radar plus large

    with col_1:
        joueur = df[df["nom"] == joueur_nom].iloc[0]
        fig = radar_player_vs_median(df, joueur, features , compare_group=compare_group)
        st.plotly_chart(fig, use_container_width=True)  # use_container_width gère mieux la responsivité

    with col_2:
        #st.image(f"images/photo_{joueur['player_id']}.jpg", caption=joueur_nom, width=200)

        img_base64 = get_image_base64(f"images/photo_{joueur['player_id']}.jpg")
        st.markdown(
                f"""
                <div>
                    <img src="data:image/webp;base64,{img_base64}" width="200" style="margin-bottom: 5px;" />
                    <div style="font-weight: 700; font-size: 18px;">
                        {joueur['nom']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
        )



        st.markdown(f"**Poste** : {joueur['poste']}")
        st.markdown(f"**Club** : {joueur['club']}")
        st.markdown(f"**Age** : {joueur['age']} ans")
        st.markdown(f"**Taille** : {joueur['taille_cm']} cm")
        st.markdown(f"**Poids** : {joueur['poids_kg']} kg")
        st.markdown(f"**Pays** : {joueur['pays']}")
        st.markdown(f"**Matchs joués** : {joueur['nombre_matchs_joues']}")
        st.markdown(f"**Minutes jouées** : {joueur['temps_jeu_min']}")

    # Espacement visuel
    st.markdown("---")

    poste = joueur["poste"]
    if compare_group == 'Poste équivalent' :
        poste_reel = poste
    else :
        poste_reel = compare_group
    st.markdown(f"### Statistiques détaillées /80min et Positionnement Centile par rapport à {poste_reel}")

    # -------- 1. sous-ensemble de référence (même logique que dans ton radar)
    sub = df.loc[
        (df["poste"] == poste_reel) & (df["temps_jeu_min"] > 200),
        features
    ].copy()

    # -------- 2. calcul des centiles pour le joueur
    centiles = {
        feat: int(round((sub[feat] < joueur[feat]).mean() * 100))
        for feat in features
    }

    # ──────────────────────────────────────────────────────────────
    # 1) Lignes « Valeur » et « Centile » alignées sur les colonnes
    # ──────────────────────────────────────────────────────────────
    val_row   = joueur[features].astype(float).rename("Valeur").round(2)
    cent_row  = pd.Series(centiles, name="Centile")          # 0-100

    table = pd.concat([val_row, cent_row], axis=1).T         # 2 lignes

    # ──────────────────────────────────────────────────────────────
    # 2) Couleur de barre = fonction linéaire rouge → vert
    # ──────────────────────────────────────────────────────────────
    def pct_to_hex(p):
        p = max(0, min(100, int(round(p))))
        r = int(255 + (0x4C-255)*p/100)      # 255→76
        g = int( 77 + (0xAF- 77)*p/100)      # 77 →175
        b = int( 77 + (0x50- 77)*p/100)      # 77 →80
        return f"#{r:02X}{g:02X}{b:02X}"

    def bar_css(val, h=28):
        if pd.isna(val):           # cas sans centile
            return f"background:#f0f0f0;height:{h}px;line-height:{h}px;"
        color = pct_to_hex(val)
        return (f"background:linear-gradient(90deg,{color} 0%,{color} {val}%,"
                f"transparent {val}%,transparent 100%);"
                f"height:{h}px;line-height:{h}px;padding:0;")

    # ──────────────────────────────────────────────────────────────
    # 3) Stylage : chiffres ±2 décimales + barre sur la ligne Centile
    # ──────────────────────────────────────────────────────────────
    styler = (
        table.style
            # formats
            .format("{:.2f}", subset=pd.IndexSlice["Valeur", :])
            .format(lambda v: f"{int(v)}%" if pd.notna(v) else "",
                    subset=pd.IndexSlice["Centile", :])

            # barre couleur sur la ligne Centile
            .applymap(bar_css, subset=pd.IndexSlice["Centile", :])

            # texte centré
            .set_properties(**{"text-align": "center", "padding": "0px"})

            # ───── NOUVEAU : teintes + bordures ──────────
            .set_table_styles(
                [
                    # gris clair sur les entêtes de colonne (th)
                    {"selector": "thead th",
                    "props": [("background-color", "#f4f4f6")]},

                    # gris clair sur la 1ʳᵉ colonne (index Valeur/Centile)
                    {"selector": "th.row_heading",
                    "props": [("background-color", "#f4f4f6")]},

                    # bordures noires fines pour toutes les cellules
                    {"selector": "th, td",
                    "props": [("border", "1px solid #555")]},
                ]
            )
    )

    # ──────────────────────────────────────────────────────────────
    # 4) Affichage (write/table pour conserver le CSS)
    # ──────────────────────────────────────────────────────────────
    st.table(styler)




# ================================================================
#  PAGE 2 : FEUILLE DE MATCH
# ================================================================
elif page=="Composer mon XV":
    st.header("Composer mon XV")

    # ───────────────────────────────────────────────────────────
    # A)  Initialisation de l'équipe en session
    # ───────────────────────────────────────────────────────────
    if "team" not in st.session_state:
        # 1ʳᵉ ouverture de l’appli → on crée toutes les clés
        st.session_state.team = {p: None for p in POSTES_ORDER}
    else:
        # L'appli avait déjà été ouverte : on ajoute les nouveaux postes
        for p in POSTES_ORDER:
            st.session_state.team.setdefault(p, None)


    # ───────────────────────────────────────────────────────────
    # B)  Table candidats par poste (filtrage unique)
    # ───────────────────────────────────────────────────────────
    CANDIDATES = {
        pos: df[df['poste'] == POSTES_BDD_NAME[pos]]
        for pos in POSTES_ORDER
    }

    # ───────────────────────────────────────────────────────────
    # C)  Callback → met à jour st.session_state.team[pos]
    # ───────────────────────────────────────────────────────────
    def update_team(pos):
        nom = st.session_state[f"select_{pos}"]

        if nom == "AUCUN":
            st.session_state.team[pos] = None
        else:
            # Cherche d'abord dans les candidats du poste
            df_candidats = CANDIDATES[pos]
            match = df_candidats[df_candidats['nom'] == nom]

            if not match.empty:
                st.session_state.team[pos] = match.iloc[0]
            else:
                # Si joueur hors poste, on cherche dans tout le df
                match_global = df[df['nom'] == nom]
                if not match_global.empty:
                    st.session_state.team[pos] = match_global.iloc[0]
                else:
                    # Sécurité ultime : fallback
                    st.session_state.team[pos] = None

    # ───────────────────────────────────────────────────────────
    # D)  Sélecteurs (3 colonnes)
    # ───────────────────────────────────────────────────────────
    st.markdown("### Sélection des postes")
    cols_sel = st.columns(3)
    for i, pos in enumerate(POSTES_ORDER):
        with cols_sel[i % 3]:
            # Checkbox : poste libre ou non
            poste_libre = st.checkbox(
                "🔁 Poste libre", 
                key=f"libre_{pos}", 
                value=False
            )

            # Détermine la source de joueurs : tous ou filtrés par poste
            if poste_libre:
                candidats = df['nom'].dropna().sort_values().tolist()
            else:
                candidats = CANDIDATES[pos]['nom'].dropna().sort_values().tolist()

            # Ajout de l’option AUCUN
            options = ["AUCUN"] + candidats

            # Récupération du joueur actuellement sélectionné
            current = st.session_state.team[pos]
            current_nom = current['nom'] if current is not None else "AUCUN"

            # Si le joueur actuellement sélectionné ne fait plus partie des options (filtrage remis)
            if not poste_libre and current_nom != "AUCUN":
                if current_nom not in CANDIDATES[pos]['nom'].values:
                    current_nom = "AUCUN"
                    st.session_state.team[pos] = None  # Réinitialise proprement

            # Choix par défaut
            default_index = options.index(current_nom) if current_nom in options else 0

            # Selectbox avec callback
            st.selectbox(
                label=pos,
                options=options,
                index=default_index,
                key=f"select_{pos}",
                on_change=update_team,
                args=(pos,)
            )

    # ───────────────────────────────────────────────────────────
    # E)  Slot d'affichage (photo + nom ou placeholder)
    # ───────────────────────────────────────────────────────────
    def player_slot(player , pos):
        if player is None:
            img_base64 = get_image_base64("images/no_player.webp")
            st.markdown(
                f"""
                <div style="display: flex; flex-direction: column; align-items: center;">
                    <img src="data:image/webp;base64,{img_base64}" width="200" style="margin-bottom: 5px;" />
                    <div style="text-align: center; font-weight: 700; font-size: 18px;">
                        {pos}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:
            img_base64 = get_image_base64(f"images/photo_{player['player_id']}.jpg")
            st.markdown(
                f"""
                <div style="display: flex; flex-direction: column; align-items: center;">
                    <img src="data:image/webp;base64,{img_base64}" width="200" style="margin-bottom: 5px;" />
                    <div style="text-align: center; font-weight: 700; font-size: 18px;">
                        {player['nom']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # ───────────────────────────────────────────────────────────
    # F)  Affiche type FFR
    # ───────────────────────────────────────────────────────────
    st.subheader("XV de départ")
    for row in POSTES_LAYOUT:
        cols_row = st.columns(len(row), gap="small")
        for col, pos in zip(cols_row, row):
            with col:
                player_slot(st.session_state.team[pos] , pos)

    # ───────────────────────────────────────────────────────────
    # G)  Radar équipe vs référence (si XV complet)
    # ───────────────────────────────────────────────────────────
    if all(v is not None for v in st.session_state.team.values()):
        st.success("XV complet !")
        # ======================================================================
        #  RADARS SÉPARÉS  : AVANTS  vs ARRIÈRES
        # ======================================================================

        col_av, col_ar = st.columns(2)       # 2 colonnes d’égale largeur

        # ----------------------------------------------------------------------
        # A)  Définition des groupes
        # ----------------------------------------------------------------------
        POSTES_AVANTS = [
            'Pilier gauche', 'Talonneur', 'Pilier droit',
            '2ème ligne gauche', '2ème ligne droite',
            '3ème ligne aile fermée', '3ème ligne centre', '3ème ligne aile ouverte'
        ]

        POSTES_ARRIERE = [
            'Mêlée', 'Ouverture',
            'Aile gauche', 'Premier centre', 'Deuxième centre', 'Aile droite',
            'Arrière'
        ]

        features_avants = [
            'taille_cm','poids_kg','ratio_poids_taille','ratio_min_matchs',
            'essais','courses','metres_parcourus','passes',
            'offloads','plaquages_casses','plaquages_reussis','ballon_grattes',
            'penales_concedees','carton_jaune'
        ]

        features_arriere = [           # adapte à tes besoins
            'taille_cm','ratio_poids_taille','ratio_min_matchs',
            'essais','courses','metres_parcourus','passes',
            'offloads','franchissements','ballon_joues_pied', 'metres_pied', 
            'plaquages_reussis'
        ]

        # ----------------------------------------------------------------------
        # B)  Helper pour récupérer stats cumulées d’un groupe
        # ----------------------------------------------------------------------
        def stats_groupe(postes, feats, team_dict, ref_team_ids):
            # ------------------------------------------------------------------
            # MON ÉQUIPE
            # ------------------------------------------------------------------
            my_rows = [ply[feats] for pos, ply in team_dict.items()
                    if pos in postes and ply is not None]
            my_stats = pd.DataFrame(my_rows).sum().to_dict()
            posts_my = [ply["poste"] for pos, ply in team_dict.items()
                        if pos in postes and ply is not None]

            # ------------------------------------------------------------------
            # ÉQUIPE RÉFÉRENCE
            # ------------------------------------------------------------------
            ref_rows = [player_row(pid)[feats] for pos, pid in ref_team_ids.items()
                        if pos in postes]
            ref_stats = pd.DataFrame(ref_rows).sum().to_dict()
            posts_ref = [player_row(pid)["poste"] for pos, pid in ref_team_ids.items()
                        if pos in postes]

            return my_stats, ref_stats, posts_my, posts_ref
        # ----------------------------------------------------------------------
        # C)  AVANTS  (colonne gauche)
        # ----------------------------------------------------------------------
        with col_av:
            st.subheader("Avants")

            my_av, ref_av, p_my_av, p_ref_av = stats_groupe(
                POSTES_AVANTS, features_avants, st.session_state.team, REF_TEAM
            )

            fig_av = radar_figure(my_av,  "Avants – Mon XV",  "#4363d8",
                                features_avants, p_my_av,  include_grid=True)
            fig_ref_av = radar_figure(ref_av, "Avants – Référence", "#ffe119",
                                    features_avants, p_ref_av, include_grid=False)
            for tr in fig_ref_av.data: fig_av.add_trace(tr)

            st.plotly_chart(fig_av, use_container_width=True)

        # ----------------------------------------------------------------------
        # D)  ARRIÈRES  (colonne droite)
        # ----------------------------------------------------------------------
        with col_ar:
            st.subheader("Arrières")

            my_ar, ref_ar, p_my_ar, p_ref_ar = stats_groupe(
                POSTES_ARRIERE, features_arriere, st.session_state.team, REF_TEAM
            )

            fig_ar = radar_figure(my_ar,  "Arrières – Mon XV",  "#3cb44b",
                                features_arriere, p_my_ar,  include_grid=True)
            fig_ref_ar = radar_figure(ref_ar, "Arrières – Référence", "#ffe119",
                                    features_arriere, p_ref_ar, include_grid=False)
            for tr in fig_ref_ar.data: fig_ar.add_trace(tr)

            st.plotly_chart(fig_ar, use_container_width=True)
    else:
        st.info("Sélectionne un joueur pour chaque poste afin d’afficher le radar d’équipe.")

    # ───────────────────────────────────────────────────────────
    # F)  Affiche type FFR de l'équipe de comparaison
    # ───────────────────────────────────────────────────────────
    st.subheader("XV de départ de l'équipe de comparaison")
    for row in POSTES_LAYOUT:
        cols_row = st.columns(len(row), gap="small")
        for col, pos in zip(cols_row, row):
            with col:
                player_slot(player_row(REF_TEAM[pos]), pos)




elif page == "Recherche similarité":
    st.header("Joueurs similaires")



    with st.sidebar:
        st.markdown("### Statistiques de pondération")
        st.multiselect(
            "Choisir les variables de pondération",
            options=ALL_STATS,
            default=st.session_state.stats_po_sel,
            key="multiselect_po_stats",
            on_change=update_po_stats          # ← déclenche la callback
        )


        st.sidebar.markdown("### Filtrer par poste (optionnel)")
        st.sidebar.multiselect(
            "Postes concernés",
            ALL_POSTES,
            default = st.session_state.pos_filter,
            key     = "multiselect_postes",
            on_change = update_post_filter
        )

    # ------------------------------------------------------------------
    # A) Choix du joueur de référence (réutilise autocomplétion existante)
    # ------------------------------------------------------------------
    noms_sorted = df.sort_values("nom")['nom'].unique().tolist()
    default_idx = noms_sorted.index(st.session_state.ref_player)

    ref_nom = st.selectbox(
        "Joueur de référence",
        options = noms_sorted,
        index   = default_idx,                   # ← valeur mémorisée
        key     = "ref_player_sel",
        on_change = update_ref_player
    )

    ref_row = df[df['nom'] == ref_nom].iloc[0]

    # ------------------------------------------------------------------
    # B) Choix interactif des POIDS
    # ------------------------------------------------------------------
    st.subheader("Pondération des statistiques")
    cols_w = st.columns(3)
    for i,var in enumerate(st.session_state.stats_po_sel):          # seulement les stats affichées
        with cols_w[i % 3]:
            st.session_state.weights[var] = st.slider(
                var, 0.0, 5.0, float(st.session_state.weights[var]), 0.1,
                key=f"w_{var}"
            )



    # --------  D)  Préparation des données ----------------------------
    feats = st.session_state.stats_po_sel
    for f in feats:                         # poids défaut si manquant
        st.session_state.weights.setdefault(f, 1.0)

    W = np.sqrt([st.session_state.weights[f] for f in feats])

    X_all = df[feats].fillna(0).to_numpy()
    scaler = StandardScaler().fit(X_all)      # centré-réduit sur toute la base

    ref_vec = scaler.transform(ref_row[feats].fillna(0).to_numpy().reshape(1, -1))[0] * W

    # --- sous-ensemble selon le filtre poste --------------------------
    if st.session_state.pos_filter:           # au moins un poste coché
        df_candidates = df[df['poste'].isin(st.session_state.pos_filter)].copy()
    else:                                     # [] ⇒ pas de filtre
        df_candidates = df.copy()

    X_cand   = scaler.transform(df_candidates[feats].fillna(0).to_numpy()) * W
    dists    = np.linalg.norm(X_cand - ref_vec, axis=1)
    df_sim   = df_candidates.assign(distance=dists)\
                             .sort_values("distance")\
                             .query("nom != @ref_nom")

    # --------  E)  Affichage des résultats ----------------------------
    n_show = st.slider("Combien de suggestions ?", 3, 30, 10)
    top_sim = df_sim.head(n_show)

    st.subheader(f"{n_show} joueurs les plus proches")
    for _, row in top_sim.iterrows():
        col1, col2 = st.columns([3,1])
        with col1:
            st.write(f"**{row['nom']}** – {row['club']} ({row['poste']}) "
                     f"— *dist. {row['distance']:.3f}*")
        with col2:
            if st.button("Voir", key=f"goto_{row['nom']}"):
                st.session_state.current_player = row['nom']     # page 1
                st.session_state.current_page   = "Visualisation joueur"

