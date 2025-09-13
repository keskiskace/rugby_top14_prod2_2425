# ----------------------
# SECTION JOUEURS
# ----------------------
st.header("🔎 Joueurs — Comparateur")

# ----------------------
# FILTRES AVANCÉS
# ----------------------
st.subheader("🎛️ Filtres")

# 1. Saison (déjà géré plus haut : selected_saisons + df_filtered)

# 2. Division (Top14 / ProD2)
if "division" in df_filtered.columns:
    divisions_dispo = sorted(df_filtered['division'].dropna().unique())
    selected_div = st.multiselect(
        "Choisir une division (optionnel)",
        divisions_dispo,
        default=[]
    )
    if selected_div:
        df_filtered = df_filtered[df_filtered['division'].isin(selected_div)]

# 3. Club
if "club" in df_filtered.columns:
    clubs_dispo = sorted(df_filtered['club'].dropna().unique())
    selected_clubs = st.multiselect(
        "Choisir un ou plusieurs clubs (optionnel)",
        clubs_dispo,
        default=[]
    )
    if selected_clubs:
        df_filtered = df_filtered[df_filtered['club'].isin(selected_clubs)]

# 4. Poste
if "poste" in df_filtered.columns:
    postes_dispo = sorted(df_filtered['poste'].dropna().unique())
    selected_postes = st.multiselect(
        "Choisir un ou plusieurs postes (optionnel)",
        postes_dispo,
        default=[]
    )
    if selected_postes:
        df_filtered = df_filtered[df_filtered['poste'].isin(selected_postes)]

# 5. Joueurs restants
player_options = df_filtered['display_name'].sort_values().unique().tolist()
if player_options:
    selected_names = st.multiselect(
        "Choisir un ou plusieurs joueurs",
        player_options,
        default=[player_options[0]]
    )
    selected_players = df_filtered[df_filtered['display_name'].isin(selected_names)].copy()
else:
    selected_players = pd.DataFrame()
    st.warning("Aucun joueur ne correspond aux filtres choisis.")

# Sélection joueurs types (si présents)
selected_types = []
if not extra_df.empty:
    types_opts = extra_df['nom'].sort_values().unique().tolist()
    selected_types = st.multiselect("Choisir un ou plusieurs joueurs types", types_opts, default=[])
    selected_type_players = extra_df[extra_df['nom'].isin(selected_types)].copy()
else:
    selected_type_players = pd.DataFrame()

# Concat sélection réelle + joueurs types
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

# Affichage des joueurs sélectionnés
for _, joueur in selected_players.iterrows():
    nom_aff = joueur.get('nom', '')
    st.subheader(nom_aff)
    if "Joueur type" not in str(nom_aff):
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

# Préparation colonnes statistiques
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

    st.download_button(
        "⬇️ Télécharger en CSV",
        result.to_csv(index=False).encode("utf-8"),
        file_name=f"{choice_type.lower().replace(' ', '_')}_{choice_stat}.csv",
        mime="text/csv"
    )

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

        club_options = clubs_filtered['display_name'].sort_values().unique().tolist()
        selected_clubs = st.multiselect(
            "Choisir un ou plusieurs clubs",
            club_options,
            default=[club_options[0]] if club_options else []
        )
        selected_clubs_df = clubs_filtered[clubs_filtered['display_name'].isin(selected_clubs)].copy()

        if not selected_clubs_df.empty:
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

