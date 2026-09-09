"""
Tableau de bord Streamlit complet (US-24) - pour la soutenance.

Regroupe TOUT le projet en un seul endroit navigable par onglets :
pipeline, resultats de clustering, regles de correction, test de
robustesse, et le calculateur interactif. Objectif : si le jury ne
comprend pas une explication orale, tout peut etre montre concretement
a l'ecran ici.

Lancement : py -m streamlit run streamlit_dashboard.py

Ce script s'attend a trouver le dossier outputs_v2/ (avec son
sous-dossier graphs/) au meme niveau, ou dans le repertoire d'ou tu
lances streamlit. Adapte OUTPUTS_DIR si besoin. Il a aussi besoin du
fichier regles_pauvrete.py dans le meme dossier.
"""

import streamlit as st
import pandas as pd
from pathlib import Path

from regles_pauvrete import (
    apply_rule1_isolement_age,
    apply_rule2_charge_locative,
    SEUIL_RATIO_LOGEMENT_PAR_PERIODE,
    SEUIL_RULE1_AGE_MIN,
    SEUIL_RULE1_PART_RETPEN_MAX,
    SEUIL_RULE1_TOTEXP_MAX,
)

st.set_page_config(page_title="Segmentation de la pauvrete - CEX", layout="wide")

OUTPUTS_DIR = Path("outputs_v2")
GRAPH_DIR = OUTPUTS_DIR / "graphs"

st.title("Segmentation comportementale de la pauvrete - CEX")
st.caption("Tableau de bord complet du projet - pipeline, clustering, regles, robustesse, calculateur")

onglet_pipeline, onglet_resultats, onglet_regles, onglet_robustesse, onglet_calculateur = st.tabs(
    ["Pipeline", "Resultats du clustering", "Regles de correction", "Test de robustesse", "Calculateur"]
)

# ========================================================================
# ONGLET 1 - PIPELINE
# ========================================================================
with onglet_pipeline:
    st.header("Vue d'ensemble du pipeline")
    st.markdown("""
    **Donnees sources** : CEX (Consumer Expenditure Survey, BLS, USA), deux
    periodes - **period1** (2017-2020) et **period2** (2021-2024).

    **Etapes du pipeline, dans l'ordre :**
    1. Selection des colonnes utiles depuis les fichiers bruts
    2. Reconstitution du trimestre complet (CQ + PQ)
    3. Nettoyage (dedoublonnage, colonnes incompletes, ecretage outliers)
    4. Encodage + standardisation (preparation de la regression)
    5. **Regression lineaire** pour estimer le revenu net manquant en 2024
       (R² = 0.9812, MAE = 3639 $)
    6. Fusion du revenu net complete avec le fichier propre
    7. Agregation : foyer-trimestre -> un foyer = une ligne
    8. Construction des features metier (echelle d'equivalence OCDE,
       parts de budget, ratios)
    9. Construction des cibles de pauvrete (seuil OCDE, revenu net et
       consommation)
    10. Preparation de l'espace de clustering (exclusions, encodage,
        standardisation - revenu et statuts de pauvrete JAMAIS vus par
        le clustering)
    11. Benchmark K-Means / GMM / CAH sur echantillon
    12. Clustering final K-Means sur toute la population
    13. Correction par regles metier
    14. Test de robustesse
    """)

    st.info(
        "Principe methodologique central : le clustering ne voit jamais le "
        "revenu ni les statuts de pauvrete pendant son execution. Ces "
        "variables sont reintroduites SEULEMENT APRES pour valider si les "
        "groupes trouves, a l'aveugle, correspondent a de la vraie pauvrete."
    )

# ========================================================================
# ONGLET 2 - RESULTATS DU CLUSTERING
# ========================================================================
with onglet_resultats:
    st.header("Resultats du clustering final")

    periode_select = st.selectbox("Periode", ["period1", "period2"], key="periode_resultats")
    k_select = st.selectbox("Nombre de categories (k)", [3, 4, 5, 6], key="k_resultats")

    col_g, col_d = st.columns(2)

    with col_g:
        img_path = GRAPH_DIR / f"final_categories_{periode_select}_k{k_select}.png"
        if img_path.exists():
            st.image(str(img_path), caption=f"Taux de pauvrete et niveau de vie par categorie - {periode_select}, k={k_select}")
        else:
            st.warning(f"Image introuvable : {img_path}")

    with col_d:
        img_path_corr = GRAPH_DIR / f"correction_regle2_{periode_select}_k{k_select}.png"
        if img_path_corr.exists():
            st.image(str(img_path_corr), caption=f"Avant / apres correction par regles - {periode_select}, k={k_select}")
        else:
            st.warning(f"Image introuvable : {img_path_corr}")

    st.subheader("Tableau des effectifs")
    resume_path = OUTPUTS_DIR / "resume_effectifs_complet.csv"
    if resume_path.exists():
        df_resume = pd.read_csv(resume_path)
        df_filtre = df_resume[(df_resume["periode"] == periode_select) & (df_resume["k"] == k_select)]
        st.dataframe(df_filtre, use_container_width=True)
    else:
        st.warning(f"Fichier introuvable : {resume_path}")

    st.subheader("Benchmark des methodes (echantillon)")
    img_benchmark = GRAPH_DIR / f"benchmark_scores_{periode_select}.png"
    if img_benchmark.exists():
        st.image(str(img_benchmark), caption=f"Silhouette et Davies-Bouldin vs k - {periode_select}")

# ========================================================================
# ONGLET 3 - REGLES DE CORRECTION
# ========================================================================
with onglet_regles:
    st.header("Regles metier de correction")

    st.subheader("Regle 1 - Isolement et age")
    st.markdown(f"""
    Un foyer verifie cette regle s'il cumule **toutes** les conditions suivantes :
    - Personne seule (FAM_TYPE == 8)
    - Age du referent >= {SEUIL_RULE1_AGE_MIN} ans
    - Cotisation retraite quasi nulle (part_retpen < {SEUIL_RULE1_PART_RETPEN_MAX:.0%})
    - Faible niveau de vie (totexp_par_equivalent_adulte < {SEUIL_RULE1_TOTEXP_MAX:,} $)

    Profil confirme par les 3 methodes de clustering independamment
    (K-Means, GMM, CAH), stable au test de robustesse a k=6.
    """)

    st.subheader("Regle 2 - Charge locative extreme")
    st.markdown(f"""
    Un foyer verifie cette regle si :
    - ratio_logement_revenu_net > seuil, **specifique par periode** :
      - Period1 : {SEUIL_RATIO_LOGEMENT_PAR_PERIODE['period1']}
      - Period2 : {SEUIL_RATIO_LOGEMENT_PAR_PERIODE['period2']}

    Seuil de period2 abaisse suite a un diagnostic en 2 temps (voir le
    detail de la demarche dans le rapport) - valide avec un taux de
    faux positifs mesure a 11.1%.

    **Priorite d'application** : la Regle 2 prime sur la Regle 1 - un
    foyer qui verifierait les deux est classe "Tres pauvre (charge
    locative)".
    """)

    st.subheader("Effet de la correction (avant / apres)")
    periode_regle = st.selectbox("Periode", ["period1", "period2"], key="periode_regles")
    k_regle = st.selectbox("k", [3, 4, 5, 6], key="k_regles")
    img_correction = GRAPH_DIR / f"correction_regle2_{periode_regle}_k{k_regle}.png"
    if img_correction.exists():
        st.image(str(img_correction), use_container_width=True)
    else:
        st.warning(f"Image introuvable : {img_correction}")

# ========================================================================
# ONGLET 4 - TEST DE ROBUSTESSE
# ========================================================================
with onglet_robustesse:
    st.header("Test de robustesse (division aleatoire 50/50)")
    st.markdown("""
    Principe : la population est divisee au hasard en deux moities
    independantes. Si une categorie garde un taux de pauvrete coherent
    dans les deux moities (ecart <= 10 points), le resultat est
    considere stable - pas du a un artefact de quelques foyers
    particuliers.
    """)

    periode_rob = st.selectbox("Periode", ["period1", "period2"], key="periode_rob")
    k_rob = st.selectbox("k", [3, 4, 5, 6], key="k_rob")

    img_rob = GRAPH_DIR / f"robustesse_{periode_rob}_k{k_rob}.png"
    if img_rob.exists():
        st.image(str(img_rob), caption=f"Comparaison des deux moities - {periode_rob}, k={k_rob}")

    csv_rob = OUTPUTS_DIR / f"robustesse_{periode_rob}_k{k_rob}.csv"
    if csv_rob.exists():
        df_rob = pd.read_csv(csv_rob)
        st.dataframe(df_rob, use_container_width=True)

        if "stable" in df_rob.columns:
            instables = df_rob[df_rob["stable"] == False]
            if len(instables) > 0:
                noms = list(instables["categorie"]) if "categorie" in instables.columns else list(instables.index)
                st.error(f"Categories instables detectees : {noms}")
            else:
                st.success("Toutes les categories sont stables pour cette combinaison periode/k.")

# ========================================================================
# ONGLET 5 - CALCULATEUR INTERACTIF
# ========================================================================
with onglet_calculateur:
    st.header("Calculateur de pauvrete pour un foyer")

    periode_calc = st.selectbox("Periode de reference", ["period1", "period2"], key="periode_calc")

    col1, col2 = st.columns(2)
    with col1:
        fam_size = st.number_input("Nombre de personnes dans le foyer", min_value=1, value=1, step=1)
    with col2:
        nb_enfants = st.number_input("Dont enfants (< 18 ans)", min_value=0, max_value=int(fam_size), value=0, step=1)

    nb_adultes = max(fam_size - nb_enfants, 1)
    echelle = 1.0 + 0.5 * max(nb_adultes - 1, 0) + 0.3 * nb_enfants
    st.info(f"Echelle d'equivalence : {echelle:.2f}")

    col3, col4, col5 = st.columns(3)
    with col3:
        revenu_net = st.number_input("Revenu net annuel ($)", min_value=0.0, value=30000.0, step=500.0)
    with col4:
        depense_totale = st.number_input("Depense totale annuelle ($)", min_value=0.0, value=30000.0, step=500.0)
    with col5:
        depense_logement = st.number_input("Depense logement annuelle ($)", min_value=0.0, value=10000.0, step=500.0)

    age_ref = st.number_input("Age du referent", min_value=18, max_value=100, value=40, step=1)
    fam_type_seul = st.checkbox("Foyer = personne seule (FAM_TYPE 8)")
    part_retpen = st.number_input("Part du budget en cotisation retraite (0 a 1)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)

    seuil_pauvrete = st.number_input(
        "Seuil de pauvrete de reference ($/an, par equivalent-adulte)",
        min_value=0.0, value=20000.0, step=500.0,
        help="A retrouver dans SEUIL_PAUVRETE de poverty_targets_period{1,2}.csv",
    )

    if st.button("Calculer"):
        revenu_net_equiv = revenu_net / echelle
        conso_equiv = depense_totale / echelle if depense_totale > 0 else None
        ratio_logement = depense_logement / revenu_net if revenu_net > 0 else None

        df_test = pd.DataFrame([{
            "FAM_TYPE": 8 if fam_type_seul else 1,
            "AGE_REF": age_ref,
            "part_retpen": part_retpen,
            "totexp_par_equivalent_adulte": conso_equiv if conso_equiv else 0,
            "ratio_logement_revenu_net": ratio_logement if ratio_logement else 0,
        }])

        rule2_ok = apply_rule2_charge_locative(df_test, periode_calc).iloc[0]
        rule1_ok = apply_rule1_isolement_age(df_test).iloc[0] and not rule2_ok

        st.divider()
        colA, colB, colC = st.columns(3)
        colA.metric("Revenu net equivalise", f"{revenu_net_equiv:,.0f} $")
        if conso_equiv:
            colB.metric("Consommation equivalisee", f"{conso_equiv:,.0f} $")
        if ratio_logement:
            colC.metric("Ratio logement/revenu", f"{ratio_logement:.2f}")

        if rule2_ok:
            st.error("**Categorie : Tres pauvre (charge locative)** - Regle 2 verifiee")
        elif rule1_ok:
            st.error("**Categorie : Tres pauvre (isolement/age)** - Regle 1 verifiee")
        elif revenu_net_equiv < seuil_pauvrete:
            st.warning("**Foyer pauvre selon le revenu net** (aucune regle specifique declenchee)")
        else:
            st.success("**Foyer non classe pauvre**")