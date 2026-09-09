"""
Application Streamlit - Calculateur de pauvrete d'un foyer (US-19)

Reprend EXACTEMENT la logique deja validee dans le projet, appliquee a
UN SEUL foyer saisi manuellement, au lieu de tout un fichier CSV :

1. Echelle d'equivalence OCDE modifiee (build_business_features_v2.py)
2. Revenu net equivalise + consommation equivalisee, compares a un
   seuil (mediane du revenu net equivalise de la population / 2,
   convention OCDE - Build_poverty_targets_v2.py)
3. Ratio logement/revenu, compare au seuil "risque logement"
   (Regle 2 finale : 2.0 en period1, 1.5 en period2 - apply_rules_
   correction_v2.py)

Lancement : streamlit run streamlit_calculateur_pauvrete.py
Installation prealable si besoin : pip install streamlit --break-system-packages
"""

import streamlit as st

st.set_page_config(page_title="Calculateur de pauvrete - foyer CEX", layout="centered")

# ----------------------------------------------------------------------
# Seuils "risque logement" retenus dans le projet, par periode (voir
# apply_rules_correction_v2.py - ajustes empiriquement, avec verification
# du taux de faux positifs avant validation)
# ----------------------------------------------------------------------
SEUIL_RATIO_LOGEMENT = {
    "Period 1 (2017-2020)": 2.0,
    "Period 2 (2021-2024)": 1.5,
}


def calculer_echelle_equivalence(fam_size: float, nb_enfants: float) -> float:
    """Meme formule que compute_equivalence_scale() dans
    build_business_features_v2.py : 1 + 0.5 x adultes supplementaires
    + 0.3 x enfants."""
    nb_adultes = max(fam_size - nb_enfants, 1.0)
    nb_adultes_supplementaires = max(nb_adultes - 1.0, 0.0)
    return 1.0 + 0.5 * nb_adultes_supplementaires + 0.3 * nb_enfants


st.title("Calculateur de pauvrete d'un foyer")
st.caption(
    "Reproduit la methodologie du projet (echelle d'equivalence OCDE, "
    "seuil de pauvrete relative, regle de charge locative) sur un foyer "
    "saisi manuellement."
)

# ----------------------------------------------------------------------
# Barre laterale : choix de la periode et du seuil de pauvrete
# ----------------------------------------------------------------------
st.sidebar.header("Parametres de reference")

periode = st.sidebar.selectbox("Periode de reference", list(SEUIL_RATIO_LOGEMENT.keys()))
seuil_ratio_logement = SEUIL_RATIO_LOGEMENT[periode]

st.sidebar.markdown(
    f"**Seuil risque logement retenu :** ratio > {seuil_ratio_logement} "
    f"(charge de logement / revenu net)"
)

mode_seuil = st.sidebar.radio(
    "Seuil de pauvrete (mediane du revenu net equivalise / 2)",
    ["Saisir le seuil manuellement", "Calculer depuis un fichier"],
)

seuil_pauvrete = None
if mode_seuil == "Saisir le seuil manuellement":
    seuil_pauvrete = st.sidebar.number_input(
        "Seuil de pauvrete ($/an, par equivalent-adulte)",
        min_value=0.0, value=20000.0, step=500.0,
        help="A retrouver dans SEUIL_PAUVRETE de poverty_targets_period{1,2}.csv, "
             "ou dans les logs de Build_poverty_targets_v2.py.",
    )
else:
    fichier = st.sidebar.file_uploader(
        "Fichier poverty_targets_period{1,2}.csv (pour recalculer la mediane)",
        type=["csv"],
    )
    if fichier is not None:
        import pandas as pd
        df_ref = pd.read_csv(fichier, low_memory=False)
        if "REVENU_NET_EQUIVALISE" in df_ref.columns:
            mediane = df_ref["REVENU_NET_EQUIVALISE"].median()
            seuil_pauvrete = mediane / 2
            st.sidebar.success(f"Seuil recalcule : {seuil_pauvrete:,.0f} $ (mediane = {mediane:,.0f} $)")
        else:
            st.sidebar.error("Colonne REVENU_NET_EQUIVALISE absente de ce fichier.")

# ----------------------------------------------------------------------
# Formulaire de saisie du foyer
# ----------------------------------------------------------------------
st.header("1. Composition du foyer")
col1, col2 = st.columns(2)
with col1:
    fam_size = st.number_input("Nombre total de personnes dans le foyer", min_value=1, value=1, step=1)
with col2:
    nb_enfants = st.number_input(
        "Dont nombre d'enfants (moins de 18 ans)", min_value=0,
        max_value=int(fam_size), value=0, step=1,
    )

echelle = calculer_echelle_equivalence(fam_size, nb_enfants)
st.info(f"Echelle d'equivalence calculee : **{echelle:.2f}**")

st.header("2. Revenu et depenses annuelles")
col3, col4 = st.columns(2)
with col3:
    revenu_net = st.number_input(
        "Revenu net annuel du foyer ($)", min_value=0.0, value=30000.0, step=500.0,
        help="Equivalent de FINATXEM dans les donnees CEX.",
    )
with col4:
    depense_totale = st.number_input(
        "Depense totale annuelle du foyer ($)", min_value=0.0, value=30000.0, step=500.0,
        help="Equivalent de TOTEXP_FULL_ANNUEL. Laisser a 0 si inconnu.",
    )

depense_logement = st.number_input(
    "Dont depense annuelle en logement ($)", min_value=0.0, value=10000.0, step=500.0,
    help="Equivalent de HOUS_FULL_ANNUEL - utilisee pour le ratio logement/revenu.",
)

# ----------------------------------------------------------------------
# Calculs
# ----------------------------------------------------------------------
st.header("3. Resultat")

if seuil_pauvrete is None:
    st.warning("Renseigne un seuil de pauvrete dans la barre laterale pour voir le resultat.")
else:
    revenu_net_equivalise = revenu_net / echelle
    consommation_equivalisee = depense_totale / echelle if depense_totale > 0 else None

    pauvre_revenu = revenu_net_equivalise < seuil_pauvrete
    pauvre_consommation = (
        consommation_equivalisee < seuil_pauvrete if consommation_equivalisee is not None else None
    )

    ratio_logement_revenu = depense_logement / revenu_net if revenu_net > 0 else None
    a_risque_logement = (
        ratio_logement_revenu > seuil_ratio_logement if ratio_logement_revenu is not None else False
    )

    colA, colB, colC = st.columns(3)
    colA.metric("Revenu net equivalise", f"{revenu_net_equivalise:,.0f} $")
    if consommation_equivalisee is not None:
        colB.metric("Consommation equivalisee", f"{consommation_equivalisee:,.0f} $")
    if ratio_logement_revenu is not None:
        colC.metric("Ratio logement / revenu", f"{ratio_logement_revenu:.2f}")

    st.divider()

    # Categorie finale, meme priorite que apply_rules_correction_v2.py :
    # la Regle 2 (charge locative) prime sur le reste.
    if a_risque_logement:
        st.error(
            "**Categorie : Tres pauvre (charge locative)**\n\n"
            f"Le ratio logement/revenu ({ratio_logement_revenu:.2f}) depasse le seuil "
            f"de risque ({seuil_ratio_logement}) - ce foyer consacre une part "
            "disproportionnee de son revenu au logement, meme si son niveau de vie "
            "global peut sembler normal."
        )
    elif pauvre_revenu:
        st.error(
            f"**Foyer classe pauvre (revenu net)**\n\n"
            f"Revenu net equivalise ({revenu_net_equivalise:,.0f} $) inferieur au "
            f"seuil de pauvrete ({seuil_pauvrete:,.0f} $)."
        )
    else:
        st.success(
            f"**Foyer non classe pauvre (revenu net)**\n\n"
            f"Revenu net equivalise ({revenu_net_equivalise:,.0f} $) superieur au "
            f"seuil de pauvrete ({seuil_pauvrete:,.0f} $)."
        )

    if pauvre_consommation is not None:
        if pauvre_consommation:
            st.warning(
                f"Consommation equivalisee ({consommation_equivalisee:,.0f} $) "
                f"egalement sous le seuil - pauvrete confirmee par la consommation."
            )
        else:
            st.info(
                f"Consommation equivalisee ({consommation_equivalisee:,.0f} $) "
                f"au-dessus du seuil - la consommation reelle du foyer reste "
                f"proche de la normale, meme si son revenu declare est bas."
            )

    with st.expander("Voir le detail des calculs"):
        st.write(f"Echelle d'equivalence = 1 + 0.5 x max({fam_size}-{nb_enfants}-1, 0) "
                 f"+ 0.3 x {nb_enfants} = **{echelle:.3f}**")
        st.write(f"Revenu net equivalise = {revenu_net:,.0f} / {echelle:.3f} = "
                 f"**{revenu_net_equivalise:,.0f} $**")
        if consommation_equivalisee is not None:
            st.write(f"Consommation equivalisee = {depense_totale:,.0f} / {echelle:.3f} = "
                     f"**{consommation_equivalisee:,.0f} $**")
        if ratio_logement_revenu is not None:
            st.write(f"Ratio logement/revenu = {depense_logement:,.0f} / {revenu_net:,.0f} = "
                     f"**{ratio_logement_revenu:.3f}**")
        st.write(f"Seuil de pauvrete utilise : **{seuil_pauvrete:,.0f} $**")
        st.write(f"Seuil de risque logement utilise ({periode}) : **{seuil_ratio_logement}**")

st.divider()
st.caption(
    "Methodologie : echelle d'equivalence OCDE modifiee, seuil de pauvrete "
    "relative (mediane du revenu net equivalise / 2, convention OCDE), "
    "regle de charge locative validee empiriquement sur les donnees CEX 2017-2024."
)