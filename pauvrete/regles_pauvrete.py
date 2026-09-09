"""
Module partage des regles metier de pauvrete (US-22).

Regroupe les 2 regles construites et validees au fil du projet, pour
qu'elles soient definies UNE SEULE FOIS et reutilisees partout
(pipeline de correction, app Streamlit) - evite la duplication et les
divergences entre les seuils utilises a differents endroits.

REGLE 1 - Isolement et age (profil dominant en period1) :
  Personne seule (FAM_TYPE == 8) ET age du referent >= 65 ans
  ET cotisation retraite quasi nulle (part_retpen < 4%)
  ET faible niveau de vie (totexp_par_equivalent_adulte < 22000 $)
  Confirmee par les 3 methodes de clustering (K-Means, GMM, CAH),
  profil stable a k=6 (test de robustesse).

REGLE 2 - Charge locative extreme (profil dominant en period2, mais
present aussi en period1) :
  ratio_logement_revenu_net > seuil, SPECIFIQUE PAR PERIODE :
    period1 : 2.0 (suffisant, resout completement l'anomalie observee)
    period2 : 1.5 (seuil abaisse suite a un 2e diagnostic - le seuil de
    2.0 laissait 194 foyers non captures, tous avec un ratio entre 1.54
    et 2.0 - seuil final valide avec un taux de faux positifs de 11.1%)

PRIORITE D'APPLICATION : Regle 2 avant Regle 1 - la charge locative est
le probleme le plus severe et le plus urgent a isoler (taux de
pauvrete-revenu proche de 90-95%, contre 34-97% pour la Regle 1 selon
le sous-groupe) ; un foyer qui verifierait les deux regles a la fois
est classe "Tres pauvre (charge locative)" en priorite.
"""

import pandas as pd

SEUIL_RATIO_LOGEMENT_PAR_PERIODE = {
    "period1": 2.0,
    "period2": 1.5,
}

SEUIL_RULE1_AGE_MIN = 65
SEUIL_RULE1_PART_RETPEN_MAX = 0.04
SEUIL_RULE1_TOTEXP_MAX = 22000

COLONNES_REQUISES_RULE1 = ["FAM_TYPE", "AGE_REF", "part_retpen", "totexp_par_equivalent_adulte"]
COLONNES_REQUISES_RULE2 = ["ratio_logement_revenu_net"]


def apply_rule1_isolement_age(df: pd.DataFrame) -> pd.Series:
    """Regle 1 : profil 'isolement et age'. Retourne un masque booleen."""
    manquantes = [c for c in COLONNES_REQUISES_RULE1 if c not in df.columns]
    if manquantes:
        raise KeyError(f"Colonnes manquantes pour la Regle 1 : {manquantes}")

    return (
        (df["FAM_TYPE"] == 8)
        & (df["AGE_REF"] >= SEUIL_RULE1_AGE_MIN)
        & (df["part_retpen"] < SEUIL_RULE1_PART_RETPEN_MAX)
        & (df["totexp_par_equivalent_adulte"] < SEUIL_RULE1_TOTEXP_MAX)
    )


def apply_rule2_charge_locative(df: pd.DataFrame, period_label: str) -> pd.Series:
    """Regle 2 : profil 'charge locative extreme'. Seuil specifique a
    la periode. Retourne un masque booleen."""
    manquantes = [c for c in COLONNES_REQUISES_RULE2 if c not in df.columns]
    if manquantes:
        raise KeyError(f"Colonnes manquantes pour la Regle 2 : {manquantes}")
    if period_label not in SEUIL_RATIO_LOGEMENT_PAR_PERIODE:
        raise ValueError(f"Periode inconnue : {period_label}")

    seuil = SEUIL_RATIO_LOGEMENT_PAR_PERIODE[period_label]
    return df["ratio_logement_revenu_net"] > seuil


def classer_avec_regles(df: pd.DataFrame, period_label: str, colonne_categorie_base: str = "categorie") -> pd.DataFrame:
    """Applique les 2 regles, dans l'ordre de priorite (Regle 2 puis
    Regle 1), par-dessus une categorie de base deja existante (issue
    du clustering). Ajoute une colonne 'categorie_finale' et deux
    colonnes booleennes de tracabilite."""
    df = df.copy()

    mask_rule2 = apply_rule2_charge_locative(df, period_label)
    mask_rule1 = apply_rule1_isolement_age(df) & ~mask_rule2  # Regle 2 prioritaire

    df["regle2_charge_locative"] = mask_rule2
    df["regle1_isolement_age"] = mask_rule1

    df["categorie_finale"] = df[colonne_categorie_base]
    df.loc[mask_rule1, "categorie_finale"] = "Tres pauvre (isolement/age)"
    df.loc[mask_rule2, "categorie_finale"] = "Tres pauvre (charge locative)"

    return df


if __name__ == "__main__":
    print(__doc__)
    print("Ce module n'est pas concu pour etre execute directement.")
    print("Importer apply_rule1_isolement_age, apply_rule2_charge_locative,")
    print("ou classer_avec_regles depuis un autre script.")