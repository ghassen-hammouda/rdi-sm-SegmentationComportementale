"""
Diagnostic cible (US-16) : pourquoi la Regle 2 ne capture-t-elle pas
TOUS les foyers pauvres restes dans la categorie "Moyen"/"Moyen
inferieur" apres la premiere correction ?

Compare deux groupes :
- Groupe A : foyers deja captures par la Regle 2 (categorie =
  "Tres pauvre (charge locative)")
- Groupe B : foyers PAS captures, mais toujours dans la categorie
  intermediaire ET classes pauvres selon le revenu net

Objectif : voir sur quelles variables ces deux groupes se ressemblent
(pour elargir la regle en consequence) ou different (pour comprendre
s'il s'agit d'un phenomene different).
"""

import pandas as pd
from pathlib import Path

INPUT_DIR = Path("outputs_v2")

RAW_FILES = {
    "period1": "poverty_targets_period1.csv",
    "period2": "poverty_targets_period2.csv",
}
ID_COLUMN = "CUID"

# Categorie intermediaire suspecte a examiner, et k correspondant, par periode
TARGET_CONFIG = {
    "period1": {"k": 6, "categorie_suspecte": "Moyen inferieur"},
    "period2": {"k": 3, "categorie_suspecte": "Moyen"},
}

PROFILE_COLUMNS = [
    "ratio_logement_revenu_net", "part_hous", "totexp_par_equivalent_adulte",
    "CUTENURE", "FAM_TYPE", "AGE_REF", "FAM_SIZE", "FINATXEM",
]


def diagnose(period_label: str) -> None:
    config = TARGET_CONFIG[period_label]
    k = config["k"]
    categorie_suspecte = config["categorie_suspecte"]

    corrected_path = INPUT_DIR / f"final_labels_corrected_{period_label}_k{k}.csv"
    raw_path = INPUT_DIR / RAW_FILES[period_label]

    df_corrected = pd.read_csv(corrected_path, low_memory=False)
    df_raw = pd.read_csv(raw_path, low_memory=False)

    df = df_corrected.merge(df_raw, on=ID_COLUMN, how="left")

    print(f"\n{'=' * 70}\n{period_label.upper()} (k={k}) - Categorie suspecte : '{categorie_suspecte}'\n{'=' * 70}")

    groupe_a = df[df["categorie"] == "Tres pauvre (charge locative)"]
    groupe_b = df[
        (df["categorie"] == categorie_suspecte)
        & (df["statut_pauvrete_revenu_net"] == 1)
    ]

    print(f"\nGroupe A (deja captures par la Regle 2) : {len(groupe_a)} foyers")
    print(f"Groupe B (pauvres, encore dans '{categorie_suspecte}', PAS captures) : {len(groupe_b)} foyers")

    if len(groupe_b) == 0:
        print("Aucun foyer dans le Groupe B - rien a diagnostiquer.")
        return

    print(f"\n--- Comparaison des profils moyens (variables continues) ---")
    comparaison = pd.DataFrame({
        "Groupe A (captures)": groupe_a[["ratio_logement_revenu_net", "part_hous",
                                          "totexp_par_equivalent_adulte", "AGE_REF",
                                          "FAM_SIZE", "FINATXEM"]].mean(),
        "Groupe B (rates)": groupe_b[["ratio_logement_revenu_net", "part_hous",
                                       "totexp_par_equivalent_adulte", "AGE_REF",
                                       "FAM_SIZE", "FINATXEM"]].mean(),
    })
    print(comparaison.round(2))

    print(f"\n--- Repartition CUTENURE (statut logement) - Groupe B ---")
    print(groupe_b["CUTENURE"].value_counts())
    print(f"    Rappel : la Regle 2 exige CUTENURE == 4 (locataire)")

    print(f"\n--- Repartition FAM_TYPE (type de foyer) - Groupe B ---")
    print(groupe_b["FAM_TYPE"].value_counts())
    print(f"    Rappel : la Regle 2 exige FAM_TYPE == 8 (personne seule)")

    print(f"\n--- Distribution du ratio_logement_revenu_net - Groupe B ---")
    print(groupe_b["ratio_logement_revenu_net"].describe())
    print(f"    Rappel : la Regle 2 exige ratio > 2.5")
    n_ratio_ok = int((groupe_b["ratio_logement_revenu_net"] > 2.5).sum())
    print(f"    Foyers du Groupe B qui ONT DEJA un ratio > 2.5 : {n_ratio_ok} / {len(groupe_b)} "
          f"({n_ratio_ok/len(groupe_b)*100:.1f}%)")


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    for period_label in TARGET_CONFIG:
        diagnose(period_label)

    print(f"\n{'=' * 70}")
    print("A EXAMINER : si la majorite du Groupe B a deja un ratio > 2.5 mais")
    print("des CUTENURE/FAM_TYPE differents de 4/8, la Regle 2 est trop stricte")
    print("sur ces deux conditions -> les retirer et garder seulement le ratio.")
    print("Si le ratio du Groupe B est plus bas (entre 1.5 et 2.5 par exemple),")
    print("il faudra plutot abaisser le seuil de ratio.")