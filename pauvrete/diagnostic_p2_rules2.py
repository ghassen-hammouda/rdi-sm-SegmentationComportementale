"""
Diagnostic de 2e niveau (US-17), cible sur PERIOD2 uniquement.

La Regle 2 elargie (ratio_logement_revenu_net > 2.0, sans CUTENURE ni
FAM_TYPE) a bien ameliore period2, mais pas completement resolu :
la categorie "Moyen" (k=3) reste a 79% de pauvrete-revenu apres
correction, contre 15-16% pour les autres categories.

Ce script compare a nouveau :
- Groupe A : foyers DEJA captures par la regle actuelle (ratio > 2.0)
- Groupe B : foyers encore dans "Moyen", encore pauvres, PAS captures

Objectif : voir si le Groupe B a un ratio juste en dessous de 2.0 (il
suffit de baisser encore un peu le seuil), ou si c'est un phenomene
different qui necessite une autre approche.
"""

import pandas as pd
from pathlib import Path

INPUT_DIR = Path("outputs_v2")

RAW_FILE = "poverty_targets_period2.csv"
CORRECTED_FILE = "final_labels_corrected_period2_k3.csv"
ID_COLUMN = "CUID"
CATEGORIE_SUSPECTE = "Moyen"


def main() -> None:
    df_corrected = pd.read_csv(INPUT_DIR / CORRECTED_FILE, low_memory=False)
    df_raw = pd.read_csv(INPUT_DIR / RAW_FILE, low_memory=False)
    df = df_corrected.merge(df_raw, on=ID_COLUMN, how="left")

    groupe_a = df[df["categorie"] == "Tres pauvre (charge locative)"]
    groupe_b = df[
        (df["categorie"] == CATEGORIE_SUSPECTE)
        & (df["statut_pauvrete_revenu_net"] == 1)
    ]

    print(f"Groupe A (deja captures, ratio > 2.0) : {len(groupe_a)} foyers")
    print(f"Groupe B (pauvres, encore dans 'Moyen', PAS captures) : {len(groupe_b)} foyers")

    if len(groupe_b) == 0:
        print("Rien a diagnostiquer, Groupe B vide.")
        return

    print(f"\n--- Distribution du ratio_logement_revenu_net - Groupe B ---")
    print(groupe_b["ratio_logement_revenu_net"].describe())

    for seuil in [1.0, 1.2, 1.5, 1.8, 2.0]:
        n = int((groupe_b["ratio_logement_revenu_net"] > seuil).sum())
        print(f"    Foyers du Groupe B avec ratio > {seuil} : {n} / {len(groupe_b)} ({n/len(groupe_b)*100:.1f}%)")

    print(f"\n--- Comparaison des profils moyens ---")
    comparaison = pd.DataFrame({
        "Groupe A (captures)": groupe_a[["ratio_logement_revenu_net", "part_hous",
                                          "totexp_par_equivalent_adulte", "FINATXEM",
                                          "AGE_REF", "FAM_SIZE"]].mean(),
        "Groupe B (rates)": groupe_b[["ratio_logement_revenu_net", "part_hous",
                                       "totexp_par_equivalent_adulte", "FINATXEM",
                                       "AGE_REF", "FAM_SIZE"]].mean(),
    })
    print(comparaison.round(2))

    print(f"\n--- FINATXEM (revenu net) du Groupe B - est-ce estime (2024) ou declare ? ---")
    if "FINATXEM_ESTIME_FLAG" in df.columns:
        print(groupe_b["FINATXEM_ESTIME_FLAG"].value_counts())
    else:
        print("Colonne FINATXEM_ESTIME_FLAG absente de ce fichier.")

    print(f"\n--- Verification avant d'abaisser le seuil a 1.5 : combien de foyers NON PAUVRES")
    print(f"    seraient capturés a tort (faux positifs) ? ---")
    for seuil_test in [1.5, 1.6, 1.8]:
        mask_seuil = df["ratio_logement_revenu_net"] > seuil_test
        n_captures_total = int(mask_seuil.sum())
        n_non_pauvres_captures = int((mask_seuil & (df["statut_pauvrete_revenu_net"] == 0)).sum())
        n_pauvres_captures = int((mask_seuil & (df["statut_pauvrete_revenu_net"] == 1)).sum())
        print(f"    Seuil {seuil_test} : {n_captures_total} foyers captures au total "
              f"({n_pauvres_captures} pauvres + {n_non_pauvres_captures} non-pauvres, "
              f"soit {n_non_pauvres_captures/n_captures_total*100:.1f}% de faux positifs)")


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    main()