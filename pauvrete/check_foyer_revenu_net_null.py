"""
Verification rapide : combien de foyers ont ratio_logement_revenu_net
manquant (NaN), et pourquoi - avant de decider quoi en faire pour le
clustering (exclusion vs imputation).
"""

import pandas as pd
from pathlib import Path

INPUT_DIR = Path("outputs_v2")

FILES = {
    "period1": "poverty_targets_period1.csv",
    "period2": "poverty_targets_period2.csv",
}

for period_label, filename in FILES.items():
    print(f"\n{'=' * 70}\n{period_label.upper()}\n{'=' * 70}")

    filepath = INPUT_DIR / filename
    df = pd.read_csv(filepath, low_memory=False)
    total = len(df)

    n_missing = int(df["ratio_logement_revenu_net"].isna().sum())
    print(f"Foyers totaux                                   : {total}")
    print(f"Foyers avec ratio_logement_revenu_net manquant  : {n_missing} ({n_missing/total*100:.2f}%)")

    if n_missing > 0:
        revenu_associe = df.loc[df["ratio_logement_revenu_net"].isna(), "FINATXEM"]
        print(f"\nStatistiques du revenu net (FINATXEM) pour ces foyers precisement :")
        print(revenu_associe.describe())

        n_revenu_negatif_ou_nul = int((revenu_associe <= 0).sum())
        print(f"\nParmi eux, combien ont bien FINATXEM <= 0 : {n_revenu_negatif_ou_nul} / {n_missing}")
        if n_revenu_negatif_ou_nul != n_missing:
            print("[ATTENTION] Certains manquants ne s'expliquent PAS par un revenu <= 0 - a investiguer.")

        print(f"\nCes foyers manquants representent-ils un profil demographique particulier ?")
        print(df.loc[df["ratio_logement_revenu_net"].isna(), ["FAM_SIZE", "AGE_REF", "CUTENURE"]].describe(include="all"))