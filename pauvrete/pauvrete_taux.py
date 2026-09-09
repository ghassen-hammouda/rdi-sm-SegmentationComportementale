"""
Comptage exact (US-21) du nombre de FOYERS pauvres et de PERSONNES
vivant dans des foyers pauvres, pour period1 et period2.

Distinction importante :
- Nombre de foyers pauvres = compte direct des lignes ou
  statut_pauvrete_revenu_net == 1
- Nombre de personnes pauvres = somme de FAM_SIZE pour ces memes
  foyers (un foyer pauvre de 4 personnes compte pour 4, pas pour 1)
"""

import pandas as pd
from pathlib import Path

INPUT_DIR = Path("outputs_v2")

RAW_FILES = {
    "period1": "poverty_targets_period1.csv",
    "period2": "poverty_targets_period2.csv",
}


def compter(period_label: str) -> None:
    path = INPUT_DIR / RAW_FILES[period_label]
    df = pd.read_csv(path, low_memory=False)

    n_foyers_total = len(df)
    n_personnes_total = int(df["FAM_SIZE"].sum())

    mask_pauvre_revenu = df["statut_pauvrete_revenu_net"] == 1
    mask_pauvre_conso = df["statut_pauvrete_consommation"] == 1

    n_foyers_pauvres_revenu = int(mask_pauvre_revenu.sum())
    n_personnes_pauvres_revenu = int(df.loc[mask_pauvre_revenu, "FAM_SIZE"].sum())

    n_foyers_pauvres_conso = int(mask_pauvre_conso.sum())
    n_personnes_pauvres_conso = int(df.loc[mask_pauvre_conso, "FAM_SIZE"].sum())

    print(f"\n{'=' * 70}\n{period_label.upper()}\n{'=' * 70}")
    print(f"Population totale : {n_foyers_total} foyers, {n_personnes_total} personnes")
    print(f"\n--- Pauvrete par REVENU NET ---")
    print(f"Foyers pauvres    : {n_foyers_pauvres_revenu} ({n_foyers_pauvres_revenu/n_foyers_total*100:.2f}%)")
    print(f"Personnes pauvres : {n_personnes_pauvres_revenu} ({n_personnes_pauvres_revenu/n_personnes_total*100:.2f}%)")
    print(f"\n--- Pauvrete par CONSOMMATION ---")
    print(f"Foyers pauvres    : {n_foyers_pauvres_conso} ({n_foyers_pauvres_conso/n_foyers_total*100:.2f}%)")
    print(f"Personnes pauvres : {n_personnes_pauvres_conso} ({n_personnes_pauvres_conso/n_personnes_total*100:.2f}%)")


if __name__ == "__main__":
    for period_label in RAW_FILES:
        compter(period_label)

    print(f"\n{'=' * 70}")
    print("Rappel important : ces chiffres viennent de l'echantillon CEX")
    print("SANS ponderation d'enquete (FINLWT21 perdue en amont du pipeline) -")
    print("ils representent la proportion dans l'echantillon, pas necessairement")
    print("la proportion exacte dans la population americaine reelle.")