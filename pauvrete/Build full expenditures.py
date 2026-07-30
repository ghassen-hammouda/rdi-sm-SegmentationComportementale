"""
Construction des depenses trimestrielles completes (_FULL = CQ + PQ)

Reprend la meme logique qu'au debut du projet : CQ (portion du trimestre
en cours) + PQ (portion du trimestre precedent) = un trimestre complet,
puisque le CEX interroge sur une fenetre glissante de 3 mois, pas sur le
trimestre calendaire strict.

FOODCQ/FOODPQ verifiees presentes sur toute la periode retenue
(2017-2020 et 2021-2023) : traitee comme un poste standard, pas de cas
particulier necessaire (le detour envisage un temps par FDHOME/FDAWAY
ne concernait que les fichiers 2024 desormais exclus).
"""

import pandas as pd
from pathlib import Path

INPUT_DIR = Path("outputs_v2")
OUTPUT_DIR = Path("outputs_v2")

FILES = {
    "period1": "interview_period1_raw_selected.csv",
    "period2": "interview_period2_raw_selected.csv",
}

EXPENDITURE_BASES = [
    "TOTEXP", "FOOD", "HOUS", "TRANS", "HEALTH", "APPAR", "ENTERT",
    "EDUCA", "ALCBEV", "TOBACC", "PERSCA", "READ", "MISC", "CASHCO",
    "LIFINS", "RETPEN",
]


def build_full_quarter_expenditure(df: pd.DataFrame, base_name: str) -> pd.DataFrame:
    """CQ + PQ = un trimestre complet, pour un poste de depense."""
    cq_col, pq_col = f"{base_name}CQ", f"{base_name}PQ"
    missing = [c for c in (cq_col, pq_col) if c not in df.columns]
    if missing:
        print(f"[ATTENTION] Colonne(s) manquante(s) pour {base_name} : {missing}, ignore.")
        return df

    full_col = f"{base_name}_FULL"
    df[full_col] = df[cq_col].fillna(0) + df[pq_col].fillna(0)
    print(f"Variable construite : {full_col} = {cq_col} + {pq_col}")
    return df


def process_period(period_label: str, filename: str) -> None:
    print(f"\n{'=' * 70}\nCONSTRUCTION DES DEPENSES COMPLETES - {period_label.upper()}\n{'=' * 70}")

    filepath = INPUT_DIR / filename
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Charge : {filepath} ({len(df)} lignes, {df.shape[1]} colonnes)")

    for base in EXPENDITURE_BASES:
        df = build_full_quarter_expenditure(df, base)

    full_cols = [c for c in df.columns if c.endswith("_FULL")]
    print(f"\n{len(full_cols)} variables _FULL construites : {full_cols}")

    output_path = OUTPUT_DIR / f"interview_{period_label}_full.csv"
    df.to_csv(output_path, index=False)
    print(f"Sauvegarde : {output_path} ({len(df)} lignes, {df.shape[1]} colonnes)")


if __name__ == "__main__":
    for period_label, filename in FILES.items():
        process_period(period_label, filename)

    print(f"\n{'=' * 70}")
    print("Termine. Fichiers dans outputs_v2/ :")
    print("  interview_period1_full.csv")
    print("  interview_period2_full.csv")
    print("  -> 16 variables _FULL ajoutees")