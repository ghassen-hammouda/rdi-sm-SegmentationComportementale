"""
Construction des features metier (US-06), sur le nouveau perimetre :
- Part de chaque poste dans le budget total (15 postes, hors TOTEXP)
- Depense par equivalent-adulte (echelle d'equivalence OCDE)
- Ratio logement/revenu (brut ET net, les deux gardees en parallele)

Echelle d'equivalence OCDE modifiee (deja validee dans le projet) :
Echelle = 1.0 + 0.5 x (adultes supplementaires) + 0.3 x (enfants, PERSLT18)

Nettoyage des features derivees : parts ecretees a [0,1] (borne logique),
ratios ecretes au 99e percentile - meme principe que la premiere version
du projet, pour eviter qu'une division par un revenu/budget proche de
zero ne produise des valeurs demesurees.
"""

import pandas as pd
import numpy as np
from pathlib import Path

INPUT_DIR = Path("outputs_v2")
OUTPUT_DIR = Path("outputs_v2")

FILES = {
    "period1": "household_period1.csv",
    "period2": "household_period2.csv",
}

EXPENDITURE_BASES = [
    "TOTEXP", "FOOD", "HOUS", "TRANS", "HEALTH", "APPAR", "ENTERT",
    "EDUCA", "ALCBEV", "TOBACC", "PERSCA", "READ", "MISC", "CASHCO",
    "LIFINS", "RETPEN",
]

INCOME_COLUMNS = ["FINATXEM", "FINCBTXM"]


def compute_equivalence_scale(df: pd.DataFrame) -> pd.Series:
    nb_adultes = (df["FAM_SIZE"] - df["PERSLT18"]).clip(lower=1)
    nb_adultes_suppl = (nb_adultes - 1).clip(lower=0)
    nb_enfants = df["PERSLT18"]
    return 1.0 + 0.5 * nb_adultes_suppl + 0.3 * nb_enfants


def add_business_features(df: pd.DataFrame, period_label: str) -> pd.DataFrame:
    df = df.copy()

    df["ECHELLE_EQUIVALENCE"] = compute_equivalence_scale(df)

    total_budget = df["TOTEXP_FULL_ANNUEL"].replace(0, np.nan)

    for base in EXPENDITURE_BASES:
        if base == "TOTEXP":
            continue
        col = f"{base}_FULL_ANNUEL"
        if col not in df.columns:
            print(f"[ATTENTION] {col} absente, features ignorees pour {base}.")
            continue
        df[f"part_{base.lower()}"] = df[col] / total_budget
        df[f"{base.lower()}_par_equivalent_adulte"] = df[col] / df["ECHELLE_EQUIVALENCE"]

    df["totexp_par_equivalent_adulte"] = df["TOTEXP_FULL_ANNUEL"] / df["ECHELLE_EQUIVALENCE"]

    for label, income_col in [("net", "FINATXEM"), ("brut", "FINCBTXM")]:
        if income_col not in df.columns:
            print(f"[ATTENTION] {income_col} absente, ratio logement/revenu {label} ignore.")
            continue
        nb_non_positif = int((df[income_col] <= 0).sum())
        if nb_non_positif > 0:
            print(f"[ATTENTION] {nb_non_positif} foyer(s) avec revenu {label} <= 0 "
                  f"-> ratio_logement_revenu_{label} sera NaN pour ces foyers.")
        revenu_positif = df[income_col].where(df[income_col] > 0, np.nan)
        df[f"ratio_logement_revenu_{label}"] = df["HOUS_FULL_ANNUEL"] / revenu_positif

    n_features = len([c for c in df.columns if c.startswith(("part_", "ratio_logement")) or c.endswith("par_equivalent_adulte")])
    print(f"[{period_label}] Features metier construites : {n_features} colonnes")

    return df


def clean_business_features(df: pd.DataFrame, period_label: str) -> pd.DataFrame:
    part_cols = [c for c in df.columns if c.startswith("part_")]
    for col in part_cols:
        n_invalid = int(((df[col] < 0) | (df[col] > 1)).sum())
        if n_invalid > 0:
            print(f"{col} : {n_invalid} valeur(s) hors de [0,1] -> ecretees a la borne logique.")
        df[col] = df[col].clip(lower=0, upper=1)

    for col in ["ratio_logement_revenu_net", "ratio_logement_revenu_brut"]:
        if col not in df.columns or df[col].dropna().empty:
            continue
        upper_bound = df[col].quantile(0.99)
        n_capped_haut = int((df[col] > upper_bound).sum())
        n_capped_bas = int((df[col] < 0).sum())
        if n_capped_haut > 0:
            print(f"{col} : {n_capped_haut} valeur(s) au-dela du 99e percentile ({upper_bound:.2f}) -> ecretees.")
        if n_capped_bas > 0:
            print(f"{col} : {n_capped_bas} valeur(s) negative(s) -> ramenees a 0.")
        df[col] = df[col].clip(lower=0, upper=upper_bound)

    return df


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    for period_label, filename in FILES.items():
        print(f"\n{'=' * 70}\nFEATURES METIER - {period_label.upper()}\n{'=' * 70}")

        filepath = INPUT_DIR / filename
        df = pd.read_csv(filepath, low_memory=False)
        print(f"Charge : {filepath} ({len(df)} foyers, {df.shape[1]} colonnes)")

        df = add_business_features(df, period_label)
        df = clean_business_features(df, period_label)

        output_path = OUTPUT_DIR / f"household_features_{period_label}.csv"
        df.to_csv(output_path, index=False)
        print(f"\nSauvegarde : {output_path} ({len(df)} foyers, {df.shape[1]} colonnes)")

    print(f"\n{'=' * 70}")
    print("Termine. Fichiers dans outputs_v2/ :")
    print("  household_features_period1.csv")
    print("  household_features_period2.csv")
    print("  -> features metier ajoutees, pretes pour le seuil et les cibles de pauvrete (US-08).")