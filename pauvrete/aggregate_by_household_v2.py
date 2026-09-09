"""
Agregation par foyer (une ligne par CUID), avec annualisation des
depenses, sur le nouveau perimetre de colonnes (repris proprement
depuis le debut).

Logique, identique a celle validee dans la premiere version du projet :
- Depenses (_FULL) : somme des trimestres disponibles x 4/nb_trimestres
- Revenu (FINATXEM, FINCBTXM) : moyenne des declarations du foyer
  (deja des mesures annuelles glissantes, pas des flux a additionner)
- Demographie : derniere valeur connue du foyer (interview la plus recente)
- FINATXEM_ESTIME_FLAG : True si AU MOINS UN trimestre du foyer avait
  un FINATXEM estime par regression (pas une vraie declaration)
"""

import pandas as pd
from pathlib import Path

INPUT_DIR = Path("outputs_v2")
OUTPUT_DIR = Path("outputs_v2")

FILES = {
    "period1": "interview_period1_ready.csv",
    "period2": "interview_period2_ready.csv",
}

ID_COLUMN = "CUID"

EXPENDITURE_BASES = [
    "TOTEXP", "FOOD", "HOUS", "TRANS", "HEALTH", "APPAR", "ENTERT",
    "EDUCA", "ALCBEV", "TOBACC", "PERSCA", "READ", "MISC", "CASHCO",
    "LIFINS", "RETPEN",
]

DEMOGRAPHIC_COLUMNS = [
    "FAM_SIZE", "PERSLT18", "PERSOT64", "FAM_TYPE", "AGE_REF",
    "EDUC_REF", "CUTENURE", "VEHQ", "REGION", "BLS_URBN", "STATE",
]

INCOME_COLUMNS = ["FINATXEM", "FINCBTXM"]


def aggregate_by_household(df: pd.DataFrame, period_label: str) -> pd.DataFrame:
    full_cols = [f"{base}_FULL" for base in EXPENDITURE_BASES if f"{base}_FULL" in df.columns]
    missing_full = [f"{base}_FULL" for base in EXPENDITURE_BASES if f"{base}_FULL" not in df.columns]
    if missing_full:
        print(f"[ATTENTION] Colonnes _FULL absentes : {missing_full}")

    nb_trimestres = df.groupby(ID_COLUMN).size().rename("NB_TRIMESTRES")

    sums = df.groupby(ID_COLUMN)[full_cols].sum()
    annualized = sums.multiply(4 / nb_trimestres, axis=0)
    annualized.columns = [f"{c}_ANNUEL" for c in annualized.columns]

    existing_income = [c for c in INCOME_COLUMNS if c in df.columns]
    income = df.groupby(ID_COLUMN)[existing_income].mean()

    existing_demo = [c for c in DEMOGRAPHIC_COLUMNS if c in df.columns]
    missing_demo = [c for c in DEMOGRAPHIC_COLUMNS if c not in df.columns]
    if missing_demo:
        print(f"[ATTENTION] Colonnes demographiques absentes : {missing_demo}")
    sort_col = "SOURCE_YEAR" if "SOURCE_YEAR" in df.columns else ID_COLUMN
    demo = df.sort_values(sort_col).groupby(ID_COLUMN)[existing_demo].last()

    parts = [nb_trimestres, annualized, income, demo]

    if "FINATXEM_ESTIME_FLAG" in df.columns:
        estime_flag = df.groupby(ID_COLUMN)["FINATXEM_ESTIME_FLAG"].any().rename("FINATXEM_ESTIME_FLAG")
        parts.append(estime_flag)

    household_df = pd.concat(parts, axis=1).reset_index()

    print(f"[{period_label}] Agregation par foyer : {len(df)} lignes foyer-trimestre "
          f"-> {len(household_df)} foyers")
    return household_df


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    for period_label, filename in FILES.items():
        print(f"\n{'=' * 70}\nAGREGATION PAR FOYER - {period_label.upper()}\n{'=' * 70}")

        filepath = INPUT_DIR / filename
        df = pd.read_csv(filepath, low_memory=False)
        print(f"Charge : {filepath} ({len(df)} lignes, {df.shape[1]} colonnes)")

        household_df = aggregate_by_household(df, period_label)

        print(f"\nRepartition du nombre de trimestres observes :")
        print(household_df["NB_TRIMESTRES"].value_counts().sort_index())

        if "FINATXEM_ESTIME_FLAG" in household_df.columns:
            print(f"\nFoyers avec au moins un trimestre a FINATXEM estime : "
                  f"{int(household_df['FINATXEM_ESTIME_FLAG'].sum())}")

        output_path = OUTPUT_DIR / f"household_{period_label}.csv"
        household_df.to_csv(output_path, index=False)
        print(f"\nSauvegarde : {output_path} ({len(household_df)} foyers, {household_df.shape[1]} colonnes)")

    print(f"\n{'=' * 70}")
    print("Termine. Fichiers dans outputs_v2/ :")
    print("  household_period1.csv")
    print("  household_period2.csv")
    print("  -> une ligne par foyer, depenses annualisees, pretes pour les features metier (US-06 suite).")