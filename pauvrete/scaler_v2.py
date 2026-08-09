"""
Mise a l'echelle (RobustScaler) des variables predictives numeriques
continues, AVANT la regression qui estimera FINATXEM pour les foyers
2024 concernes.

Point important : seules les variables PREDICTIVES sont mises a
l'echelle (FINCBTXM, PERSLT18, PERSOT64, AGE_REF) - jamais la cible
(FINATXEM), puisqu'on veut au final de vraies valeurs en dollars pour
completer le revenu net, pas des valeurs normalisees.

FINCBTXM_SCALED_SQ (le terme au carre de la formule de regression) est
construit A PARTIR de la version deja mise a l'echelle, pas l'inverse -
evite qu'un revenu eleve (ex: 100 000$) produise un carre demesurement
grand (10 milliards) qui destabiliserait le calcul de la regression.
"""

import pandas as pd
from pathlib import Path
from sklearn.preprocessing import RobustScaler

INPUT_DIR = Path("outputs_v2")
OUTPUT_DIR = Path("outputs_v2")

FILES = {
    "period1": "interview_period1_encoded.csv",
    "period2": "interview_period2_encoded.csv",
}

# Variables predictives numeriques continues, a mettre a l'echelle.
# FINATXEM (la cible) est volontairement exclue.
NUMERIC_PREDICTORS = ["FINCBTXM", "PERSLT18", "PERSOT64", "AGE_REF"]


def scale_predictors(df: pd.DataFrame, period_label: str) -> pd.DataFrame:
    df = df.copy()

    existing = [c for c in NUMERIC_PREDICTORS if c in df.columns]
    missing = [c for c in NUMERIC_PREDICTORS if c not in df.columns]
    if missing:
        print(f"[ATTENTION] Variables predictives absentes : {missing}")

    scaler = RobustScaler()
    scaled_cols = [f"{c}_SCALED" for c in existing]
    df[scaled_cols] = scaler.fit_transform(df[existing])
    print(f"[{period_label}] RobustScaler applique sur : {existing}")

    if "FINCBTXM_SCALED" in df.columns:
        df["FINCBTXM_SCALED_SQ"] = df["FINCBTXM_SCALED"] ** 2
        print(f"[{period_label}] FINCBTXM_SCALED_SQ construite a partir de la version deja mise a l'echelle.")
    else:
        print("[ATTENTION] FINCBTXM_SCALED absente, impossible de construire le terme au carre.")

    print(f"[{period_label}] FINATXEM (cible) laissee inchangee, en dollars reels - jamais mise a l'echelle.")

    return df


if __name__ == "__main__":
    for period_label, filename in FILES.items():
        print(f"\n{'=' * 70}\nMISE A L'ECHELLE - {period_label.upper()}\n{'=' * 70}")

        filepath = INPUT_DIR / filename
        df = pd.read_csv(filepath, low_memory=False)
        print(f"Charge : {filepath} ({len(df)} lignes, {df.shape[1]} colonnes)")

        df_scaled = scale_predictors(df, period_label)

        output_path = OUTPUT_DIR / f"interview_{period_label}_scaled.csv"
        df_scaled.to_csv(output_path, index=False)
        print(f"Sauvegarde : {output_path} ({len(df_scaled)} lignes, {df_scaled.shape[1]} colonnes)")

    print(f"\n{'=' * 70}")
    print("Termine. Fichiers dans outputs_v2/ :")
    print("  interview_period1_scaled.csv")
    print("  interview_period2_scaled.csv (2024 incluse, pret pour la regression FINATXEM)")