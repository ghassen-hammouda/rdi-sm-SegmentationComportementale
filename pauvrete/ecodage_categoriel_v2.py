"""
Encodage des variables categorielles (US-07), sur le nouveau perimetre
revalide - AVANT la regression qui estimera FINATXEM pour les foyers
2024 concernes.

Variables encodees :
- One-hot : REGION (4), FAM_TYPE (9), CUTENURE (6), STATE (jusqu'a 51),
  SOURCE_YEAR (les regles fiscales changent par annee de facon discrete,
  pas continue - traitee comme categorielle plutot que numerique)
- Ordinale (inchangee) : EDUC_REF (ordre croissant deja verifie)
- Binaire recodee 0/1 : BLS_URBN

Rien d'autre n'est modifie ici : ni le nettoyage, ni la regression
FINATXEM (etape suivante), qui utilisera ce fichier encode en entree.
"""

import pandas as pd
from pathlib import Path

INPUT_DIR = Path("outputs_v2")
OUTPUT_DIR = Path("outputs_v2")

FILES = {
    "period1": "interview_period1_clean_v2.csv",
    "period2": "interview_period2_clean_v2.csv",
}

ONEHOT_COLUMNS = ["REGION", "FAM_TYPE", "CUTENURE", "STATE", "SOURCE_YEAR"]
ORDINAL_COLUMNS = ["EDUC_REF"]
BINARY_COLUMN = "BLS_URBN"


def encode_categorical(df: pd.DataFrame, period_label: str) -> pd.DataFrame:
    df = df.copy()

    existing_onehot = [c for c in ONEHOT_COLUMNS if c in df.columns]
    missing_onehot = [c for c in ONEHOT_COLUMNS if c not in df.columns]
    if missing_onehot:
        print(f"[ATTENTION] Colonnes one-hot attendues absentes : {missing_onehot}")

    n_avant = df.shape[1]
    # drop_first=True : retire une categorie de reference par groupe,
    # pour eviter le "piege des variables muettes" (dummy variable trap) -
    # sans ca, la somme des colonnes d'un meme groupe vaut toujours 1,
    # une redondance parfaite avec l'intercept du modele qui rend une
    # regression lineaire numeriquement instable (coefficients demesures
    # qui s'annulent par coincidence sur les donnees d'entrainement, puis
    # explosent sur de nouvelles donnees).
    df = pd.get_dummies(df, columns=existing_onehot, prefix=existing_onehot, dtype=int, drop_first=True)
    n_apres = df.shape[1]
    print(f"[{period_label}] One-hot applique sur {existing_onehot} : "
          f"{n_avant} -> {n_apres} colonnes ({n_apres - n_avant} colonnes creees)")

    if BINARY_COLUMN in df.columns:
        df[BINARY_COLUMN] = df[BINARY_COLUMN].map({1: 1, 2: 0})
        n_missing = df[BINARY_COLUMN].isna().sum()
        if n_missing > 0:
            print(f"[ATTENTION] {n_missing} valeur(s) de {BINARY_COLUMN} hors {{1,2}}, a examiner.")
    else:
        print(f"[ATTENTION] {BINARY_COLUMN} absente.")

    for col in ORDINAL_COLUMNS:
        if col not in df.columns:
            print(f"[ATTENTION] {col} (ordinale) absente.")
        else:
            print(f"[{period_label}] {col} laissee inchangee (ordinale, ordre deja verifie).")

    return df


if __name__ == "__main__":
    for period_label, filename in FILES.items():
        print(f"\n{'=' * 70}\nENCODAGE - {period_label.upper()}\n{'=' * 70}")

        filepath = INPUT_DIR / filename
        df = pd.read_csv(filepath, low_memory=False)
        print(f"Charge : {filepath} ({len(df)} lignes, {df.shape[1]} colonnes)")

        if "SOURCE_YEAR" in df.columns:
            print(f"Repartition SOURCE_YEAR avant encodage :")
            print(df["SOURCE_YEAR"].value_counts().sort_index())

        df_encoded = encode_categorical(df, period_label)

        output_path = OUTPUT_DIR / f"interview_{period_label}_encoded.csv"
        df_encoded.to_csv(output_path, index=False)
        print(f"Sauvegarde : {output_path} ({len(df_encoded)} lignes, {df_encoded.shape[1]} colonnes)")

    print(f"\n{'=' * 70}")
    print("Termine. Fichiers dans outputs_v2/ :")
    print("  interview_period1_encoded.csv")
    print("  interview_period2_encoded.csv (2024 incluse)")