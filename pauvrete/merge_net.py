"""
Fusion du resultat de la regression FINATXEM (2024) avec le fichier
nettoye lisible (colonnes non encodees), pour obtenir un seul fichier
complet et pret pour l'agregation par foyer (etape suivante).

Periode 1 : FINATXEM deja complete par mediane dans clean_data_v2.py -
             ajout simple de FINATXEM_ESTIME_FLAG=False pour coherence
             de structure avec la Periode 2.
Periode 2 : FINATXEM (incomplete dans clean_v2.csv) remplacee par la
             version complete issue de la regression, fusion sur NEWID.
"""

import pandas as pd
from pathlib import Path

INPUT_DIR = Path("outputs_v2")
OUTPUT_DIR = Path("outputs_v2")


def process_period1() -> None:
    print(f"\n{'=' * 70}\nPERIODE 1 : ajout de FINATXEM_ESTIME_FLAG (coherence de structure)\n{'=' * 70}")

    filepath = INPUT_DIR / "interview_period1_clean_v2.csv"
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Charge : {filepath} ({len(df)} lignes, {df.shape[1]} colonnes)")

    n_missing = int(df["FINATXEM"].isna().sum())
    print(f"FINATXEM manquant : {n_missing} (attendu : 0, deja impute par mediane en amont)")

    df["FINATXEM_ESTIME_FLAG"] = False

    output_path = OUTPUT_DIR / "interview_period1_ready.csv"
    df.to_csv(output_path, index=False)
    print(f"Sauvegarde : {output_path} ({len(df)} lignes, {df.shape[1]} colonnes)")


def process_period2() -> None:
    print(f"\n{'=' * 70}\nPERIODE 2 : fusion de FINATXEM (regression) avec le fichier lisible\n{'=' * 70}")

    clean_path = INPUT_DIR / "interview_period2_clean_v2.csv"
    reg_path = INPUT_DIR / "interview_period2_finatxem_regression.csv"

    df_clean = pd.read_csv(clean_path, low_memory=False)
    print(f"Charge (lisible)  : {clean_path} ({len(df_clean)} lignes, {df_clean.shape[1]} colonnes)")
    print(f"FINATXEM manquant avant fusion : {df_clean['FINATXEM'].isna().sum()}")

    df_reg = pd.read_csv(reg_path, low_memory=False)
    print(f"Charge (regression) : {reg_path} ({len(df_reg)} lignes, {df_reg.shape[1]} colonnes)")

    if "NEWID" not in df_clean.columns or "NEWID" not in df_reg.columns:
        raise KeyError("NEWID absente d'un des deux fichiers, fusion impossible.")

    finatxem_complet = df_reg[["NEWID", "FINATXEM", "FINATXEM_ESTIME_FLAG"]].copy()

    n_avant = len(df_clean)
    df_merged = df_clean.drop(columns=["FINATXEM"]).merge(finatxem_complet, on="NEWID", how="left")
    n_apres = len(df_merged)

    if n_avant != n_apres:
        print(f"[ATTENTION] Le nombre de lignes a change pendant la fusion : {n_avant} -> {n_apres} "
              f"(verifier les doublons de NEWID cote regression).")
    else:
        print(f"Fusion verifiee : {n_apres} lignes avant et apres, aucune ligne perdue ou dupliquee.")

    n_missing_apres = int(df_merged["FINATXEM"].isna().sum())
    print(f"FINATXEM manquant apres fusion : {n_missing_apres} (attendu : 0)")
    if n_missing_apres > 0:
        print(f"[ATTENTION] {n_missing_apres} foyer(s) toujours sans FINATXEM apres fusion - "
              f"probablement des NEWID presents dans clean_v2.csv mais absents du fichier de regression.")

    print(f"Foyers avec FINATXEM estimee (regression) : {int(df_merged['FINATXEM_ESTIME_FLAG'].sum())}")

    output_path = OUTPUT_DIR / "interview_period2_ready.csv"
    df_merged.to_csv(output_path, index=False)
    print(f"Sauvegarde : {output_path} ({len(df_merged)} lignes, {df_merged.shape[1]} colonnes)")


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    process_period1()
    process_period2()

    print(f"\n{'=' * 70}")
    print("Termine. Fichiers dans outputs_v2/ :")
    print("  interview_period1_ready.csv")
    print("  interview_period2_ready.csv")
    print("  -> FINATXEM complete sur les deux periodes, colonnes lisibles (non encodees),")
    print("     prets pour l'agregation par foyer (etape suivante).")