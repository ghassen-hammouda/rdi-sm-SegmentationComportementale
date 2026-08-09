"""
Nettoyage complet du nouveau perimetre (repris proprement, dans le bon
ordre cette fois : NETTOYAGE avant encodage/scaling, pas apres).

Etapes, dans l'ordre :
1. Dedoublonnage (NEWID) - meme logique que le projet depuis le debut :
   garder la version la plus recente (fichier "x") en cas de doublon
2. Suppression des colonnes trop incompletes (epargne : IRAYRX, IRAX,
   LIQUDYRX, LIQUIDX - 86 a 93% de manquant, deja diagnostique en EDA)
3. Imputation par le mode de REGION et STATE (categorielles, taux de
   manquant raisonnable : 3,65% et 10,44%)
4. Ecretage des valeurs aberrantes (IQR x3 + z-score > 3) sur les 16
   postes de depense _FULL, avec tracabilite (_CAPPED_FLAG)

FINATXEM n'est PAS touchee ici : ses valeurs manquantes (uniquement sur
les foyers 2024 concernes) seront completees par regression, dans un
script separe, apres l'encodage et le scaling.
"""

import pandas as pd
import numpy as np
from pathlib import Path

INPUT_DIR = Path("outputs_v2")
OUTPUT_DIR = Path("outputs_v2")

FILES = {
    "period1": "interview_period1_full.csv",
    "period2": "interview_period2_full.csv",
}

ID_COLUMNS = ["NEWID", "CUID"]

# Colonnes trop incompletes (deja diagnostiquees en EDA : 86 a 93% de manquant)
COLUMNS_TO_DROP_HIGH_MISSING = ["IRAYRX", "IRAX", "LIQUDYRX", "LIQUIDX"]

# Colonnes categorielles a imputer par le mode
COLUMNS_TO_IMPUTE_MODE = ["REGION", "STATE"]

EXPENDITURE_FULL_COLUMNS = [
    "TOTEXP_FULL", "FOOD_FULL", "HOUS_FULL", "TRANS_FULL", "HEALTH_FULL",
    "APPAR_FULL", "ENTERT_FULL", "EDUCA_FULL", "ALCBEV_FULL", "TOBACC_FULL",
    "PERSCA_FULL", "READ_FULL", "MISC_FULL", "CASHCO_FULL", "LIFINS_FULL",
    "RETPEN_FULL",
]


# ---------------------------------------------------------------------------
# 1. DEDOUBLONNAGE
# ---------------------------------------------------------------------------

def deduplicate_republished_quarters(df: pd.DataFrame) -> pd.DataFrame:
    """Retire les doublons de NEWID en gardant la version la plus recente
    (fichier contenant 'x' dans son nom, publie apres la version standard).
    """
    before = len(df)
    if "SOURCE_FILE" not in df.columns or "NEWID" not in df.columns:
        print("[ATTENTION] SOURCE_FILE ou NEWID absente, dedoublonnage ignore.")
        return df

    is_x_version = df["SOURCE_FILE"].str.contains("x", case=False, na=False)
    df_sorted = df.assign(_is_x=is_x_version).sort_values("_is_x")
    df_dedup = df_sorted.drop_duplicates(subset="NEWID", keep="last").drop(columns="_is_x")

    after = len(df_dedup)
    print(f"Dedoublonnage : {before} lignes -> {after} lignes ({before - after} doublons retires)")
    return df_dedup.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. SUPPRESSION DES COLONNES TROP INCOMPLETES
# ---------------------------------------------------------------------------

def drop_high_missing_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    existing = [c for c in columns if c in df.columns]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        print(f"[ATTENTION] Colonnes deja absentes : {missing}")
    if existing:
        df = df.drop(columns=existing)
        print(f"Colonnes retirees (trop incompletes) : {existing}")
    return df


# ---------------------------------------------------------------------------
# 3. IMPUTATION PAR LE MODE
# ---------------------------------------------------------------------------

def impute_mode(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            print(f"[ATTENTION] {col} absente, imputation ignoree.")
            continue
        n = int(df[col].isna().sum())
        if n == 0:
            print(f"{col} : aucune valeur manquante, rien a imputer.")
            continue
        mode_series = df[col].mode(dropna=True)
        if mode_series.empty:
            print(f"[ATTENTION] {col} : aucun mode calculable, ignoree.")
            continue
        mode_val = mode_series.iloc[0]
        df[col] = df[col].fillna(mode_val)
        print(f"{col} : {n} valeur(s) imputee(s) par le mode ({mode_val})")
    return df


# ---------------------------------------------------------------------------
# 4. ECRETAGE DES VALEURS ABERRANTES (IQR x3 + z-score > 3)
# ---------------------------------------------------------------------------

def cap_outliers(df: pd.DataFrame, columns: list[str],
                  iqr_multiplier: float = 3.0, z_threshold: float = 3.0) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            print(f"[ATTENTION] {col} absente, ecretage ignore.")
            continue

        s = df[col]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1

        # Cas particulier : postes de depense ou la majorite des foyers
        # ont une valeur nulle sur le trimestre (education, tabac,
        # lecture...) -> Q1 = Q3 = 0, IQR = 0. La formule habituelle
        # donnerait alors des bornes [0, 0], ce qui ecraserait a zero
        # TOUTES les valeurs non nulles, meme parfaitement normales -
        # une perte de vraie donnee, pas une correction. On saute
        # l'ecretage IQR dans ce cas et on ne garde que le z-score
        # (qui reste valide meme quand l'IQR est nul).
        if iqr == 0:
            mean, std = s.mean(), s.std()
            is_outlier = (s - mean).abs() > z_threshold * std if std > 0 else pd.Series(False, index=s.index)
            n_outliers = int(is_outlier.sum())
            df[f"{col}_CAPPED_FLAG"] = is_outlier
            if std > 0:
                low_z, high_z = mean - z_threshold * std, mean + z_threshold * std
                df[col] = s.clip(lower=low_z, upper=high_z)
                print(f"[ATTENTION] {col} : IQR nul (Q1=Q3=0, poste tres majoritairement a zero) "
                      f"-> ecretage base sur le z-score uniquement, pas l'IQR. "
                      f"{n_outliers} valeur(s) ecretee(s) [{low_z:,.2f} ; {high_z:,.2f}]")
            else:
                print(f"[ATTENTION] {col} : IQR et ecart-type nuls (colonne constante) -> aucun ecretage applique.")
            continue

        low_iqr, high_iqr = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr

        mean, std = s.mean(), s.std()
        is_outlier_z = (s - mean).abs() > z_threshold * std if std > 0 else pd.Series(False, index=s.index)
        is_outlier_iqr = (s < low_iqr) | (s > high_iqr)
        is_outlier = is_outlier_iqr | is_outlier_z
        n_outliers = int(is_outlier.sum())

        df[f"{col}_CAPPED_FLAG"] = is_outlier
        df[col] = s.clip(lower=low_iqr, upper=high_iqr)

        if n_outliers > 0:
            print(f"{col} : {n_outliers} valeur(s) ecretee(s) [{low_iqr:,.2f} ; {high_iqr:,.2f}]")

    return df


# ---------------------------------------------------------------------------
# PIPELINE PAR PERIODE
# ---------------------------------------------------------------------------

def process_period(period_label: str, filename: str) -> None:
    print(f"\n{'=' * 70}\nNETTOYAGE - {period_label.upper()}\n{'=' * 70}")

    filepath = INPUT_DIR / filename
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Charge : {filepath} ({len(df)} lignes, {df.shape[1]} colonnes)")

    print(f"\n--- Etape 1 : dedoublonnage ---")
    df = deduplicate_republished_quarters(df)

    print(f"\n--- Etape 2 : suppression des colonnes trop incompletes ---")
    df = drop_high_missing_columns(df, COLUMNS_TO_DROP_HIGH_MISSING)

    print(f"\n--- Etape 3 : imputation par le mode (REGION, STATE) ---")
    df = impute_mode(df, COLUMNS_TO_IMPUTE_MODE)

    print(f"\n--- Etape 4 : ecretage des valeurs aberrantes (postes de depense) ---")
    df = cap_outliers(df, EXPENDITURE_FULL_COLUMNS)

    if "FINATXEM" in df.columns:
        n_missing_finatxem = int(df["FINATXEM"].isna().sum())
        if period_label == "period1":
            # Periode 1 : non-reponse individuelle ordinaire, dispersee,
            # faible volume (< 1%) -> la politique du projet depuis l'US-05
            # s'applique normalement : imputation par la mediane.
            if n_missing_finatxem > 0:
                mediane = df["FINATXEM"].median()
                df["FINATXEM"] = df["FINATXEM"].fillna(mediane)
                print(f"\n[INFO] FINATXEM (Periode 1) : {n_missing_finatxem} valeur(s) manquante(s) "
                      f"(non-reponse ordinaire, faible volume) -> imputee(s) par la mediane ({mediane:,.2f}).")
            else:
                print(f"\n[INFO] FINATXEM (Periode 1) : aucune valeur manquante.")
        else:
            # Periode 2 : manquant concentre sur 5 fichiers 2024 (transition
            # methodologique du BLS, ~28% de la periode) -> volontairement
            # NON traite ici, necessite la regression dediee (etape suivante).
            print(f"\n[INFO] FINATXEM (Periode 2) : {n_missing_finatxem} valeur(s) manquante(s), "
                  f"concentrees sur les fichiers 2024 -> volontairement NON traitees ici "
                  f"(regression prevue apres encodage/scaling).")

    output_path = OUTPUT_DIR / f"interview_{period_label}_clean_v2.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSauvegarde : {output_path} ({len(df)} lignes, {df.shape[1]} colonnes)")


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    for period_label, filename in FILES.items():
        process_period(period_label, filename)

    print(f"\n{'=' * 70}")
    print("Termine. Fichiers dans outputs_v2/ :")
    print("  interview_period1_clean_v2.csv")
    print("  interview_period2_clean_v2.csv (2024 incluse, FINATXEM encore manquante sur certains foyers)")
    print("\nProchaine etape : encodage (encode_categorical_v2.py), sur CES fichiers nettoyes.")