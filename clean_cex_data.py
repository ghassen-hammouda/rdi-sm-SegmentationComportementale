"""
US-05 : Nettoyage des donnees CEX (Interview Survey FMLI + Diary Survey FMLD)

Ce script applique les corrections identifiees et priorisees dans le
rapport d'audit US-04, dans l'ordre de priorite retenu, pour les DEUX
composantes du CEX (Interview et Diary) :

1. Dedoublonnage des trimestres/vagues publies deux fois (paires
   fmli181/181x, fmli191/191x cote Interview), en conservant la version
   la plus recente ("x")
2. Reconstitution des depenses trimestrielles completes (CQ + PQ) cote
   Interview, deja validee en US-04 (fonction reutilisee depuis
   load_and_audit_fmli.py). Sans objet cote Diary (FOODTOT est deja une
   variable agregee, pas de decoupage CQ/PQ dans cette enquete).
3. Suppression des colonnes structurellement vides (100% manquant sur
   TOUTE la periode ET sur CHAQUE annee prise individuellement), apres
   verification explicite qu'aucune annee ne fait exception
4. Signalement (sans suppression) des valeurs negatives sur les variables
   de depense et de revenu, pour examen manuel avant toute decision
   de suppression definitive

Le resultat est sauvegarde dans des fichiers CSV nettoyes, distincts des
donnees brutes, pour ne jamais ecraser la source originale.
"""

import pandas as pd
from pathlib import Path

# Reutilisation du travail deja fait en US-04, plutot que de dupliquer
# le code de chargement des donnees.
from load_and_audit_fmli import (
    load_survey_for_years,
    build_full_quarter_expenditure,
    missing_rate_by_group,
    PERIOD_1_YEARS,
    PERIOD_2_YEARS,
    ID_COLUMNS,
)

OUTPUT_DIR = Path("outputs_clean")

# Variables de depense a reconstituer en CQ+PQ (Interview uniquement, cf.
# docstring ci-dessus : Diary ne suit pas ce decoupage).
EXPENDITURE_BASES = ["TOTEXP", "FOOD", "HOUS"]

# Variables Interview candidates a la suppression (US-04, rapport section
# "taux de valeurs manquantes") : DESFLG/INT_PHON/INT_HOME sont vides a
# 100% sur la Periode 1 (elles disparaissent ensuite du schema en P2 -
# un mecanisme different, deja documente en US-04, pas traite ici).
# OTRINCBX est vide a 100% sur LES DEUX periodes 2017-2024 : c'est le
# seul cas de "colonne structurellement vide sur toute la duree" au sens
# strict, elle avait ete omise par erreur dans la premiere version de ce
# script alors qu'elle correspond exactement au critere de l'US-05.
COLUMNS_TO_DROP_KNOWN_EMPTY_INTERVIEW = ["DESFLG", "INT_PHON", "INT_HOME", "OTRINCBX"]

# Diary : MARI_AL1 est la seule variable identifiee en US-04 comme vide a
# 100% sur les deux periodes (rapport US-04, taux de manquant Diary).
COLUMNS_TO_DROP_KNOWN_EMPTY_DIARY = ["MARI_AL1"]

# Variables sur lesquelles on signale les valeurs negatives sans les
# supprimer automatiquement (a examiner manuellement avant toute action).
COLUMNS_TO_FLAG_NEGATIVE_INTERVIEW = ["TOTEXP_FULL", "HOUS_FULL", "FINCBTXM"]
COLUMNS_TO_FLAG_NEGATIVE_DIARY = ["FOODTOT"]


# ---------------------------------------------------------------------------
# 0. VERIFICATION AVANT SUPPRESSION : la colonne est-elle vide sur CHAQUE
#    annee de la periode, pas seulement en moyenne cumulee ?
# ---------------------------------------------------------------------------

def verify_fully_empty_by_year(df: pd.DataFrame, columns: list[str],
                                tolerance_pct: float = 99.9) -> list[str]:
    """Avant de supprimer une colonne jugee 'structurellement vide' en
    US-04, verifie qu'elle est bien vide sur CHAQUE annee de la periode,
    pas seulement en moyenne cumulee sur l'ensemble des annees. Une
    colonne pourrait etre vide sur 2017-2019 mais renseignee en 2020 par
    exemple : la supprimer sans ce controle ferait perdre de la vraie
    donnee sans qu'on s'en rende compte.

    Retourne la liste des colonnes confirmees vides sur toutes les
    annees (donc sures a supprimer). Celles qui ne le sont pas sont
    signalees en detail et exclues de la liste retournee, pour examen
    manuel avant toute suppression.
    """
    confirmed = []
    for col in columns:
        if col not in df.columns:
            print(f"[ATTENTION] {col} absente du DataFrame, verification ignoree.")
            continue

        by_year = missing_rate_by_group(df, col, "SOURCE_YEAR")
        min_missing = by_year["taux_manquant_pct"].min()

        if min_missing >= tolerance_pct:
            confirmed.append(col)
        else:
            print(f"[ATTENTION] {col} n'est PAS vide sur toutes les annees "
                  f"(minimum observe = {min_missing:.2f}%) -> NON supprimee, "
                  f"a examiner manuellement avant toute decision :")
            print(by_year)

    return confirmed


# ---------------------------------------------------------------------------
# 1. DEDUPLICATION DES VAGUES PUBLIEES DEUX FOIS
# ---------------------------------------------------------------------------

def deduplicate_republished_quarters(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les doublons de NEWID en conservant la version la plus
    recente de chaque vague republiee (fichier dont le nom contient un
    'x', ex: fmli181x.csv, publie apres fmli181.csv).

    Principe : pour chaque NEWID en double, on trie les lignes pour que
    la version "x" (str contenant 'x' dans SOURCE_FILE) passe en dernier,
    puis on garde uniquement la derniere occurrence de chaque NEWID.

    Cote Diary, aucun fichier "x" n'a ete observe en US-04 (0 doublon
    NEWID detecte) : cette fonction agit alors comme un filet de securite
    sans effet, plutot qu'un traitement specifique a ajouter separement.
    """
    before = len(df)

    is_x_version = df["SOURCE_FILE"].str.contains("x", case=False, na=False)
    df_sorted = df.assign(_is_x=is_x_version).sort_values("_is_x")
    df_dedup = df_sorted.drop_duplicates(subset="NEWID", keep="last").drop(columns="_is_x")

    after = len(df_dedup)
    print(f"Dedoublonnage : {before} lignes -> {after} lignes ({before - after} doublons retires)")
    return df_dedup.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. SUPPRESSION DES COLONNES STRUCTURELLEMENT VIDES (apres verification)
# ---------------------------------------------------------------------------

def drop_known_empty_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Retire les colonnes confirmees entierement vides sur toutes les
    annees par verify_fully_empty_by_year. On les retire plutot que de
    les imputer, faute de toute valeur reelle permettant une imputation
    fiable.
    """
    existing = [c for c in columns if c in df.columns]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        print(f"[ATTENTION] Colonnes deja absentes, rien a retirer : {missing}")
    if existing:
        df = df.drop(columns=existing)
        print(f"Colonnes retirees (structurellement vides, confirmees par annee) : {existing}")
    return df


# ---------------------------------------------------------------------------
# 3. SIGNALEMENT DES VALEURS NEGATIVES (SANS SUPPRESSION AUTOMATIQUE)
# ---------------------------------------------------------------------------

def flag_negative_values(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Ajoute, pour chaque variable listee, une colonne booleenne
    '{variable}_NEGATIVE_FLAG' signalant les valeurs negatives, sans les
    modifier ni les supprimer. Objectif : permettre un examen manuel
    (ex: distinguer un remboursement legitime d'une erreur de saisie)
    avant de decider d'un traitement definitif, conformement a la
    recommandation du rapport US-04.
    """
    for col in columns:
        if col not in df.columns:
            print(f"[ATTENTION] {col} absente, signalement ignore.")
            continue
        flag_col = f"{col}_NEGATIVE_FLAG"
        df[flag_col] = df[col] < 0
        nb_negative = df[flag_col].sum()
        pct_negative = df[flag_col].mean() * 100
        print(f"{col} : {nb_negative} valeurs negatives signalees ({pct_negative:.2f}%)")
    return df


# ---------------------------------------------------------------------------
# 4. PIPELINES DE NETTOYAGE COMPLETS
# ---------------------------------------------------------------------------

def clean_interview_data(years: list[str], period_label: str) -> pd.DataFrame:
    print(f"\n========== Nettoyage Interview (FMLI) — {period_label} ==========")

    print("\n--- Chargement des donnees brutes ---")
    df = load_survey_for_years(years, survey="interview")

    print("\n--- Etape 1 : dedoublonnage des vagues republiees ---")
    df = deduplicate_republished_quarters(df)

    print("\n--- Etape 2 : reconstitution des depenses trimestrielles (CQ+PQ) ---")
    for base in EXPENDITURE_BASES:
        df = build_full_quarter_expenditure(df, base)

    print("\n--- Etape 3 : verification des colonnes candidates a la suppression ---")
    confirmed_empty = verify_fully_empty_by_year(df, COLUMNS_TO_DROP_KNOWN_EMPTY_INTERVIEW)
    df = drop_known_empty_columns(df, confirmed_empty)

    print("\n--- Etape 4 : signalement des valeurs negatives ---")
    df = flag_negative_values(df, COLUMNS_TO_FLAG_NEGATIVE_INTERVIEW)

    return df


def clean_diary_data(years: list[str], period_label: str) -> pd.DataFrame:
    print(f"\n========== Nettoyage Diary (FMLD) — {period_label} ==========")

    print("\n--- Chargement des donnees brutes ---")
    df = load_survey_for_years(years, survey="diary")

    print("\n--- Etape 1 : dedoublonnage des vagues republiees (filet de securite) ---")
    df = deduplicate_republished_quarters(df)

    print("\n--- Etape 2 : verification des colonnes candidates a la suppression ---")
    confirmed_empty = verify_fully_empty_by_year(df, COLUMNS_TO_DROP_KNOWN_EMPTY_DIARY)
    df = drop_known_empty_columns(df, confirmed_empty)

    print("\n--- Etape 3 : signalement des valeurs negatives ---")
    df = flag_negative_values(df, COLUMNS_TO_FLAG_NEGATIVE_DIARY)

    return df


def save_clean_data(df: pd.DataFrame, filename: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / filename
    df.to_csv(output_path, index=False)
    print(f"\nDonnees nettoyees sauvegardees : {output_path} ({len(df)} lignes, {df.shape[1]} colonnes)")


# ---------------------------------------------------------------------------
# 5. EXECUTION PRINCIPALE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ============================================================
    # INTERVIEW SURVEY (FMLI)
    # ============================================================
    df_int_p1_clean = clean_interview_data(PERIOD_1_YEARS, "Periode 1 (2017-2020)")
    save_clean_data(df_int_p1_clean, "interview_period1_clean.csv")

    df_int_p2_clean = clean_interview_data(PERIOD_2_YEARS, "Periode 2 (2021-2024)")
    save_clean_data(df_int_p2_clean, "interview_period2_clean.csv")

    # ============================================================
    # DIARY SURVEY (FMLD)
    # ============================================================
    df_dia_p1_clean = clean_diary_data(PERIOD_1_YEARS, "Periode 1 (2017-2020)")
    save_clean_data(df_dia_p1_clean, "diary_period1_clean.csv")

    df_dia_p2_clean = clean_diary_data(PERIOD_2_YEARS, "Periode 2 (2021-2024)")
    save_clean_data(df_dia_p2_clean, "diary_period2_clean.csv")

    print("\n========== Recapitulatif ==========")
    print(f"Interview Periode 1 : {len(df_int_p1_clean)} lignes apres nettoyage")
    print(f"Interview Periode 2 : {len(df_int_p2_clean)} lignes apres nettoyage")
    print(f"Diary Periode 1     : {len(df_dia_p1_clean)} lignes apres nettoyage")
    print(f"Diary Periode 2     : {len(df_dia_p2_clean)} lignes apres nettoyage")