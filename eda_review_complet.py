"""
Revue consolidee de l'EDA : Interview (FMLI) + Diary (FMLD), Periode 1 + Periode 2

Ce script rassemble en un seul endroit, dans l'ordre suivant, les elements
d'audit necessaires avant de passer au nettoyage :

1. Nombre de colonnes (par enquete, par periode)
2. Colonnes communes entre les deux periodes / colonnes uniquement en P1 / uniquement en P2
3. Type de chaque colonne (resume + export detaille)
4. Doublons (NEWID)
5. Valeurs manquantes (taux par colonne)
6. Valeurs aberrantes : scan IQR sur les variables numeriques cles + histogrammes
7. Correlation entre les variables numeriques cles (Interview uniquement, tableau + heatmap)

Chaque etape est executee pour Interview ET Diary, sur les DEUX periodes,
avec les memes fonctions pour garantir un traitement strictement symetrique.

Perimetre : AUDIT UNIQUEMENT. Rien n'est modifie, corrige ou supprime ici.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from load_and_audit_fmli import (
    load_survey_for_years,
    missing_rate_report,
    detect_duplicate_newid,
    list_all_columns,
    plot_distribution,
    PERIOD_1_YEARS,
    PERIOD_2_YEARS,
    ID_COLUMNS,
)

OUTPUT_DIR = Path("outputs_eda_review")
OUTPUT_DIR.mkdir(exist_ok=True)

# Variables numeriques cles a examiner en detail (histogrammes + scan
# aberrantes). Variables brutes CQ/PQ ici (pas encore reconstruites en
# _FULL), puisqu'on est dans la phase d'audit avant tout traitement.
CORE_NUMERIC_INTERVIEW = ["TOTEXPCQ", "TOTEXPPQ", "FOODCQ", "FOODPQ", "HOUSCQ", "HOUSPQ", "FINCBTXM"]
CORE_NUMERIC_DIARY = ["FOODTOT"]


# ---------------------------------------------------------------------------
# 1. NOMBRE DE COLONNES
# ---------------------------------------------------------------------------

def report_column_count(df_p1: pd.DataFrame, df_p2: pd.DataFrame, survey_label: str) -> None:
    print(f"\n=== [1] Nombre de colonnes - {survey_label} ===")
    print(f"Periode 1 : {df_p1.shape[1]} colonnes, {len(df_p1)} lignes")
    print(f"Periode 2 : {df_p2.shape[1]} colonnes, {len(df_p2)} lignes")


# ---------------------------------------------------------------------------
# 2. COLONNES COMMUNES / UNIQUEMENT P1 / UNIQUEMENT P2
# ---------------------------------------------------------------------------

def report_column_overlap(df_p1: pd.DataFrame, df_p2: pd.DataFrame, survey_label: str):
    """Compare les colonnes des deux DataFrames deja charges en memoire :
    communes aux deux periodes, presentes seulement en P1, presentes
    seulement en P2. Exporte les listes completes en CSV (trop longues
    pour tenir dans la console).
    """
    cols_p1 = set(df_p1.columns)
    cols_p2 = set(df_p2.columns)
    common = cols_p1 & cols_p2
    only_p1 = sorted(cols_p1 - cols_p2)
    only_p2 = sorted(cols_p2 - cols_p1)

    print(f"\n=== [2] Colonnes communes / uniques - {survey_label} ===")
    print(f"Colonnes communes aux deux periodes : {len(common)}")
    print(f"Colonnes presentes seulement en P1 ({len(only_p1)}) : {only_p1}")
    print(f"Colonnes presentes seulement en P2 ({len(only_p2)}) : {only_p2}")

    pd.DataFrame({"variable": only_p1}).to_csv(OUTPUT_DIR / f"columns_only_p1_{survey_label}.csv", index=False)
    pd.DataFrame({"variable": only_p2}).to_csv(OUTPUT_DIR / f"columns_only_p2_{survey_label}.csv", index=False)

    return common, only_p1, only_p2


# ---------------------------------------------------------------------------
# 3. TYPE DE CHAQUE COLONNE
# ---------------------------------------------------------------------------

def report_column_types(df: pd.DataFrame, period_label: str, survey_label: str) -> pd.DataFrame:
    """Resume les types de donnees (combien de colonnes float64, int64,
    object...) et exporte le detail complet (nom + type de chaque
    colonne) en CSV via list_all_columns, deja ecrite en US-04.
    """
    print(f"\n=== [3] Types de colonnes - {survey_label} {period_label} ===")
    type_counts = df.dtypes.astype(str).value_counts()
    print(type_counts)

    cols_df = list_all_columns(df, output_path=str(OUTPUT_DIR / f"columns_types_{survey_label}_{period_label}.csv"))
    return cols_df


# ---------------------------------------------------------------------------
# 4. DOUBLONS
# ---------------------------------------------------------------------------

def report_duplicates(df: pd.DataFrame, period_label: str, survey_label: str) -> pd.DataFrame:
    print(f"\n=== [4] Doublons NEWID - {survey_label} {period_label} ===")
    dup = detect_duplicate_newid(df)
    pct = len(dup) / len(df) * 100 if len(df) else 0
    print(f"Nombre de NEWID dupliques : {len(dup)} (sur {len(df)} lignes, soit {pct:.2f}%)")
    return dup


# ---------------------------------------------------------------------------
# 5. VALEURS MANQUANTES
# ---------------------------------------------------------------------------

def report_missing_values(df: pd.DataFrame, period_label: str, survey_label: str) -> pd.DataFrame:
    print(f"\n=== [5] Valeurs manquantes - {survey_label} {period_label} ===")
    report = missing_rate_report(df, exclude_cols=ID_COLUMNS)
    print(report.head(15))
    report.to_csv(OUTPUT_DIR / f"missing_rate_{survey_label}_{period_label}.csv", index=False)
    return report


# ---------------------------------------------------------------------------
# 6. VALEURS ABERRANTES (scan IQR + histogrammes)
# ---------------------------------------------------------------------------

def detect_outliers_iqr(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Detecte les valeurs aberrantes par la methode IQR (1.5x l'ecart
    interquartile) sur une liste de colonnes numeriques. Detection
    uniquement : rien n'est modifie ou supprime.
    """
    rows = []
    for col in numeric_cols:
        if col not in df.columns:
            print(f"[ATTENTION] {col} absente, scan aberrantes ignore.")
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = int(((s < low) | (s > high)).sum())
        rows.append({
            "variable": col,
            "Q1": round(q1, 2), "Q3": round(q3, 2), "IQR": round(iqr, 2),
            "borne_basse": round(low, 2), "borne_haute": round(high, 2),
            "nb_valeurs_aberrantes": n_outliers,
            "pct_valeurs_aberrantes": round(n_outliers / len(s) * 100, 2),
        })
    return pd.DataFrame(rows)


def report_outliers(df: pd.DataFrame, period_label: str, survey_label: str, core_cols: list[str]) -> pd.DataFrame:
    print(f"\n=== [6] Valeurs aberrantes (IQR) - {survey_label} {period_label} ===")
    outliers = detect_outliers_iqr(df, core_cols)
    print(outliers)
    outliers.to_csv(OUTPUT_DIR / f"outliers_iqr_{survey_label}_{period_label}.csv", index=False)

    print(f"Generation des histogrammes - {survey_label} {period_label}")
    for col in core_cols:
        plot_distribution(df, col, output_path=str(OUTPUT_DIR / f"hist_{survey_label}_{col}_{period_label}.png"))

    return outliers


# ---------------------------------------------------------------------------
# 7. CORRELATION ENTRE VARIABLES NUMERIQUES CLES (+ heatmap graphique)
# ---------------------------------------------------------------------------

def correlation_matrix(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    existing = [c for c in numeric_cols if c in df.columns]
    return df[existing].corr()


def plot_correlation_heatmap(corr_df: pd.DataFrame, title: str, output_path: str) -> None:
    """Version graphique de la matrice de correlation : plus rapide a
    lire qu'un tableau de chiffres, chaque case est annotee avec sa
    valeur exacte pour ne pas se fier uniquement a la couleur.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if corr_df.empty:
        print("Matrice de correlation vide, graphique ignore.")
        return

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr_df.values, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_yticks(range(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_df.columns)

    for i in range(len(corr_df.columns)):
        for j in range(len(corr_df.columns)):
            ax.text(j, i, f"{corr_df.values[i, j]:.2f}", ha="center", va="center", color="black")

    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Coefficient de correlation")
    plt.tight_layout()

    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Graphique sauvegarde : {output_path}")


def report_correlation(df: pd.DataFrame, period_label: str, survey_label: str,
                        numeric_cols: list[str]) -> pd.DataFrame:
    print(f"\n=== [7] Correlation entre variables numeriques - {survey_label} {period_label} ===")

    existing = [c for c in numeric_cols if c in df.columns]
    if len(existing) < 2:
        print(f"Moins de 2 variables numeriques disponibles ({existing}) -> correlation non pertinente, ignoree.")
        return pd.DataFrame()

    corr = correlation_matrix(df, numeric_cols)
    print(corr)
    corr.to_csv(OUTPUT_DIR / f"correlation_{survey_label}_{period_label}.csv")

    plot_correlation_heatmap(
        corr,
        title=f"{survey_label.capitalize()} - Correlation ({period_label})",
        output_path=str(OUTPUT_DIR / f"heatmap_correlation_{survey_label}_{period_label}.png"),
    )
    return corr


# ---------------------------------------------------------------------------
# PIPELINE PAR ENQUETE (identique pour Interview et Diary)
# ---------------------------------------------------------------------------

def review_survey(survey: str, core_cols: list[str], survey_label: str):
    print(f"\n{'=' * 70}\nREVUE EDA - {survey_label.upper()}\n{'=' * 70}")

    df_p1 = load_survey_for_years(PERIOD_1_YEARS, survey=survey)
    df_p2 = load_survey_for_years(PERIOD_2_YEARS, survey=survey)

    # 1. Nombre de colonnes
    report_column_count(df_p1, df_p2, survey_label)

    # 2. Colonnes communes / uniquement P1 / uniquement P2
    report_column_overlap(df_p1, df_p2, survey_label)

    # 3. Type de chaque colonne
    report_column_types(df_p1, "period1", survey_label)
    report_column_types(df_p2, "period2", survey_label)

    # 4. Doublons
    report_duplicates(df_p1, "period1", survey_label)
    report_duplicates(df_p2, "period2", survey_label)

    # 5. Valeurs manquantes
    report_missing_values(df_p1, "period1", survey_label)
    report_missing_values(df_p2, "period2", survey_label)

    # 6. Valeurs aberrantes (scan + histogrammes)
    report_outliers(df_p1, "period1", survey_label, core_cols)
    report_outliers(df_p2, "period2", survey_label, core_cols)

    # 7. Correlation entre variables numeriques cles (+ heatmap), et ecart P1/P2
    corr_p1 = report_correlation(df_p1, "period1", survey_label, core_cols)
    corr_p2 = report_correlation(df_p2, "period2", survey_label, core_cols)

    if not corr_p1.empty and not corr_p2.empty:
        common_cols = [c for c in corr_p1.columns if c in corr_p2.columns]
        corr_diff = corr_p2.loc[common_cols, common_cols] - corr_p1.loc[common_cols, common_cols]
        print(f"\n=== [7bis] Ecart de correlation Periode 2 - Periode 1 - {survey_label} ===")
        print(corr_diff)
        corr_diff.to_csv(OUTPUT_DIR / f"correlation_diff_{survey_label}.csv")

    return df_p1, df_p2


# ---------------------------------------------------------------------------
# EXECUTION PRINCIPALE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    df_int_p1, df_int_p2 = review_survey("interview", CORE_NUMERIC_INTERVIEW, "interview")
    df_dia_p1, df_dia_p2 = review_survey("diary", CORE_NUMERIC_DIARY, "diary")

    print(f"\n{'=' * 70}")
    print(f"Termine. Tous les rapports (.csv) et graphiques (.png) sont dans : {OUTPUT_DIR}/")
    print(f"{'=' * 70}")