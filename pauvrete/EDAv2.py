"""
EDA complete sur le nouveau perimetre de colonnes (revalide famille par
famille) : manquantes, doublons, valeurs aberrantes (histogrammes +
boxplots), correlation, statistiques descriptives ponderees.

Perimetre : AUDIT UNIQUEMENT, aucune correction ici (coherent avec la
philosophie du projet depuis le debut).
"""

import pandas as pd
import numpy as np
from pathlib import Path

INPUT_DIR = Path("outputs_v2")
OUTPUT_DIR = Path("outputs_eda_v2")
OUTPUT_DIR.mkdir(exist_ok=True)

FILES = {
    "period1": "interview_period1_full.csv",
    "period2": "interview_period2_full.csv",
}

ID_COLUMNS = ["NEWID", "CUID"]
WEIGHT_COLUMN = "FINLWT21"

# Les 16 postes de depense reconstruits + les deux revenus (net et brut),
# coeur de l'analyse statistique.
EXPENDITURE_FULL_COLUMNS = [
    "TOTEXP_FULL", "FOOD_FULL", "HOUS_FULL", "TRANS_FULL", "HEALTH_FULL",
    "APPAR_FULL", "ENTERT_FULL", "EDUCA_FULL", "ALCBEV_FULL", "TOBACC_FULL",
    "PERSCA_FULL", "READ_FULL", "MISC_FULL", "CASHCO_FULL", "LIFINS_FULL",
    "RETPEN_FULL", "FINATXEM", "FINCBTXM",
]


# ---------------------------------------------------------------------------
# 1. VALEURS MANQUANTES
# ---------------------------------------------------------------------------

def missing_rate_report(df: pd.DataFrame, exclude_cols: list[str]) -> pd.DataFrame:
    cols = [c for c in df.columns if c not in exclude_cols]
    report = pd.DataFrame({
        "variable": cols,
        "taux_manquant_pct": [df[c].isna().mean() * 100 for c in cols],
        "type": [str(df[c].dtype) for c in cols],
    })
    return report.sort_values("taux_manquant_pct", ascending=False).reset_index(drop=True)


def plot_missing_rate(report_df: pd.DataFrame, period_label: str, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = report_df[report_df["taux_manquant_pct"] > 0].head(20).sort_values("taux_manquant_pct")
    if top.empty:
        print("Aucune valeur manquante a afficher.")
        return

    plt.figure(figsize=(9, max(4, 0.35 * len(top))))
    plt.barh(top["variable"], top["taux_manquant_pct"], color="#c0392b")
    plt.xlabel("Taux de valeurs manquantes (%)")
    plt.title(f"Valeurs manquantes - {period_label}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Graphique sauvegarde : {output_path}")


# ---------------------------------------------------------------------------
# 2. DOUBLONS
# ---------------------------------------------------------------------------

def report_duplicates(df: pd.DataFrame, period_label: str) -> None:
    if "NEWID" not in df.columns:
        print("[ATTENTION] NEWID absente, doublons non verifiables.")
        return
    dup = df["NEWID"].duplicated().sum()
    pct = dup / len(df) * 100 if len(df) else 0
    print(f"[{period_label}] NEWID dupliques : {dup} ({pct:.2f}% des lignes)")


# ---------------------------------------------------------------------------
# 3. VALEURS ABERRANTES (scan IQR + histogrammes + boxplots)
# ---------------------------------------------------------------------------

def detect_outliers_iqr(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
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
            "variable": col, "Q1": round(q1, 2), "Q3": round(q3, 2),
            "borne_basse": round(low, 2), "borne_haute": round(high, 2),
            "nb_aberrantes": n_outliers, "pct_aberrantes": round(n_outliers / len(s) * 100, 2),
        })
    return pd.DataFrame(rows)


def plot_distribution(df: pd.DataFrame, col: str, output_path: Path, clip_percentiles=(1, 99)) -> None:
    """Histogramme avec moyenne/mediane, deja utilise depuis le debut du projet."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = df[col].dropna()
    if s.empty:
        return
    lo, hi = np.percentile(s, clip_percentiles)

    plt.figure(figsize=(7, 4.5))
    plt.hist(s, bins=60, range=(lo, hi), color="#2980b9", edgecolor="white")
    plt.axvline(s.mean(), color="#c0392b", linestyle="--", label=f"Moyenne = {s.mean():.0f}")
    plt.axvline(s.median(), color="#27ae60", linestyle="--", label=f"Mediane = {s.median():.0f}")
    plt.xlabel(col)
    plt.ylabel("Nombre de foyers")
    plt.title(f"Distribution de {col}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_boxplot_single(df: pd.DataFrame, col: str, output_path: Path) -> None:
    """Boxplot individuel pour une variable : montre la mediane (barre
    centrale), les quartiles (boite), les moustaches (1.5x IQR), et
    chaque valeur aberrante au-dela comme un point isole - une lecture
    plus directe des aberrantes que l'histogramme seul.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = df[col].dropna()
    if s.empty:
        return

    plt.figure(figsize=(6, 4))
    box = plt.boxplot(s, vert=False, patch_artist=True, widths=0.5,
                       flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "#c0392b",
                                   "markeredgecolor": "#c0392b", "alpha": 0.4})
    box["boxes"][0].set_facecolor("#aed6f1")
    box["medians"][0].set_color("#c0392b")
    plt.xlabel(col)
    plt.title(f"Boxplot de {col}")
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_boxplot_comparison(df: pd.DataFrame, cols: list[str], period_label: str, output_path: Path) -> None:
    """Un seul graphique avec tous les postes cote a cote, pour comparer
    directement leur dispersion et le nombre de valeurs aberrantes entre
    eux (plus rapide a lire que 18 boxplots individuels separes).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = [df[c].dropna() for c in cols if c in df.columns]
    labels = [c for c in cols if c in df.columns]
    if not data:
        return

    plt.figure(figsize=(max(10, len(labels) * 0.8), 6))
    box = plt.boxplot(data, vert=True, patch_artist=True, widths=0.6,
                       flierprops={"marker": "o", "markersize": 2, "markerfacecolor": "#c0392b",
                                   "markeredgecolor": "#c0392b", "alpha": 0.3})
    for patch in box["boxes"]:
        patch.set_facecolor("#aed6f1")
    for median in box["medians"]:
        median.set_color("#c0392b")

    plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("Montant ($)")
    plt.title(f"Boxplots compares - depenses et revenu ({period_label})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Graphique sauvegarde : {output_path}")


# ---------------------------------------------------------------------------
# 4. CORRELATION
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(corr_df: pd.DataFrame, period_label: str, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    im = ax.imshow(corr_df.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_yticks(range(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr_df.columns, fontsize=8)
    for i in range(len(corr_df.columns)):
        for j in range(len(corr_df.columns)):
            ax.text(j, i, f"{corr_df.values[i, j]:.2f}", ha="center", va="center", color="black", fontsize=6)
    ax.set_title(f"Correlation entre depenses et revenu (net/brut) - {period_label}")
    fig.colorbar(im, ax=ax, label="Coefficient de correlation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Graphique sauvegarde : {output_path}")


# ---------------------------------------------------------------------------
# 5. STATISTIQUES DESCRIPTIVES PONDEREES
# ---------------------------------------------------------------------------

def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna()
    if mask.sum() == 0:
        return np.nan
    return np.average(values[mask], weights=weights[mask])


def weighted_std(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna()
    if mask.sum() == 0:
        return np.nan
    avg = weighted_mean(values, weights)
    variance = np.average((values[mask] - avg) ** 2, weights=weights[mask])
    return np.sqrt(variance)


def descriptive_stats(df: pd.DataFrame, cols: list[str], weight_col: str) -> pd.DataFrame:
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col]
        rows.append({
            "variable": col,
            "moyenne_non_ponderee": round(s.mean(), 2),
            "moyenne_ponderee": round(weighted_mean(s, df[weight_col]), 2) if weight_col in df.columns else np.nan,
            "mediane": round(s.median(), 2),
            "ecart_type_pondere": round(weighted_std(s, df[weight_col]), 2) if weight_col in df.columns else np.nan,
            "min": round(s.min(), 2),
            "max": round(s.max(), 2),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PIPELINE PAR PERIODE
# ---------------------------------------------------------------------------

def process_period(period_label: str, filename: str) -> None:
    print(f"\n{'=' * 70}\nEDA - {period_label.upper()}\n{'=' * 70}")

    df = pd.read_csv(INPUT_DIR / filename, low_memory=False)
    print(f"Charge : {len(df)} lignes, {df.shape[1]} colonnes")

    print(f"\n--- [1] Valeurs manquantes ---")
    missing = missing_rate_report(df, ID_COLUMNS)
    print(missing[missing["taux_manquant_pct"] > 0].head(15))
    missing.to_csv(OUTPUT_DIR / f"missing_{period_label}.csv", index=False)
    plot_missing_rate(missing, period_label, OUTPUT_DIR / f"missing_{period_label}.png")

    print(f"\n--- [2] Doublons ---")
    report_duplicates(df, period_label)

    print(f"\n--- [3] Valeurs aberrantes (IQR) : histogrammes + boxplots ---")
    outliers = detect_outliers_iqr(df, EXPENDITURE_FULL_COLUMNS)
    print(outliers)
    outliers.to_csv(OUTPUT_DIR / f"outliers_{period_label}.csv", index=False)

    for col in EXPENDITURE_FULL_COLUMNS:
        if col not in df.columns:
            print(f"[ATTENTION] {col} absente, graphiques ignores.")
            continue
        plot_distribution(df, col, OUTPUT_DIR / f"hist_{period_label}_{col}.png")
        plot_boxplot_single(df, col, OUTPUT_DIR / f"boxplot_{period_label}_{col}.png")
    print("Histogrammes et boxplots individuels sauvegardes pour les colonnes disponibles.")

    plot_boxplot_comparison(df, EXPENDITURE_FULL_COLUMNS, period_label,
                             OUTPUT_DIR / f"boxplot_comparaison_{period_label}.png")

    print(f"\n--- [4] Correlation entre depenses et revenu (net/brut) ---")
    existing_cols = [c for c in EXPENDITURE_FULL_COLUMNS if c in df.columns]
    corr = df[existing_cols].corr()
    print(corr.round(2))
    corr.to_csv(OUTPUT_DIR / f"correlation_{period_label}.csv")
    plot_correlation_heatmap(corr, period_label, OUTPUT_DIR / f"correlation_{period_label}.png")

    print(f"\n--- [5] Statistiques descriptives ponderees (depenses + revenu) ---")
    stats = descriptive_stats(df, EXPENDITURE_FULL_COLUMNS, WEIGHT_COLUMN)
    print(stats)
    stats.to_csv(OUTPUT_DIR / f"descriptive_stats_{period_label}.csv", index=False)


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    for period_label, filename in FILES.items():
        process_period(period_label, filename)

    print(f"\n{'=' * 70}")
    print(f"Termine. Tous les rapports (.csv) et graphiques (.png) sont dans : {OUTPUT_DIR}/")
    print("  hist_*.png                -> histogrammes (deja existants)")
    print("  boxplot_*.png             -> boxplots individuels (nouveau)")
    print("  boxplot_comparaison_*.png -> tous les postes + revenu compares sur un seul graphique (nouveau)")