"""
Profilage detaille des clusters retenus (US-11 v2), sur l'echantillon
de 5000 foyers, pour les 3 methodes a la fois - AVEC GRAPHIQUES.

Objectif : comprendre CONCRETEMENT ce qui differencie les clusters de
chaque methode (K-Means, GMM, CAH), en valeurs REELLES (non
standardisees), avant de choisir quelle methode retenir pour le
clustering final.

Pour chaque periode, au k retenu (BEST_K_OVERRIDE, coherent avec
benchmark_clustering_sample_v2.py) :
1. Reclustering des 3 methodes sur le MEME echantillon
2. Reassociation aux valeurs reelles (poverty_targets_period{1,2}.csv)
3. Pour chaque methode : graphique des moyennes de variables cles par
   cluster (barres groupees) + tableau CSV complet
4. Repartition des categorielles dominantes par cluster (graphique)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster

INPUT_DIR = Path("outputs_v2")
OUTPUT_DIR = Path("outputs_v2")
GRAPH_DIR = OUTPUT_DIR / "graphs"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_FILES = {
    "period1": "clustering_features_period1.csv",
    "period2": "clustering_features_period2.csv",
}
RAW_FILES = {
    "period1": "poverty_targets_period1.csv",
    "period2": "poverty_targets_period2.csv",
}

ID_COLUMN = "CUID"
SAMPLE_SIZE = 5000
RANDOM_STATE = 42
KMEANS_N_INIT = 5
GMM_N_INIT = 3
POVERTY_TARGETS = ["statut_pauvrete_revenu_net", "statut_pauvrete_consommation"]

BEST_K_OVERRIDE = {
    "period1": 4,
    "period2": 2,
}

# Variables cles a profiler visuellement (sous-ensemble lisible, pas les 42)
KEY_CONTINUOUS = [
    "totexp_par_equivalent_adulte", "ratio_logement_revenu_net",
    "part_hous", "part_food", "part_health", "part_retpen",
    "FAM_SIZE", "PERSLT18", "PERSOT64", "AGE_REF",
]
CATEGORICAL_TO_PROFILE = ["REGION", "FAM_TYPE", "CUTENURE"]


def load_sample_with_raw(period_label: str) -> tuple[pd.DataFrame, np.ndarray]:
    features_path = INPUT_DIR / FEATURES_FILES[period_label]
    raw_path = INPUT_DIR / RAW_FILES[period_label]

    df_features = pd.read_csv(features_path, low_memory=False)
    df_raw = pd.read_csv(raw_path, low_memory=False)

    n = min(SAMPLE_SIZE, len(df_features))
    sample_features = df_features.sample(n=n, random_state=RANDOM_STATE).reset_index(drop=True)

    X = sample_features.drop(columns=[ID_COLUMN]).values

    sample_full = sample_features[[ID_COLUMN]].merge(df_raw, on=ID_COLUMN, how="left")

    return sample_full, X


def fit_all_methods(X: np.ndarray, k: int) -> dict:
    labels_dict = {}

    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=KMEANS_N_INIT).fit(X)
    labels_dict["KMeans"] = km.labels_

    gmm = GaussianMixture(n_components=k, random_state=RANDOM_STATE, n_init=GMM_N_INIT).fit(X)
    labels_dict["GMM"] = gmm.predict(X)

    Z = linkage(X, method="ward")
    labels_dict["CAH"] = fcluster(Z, k, criterion="maxclust")

    return labels_dict


def plot_continuous_profile(sample_full: pd.DataFrame, labels_dict: dict, period_label: str, k: int) -> None:
    n_vars = len(KEY_CONTINUOUS)
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    axes = axes.flatten()

    for i, var in enumerate(KEY_CONTINUOUS):
        ax = axes[i]
        for methode, labels in labels_dict.items():
            tmp = sample_full.copy()
            tmp["cluster"] = labels
            means = tmp.groupby("cluster")[var].mean().sort_index()
            ax.plot(means.index.astype(str), means.values, marker="o", label=methode)
        ax.set_title(var, fontsize=10)
        ax.tick_params(axis="x", labelsize=8)

    axes[0].legend(fontsize=8)
    fig.suptitle(f"Profil des clusters (moyennes reelles) - {period_label}, k={k}")
    plt.tight_layout()
    path = GRAPH_DIR / f"cluster_profile_continuous_{period_label}_k{k}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Graphique sauvegarde : {path}")


def plot_categorical_profile(sample_full: pd.DataFrame, labels_dict: dict, period_label: str, k: int) -> None:
    fig, axes = plt.subplots(len(CATEGORICAL_TO_PROFILE), len(labels_dict),
                              figsize=(5 * len(labels_dict), 4 * len(CATEGORICAL_TO_PROFILE)))

    for row, cat_var in enumerate(CATEGORICAL_TO_PROFILE):
        for col, (methode, labels) in enumerate(labels_dict.items()):
            ax = axes[row, col]
            tmp = sample_full.copy()
            tmp["cluster"] = labels
            crosstab = pd.crosstab(tmp["cluster"], tmp[cat_var], normalize="index") * 100
            crosstab.plot(kind="bar", stacked=True, ax=ax, legend=(col == len(labels_dict) - 1), width=0.8)
            ax.set_title(f"{methode} - {cat_var}", fontsize=10)
            ax.set_ylabel("% du cluster")
            if col == len(labels_dict) - 1:
                ax.legend(fontsize=6, bbox_to_anchor=(1.05, 1), loc="upper left")

    fig.suptitle(f"Repartition des categories par cluster - {period_label}, k={k}")
    plt.tight_layout()
    path = GRAPH_DIR / f"cluster_profile_categorical_{period_label}_k{k}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Graphique sauvegarde : {path}")


def save_full_profile_table(sample_full: pd.DataFrame, labels_dict: dict, period_label: str, k: int) -> None:
    for methode, labels in labels_dict.items():
        tmp = sample_full.copy()
        tmp["cluster"] = labels
        table = tmp.groupby("cluster")[KEY_CONTINUOUS + POVERTY_TARGETS].mean()
        table["n_foyers"] = tmp.groupby("cluster").size()
        path = OUTPUT_DIR / f"cluster_profile_{methode}_{period_label}_k{k}.csv"
        table.to_csv(path)
        print(f"Sauvegarde : {path}")
        print(f"\n--- {methode} - {period_label} (k={k}) ---")
        print(table.round(3))


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    for period_label in FEATURES_FILES:
        k = BEST_K_OVERRIDE[period_label]
        print(f"\n{'=' * 70}\nPROFILAGE DETAILLE - {period_label.upper()} (k={k})\n{'=' * 70}")

        sample_full, X = load_sample_with_raw(period_label)
        print(f"Echantillon charge : {len(sample_full)} foyers")

        labels_dict = fit_all_methods(X, k)

        save_full_profile_table(sample_full, labels_dict, period_label, k)
        plot_continuous_profile(sample_full, labels_dict, period_label, k)
        plot_categorical_profile(sample_full, labels_dict, period_label, k)

    print(f"\n{'=' * 70}")
    print("Termine. A examiner :")
    print(f"  - {GRAPH_DIR}/cluster_profile_continuous_period{{1,2}}_k*.png")
    print(f"  - {GRAPH_DIR}/cluster_profile_categorical_period{{1,2}}_k*.png")
    print("  - cluster_profile_{methode}_period{1,2}_k*.csv (tableaux complets)")