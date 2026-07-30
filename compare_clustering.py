"""
Comparaison de plusieurs algorithmes de clustering non supervise

Ce script applique 4 algorithmes differents sur EXACTEMENT le meme
echantillon et la meme reduction PCA (pour une comparaison honnete) :
- K-means (deja utilise, repris ici en reference)
- Clustering hierarchique (agglomeratif)
- DBSCAN (detecte le bruit/les cas atypiques, pas de k impose)
- Gaussian Mixture Model (probabiliste, plus souple que K-means)

Meme principe que pour K-means : aucun algorithme ne voit jamais
statut_pauvrete_* pendant le clustering. La comparaison avec la pauvrete
n'intervient qu'a la toute fin, pour chaque modele separement.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

INPUT_DIR = Path("outputs_features")
OUTPUT_DIR = Path("outputs_clustering")
OUTPUT_DIR.mkdir(exist_ok=True)

FILES = {
    "period1": "features_period1.csv",
    "period2": "features_period2.csv",
}

ID_COLUMN = "CUID"
TARGET_COLUMNS = [
    "statut_pauvrete_revenu_brut",
    "statut_pauvrete_revenu_net",
    "statut_pauvrete_consommation",
]

SAMPLE_SIZE = 5000
RANDOM_STATE = 42
PCA_VARIANCE_TARGET = 0.90
FIXED_K = 3   # coherent avec le k retenu pour KMeans (silhouette + coude)


# ---------------------------------------------------------------------------
# CHARGEMENT ET REDUCTION (identique au script K-means, pour comparaison equitable)
# ---------------------------------------------------------------------------

def load_sample(filepath: Path, sample_size: int, random_state: int) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    n = min(sample_size, len(df))
    sample = df.sample(n=n, random_state=random_state).reset_index(drop=True)
    print(f"Charge : {filepath} ({len(df)} foyers) -> echantillon de {len(sample)} foyers")
    return sample


def reduce_dimensions(df: pd.DataFrame, feature_cols: list[str],
                       variance_target: float, random_state: int) -> np.ndarray:
    pca_full = PCA(random_state=random_state)
    pca_full.fit(df[feature_cols])
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.argmax(cumulative >= variance_target) + 1)

    pca = PCA(n_components=n_components, random_state=random_state)
    X_reduced = pca.fit_transform(df[feature_cols])
    print(f"PCA : {n_components} composantes retenues pour {variance_target*100:.0f}% de variance")
    return X_reduced


# ---------------------------------------------------------------------------
# ESTIMATION AUTOMATIQUE DE eps POUR DBSCAN
# ---------------------------------------------------------------------------

def estimate_dbscan_eps(X: np.ndarray, k_neighbors: int = 5, percentile: float = 90) -> float:
    """DBSCAN n'a pas de k a choisir, mais un rayon (eps) et un nombre
    minimal de voisins (min_samples). Estimation heuristique de eps :
    pour chaque point, on mesure la distance a son k-ieme plus proche
    voisin, puis on prend le 90e percentile de ces distances - une
    valeur qui capture la densite typique des points "normaux" sans etre
    trop influencee par les quelques points isoles.

    Cette estimation est un point de depart raisonnable, pas une valeur
    optimale garantie : un ajustement manuel peut ameliorer le resultat.
    """
    neighbors = NearestNeighbors(n_neighbors=k_neighbors)
    neighbors.fit(X)
    distances, _ = neighbors.kneighbors(X)
    k_distances = distances[:, -1]
    eps = float(np.percentile(k_distances, percentile))
    print(f"  eps estime automatiquement : {eps:.3f} (90e percentile des distances au {k_neighbors}e voisin)")
    return eps


# ---------------------------------------------------------------------------
# CROISEMENT CLUSTERS x PAUVRETE (identique au script K-means)
# ---------------------------------------------------------------------------

def crosstab_clusters_vs_poverty(df: pd.DataFrame, labels: np.ndarray,
                                  model_name: str, period_label: str) -> pd.DataFrame:
    df = df.copy()
    df["cluster"] = labels

    rows = []
    for cluster_id, group in df.groupby("cluster"):
        row = {"modele": model_name, "cluster": cluster_id, "nb_foyers": len(group)}
        for target in TARGET_COLUMNS:
            row[f"taux_{target}"] = round(group[target].mean() * 100, 2)
        rows.append(row)

    summary = pd.DataFrame(rows)
    print(f"\n[{period_label}] {model_name} - repartition et taux de pauvrete par cluster :")
    print(summary)
    return summary


# ---------------------------------------------------------------------------
# PIPELINE PAR PERIODE : LES 4 ALGORITHMES
# ---------------------------------------------------------------------------

def process_period(period_label: str, filename: str) -> pd.DataFrame:
    print(f"\n{'=' * 70}\nCOMPARAISON DE MODELES DE CLUSTERING - {period_label.upper()}\n{'=' * 70}")

    df = load_sample(INPUT_DIR / filename, SAMPLE_SIZE, RANDOM_STATE)
    feature_cols = [c for c in df.columns if c not in [ID_COLUMN] + TARGET_COLUMNS]
    X = reduce_dimensions(df, feature_cols, PCA_VARIANCE_TARGET, RANDOM_STATE)

    all_summaries = []
    model_comparison_rows = []

    # --- 1. K-MEANS (reference) ---
    print(f"\n--- 1. K-means (k={FIXED_K}) ---")
    kmeans = KMeans(n_clusters=FIXED_K, random_state=RANDOM_STATE, n_init=10)
    labels_kmeans = kmeans.fit_predict(X)
    sil_kmeans = silhouette_score(X, labels_kmeans)
    print(f"Score de silhouette : {sil_kmeans:.4f}")
    summary_kmeans = crosstab_clusters_vs_poverty(df, labels_kmeans, "KMeans", period_label)
    all_summaries.append(summary_kmeans)
    model_comparison_rows.append({
        "modele": "KMeans", "n_clusters_trouves": FIXED_K, "silhouette": round(sil_kmeans, 4),
        "taux_pauvrete_max_cluster": summary_kmeans["taux_statut_pauvrete_revenu_net"].max(),
    })

    # --- 2. CLUSTERING HIERARCHIQUE (agglomeratif) ---
    print(f"\n--- 2. Clustering hierarchique (k={FIXED_K}) ---")
    hierarchical = AgglomerativeClustering(n_clusters=FIXED_K)
    labels_hier = hierarchical.fit_predict(X)
    sil_hier = silhouette_score(X, labels_hier)
    print(f"Score de silhouette : {sil_hier:.4f}")
    summary_hier = crosstab_clusters_vs_poverty(df, labels_hier, "Hierarchique", period_label)
    all_summaries.append(summary_hier)
    model_comparison_rows.append({
        "modele": "Hierarchique", "n_clusters_trouves": FIXED_K, "silhouette": round(sil_hier, 4),
        "taux_pauvrete_max_cluster": summary_hier["taux_statut_pauvrete_revenu_net"].max(),
    })

    # --- 3. DBSCAN (pas de k impose, detecte le bruit) ---
    print(f"\n--- 3. DBSCAN ---")
    eps = estimate_dbscan_eps(X)
    dbscan = DBSCAN(eps=eps, min_samples=5)
    labels_dbscan = dbscan.fit_predict(X)
    n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
    n_noise = int((labels_dbscan == -1).sum())
    print(f"Clusters trouves : {n_clusters_dbscan} (+ {n_noise} points consideres comme bruit, etiquette -1)")
    if n_clusters_dbscan >= 2:
        mask = labels_dbscan != -1
        sil_dbscan = silhouette_score(X[mask], labels_dbscan[mask]) if mask.sum() > 0 else np.nan
        print(f"Score de silhouette (hors bruit) : {sil_dbscan:.4f}")
    else:
        sil_dbscan = np.nan
        print("[ATTENTION] Moins de 2 clusters trouves, silhouette non calculable. "
              "Ajustez eps/min_samples si ce resultat n'est pas exploitable.")
    summary_dbscan = crosstab_clusters_vs_poverty(df, labels_dbscan, "DBSCAN", period_label)
    all_summaries.append(summary_dbscan)
    model_comparison_rows.append({
        "modele": "DBSCAN", "n_clusters_trouves": n_clusters_dbscan,
        "silhouette": round(sil_dbscan, 4) if not np.isnan(sil_dbscan) else None,
        "taux_pauvrete_max_cluster": summary_dbscan["taux_statut_pauvrete_revenu_net"].max(),
    })

    # --- 4. GAUSSIAN MIXTURE MODEL ---
    print(f"\n--- 4. Gaussian Mixture Model (k={FIXED_K}) ---")
    gmm = GaussianMixture(n_components=FIXED_K, random_state=RANDOM_STATE)
    labels_gmm = gmm.fit_predict(X)
    sil_gmm = silhouette_score(X, labels_gmm)
    print(f"Score de silhouette : {sil_gmm:.4f}")
    summary_gmm = crosstab_clusters_vs_poverty(df, labels_gmm, "GaussianMixture", period_label)
    all_summaries.append(summary_gmm)
    model_comparison_rows.append({
        "modele": "GaussianMixture", "n_clusters_trouves": FIXED_K, "silhouette": round(sil_gmm, 4),
        "taux_pauvrete_max_cluster": summary_gmm["taux_statut_pauvrete_revenu_net"].max(),
    })

    # --- TABLEAU COMPARATIF FINAL ---
    comparison = pd.DataFrame(model_comparison_rows)
    print(f"\n{'=' * 70}\n[{period_label}] TABLEAU COMPARATIF DES 4 MODELES\n{'=' * 70}")
    print(comparison)

    detail = pd.concat(all_summaries, ignore_index=True)
    detail.to_csv(OUTPUT_DIR / f"comparison_clusters_detail_{period_label}.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / f"comparison_models_{period_label}.csv", index=False)

    return comparison


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    for period_label, filename in FILES.items():
        process_period(period_label, filename)

    print(f"\n{'=' * 70}")
    print("Termine. Fichiers dans outputs_clustering/ :")
    print("  comparison_models_period*.csv        -> tableau comparatif des 4 modeles")
    print("  comparison_clusters_detail_period*.csv -> detail par cluster, tous modeles confondus")