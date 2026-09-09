"""
Benchmark de methodes de clustering (US-10), sur l'espace de variables
prepare dans build_clustering_features_v2.py.

Methodes comparees :
- K-Means : rapide, hypothese de clusters spheriques
- GMM (Gaussian Mixture Model) : clusters elliptiques, versions "douces"
  (probabilites d'appartenance plutot qu'une affectation dure)
- CAH (Classification Ascendante Hierarchique, agglomerative) : pas
  d'hypothese de forme, mais couteuse en memoire/temps sur un gros
  volume - executee sur un ECHANTILLON ALEATOIRE (voir CAH_SAMPLE_SIZE),
  pas sur la totalite des foyers. A mentionner explicitement comme
  limite si le resultat est presente.

Plage de k testee : K_RANGE (par defaut 2 a 6).

Metriques calculees pour chaque (methode, k) :
- Inertie (K-Means uniquement) : pour la methode du coude (elbow)
- BIC (GMM uniquement) : plus bas = meilleur compromis qualite/complexite
- Silhouette score : cohesion intra-cluster vs separation inter-cluster,
  entre -1 et 1, plus haut = mieux. Calcule sur un ECHANTILLON (le calcul
  exact sur 38000+ foyers serait trop couteux, O(n^2))
- Davies-Bouldin : plus bas = mieux, mesure la separation des clusters

Ce script NE CHOISIT PAS automatiquement la meilleure combinaison
methode/k : il produit un tableau recapitulatif (CSV) a examiner
manuellement, en croisant les scores avec l'interpretabilite metier
des clusters obtenus (un cluster illisible n'est pas utile meme s'il
score bien).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score

INPUT_DIR = Path("outputs_v2")
OUTPUT_DIR = Path("outputs_v2")

FILES = {
    "period1": "clustering_features_period1.csv",
    "period2": "clustering_features_period2.csv",
}

ID_COLUMN = "CUID"
K_RANGE = range(2, 7)  # reduit de 2-8 a 2-6 pour accelerer le benchmark
RANDOM_STATE = 42

# n_init reduit par rapport aux valeurs par defaut de scikit-learn, pour
# accelerer le benchmark exploratoire (KMeans : 10->2, GMM : 3->1).
# Compromis assume : legerement moins robuste au hasard de l'initialisation,
# mais random_state fixe -> resultats reproductibles d'une execution a l'autre.
KMEANS_N_INIT = 2
GMM_N_INIT = 1
SILHOUETTE_SAMPLE_SIZE = 5000  # sous-echantillon pour le calcul du silhouette (cout O(n^2))
CAH_SAMPLE_SIZE = 3000  # sous-echantillon pour la CAH (cout memoire/temps du linkage)


def load_feature_matrix(filepath: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(filepath, low_memory=False)
    ids = df[ID_COLUMN].values
    X = df.drop(columns=[ID_COLUMN]).values
    return df, X


def evaluate_kmeans(X: np.ndarray, k: int) -> dict:
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=KMEANS_N_INIT)
    labels = model.fit_predict(X)

    sample_idx = np.random.RandomState(RANDOM_STATE).choice(
        len(X), size=min(SILHOUETTE_SAMPLE_SIZE, len(X)), replace=False
    )
    silhouette = silhouette_score(X[sample_idx], labels[sample_idx])
    davies_bouldin = davies_bouldin_score(X, labels)

    return {
        "methode": "KMeans", "k": k,
        "inertie": model.inertia_, "bic": np.nan,
        "silhouette": silhouette, "davies_bouldin": davies_bouldin,
    }


def evaluate_gmm(X: np.ndarray, k: int) -> dict:
    model = GaussianMixture(n_components=k, random_state=RANDOM_STATE, n_init=GMM_N_INIT)
    labels = model.fit_predict(X)

    sample_idx = np.random.RandomState(RANDOM_STATE).choice(
        len(X), size=min(SILHOUETTE_SAMPLE_SIZE, len(X)), replace=False
    )
    silhouette = silhouette_score(X[sample_idx], labels[sample_idx])
    davies_bouldin = davies_bouldin_score(X, labels)

    return {
        "methode": "GMM", "k": k,
        "inertie": np.nan, "bic": model.bic(X),
        "silhouette": silhouette, "davies_bouldin": davies_bouldin,
    }


def evaluate_cah(X: np.ndarray, k: int) -> dict:
    """CAH executee sur un sous-echantillon aleatoire (cout memoire/temps
    du linkage hierarchique prohibitif sur la totalite des foyers)."""
    sample_idx = np.random.RandomState(RANDOM_STATE).choice(
        len(X), size=min(CAH_SAMPLE_SIZE, len(X)), replace=False
    )
    X_sample = X[sample_idx]

    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = model.fit_predict(X_sample)

    silhouette = silhouette_score(X_sample, labels)
    davies_bouldin = davies_bouldin_score(X_sample, labels)

    return {
        "methode": "CAH (echantillon)", "k": k,
        "inertie": np.nan, "bic": np.nan,
        "silhouette": silhouette, "davies_bouldin": davies_bouldin,
    }


def run_benchmark(X: np.ndarray, period_label: str) -> pd.DataFrame:
    results = []
    for k in K_RANGE:
        print(f"[{period_label}] k={k} : KMeans...")
        results.append(evaluate_kmeans(X, k))

        print(f"[{period_label}] k={k} : GMM...")
        results.append(evaluate_gmm(X, k))

        print(f"[{period_label}] k={k} : CAH (echantillon de {min(CAH_SAMPLE_SIZE, len(X))} foyers)...")
        results.append(evaluate_cah(X, k))

    return pd.DataFrame(results)


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    for period_label, filename in FILES.items():
        print(f"\n{'=' * 70}\nBENCHMARK CLUSTERING - {period_label.upper()}\n{'=' * 70}")

        filepath = INPUT_DIR / filename
        df, X = load_feature_matrix(filepath)
        print(f"Charge : {filepath} ({len(df)} foyers, {X.shape[1]} variables)")

        results_df = run_benchmark(X, period_label)

        print(f"\n[{period_label}] Resultats (tries par silhouette decroissant) :")
        print(results_df.sort_values("silhouette", ascending=False).to_string(index=False))

        output_path = OUTPUT_DIR / f"benchmark_clustering_{period_label}.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\nSauvegarde : {output_path}")

    print(f"\n{'=' * 70}")
    print("Termine. Fichiers dans outputs_v2/ :")
    print("  benchmark_clustering_period{1,2}.csv")
    print("  -> A examiner manuellement : croiser silhouette/davies_bouldin avec")
    print("     l'interpretabilite metier des clusters avant de choisir methode + k.")
    print("  Rappel : CAH executee sur un ECHANTILLON, pas la totalite des foyers -")
    print("  a mentionner comme limite si ces resultats sont presentes.")