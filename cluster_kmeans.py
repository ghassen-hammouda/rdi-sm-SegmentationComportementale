"""
Clustering non supervise (K-means) sur echantillon, avec choix de k par
la methode du coude

Ce script :
1. Charge les features encodees/mises a l'echelle (US-07)
2. Tire un echantillon aleatoire (par defaut 5000 foyers, ajustable)
3. Applique une PCA sur cet echantillon (16-17 composantes ~ 90% de
   variance, cf. resultats US-07) pour reduire la dimension avant clustering
4. Teste plusieurs valeurs de k (2 a 8 par defaut)
5. Pour chaque k : inertie (methode du coude) + score de silhouette
6. Une fois k choisi : croise les clusters obtenus avec les statuts de
   pauvrete (jamais utilises pendant le clustering lui-meme)

Le clustering reste strictement non supervise jusqu'a l'etape 6 : les
colonnes statut_pauvrete_* ne sont jamais donnees a KMeans, seulement
utilisees APRES pour interpreter les clusters obtenus.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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

SAMPLE_SIZE = 5000       # nombre de foyers tires pour le test (ajustable)
RANDOM_STATE = 42
K_RANGE = range(2, 9)    # valeurs de k testees (2 a 8 inclus)
PCA_VARIANCE_TARGET = 0.90  # coherent avec le seuil retenu en US-07


def load_sample(filepath: Path, sample_size: int, random_state: int) -> pd.DataFrame:
    """Charge le fichier de features et tire un echantillon aleatoire,
    plutot que de travailler sur la totalite des foyers (rapide a
    iterer, et prepare la meme logique pour des algorithmes plus lourds
    comme le clustering hierarchique plus tard).
    """
    df = pd.read_csv(filepath)
    n = min(sample_size, len(df))
    sample = df.sample(n=n, random_state=random_state).reset_index(drop=True)
    print(f"Charge : {filepath} ({len(df)} foyers) -> echantillon de {len(sample)} foyers")
    return sample


def reduce_dimensions(df: pd.DataFrame, feature_cols: list[str],
                       variance_target: float, random_state: int) -> tuple[np.ndarray, int]:
    """PCA ajustee sur l'echantillon, avec le nombre de composantes
    necessaires pour atteindre variance_target (coherent avec les
    resultats deja obtenus en US-07 sur la population complete : 16-17
    composantes pour 90%).
    """
    pca_full = PCA(random_state=random_state)
    pca_full.fit(df[feature_cols])
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.argmax(cumulative >= variance_target) + 1)

    pca = PCA(n_components=n_components, random_state=random_state)
    X_reduced = pca.fit_transform(df[feature_cols])

    print(f"PCA : {n_components} composantes retenues pour {variance_target*100:.0f}% de variance "
          f"(sur {len(feature_cols)} features de depart)")
    return X_reduced, n_components


def evaluate_k_range(X: np.ndarray, k_range: range, random_state: int) -> pd.DataFrame:
    """Pour chaque k teste : inertie (methode du coude) et score de
    silhouette (qualite de separation des clusters). Aucune information
    sur la pauvrete n'intervient a cette etape.
    """
    rows = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        inertia = kmeans.inertia_
        silhouette = silhouette_score(X, labels) if k > 1 else np.nan

        print(f"  k={k} : inertie={inertia:,.1f}, silhouette={silhouette:.4f}")
        rows.append({"k": k, "inertie": inertia, "silhouette": silhouette})

    return pd.DataFrame(rows)


def plot_elbow_and_silhouette(report: pd.DataFrame, period_label: str, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(report["k"], report["inertie"], marker="o", color="#2980b9")
    axes[0].set_xlabel("Nombre de clusters (k)")
    axes[0].set_ylabel("Inertie intra-cluster")
    axes[0].set_title(f"Methode du coude ({period_label})")
    axes[0].grid(alpha=0.3)

    axes[1].plot(report["k"], report["silhouette"], marker="o", color="#27ae60")
    axes[1].set_xlabel("Nombre de clusters (k)")
    axes[1].set_ylabel("Score de silhouette")
    axes[1].set_title(f"Score de silhouette ({period_label})")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Graphique sauvegarde : {output_path}")


def crosstab_clusters_vs_poverty(df: pd.DataFrame, labels: np.ndarray, period_label: str) -> pd.DataFrame:
    """Une fois les clusters obtenus (sans jamais avoir vu le statut de
    pauvrete), on croise chaque cluster avec les statuts de pauvrete
    pour interpreter les groupes a posteriori - jamais pendant le
    clustering lui-meme.
    """
    df = df.copy()
    df["cluster"] = labels

    print(f"\n[{period_label}] Repartition des foyers par cluster :")
    print(df["cluster"].value_counts().sort_index())

    rows = []
    for cluster_id, group in df.groupby("cluster"):
        row = {"cluster": cluster_id, "nb_foyers": len(group)}
        for target in TARGET_COLUMNS:
            row[f"taux_{target}"] = round(group[target].mean() * 100, 2)
        rows.append(row)

    summary = pd.DataFrame(rows)
    print(f"\n[{period_label}] Taux de pauvrete par cluster :")
    print(summary)
    return summary


# ---------------------------------------------------------------------------
# PIPELINE PAR PERIODE
# ---------------------------------------------------------------------------

def process_period(period_label: str, filename: str) -> None:
    print(f"\n{'=' * 70}\nCLUSTERING K-MEANS - {period_label.upper()}\n{'=' * 70}")

    df = load_sample(INPUT_DIR / filename, SAMPLE_SIZE, RANDOM_STATE)

    feature_cols = [c for c in df.columns if c not in [ID_COLUMN] + TARGET_COLUMNS]
    print(f"{len(feature_cols)} features utilisees pour le clustering (cibles exclues).")

    X_reduced, n_components = reduce_dimensions(df, feature_cols, PCA_VARIANCE_TARGET, RANDOM_STATE)

    print(f"\n--- Test de plusieurs k ({list(K_RANGE)}) ---")
    report = evaluate_k_range(X_reduced, K_RANGE, RANDOM_STATE)
    report.to_csv(OUTPUT_DIR / f"kmeans_elbow_{period_label}.csv", index=False)
    plot_elbow_and_silhouette(report, period_label, OUTPUT_DIR / f"kmeans_elbow_{period_label}.png")

    best_k_silhouette = int(report.loc[report["silhouette"].idxmax(), "k"])
    print(f"\n[{period_label}] Meilleur k selon le score de silhouette : {best_k_silhouette}")
    print(f"[{period_label}] Regardez aussi le graphique du coude pour confirmer visuellement ce choix.")

    print(f"\n--- Clustering final avec k={best_k_silhouette} ---")
    kmeans_final = KMeans(n_clusters=best_k_silhouette, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans_final.fit_predict(X_reduced)

    summary = crosstab_clusters_vs_poverty(df, labels, period_label)
    summary.to_csv(OUTPUT_DIR / f"clusters_vs_pauvrete_{period_label}.csv", index=False)


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    for period_label, filename in FILES.items():
        process_period(period_label, filename)

    print(f"\n{'=' * 70}")
    print("Termine. Fichiers dans outputs_clustering/ :")
    print("  kmeans_elbow_period*.csv/.png       -> inertie + silhouette par k")
    print("  clusters_vs_pauvrete_period*.csv    -> taux de pauvrete par cluster (k retenu)")