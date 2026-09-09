"""
Benchmark de clustering sur echantillon, AVEC SORTIES GRAPHIQUES (US-10 v2).
Remplace benchmark_clustering_v2.py et profile_clusters_v2.py par un
protocole unifie.

Nouveau protocole :
1. Tirage d'un echantillon aleatoire de SAMPLE_SIZE foyers par periode -
   LE MEME echantillon sert aux 3 methodes (K-Means, GMM, CAH), pour
   une comparaison equitable et rapide (fini le melange "clustering sur
   tous les foyers, evaluation sur echantillon" de la version precedente)
2. Benchmark des 3 methodes sur cet echantillon, k de K_MIN a K_MAX ->
   tableau de scores (silhouette, Davies-Bouldin, inertie/BIC) + deux
   graphiques (silhouette vs k, Davies-Bouldin vs k) + un dendrogramme
   pour la CAH
3. Identification du meilleur (methode, k) par silhouette
4. A ce meme k, reclustering avec CHACUNE des 3 methodes sur
   l'echantillon, visualise en projection PCA 2D (un point = un foyer,
   couleur = cluster), pour comparer visuellement comment chaque
   methode decoupe l'espace
5. Croisement de chaque clustering avec LES DEUX cibles de pauvrete
   (statut_pauvrete_revenu_net ET statut_pauvrete_consommation, jamais
   vues par le clustering, reintroduites ici uniquement pour la
   lecture des resultats), en tableau CSV ET en graphique en barres

Ce script NE CHOISIT PAS l'algorithme final : il produit les elements
de comparaison (scores, visuels, taux de pauvrete par cluster) pour
une decision manuelle. Une fois l'algorithme choisi, un clustering
final sur TOUS les foyers (pas seulement l'echantillon) sera fait dans
un script separe.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

INPUT_DIR = Path("outputs_v2")
OUTPUT_DIR = Path("outputs_v2")
GRAPH_DIR = OUTPUT_DIR / "graphs"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_FILES = {
    "period1": "clustering_features_period1.csv",
    "period2": "clustering_features_period2.csv",
}
RESERVED_FILES = {
    "period1": "clustering_reserved_period1.csv",
    "period2": "clustering_reserved_period2.csv",
}

ID_COLUMN = "CUID"
SAMPLE_SIZE = 5000
K_RANGE = range(2, 7)
RANDOM_STATE = 42
KMEANS_N_INIT = 5  # echantillon petit -> peut se permettre plus d'initialisations
GMM_N_INIT = 3
POVERTY_TARGETS = ["statut_pauvrete_revenu_net", "statut_pauvrete_consommation"]

# Override manuel du k retenu pour l'etape de reclustering + PCA + poverty
# plot : le meilleur silhouette (k=2) n'est PAS retenu en period1, car
# confirme (profilage sur tous les foyers + echantillon + PCA + ecretage
# teste) comme un clivage demographique (age, taille du foyer), sans lien
# avec la pauvrete. k=4 avait deja fait ressortir un sous-groupe a fort
# taux de pauvrete lors du profilage complet. Period2 garde le meilleur
# silhouette (k=2), deja valide comme pertinent (ecart de pauvrete marque).
BEST_K_OVERRIDE = {
    "period1": 4,
    "period2": 2,
}


def load_sample(period_label: str) -> pd.DataFrame:
    features_path = INPUT_DIR / FEATURES_FILES[period_label]
    reserved_path = INPUT_DIR / RESERVED_FILES[period_label]

    df_features = pd.read_csv(features_path, low_memory=False)
    df_reserved = pd.read_csv(reserved_path, low_memory=False)

    n = min(SAMPLE_SIZE, len(df_features))
    sample = df_features.sample(n=n, random_state=RANDOM_STATE).reset_index(drop=True)
    sample = sample.merge(df_reserved[[ID_COLUMN] + POVERTY_TARGETS], on=ID_COLUMN, how="left")

    print(f"[{period_label}] Echantillon tire : {len(sample)} foyers (sur {len(df_features)} disponibles)")
    return sample


def run_benchmark(X: np.ndarray, period_label: str):
    results = []

    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=KMEANS_N_INIT).fit(X)
        results.append({
            "methode": "KMeans", "k": k,
            "silhouette": silhouette_score(X, km.labels_),
            "davies_bouldin": davies_bouldin_score(X, km.labels_),
            "inertie": km.inertia_, "bic": np.nan,
        })

        gmm = GaussianMixture(n_components=k, random_state=RANDOM_STATE, n_init=GMM_N_INIT).fit(X)
        labels_gmm = gmm.predict(X)
        results.append({
            "methode": "GMM", "k": k,
            "silhouette": silhouette_score(X, labels_gmm),
            "davies_bouldin": davies_bouldin_score(X, labels_gmm),
            "inertie": np.nan, "bic": gmm.bic(X),
        })

        print(f"[{period_label}] k={k} : KMeans et GMM evalues")

    # CAH : un seul linkage calcule (le plus couteux), puis coupe a differents k
    print(f"[{period_label}] Calcul du linkage hierarchique (CAH)...")
    Z = linkage(X, method="ward")
    for k in K_RANGE:
        labels_cah = fcluster(Z, k, criterion="maxclust")
        results.append({
            "methode": "CAH", "k": k,
            "silhouette": silhouette_score(X, labels_cah),
            "davies_bouldin": davies_bouldin_score(X, labels_cah),
            "inertie": np.nan, "bic": np.nan,
        })

    return pd.DataFrame(results), Z


def plot_scores(results_df: pd.DataFrame, period_label: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for methode in results_df["methode"].unique():
        subset = results_df[results_df["methode"] == methode].sort_values("k")
        axes[0].plot(subset["k"], subset["silhouette"], marker="o", label=methode)
        axes[1].plot(subset["k"], subset["davies_bouldin"], marker="o", label=methode)

    axes[0].set_title(f"Silhouette score vs k - {period_label}")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("Silhouette (plus haut = mieux)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].set_title(f"Davies-Bouldin vs k - {period_label}")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("Davies-Bouldin (plus bas = mieux)")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = GRAPH_DIR / f"benchmark_scores_{period_label}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Graphique sauvegarde : {path}")


def plot_dendrogram(Z: np.ndarray, period_label: str) -> None:
    plt.figure(figsize=(10, 5))
    dendrogram(Z, truncate_mode="lastp", p=30, show_leaf_counts=True)
    plt.title(f"Dendrogramme CAH (30 dernieres fusions) - {period_label}")
    plt.xlabel("Foyers (ou groupes de foyers)")
    plt.ylabel("Distance")
    plt.tight_layout()
    path = GRAPH_DIR / f"dendrogram_{period_label}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Graphique sauvegarde : {path}")


def plot_clusters_pca(X: np.ndarray, labels_dict: dict, period_label: str, k: int) -> None:
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X)
    var_expl = pca.explained_variance_ratio_.sum() * 100

    fig, axes = plt.subplots(1, len(labels_dict), figsize=(6 * len(labels_dict), 5))
    for ax, (methode, labels) in zip(axes, labels_dict.items()):
        ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=10, alpha=0.6)
        ax.set_title(f"{methode} (k={k})")
        ax.set_xlabel("Composante 1"); ax.set_ylabel("Composante 2")

    fig.suptitle(f"Projection PCA 2D - {period_label} (variance expliquee : {var_expl:.1f}%)")
    plt.tight_layout()
    path = GRAPH_DIR / f"clusters_pca_{period_label}_k{k}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Graphique sauvegarde : {path}")


def plot_poverty_by_cluster(sample: pd.DataFrame, labels_dict: dict, period_label: str, k: int) -> dict:
    fig, axes = plt.subplots(1, len(labels_dict), figsize=(6 * len(labels_dict), 5))
    tables = {}

    for ax, (methode, labels) in zip(axes, labels_dict.items()):
        tmp = sample.copy()
        tmp["cluster"] = labels
        taux = tmp.groupby("cluster")[POVERTY_TARGETS].mean() * 100
        tables[methode] = taux

        taux.plot(kind="bar", ax=ax, legend=False)
        ax.set_title(f"{methode} (k={k})")
        ax.set_ylabel("Taux de pauvrete (%)")
        ax.set_xlabel("Cluster")

    axes[-1].legend(["Revenu net", "Consommation"], loc="upper right")
    fig.suptitle(f"Taux de pauvrete par cluster - {period_label}")
    plt.tight_layout()
    path = GRAPH_DIR / f"poverty_by_cluster_{period_label}_k{k}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Graphique sauvegarde : {path}")

    return tables


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    for period_label in FEATURES_FILES:
        print(f"\n{'=' * 70}\nBENCHMARK SUR ECHANTILLON - {period_label.upper()}\n{'=' * 70}")

        sample = load_sample(period_label)
        X = sample.drop(columns=[ID_COLUMN] + POVERTY_TARGETS).values

        results_df, Z = run_benchmark(X, period_label)

        print(f"\n[{period_label}] Resultats (tries par silhouette decroissant) :")
        print(results_df.sort_values("silhouette", ascending=False).to_string(index=False))

        table_path = OUTPUT_DIR / f"benchmark_sample_{period_label}.csv"
        results_df.to_csv(table_path, index=False)
        print(f"\nSauvegarde : {table_path}")

        plot_scores(results_df, period_label)
        plot_dendrogram(Z, period_label)

        best_row = results_df.loc[results_df["silhouette"].idxmax()]
        print(f"\n[{period_label}] Meilleur (methode, k) par silhouette (info) : "
              f"{best_row['methode']} / k={int(best_row['k'])} (silhouette={best_row['silhouette']:.3f})")

        best_k = BEST_K_OVERRIDE.get(period_label, int(best_row["k"]))
        if period_label in BEST_K_OVERRIDE:
            print(f"[{period_label}] k RETENU (override manuel, pas le meilleur silhouette) : k={best_k} "
                  f"- voir BEST_K_OVERRIDE en tete de script pour la justification")
        else:
            print(f"[{period_label}] k retenu (silhouette) : k={best_k}")

        # Reclustering avec les 3 methodes, au MEME k, pour comparaison directe
        labels_dict = {}

        km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=KMEANS_N_INIT).fit(X)
        labels_dict["KMeans"] = km.labels_

        gmm = GaussianMixture(n_components=best_k, random_state=RANDOM_STATE, n_init=GMM_N_INIT).fit(X)
        labels_dict["GMM"] = gmm.predict(X)

        labels_dict["CAH"] = fcluster(Z, best_k, criterion="maxclust")

        plot_clusters_pca(X, labels_dict, period_label, best_k)
        poverty_tables = plot_poverty_by_cluster(sample, labels_dict, period_label, best_k)

        for methode, table in poverty_tables.items():
            print(f"\n[{period_label}] Taux de pauvrete par cluster - {methode} :")
            print(table.round(2))
            csv_path = OUTPUT_DIR / f"poverty_by_cluster_{methode}_{period_label}_k{best_k}.csv"
            table.to_csv(csv_path)
            print(f"Sauvegarde : {csv_path}")

    print(f"\n{'=' * 70}")
    print("Termine. A examiner avant de choisir l'algorithme final :")
    print(f"  - {GRAPH_DIR}/benchmark_scores_period{{1,2}}.png (scores vs k)")
    print(f"  - {GRAPH_DIR}/dendrogram_period{{1,2}}.png (structure CAH)")
    print(f"  - {GRAPH_DIR}/clusters_pca_period{{1,2}}_k*.png (visuel des 3 methodes)")
    print(f"  - {GRAPH_DIR}/poverty_by_cluster_period{{1,2}}_k*.png (lien avec la pauvrete)")
    print("  Choix manuel de l'algorithme le plus coherent (score + visuel + lien")
    print("  avec la pauvrete) avant le clustering final sur TOUS les foyers.")