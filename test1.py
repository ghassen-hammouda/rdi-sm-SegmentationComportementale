"""
Test de stabilite du clustering K-means sur plusieurs echantillons

Ce script relance le clustering K-means (k=3) sur PLUSIEURS echantillons
aleatoires differents (pas juste un seul), pour verifier que le cluster
concentre en pauvrete ressort de facon coherente d'un echantillon a
l'autre - plutot que d'etre un resultat qui ne tient qu'a un seul tirage
aleatoire particulier.

Principe : si le taux de pauvrete du cluster le plus concentre varie
enormement d'un echantillon a l'autre (ex: 92% une fois, 40% une autre
fois), le resultat n'est pas fiable. S'il reste stable (ex: toujours
entre 85% et 95%), c'est un signal robuste, pas un coup de chance.
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

SAMPLE_SIZE = 5000
FIXED_K = 3
PCA_VARIANCE_TARGET = 0.90

# Plusieurs graines aleatoires differentes = plusieurs echantillons
# differents tires du meme fichier. 42 est la graine utilisee dans tous
# les scripts precedents ; les 4 autres sont nouvelles, pour verifier
# que le resultat ne depend pas d'un tirage particulier.
RANDOM_SEEDS = [42, 7, 123, 2024, 99]


def run_one_sample(df_full: pd.DataFrame, seed: int) -> dict:
    """Tire un echantillon avec une graine donnee, applique PCA + KMeans
    (k=3), et renvoie les caracteristiques du cluster le plus concentre
    en pauvrete (selon le revenu net).
    """
    sample = df_full.sample(n=min(SAMPLE_SIZE, len(df_full)), random_state=seed).reset_index(drop=True)

    feature_cols = [c for c in sample.columns if c not in [ID_COLUMN] + TARGET_COLUMNS]

    pca_full = PCA(random_state=seed)
    pca_full.fit(sample[feature_cols])
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.argmax(cumulative >= PCA_VARIANCE_TARGET) + 1)

    pca = PCA(n_components=n_components, random_state=seed)
    X = pca.fit_transform(sample[feature_cols])

    kmeans = KMeans(n_clusters=FIXED_K, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(X)
    silhouette = silhouette_score(X, labels)

    sample = sample.copy()
    sample["cluster"] = labels

    # Identifie le cluster avec le plus haut taux de pauvrete (revenu net)
    taux_par_cluster = sample.groupby("cluster")["statut_pauvrete_revenu_net"].mean() * 100
    cluster_pauvre_id = taux_par_cluster.idxmax()
    cluster_pauvre = sample[sample["cluster"] == cluster_pauvre_id]

    return {
        "graine": seed,
        "n_composantes_pca": n_components,
        "silhouette": round(silhouette, 4),
        "taille_cluster_pauvre": len(cluster_pauvre),
        "pct_echantillon": round(len(cluster_pauvre) / len(sample) * 100, 2),
        "taux_pauvrete_brut": round(cluster_pauvre["statut_pauvrete_revenu_brut"].mean() * 100, 2),
        "taux_pauvrete_net": round(cluster_pauvre["statut_pauvrete_revenu_net"].mean() * 100, 2),
        "taux_pauvrete_conso": round(cluster_pauvre["statut_pauvrete_consommation"].mean() * 100, 2),
    }


def test_stability_for_period(period_label: str, filename: str) -> pd.DataFrame:
    print(f"\n{'=' * 70}\nTEST DE STABILITE - {period_label.upper()}\n{'=' * 70}")

    df_full = pd.read_csv(INPUT_DIR / filename)
    print(f"Fichier charge : {len(df_full)} foyers au total.")
    print(f"Test sur {len(RANDOM_SEEDS)} echantillons differents de {SAMPLE_SIZE} foyers chacun : {RANDOM_SEEDS}\n")

    rows = [run_one_sample(df_full, seed) for seed in RANDOM_SEEDS]
    results = pd.DataFrame(rows)
    print(results)

    print(f"\n--- Statistiques de stabilite ({period_label}) ---")
    print(f"Taux de pauvrete (revenu net) du cluster concentre :")
    print(f"  Moyenne sur les {len(RANDOM_SEEDS)} echantillons : {results['taux_pauvrete_net'].mean():.2f}%")
    print(f"  Ecart-type                                       : {results['taux_pauvrete_net'].std():.2f} points")
    print(f"  Min - Max                                        : {results['taux_pauvrete_net'].min():.2f}% - {results['taux_pauvrete_net'].max():.2f}%")

    ecart_type = results["taux_pauvrete_net"].std()
    if ecart_type < 5:
        verdict = "STABLE : le resultat varie peu d'un echantillon a l'autre, signal fiable."
    elif ecart_type < 15:
        verdict = "MODEREMENT STABLE : variation notable, a mentionner comme limite."
    else:
        verdict = "INSTABLE : le resultat depend fortement de l'echantillon tire, prudence requise."
    print(f"  Verdict : {verdict}")

    results.to_csv(OUTPUT_DIR / f"stability_test_{period_label}.csv", index=False)
    return results


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    for period_label, filename in FILES.items():
        test_stability_for_period(period_label, filename)

    print(f"\n{'=' * 70}")
    print("Termine. Fichiers dans outputs_clustering/ : stability_test_period*.csv")