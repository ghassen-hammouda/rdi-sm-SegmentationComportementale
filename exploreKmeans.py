"""
Export du detail des foyers appartenant au cluster concentre en pauvrete

Ce script relance le clustering K-means (k=3, meme protocole que
cluster_kmeans_elbow.py), identifie le cluster le plus concentre en
pauvrete, puis exporte la liste COMPLETE de ces foyers avec leurs
vraies valeurs (dollars, taille du foyer, region...) plutot que les
colonnes mises a l'echelle - pour permettre un examen manuel, ligne par
ligne, dans Excel.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

FEATURES_DIR = Path("outputs_features")
POVERTY_DIR = Path("outputs_poverty")
OUTPUT_DIR = Path("outputs_clustering")
OUTPUT_DIR.mkdir(exist_ok=True)

FILES_FEATURES = {
    "period1": "features_period1.csv",
    "period2": "features_period2.csv",
}
FILES_POVERTY = {
    "period1": "poverty_dataset_period1.csv",
    "period2": "poverty_dataset_period2.csv",
}

ID_COLUMN = "CUID"
TARGET_COLUMNS = [
    "statut_pauvrete_revenu_brut",
    "statut_pauvrete_revenu_net",
    "statut_pauvrete_consommation",
]

SAMPLE_SIZE = 5000
RANDOM_STATE = 42
FIXED_K = 3
PCA_VARIANCE_TARGET = 0.90

# Colonnes lisibles a exporter pour l'examen manuel (valeurs reelles,
# pas les colonnes mises a l'echelle utilisees pour le calcul).
READABLE_COLUMNS = [
    "CUID", "NB_TRIMESTRES",
    "FINCBTXM", "FINATXEM",
    "TOTEXP_FULL_ANNUEL", "FOOD_FULL_ANNUEL", "HOUS_FULL_ANNUEL", "TRANS_FULL_ANNUEL", "HEALTH_FULL_ANNUEL",
    "FAM_SIZE", "PERSLT18", "PERSOT64", "FAM_TYPE", "AGE_REF", "EDUC_REF", "CUTENURE", "VEHQ",
    "REGION", "BLS_URBN",
    "ECHELLE_EQUIVALENCE", "REVENU_BRUT_EQUIVALISE", "REVENU_NET_EQUIVALISE", "CONSOMMATION_EQUIVALISEE",
    "SEUIL_PAUVRETE",
    "statut_pauvrete_revenu_brut", "statut_pauvrete_revenu_net", "statut_pauvrete_consommation",
]


def find_poor_cluster(period_label: str) -> pd.DataFrame:
    print(f"\n{'=' * 70}\nEXPORT DU CLUSTER PAUVRE - {period_label.upper()}\n{'=' * 70}")

    features_df = pd.read_csv(FEATURES_DIR / FILES_FEATURES[period_label])
    poverty_df = pd.read_csv(POVERTY_DIR / FILES_POVERTY[period_label])

    sample = features_df.sample(n=min(SAMPLE_SIZE, len(features_df)), random_state=RANDOM_STATE).reset_index(drop=True)
    feature_cols = [c for c in sample.columns if c not in [ID_COLUMN] + TARGET_COLUMNS]

    pca_full = PCA(random_state=RANDOM_STATE)
    pca_full.fit(sample[feature_cols])
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.argmax(cumulative >= PCA_VARIANCE_TARGET) + 1)

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X = pca.fit_transform(sample[feature_cols])

    kmeans = KMeans(n_clusters=FIXED_K, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X)
    sample = sample.copy()
    sample["cluster"] = labels

    taux_par_cluster = sample.groupby("cluster")["statut_pauvrete_revenu_net"].mean() * 100
    cluster_pauvre_id = taux_par_cluster.idxmax()
    print(f"Cluster identifie comme le plus pauvre : cluster {cluster_pauvre_id} "
          f"({taux_par_cluster[cluster_pauvre_id]:.2f}% de pauvrete, revenu net)")

    cuids_pauvres = sample.loc[sample["cluster"] == cluster_pauvre_id, ID_COLUMN]
    print(f"Nombre de foyers dans ce cluster : {len(cuids_pauvres)}")

    detail = poverty_df[poverty_df[ID_COLUMN].isin(cuids_pauvres)][READABLE_COLUMNS].copy()
    detail = detail.sort_values("REVENU_NET_EQUIVALISE").reset_index(drop=True)

    output_path = OUTPUT_DIR / f"foyers_cluster_pauvre_{period_label}.csv"
    detail.to_csv(output_path, index=False)
    print(f"Detail exporte : {output_path} ({len(detail)} foyers, {detail.shape[1]} colonnes lisibles)")

    print(f"\nApercu des 5 foyers avec le revenu equivalise le plus bas :")
    print(detail[["CUID", "FINCBTXM", "FINATXEM", "TOTEXP_FULL_ANNUEL", "FAM_SIZE",
                  "REVENU_NET_EQUIVALISE", "CONSOMMATION_EQUIVALISEE"]].head())

    return detail


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    for period_label in FILES_FEATURES:
        find_poor_cluster(period_label)

    print(f"\n{'=' * 70}")
    print("Termine. Fichiers dans outputs_clustering/ :")
    print("  foyers_cluster_pauvre_period1.csv / _period2.csv")
    print("  -> liste complete des foyers du cluster concentre, avec leurs vraies valeurs (dollars, taille, region...)")
    print("  -> a ouvrir dans Excel pour examen manuel, tries du revenu equivalise le plus bas au plus haut")