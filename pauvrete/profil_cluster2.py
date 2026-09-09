"""
Profilage des clusters K-Means (US-11), pour comprendre CE QUE les
clusters representent concretement - pas seulement leur score.

Pour chaque k teste (2 et 4 par defaut, voir K_VALUES) :
1. Reentraine K-Means sur l'espace standardise (clustering_features_
   period{1,2}.csv), avec les memes parametres que le benchmark
2. Reassocie chaque foyer a son cluster ET a ses valeurs REELLES
   (non standardisees, issues de poverty_targets_period{1,2}.csv) -
   un score standardise n'est pas lisible, une moyenne en dollars ou
   en % de budget l'est
3. Affiche, par cluster : la taille, la moyenne des variables continues
   (parts de budget, ratio logement/revenu, demographie...), la
   repartition des variables categorielles (REGION, FAM_TYPE,
   CUTENURE), ET le taux de pauvrete selon statut_pauvrete_revenu_net
   et statut_pauvrete_consommation (issus de clustering_reserved_
   period{1,2}.csv - JAMAIS vus par le clustering, uniquement utilises
   ici pour la lecture des resultats)

Objectif : verifier si le clivage trouve par l'algorithme correspond a
un vrai profil de pauvrete, ou a autre chose (ex. proprietaire vs
locataire, region...) qui n'a rien a voir avec la pauvrete.
"""

import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans

INPUT_DIR = Path("outputs_v2")
OUTPUT_DIR = Path("outputs_v2")

FEATURES_FILES = {
    "period1": "clustering_features_period1.csv",
    "period2": "clustering_features_period2.csv",
}
RAW_FILES = {
    "period1": "poverty_targets_period1.csv",
    "period2": "poverty_targets_period2.csv",
}
RESERVED_FILES = {
    "period1": "clustering_reserved_period1.csv",
    "period2": "clustering_reserved_period2.csv",
}

ID_COLUMN = "CUID"
K_VALUES = [2, 4]
RANDOM_STATE = 42
KMEANS_N_INIT = 2  # coherent avec benchmark_clustering_v2.py

CONTINUOUS_TO_PROFILE = [
    "part_food", "part_hous", "part_trans", "part_health", "part_appar",
    "part_entert", "part_educa", "part_alcbev", "part_tobacc", "part_persca",
    "part_read", "part_misc", "part_cashco", "part_lifins", "part_retpen",
    "totexp_par_equivalent_adulte", "ratio_logement_revenu_net",
    "FAM_SIZE", "PERSLT18", "PERSOT64", "AGE_REF", "VEHQ", "EDUC_REF",
]
CATEGORICAL_TO_PROFILE = ["REGION", "FAM_TYPE", "CUTENURE", "BLS_URBN"]
POVERTY_TARGETS = ["statut_pauvrete_revenu_net", "statut_pauvrete_consommation"]


def fit_kmeans_labels(X: pd.DataFrame, k: int) -> pd.Series:
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=KMEANS_N_INIT)
    labels = model.fit_predict(X)
    return pd.Series(labels, index=X.index, name="cluster")


def profile_clusters(df_labeled: pd.DataFrame, period_label: str, k: int) -> pd.DataFrame:
    print(f"\n--- Taille des clusters (k={k}) ---")
    tailles = df_labeled["cluster"].value_counts().sort_index()
    for cluster_id, n in tailles.items():
        print(f"  Cluster {cluster_id} : {n} foyers ({n/len(df_labeled)*100:.1f}%)")

    print(f"\n--- Moyennes des variables continues, par cluster (valeurs REELLES) ---")
    profil_continu = df_labeled.groupby("cluster")[CONTINUOUS_TO_PROFILE].mean().T
    print(profil_continu.round(3))

    print(f"\n--- Categorie la plus frequente, par cluster ---")
    for col in CATEGORICAL_TO_PROFILE:
        if col not in df_labeled.columns:
            continue
        print(f"\n  {col} :")
        repartition = df_labeled.groupby("cluster")[col].agg(
            lambda s: f"{s.mode().iloc[0]} ({(s == s.mode().iloc[0]).mean()*100:.1f}%)"
        )
        for cluster_id, val in repartition.items():
            print(f"    Cluster {cluster_id} : {val}")

    print(f"\n--- Taux de pauvrete par cluster (variables JAMAIS vues par le clustering) ---")
    profil_pauvrete = df_labeled.groupby("cluster")[POVERTY_TARGETS].mean() * 100
    print(profil_pauvrete.round(2))

    return profil_continu


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    for period_label in FEATURES_FILES:
        print(f"\n{'=' * 70}\nPROFILAGE DES CLUSTERS - {period_label.upper()}\n{'=' * 70}")

        features_path = INPUT_DIR / FEATURES_FILES[period_label]
        raw_path = INPUT_DIR / RAW_FILES[period_label]
        reserved_path = INPUT_DIR / RESERVED_FILES[period_label]

        df_features = pd.read_csv(features_path, low_memory=False)
        df_raw = pd.read_csv(raw_path, low_memory=False)
        df_reserved = pd.read_csv(reserved_path, low_memory=False)
        print(f"Charge : {features_path} ({len(df_features)} foyers, espace standardise)")

        X = df_features.drop(columns=[ID_COLUMN])

        for k in K_VALUES:
            print(f"\n{'-' * 70}\nK-Means k={k} - {period_label.upper()}\n{'-' * 70}")

            labels = fit_kmeans_labels(X, k)

            df_labeled = df_features[[ID_COLUMN]].copy()
            df_labeled["cluster"] = labels.values

            # Reassociation aux valeurs REELLES (non standardisees) + aux cibles de pauvrete
            df_labeled = df_labeled.merge(df_raw, on=ID_COLUMN, how="left")
            df_labeled = df_labeled.merge(
                df_reserved[[ID_COLUMN] + POVERTY_TARGETS], on=ID_COLUMN, how="left", suffixes=("", "_reserved")
            )

            profile_clusters(df_labeled, period_label, k)

            output_path = OUTPUT_DIR / f"clusters_labeled_{period_label}_k{k}.csv"
            df_labeled[[ID_COLUMN, "cluster"]].to_csv(output_path, index=False)
            print(f"\nSauvegarde des labels : {output_path}")

    print(f"\n{'=' * 70}")
    print("Termine. Fichiers dans outputs_v2/ :")
    print("  clusters_labeled_period{1,2}_k{2,4}.csv -> CUID + cluster assigne")
    print("  A examiner : est-ce que le clivage trouve correspond a un vrai profil")
    print("  de pauvrete (ecart de taux de pauvrete marque entre clusters), ou a")
    print("  autre chose (proprietaire/locataire, region...) sans rapport direct ?")