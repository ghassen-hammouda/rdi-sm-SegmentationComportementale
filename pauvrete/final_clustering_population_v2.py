"""
Clustering K-Means final sur TOUTE LA POPULATION (US-14, Option A -
etape finale), teste pour plusieurs k (3, 4, 5, 6).

Contrairement au benchmark precedent (echantillon de 5000, pour
comparer les methodes), ce script tourne sur la totalite des foyers
(38439 en period1, 32930 en period2) - fini le probleme des tout
petits clusters (17 foyers sur 5000) : sur la population complete, un
meme profil economique regroupera naturellement beaucoup plus de
foyers, donc des resultats plus robustes.

Pour chaque k teste :
1. K-Means sur tous les foyers
2. Tri des clusters par niveau de vie moyen (totexp_par_equivalent_
   adulte), et attribution d'un LABEL economique (tres pauvre -> riche)
3. Profil complet (moyennes reelles, taux de pauvrete, taille) + graphiques
4. Sauvegarde des labels par foyer, pour la suite (regles d'affinage)

A la fin : comparaison des k entre eux pour choisir celui qui donne le
gradient le plus clair et des groupes de taille raisonnable.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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
K_VALUES_TO_TEST = [3, 4, 5, 6]
RANDOM_STATE = 42
KMEANS_N_INIT = 10  # population complete -> peut se permettre plus d'initialisations
POVERTY_TARGETS = ["statut_pauvrete_revenu_net", "statut_pauvrete_consommation"]
RISQUE_LOGEMENT_SEUIL = 2.5  # coherent avec la Regle 2 deja definie

KEY_CONTINUOUS = [
    "totexp_par_equivalent_adulte", "ratio_logement_revenu_net",
    "part_hous", "part_food", "part_health", "part_retpen",
    "FAM_SIZE", "AGE_REF",
]

LABELS_BY_K = {
    3: ["Pauvre", "Moyen", "Aise"],
    4: ["Tres pauvre", "Pauvre", "Moyen", "Aise"],
    5: ["Tres pauvre", "Pauvre", "Moyen", "Aise", "Riche"],
    6: ["Tres pauvre", "Pauvre", "Moyen inferieur", "Moyen superieur", "Aise", "Riche"],
}


def load_full_population(period_label: str) -> tuple[pd.DataFrame, np.ndarray]:
    features_path = INPUT_DIR / FEATURES_FILES[period_label]
    raw_path = INPUT_DIR / RAW_FILES[period_label]

    df_features = pd.read_csv(features_path, low_memory=False)
    df_raw = pd.read_csv(raw_path, low_memory=False)

    X = df_features.drop(columns=[ID_COLUMN]).values
    df_full = df_features[[ID_COLUMN]].merge(df_raw, on=ID_COLUMN, how="left")

    return df_full, X


def cluster_and_label(df_full: pd.DataFrame, X: np.ndarray, k: int, period_label: str) -> pd.DataFrame:
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=KMEANS_N_INIT)
    raw_labels = model.fit_predict(X)

    df = df_full.copy()
    df["cluster_raw"] = raw_labels

    # Tri des clusters par niveau de vie moyen croissant -> attribution du label economique
    ordre = df.groupby("cluster_raw")["totexp_par_equivalent_adulte"].mean().sort_values().index
    mapping = {cluster_id: LABELS_BY_K[k][rank] for rank, cluster_id in enumerate(ordre)}
    df["categorie"] = df["cluster_raw"].map(mapping)

    # Flag SEPARE "a risque logement" - le niveau de vie global (totexp_par_
    # equivalent_adulte) ne capture pas les foyers dont le desequilibre vient
    # de la REPARTITION du budget (trop de logement par rapport au revenu),
    # pas du niveau de depense global. Ce flag coexiste avec la categorie de
    # niveau de vie, il ne la remplace pas - un foyer peut etre "Moyen" ET
    # "a risque logement" en meme temps.
    df["a_risque_logement"] = df["ratio_logement_revenu_net"] > RISQUE_LOGEMENT_SEUIL

    print(f"\n[{period_label}] k={k} - Repartition des categories :")
    tailles = df["categorie"].value_counts().reindex(LABELS_BY_K[k])
    for cat, n in tailles.items():
        print(f"    {cat} : {n} foyers ({n/len(df)*100:.2f}%)")

    n_risque = int(df["a_risque_logement"].sum())
    print(f"[{period_label}] k={k} - Foyers a risque logement (ratio > {RISQUE_LOGEMENT_SEUIL}) : "
          f"{n_risque} ({n_risque/len(df)*100:.2f}%)")
    print(f"[{period_label}] k={k} - Repartition du risque logement PAR categorie :")
    for cat in LABELS_BY_K[k]:
        sous_ensemble = df.loc[df["categorie"] == cat, "a_risque_logement"]
        if len(sous_ensemble) > 0:
            print(f"    {cat} : {sous_ensemble.sum()} foyers a risque logement "
                  f"({sous_ensemble.mean()*100:.2f}% de cette categorie)")

    return df


def profile_and_plot(df: pd.DataFrame, k: int, period_label: str) -> pd.DataFrame:
    table = df.groupby("categorie")[KEY_CONTINUOUS + POVERTY_TARGETS].mean()
    table["n_foyers"] = df.groupby("categorie").size()
    table = table.reindex(LABELS_BY_K[k])

    csv_path = OUTPUT_DIR / f"final_categories_{period_label}_k{k}.csv"
    table.to_csv(csv_path)
    print(f"Sauvegarde : {csv_path}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    table[POVERTY_TARGETS].plot(kind="bar", ax=axes[0])
    axes[0].set_title(f"Taux de pauvrete par categorie - {period_label}, k={k}")
    axes[0].set_ylabel("Taux de pauvrete (%)" if table[POVERTY_TARGETS].max().max() > 1 else "Proportion")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend(["Revenu net", "Consommation"])

    table["totexp_par_equivalent_adulte"].plot(kind="bar", ax=axes[1], color="darkgreen")
    axes[1].set_title(f"Niveau de vie par categorie - {period_label}, k={k}")
    axes[1].set_ylabel("$ / equivalent-adulte")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    graph_path = GRAPH_DIR / f"final_categories_{period_label}_k{k}.png"
    plt.savefig(graph_path, dpi=150)
    plt.close()
    print(f"Graphique sauvegarde : {graph_path}")

    return table


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    all_tables = {}

    for period_label in FEATURES_FILES:
        print(f"\n{'=' * 70}\nCLUSTERING FINAL (POPULATION COMPLETE) - {period_label.upper()}\n{'=' * 70}")

        df_full, X = load_full_population(period_label)
        print(f"Population complete chargee : {len(df_full)} foyers")

        for k in K_VALUES_TO_TEST:
            df_labeled = cluster_and_label(df_full, X, k, period_label)
            table = profile_and_plot(df_labeled, k, period_label)
            all_tables[(period_label, k)] = table

            # Sauvegarde des labels foyer par foyer, pour la suite (regles)
            labels_path = OUTPUT_DIR / f"final_labels_{period_label}_k{k}.csv"
            df_labeled[[ID_COLUMN, "categorie", "a_risque_logement"]].to_csv(labels_path, index=False)

    print(f"\n{'=' * 70}")
    print("Termine. Pour choisir le meilleur k, comparer :")
    print("  1. La taille du plus petit groupe (doit rester significative, pas 17 foyers)")
    print("  2. La clarte du gradient de pauvrete entre categories (doit croitre/decroitre")
    print("     regulierement, pas de saut bizarre)")
    print("  3. La lisibilite pour la presentation (5-6 categories max recommande)")
    print(f"\nFichiers : final_categories_period{{1,2}}_k{{3,4,5,6}}.csv")
    print(f"Graphiques : {GRAPH_DIR}/final_categories_period{{1,2}}_k{{3,4,5,6}}.png")