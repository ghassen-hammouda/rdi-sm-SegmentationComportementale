"""
Determination objective du k optimal (US-13, Option A), via l'ecart
entre hauteurs de fusion du dendrogramme CAH.

Principe : les dernieres fusions du dendrogramme (les plus hautes)
representent les regroupements les plus "couteux" - fusionner des
groupes deja tres differents entre eux. Le plus grand ECART entre deux
hauteurs de fusion consecutives indique la coupure la plus naturelle :
juste en dessous de cet ecart, on a des groupes coherents ; juste au
dessus, on force la fusion de groupes qui n'ont plus grand-chose en
commun.

Methode : sur les N dernieres fusions du linkage CAH (echantillon de
5000 foyers), calculer l'ecart entre chaque paire de hauteurs
consecutives, et proposer k = nombre de groupes juste avant le plus
grand ecart.

Ce k sert ensuite de base a la version finale (K-Means sur toute la
population), dans un script separe.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage

INPUT_DIR = Path("outputs_v2")
OUTPUT_DIR = Path("outputs_v2")
GRAPH_DIR = OUTPUT_DIR / "graphs"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_FILES = {
    "period1": "clustering_features_period1.csv",
    "period2": "clustering_features_period2.csv",
}

ID_COLUMN = "CUID"
SAMPLE_SIZE = 5000
RANDOM_STATE = 42
N_LAST_MERGES = 15  # nombre de dernieres fusions a examiner


def load_sample(period_label: str) -> np.ndarray:
    features_path = INPUT_DIR / FEATURES_FILES[period_label]
    df_features = pd.read_csv(features_path, low_memory=False)
    n = min(SAMPLE_SIZE, len(df_features))
    sample = df_features.sample(n=n, random_state=RANDOM_STATE).reset_index(drop=True)
    return sample.drop(columns=[ID_COLUMN]).values


def find_optimal_k(Z: np.ndarray, period_label: str) -> int:
    n_total_merges = Z.shape[0]
    last_heights = Z[-N_LAST_MERGES:, 2]  # hauteurs des N dernieres fusions, ordre croissant

    # Ecarts entre hauteurs consecutives
    gaps = np.diff(last_heights)

    # IMPORTANT : la toute derniere fusion (qui donne k=2) est TOUJOURS la
    # plus haute par construction du dendrogramme (elle reunit les 2 derniers
    # gros blocs, mecaniquement tres differents). L'ecart menant a k=2 est
    # donc artificiellement gonfle et biaise systematiquement vers k=2,
    # sans rapport avec une vraie structure economique. On l'exclut de la
    # recherche du plus grand ecart.
    gaps_sans_racine = gaps[:-1]
    idx_max_gap = np.argmax(gaps_sans_racine)

    n_points = n_total_merges + 1
    merge_rank_before_gap = n_total_merges - N_LAST_MERGES + idx_max_gap + 1
    optimal_k = n_points - merge_rank_before_gap

    print(f"[{period_label}] {N_LAST_MERGES} dernieres hauteurs de fusion : {last_heights.round(1)}")
    print(f"[{period_label}] Ecarts entre fusions consecutives : {gaps.round(1)}")
    print(f"[{period_label}] (Ecart vers k=2 exclu de la recherche : {gaps[-1]:.1f} - "
          f"toujours artificiellement le plus grand par construction)")
    print(f"[{period_label}] Plus grand ecart (hors k=2) trouve entre les hauteurs "
          f"{last_heights[idx_max_gap]:.1f} et {last_heights[idx_max_gap + 1]:.1f}")
    print(f"[{period_label}] k optimal propose : {optimal_k}")

    return optimal_k, last_heights, gaps


def plot_gap_analysis(last_heights: np.ndarray, gaps: np.ndarray, optimal_k: int, period_label: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    n_points_plotted = len(last_heights)
    k_values = list(range(n_points_plotted, 0, -1))  # nb de clusters a chaque etape

    axes[0].plot(k_values, last_heights, marker="o")
    axes[0].axvline(optimal_k, color="red", linestyle="--", label=f"k optimal = {optimal_k}")
    axes[0].set_xlabel("Nombre de clusters (k)")
    axes[0].set_ylabel("Hauteur de fusion")
    axes[0].set_title(f"Hauteur de fusion vs k - {period_label}")
    axes[0].invert_xaxis()
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    gap_k_values = k_values[:-1]
    axes[1].bar([str(k) for k in gap_k_values], gaps)
    axes[1].set_xlabel("Nombre de clusters (k), cote gauche de l'ecart")
    axes[1].set_ylabel("Ecart avec la fusion suivante")
    axes[1].set_title(f"Ecarts entre fusions - {period_label}")
    axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    path = GRAPH_DIR / f"optimal_k_gap_{period_label}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Graphique sauvegarde : {path}")


if __name__ == "__main__":
    results = {}
    for period_label in FEATURES_FILES:
        print(f"\n{'=' * 70}\nDETERMINATION DU K OPTIMAL - {period_label.upper()}\n{'=' * 70}")

        X = load_sample(period_label)
        print(f"Echantillon charge : {len(X)} foyers")

        Z = linkage(X, method="ward")
        optimal_k, last_heights, gaps = find_optimal_k(Z, period_label)
        plot_gap_analysis(last_heights, gaps, optimal_k, period_label)

        results[period_label] = optimal_k

    print(f"\n{'=' * 70}")
    print("Termine. k optimal propose par periode :")
    for period_label, k in results.items():
        print(f"  {period_label} : k = {k}")
    print(f"\nGraphiques : {GRAPH_DIR}/optimal_k_gap_period{{1,2}}.png")
    print("A EXAMINER : le k propose est une suggestion objective, mais reste a valider")
    print("par le profilage (est-ce que ce k donne des categories economiques lisibles ?).")