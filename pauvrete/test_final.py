"""
Test de robustesse (US-18) des categories finales (K-Means + correction
par regles), pour les 4 versions de k (3, 4, 5, 6), dans les deux
periodes.

Principe : le clustering final a ete fait sur TOUTE la population -
il n'y a donc pas d'echantillon "jamais vu" au sens strict pour tester
la generalisation classique. A la place, on teste la ROBUSTESSE du
resultat : si on divise la population en 2 moities aleatoires
independantes, est-ce que chaque categorie garde un taux de pauvrete
coherent dans les deux moities ? Si oui, le resultat n'est pas cause
par quelques foyers particuliers, c'est un vrai signal stable dans les
donnees.

Pour chaque (periode, k) : split 50/50 aleatoire, calcul du taux de
pauvrete par categorie dans chaque moitie, comparaison cote a cote +
graphique. Un ecart de plus de 10 points entre les deux moities pour
une categorie est signale comme instable.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

INPUT_DIR = Path("outputs_v2")
OUTPUT_DIR = Path("outputs_v2")
GRAPH_DIR = OUTPUT_DIR / "graphs"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

RAW_FILES = {
    "period1": "poverty_targets_period1.csv",
    "period2": "poverty_targets_period2.csv",
}
ID_COLUMN = "CUID"
K_VALUES = [3, 4, 5, 6]
RANDOM_STATE = 42
POVERTY_TARGETS = ["statut_pauvrete_revenu_net", "statut_pauvrete_consommation"]
SEUIL_ALERTE_ECART = 10.0  # points de %, au-dela -> instable


def test_robustesse(period_label: str, k: int) -> None:
    corrected_path = INPUT_DIR / f"final_labels_corrected_{period_label}_k{k}.csv"
    raw_path = INPUT_DIR / RAW_FILES[period_label]

    df_corrected = pd.read_csv(corrected_path, low_memory=False)
    df_raw = pd.read_csv(raw_path, low_memory=False)
    df = df_corrected.merge(df_raw[[ID_COLUMN] + POVERTY_TARGETS], on=ID_COLUMN, how="left")

    rng = np.random.RandomState(RANDOM_STATE)
    split_mask = rng.rand(len(df)) < 0.5
    moitie_1 = df[split_mask]
    moitie_2 = df[~split_mask]

    print(f"\n[{period_label}] k={k} - Split : {len(moitie_1)} foyers (moitie 1), "
          f"{len(moitie_2)} foyers (moitie 2)")

    taux_1 = moitie_1.groupby("categorie")[POVERTY_TARGETS].mean() * 100
    taux_2 = moitie_2.groupby("categorie")[POVERTY_TARGETS].mean() * 100

    comparaison = pd.DataFrame({
        "Moitie1_revenu_net": taux_1["statut_pauvrete_revenu_net"],
        "Moitie2_revenu_net": taux_2["statut_pauvrete_revenu_net"],
    })
    comparaison["ecart"] = (comparaison["Moitie1_revenu_net"] - comparaison["Moitie2_revenu_net"]).abs()
    comparaison["stable"] = comparaison["ecart"] <= SEUIL_ALERTE_ECART

    print(comparaison.round(2))

    instables = comparaison[~comparaison["stable"]]
    if len(instables) > 0:
        print(f"[ATTENTION] Categories instables (ecart > {SEUIL_ALERTE_ECART} points) : "
              f"{list(instables.index)}")
    else:
        print(f"Toutes les categories sont stables (ecart <= {SEUIL_ALERTE_ECART} points).")

    fig, ax = plt.subplots(figsize=(8, 5))
    comparaison[["Moitie1_revenu_net", "Moitie2_revenu_net"]].plot(kind="bar", ax=ax)
    ax.set_title(f"Robustesse : taux de pauvrete par categorie, 2 moities independantes\n{period_label}, k={k}")
    ax.set_ylabel("Taux de pauvrete-revenu-net (%)")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    path = GRAPH_DIR / f"robustesse_{period_label}_k{k}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Graphique sauvegarde : {path}")

    csv_path = OUTPUT_DIR / f"robustesse_{period_label}_k{k}.csv"
    comparaison.to_csv(csv_path)


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    for period_label in RAW_FILES:
        print(f"\n{'=' * 70}\nTEST DE ROBUSTESSE - {period_label.upper()}\n{'=' * 70}")
        for k in K_VALUES:
            test_robustesse(period_label, k)

    print(f"\n{'=' * 70}")
    print("Termine. Si toutes les categories sont stables (ecart <= 10 points)")
    print("entre les deux moities aleatoires, le resultat final est robuste,")
    print("pas du a un artefact de quelques foyers particuliers.")