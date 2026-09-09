"""
Correction par regles (US-15), appliquee EN AVAL du clustering K-Means
sur toute la population (final_clustering_population_v2.py).

Principe methodologique du projet : clustering + regles.
- Le clustering K-Means donne un premier decoupage en categories
  economiques (Tres pauvre -> Riche), trie par niveau de vie global
  (totexp_par_equivalent_adulte).
- Limite connue : ce tri ne capture pas les foyers dont le probleme
  vient de la REPARTITION du budget (charge de logement ecrasante par
  rapport au revenu), pas du niveau de depense global - ces foyers
  atterrissent au milieu du classement alors qu'ils sont en realite
  tres vulnerables (voir Regle 2, deja definie et testee separement).
- Ce script applique la Regle 2 comme CORRECTION explicite, en aval :
  tout foyer qui verifie la regle (charge locative extreme) est
  reclasse en "Tres pauvre", quelle que soit sa categorie d'origine
  issue du seul tri par niveau de vie.

Rien n'est modifie dans le clustering lui-meme - c'est une couche de
correction separee et transparente, tracee explicitement (avant/apres).
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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
POVERTY_TARGETS = ["statut_pauvrete_revenu_net", "statut_pauvrete_consommation"]

# Regle 2 ELARGIE - version finale, seuil SPECIFIQUE PAR PERIODE.
# period1 : seuil 2.0 suffit, resout completement l'anomalie (verifie
# par correction_regle2_period1_k*.png - plus aucun pic anormal).
# period2 : seuil 2.0 laissait 194 foyers non captures (ratio entre
# 1.54 et 2.0, diagnose_period2_v2.py) qui maintenaient la categorie
# "Moyen" a un taux de pauvrete anormal (~79%). Seuil abaisse a 1.5
# pour period2 uniquement, ce qui capture 100% de ce groupe residuel,
# au prix d'un taux de faux positifs mesure a 11.1% (foyers non-pauvres
# mais quand meme en tension logement significative - juge acceptable).
RATIO_SEUIL_PAR_PERIODE = {
    "period1": 2.0,
    "period2": 1.5,
}


def apply_rule2(df: pd.DataFrame, period_label: str) -> pd.Series:
    seuil = RATIO_SEUIL_PAR_PERIODE[period_label]
    return df["ratio_logement_revenu_net"] > seuil


def correct_categories(period_label: str, k: int) -> pd.DataFrame:
    labels_path = INPUT_DIR / f"final_labels_{period_label}_k{k}.csv"
    raw_path = INPUT_DIR / RAW_FILES[period_label]

    df_labels = pd.read_csv(labels_path, low_memory=False)
    df_raw = pd.read_csv(raw_path, low_memory=False)

    df = df_labels.merge(df_raw, on=ID_COLUMN, how="left")

    mask_rule2 = apply_rule2(df, period_label)
    n_corriges = int(mask_rule2.sum())

    df["categorie_avant_correction"] = df["categorie"]
    df.loc[mask_rule2, "categorie"] = "Tres pauvre (charge locative)"

    print(f"\n[{period_label}] k={k} - Foyers corriges par la Regle 2 (charge locative) : "
          f"{n_corriges} ({n_corriges/len(df)*100:.2f}%)")

    if n_corriges > 0:
        print(f"[{period_label}] k={k} - D'ou venaient ces foyers avant correction :")
        avant = df.loc[mask_rule2, "categorie_avant_correction"].value_counts()
        for cat, n in avant.items():
            print(f"    {n} foyers venaient de la categorie '{cat}'")

    return df


def compare_before_after(df: pd.DataFrame, period_label: str, k: int) -> None:
    print(f"\n[{period_label}] k={k} - Taux de pauvrete AVANT correction :")
    avant = df.groupby("categorie_avant_correction")[POVERTY_TARGETS].mean() * 100
    print(avant.round(2))

    print(f"\n[{period_label}] k={k} - Taux de pauvrete APRES correction :")
    apres = df.groupby("categorie")[POVERTY_TARGETS].mean() * 100
    print(apres.round(2))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    avant[POVERTY_TARGETS].plot(kind="bar", ax=axes[0])
    axes[0].set_title(f"AVANT correction - {period_label}, k={k}")
    axes[0].set_ylabel("Taux de pauvrete (%)")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend(["Revenu net", "Consommation"], fontsize=8)

    apres[POVERTY_TARGETS].plot(kind="bar", ax=axes[1])
    axes[1].set_title(f"APRES correction (Regle 2) - {period_label}, k={k}")
    axes[1].set_ylabel("Taux de pauvrete (%)")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend(["Revenu net", "Consommation"], fontsize=8)

    plt.tight_layout()
    path = GRAPH_DIR / f"correction_regle2_{period_label}_k{k}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Graphique sauvegarde : {path}")


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    for period_label in RAW_FILES:
        print(f"\n{'=' * 70}\nCORRECTION PAR REGLES - {period_label.upper()}\n{'=' * 70}")

        for k in K_VALUES:
            df_corrige = correct_categories(period_label, k)
            compare_before_after(df_corrige, period_label, k)

            output_path = OUTPUT_DIR / f"final_labels_corrected_{period_label}_k{k}.csv"
            df_corrige[[ID_COLUMN, "categorie_avant_correction", "categorie"]].to_csv(output_path, index=False)
            print(f"Sauvegarde : {output_path}")

    print(f"\n{'=' * 70}")
    print("Termine. A verifier : la categorie 'Tres pauvre (charge locative)' doit")
    print("desormais avoir un taux de pauvrete-revenu tres eleve et coherent, et les")
    print("categories restantes (Pauvre/Moyen/Aise/Riche) doivent retrouver un vrai")
    print("gradient croissant, sans plus de pic anormal au milieu.")