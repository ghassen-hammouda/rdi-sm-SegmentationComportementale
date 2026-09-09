"""
Consolidation des effectifs (US-20) - regenere un tableau propre et
fiable des effectifs par categorie, AVANT et APRES la correction par
regles, pour les 4 valeurs de k et les 2 periodes.

Lit directement final_labels_corrected_period{1,2}_k{3,4,5,6}.csv, qui
contient deja categorie_avant_correction ET categorie (finale) - pas
besoin de recalculer ou de reconstituer a la main.
"""

import pandas as pd
from pathlib import Path

INPUT_DIR = Path("outputs_v2")
OUTPUT_DIR = Path("outputs_v2")

K_VALUES = [3, 4, 5, 6]
PERIODS = ["period1", "period2"]

# Ordre d'affichage souhaite (les categories "de base" dans l'ordre
# croissant, la categorie de correction toujours en dernier)
ORDRE_CATEGORIES = {
    3: ["Tres pauvre", "Pauvre", "Moyen", "Aise", "Riche"],
    4: ["Tres pauvre", "Pauvre", "Moyen", "Aise", "Riche"],
    5: ["Tres pauvre", "Pauvre", "Moyen", "Aise", "Riche"],
    6: ["Tres pauvre", "Pauvre", "Moyen inferieur", "Moyen superieur", "Aise", "Riche"],
}


def build_summary(period_label: str, k: int) -> pd.DataFrame:
    path = INPUT_DIR / f"final_labels_corrected_{period_label}_k{k}.csv"
    df = pd.read_csv(path, low_memory=False)

    total = len(df)

    avant = df["categorie_avant_correction"].value_counts()
    apres = df["categorie"].value_counts()

    toutes_categories = sorted(set(avant.index) | set(apres.index),
                                key=lambda c: (c not in ORDRE_CATEGORIES.get(k, []),
                                               ORDRE_CATEGORIES.get(k, []).index(c)
                                               if c in ORDRE_CATEGORIES.get(k, []) else 999,
                                               c))

    rows = []
    for cat in toutes_categories:
        n_avant = int(avant.get(cat, 0))
        n_apres = int(apres.get(cat, 0))
        rows.append({
            "periode": period_label,
            "k": k,
            "categorie": cat,
            "foyers_avant": n_avant,
            "pct_avant": round(n_avant / total * 100, 2),
            "foyers_apres": n_apres,
            "pct_apres": round(n_apres / total * 100, 2),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    tous_les_tableaux = []

    for period_label in PERIODS:
        print(f"\n{'=' * 90}\n{period_label.upper()}\n{'=' * 90}")
        for k in K_VALUES:
            table = build_summary(period_label, k)
            tous_les_tableaux.append(table)

            print(f"\n--- k={k} ---")
            print(table[["categorie", "foyers_avant", "pct_avant", "foyers_apres", "pct_apres"]]
                  .to_string(index=False))

    resume_complet = pd.concat(tous_les_tableaux, ignore_index=True)
    output_path = OUTPUT_DIR / "resume_effectifs_complet.csv"
    resume_complet.to_csv(output_path, index=False)

    print(f"\n{'=' * 90}")
    print(f"Termine. Tableau complet et fiable sauvegarde : {output_path}")
    print("Colonnes : periode, k, categorie, foyers_avant, pct_avant, foyers_apres, pct_apres")