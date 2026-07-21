"""
Rapport de profiling visuel des datasets pauvrete (poverty_dataset_period*.csv)

Genere un rapport HTML interactif par periode, avec pour chaque colonne :
histogramme, taux de manquant, statistiques descriptives, alertes
automatiques sur les valeurs suspectes (constantes, fortement
correlees, zeros excessifs...), plus une matrice de correlation globale.

Installation prealable (une seule fois) :
    py -m pip install ydata-profiling

Lancement :
    py -u generate_profiling_report.py
"""

import pandas as pd
from pathlib import Path
from ydata_profiling import ProfileReport

INPUT_DIR = Path("outputs_poverty")
OUTPUT_DIR = Path("outputs_profiling")
OUTPUT_DIR.mkdir(exist_ok=True)

FILES = {
    "period1": "poverty_dataset_period1.csv",
    "period2": "poverty_dataset_period2.csv",
}


def generate_report(period_label: str, filename: str) -> None:
    filepath = INPUT_DIR / filename
    if not filepath.exists():
        print(f"[ATTENTION] Fichier introuvable : {filepath}")
        return

    print(f"\nChargement : {filepath}")
    df = pd.read_csv(filepath)

    print(f"Generation du rapport de profiling ({len(df)} lignes, {df.shape[1]} colonnes)...")
    print("Peut prendre 1 a 3 minutes selon la taille du dataset.")

    profile = ProfileReport(
        df,
        title=f"Rapport de profiling - {period_label}",
        explorative=True,  # active les analyses approfondies (correlations, alertes)
        interactions={"continuous": False},  # desactive les nuages de points par paire
        # (avec 70+ colonnes, ca genere des milliers de graphiques et un fichier
        # tres lourd sans reel apport ; les correlations globales suffisent)
    )

    output_path = OUTPUT_DIR / f"profiling_{period_label}.html"
    profile.to_file(output_path)
    print(f"Rapport sauvegarde : {output_path}")


if __name__ == "__main__":
    for period_label, filename in FILES.items():
        generate_report(period_label, filename)

    print("\nOuvrez les fichiers .html generes dans un navigateur pour explorer visuellement chaque colonne.")