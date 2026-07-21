"""
Verification de l'existence des colonnes candidates pour la modelisation
(US-06 / US-08), AVANT toute construction de features ou de cible.

Ne charge que les en-tetes des fichiers nettoyes (rapide), pas les
donnees completes. A lancer une seule fois, pour figer la liste reelle
des colonnes disponibles avant de continuer.
"""

import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("outputs_clean")

# Fichiers nettoyes les plus recents (adapter le suffixe de date si besoin)
FILES = {
    "Interview P1": "interview_period1_clean_v20260719.csv",
    "Interview P2": "interview_period2_clean_v20260719.csv",
}

# Mots-cles a rechercher, groupes par categorie. Bases sur les noms de
# variables agregees connues du dictionnaire CEX (a confirmer une fois
# les vraies colonnes obtenues, certains noms exacts peuvent varier).
CANDIDATE_KEYWORDS = {
    "Transport": ["TRANS"],
    "Sante": ["HEALTH"],
    "Habillement": ["APPAR"],
    "Loisirs / divertissement": ["ENTERT"],
    "Education": ["EDUCA"],
    "Alcool": ["ALCB", "ALCOHOL"],
    "Tabac": ["TOBAC"],
    "Soins personnels": ["PERSCA"],
    "Lecture": ["READ"],
    "Divers": ["MISC"],
    "Dons / contributions": ["CASHCO"],
    "Assurance-vie / pension": ["LIFINS", "PENS", "RETPEN"],
    "Taille du foyer": ["FAM_SIZE", "FAMSIZE"],
    "Personnes < 18 ans": ["PERSLT18"],
    "Personnes 65+ ans": ["PERSOT64"],
    "Type de foyer": ["FAM_TYPE"],
    "Age personne de reference": ["AGE_REF", "AGEREF"],
    "Education personne de reference": ["EDUC_REF", "EDUCREF"],
    "Statut d'occupation logement (proprietaire/locataire)": ["CUTENURE", "TYPOWND"],
    "Nombre de vehicules": ["VEHQ"],
}


def check_columns(filepath: Path) -> set:
    """Lit uniquement l'en-tete du fichier (nrows=0), retourne l'ensemble des colonnes."""
    if not filepath.exists():
        print(f"[ATTENTION] Fichier introuvable : {filepath}")
        return set()
    return set(pd.read_csv(filepath, nrows=0).columns)


if __name__ == "__main__":
    columns_by_file = {label: check_columns(OUTPUT_DIR / fname) for label, fname in FILES.items()}

    for label, cols in columns_by_file.items():
        print(f"{label} : {len(cols)} colonnes chargees" if cols else f"{label} : non trouve")

    print("\n" + "=" * 70)
    print("RECHERCHE DES COLONNES CANDIDATES PAR CATEGORIE")
    print("=" * 70)

    for category, keywords in CANDIDATE_KEYWORDS.items():
        print(f"\n--- {category} ---")
        for label, cols in columns_by_file.items():
            matches = sorted([c for c in cols if any(kw in c.upper() for kw in keywords)])
            print(f"  {label} : {matches if matches else 'AUCUNE COLONNE TROUVEE'}")