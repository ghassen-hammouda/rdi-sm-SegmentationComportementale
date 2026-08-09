"""
Chargement des donnees Interview (FMLI) avec le PERIMETRE DE COLONNES
REVALIDE (repris de zero, une famille a la fois) :

Famille 1 - Identifiants/poids : CUID, NEWID, FINLWT21
Famille 2 - Revenu             : FINATXEM (revenu net uniquement)
Famille 2bis - Epargne          : LIQUIDX, LIQUDYRX, IRAX, IRAYRX
Famille 3 - Depenses par poste   : 16 postes x CQ/PQ (32 colonnes)
Famille 4 - Demographie          : FAM_SIZE, PERSLT18, PERSOT64, AGE_REF,
                                    VEHQ, FAM_TYPE, CUTENURE, REGION,
                                    BLS_URBN, EDUC_REF
Famille 5 - Technique             : STATE

Ce script ne fait QUE charger et selectionner ces colonnes - aucun
nettoyage, aucune imputation, aucun calcul ici. Objectif : produire deux
CSV de depart propres (un par periode), prets pour une EDA et un
nettoyage repris de zero sur ce nouveau perimetre.
"""

import pandas as pd
from pathlib import Path

DATA_ROOT = Path("data")
OUTPUT_DIR = Path("outputs_v2")
OUTPUT_DIR.mkdir(exist_ok=True)

PERIOD_1_YEARS = ["2017", "2018", "2019", "2020"]
# 2024 exclue pour l'instant : changement structurel du questionnaire BLS
# en cours d'annee (transition methodologique, cf. "Gemini Project"),
# FINATXEM et ses composantes fiscales absentes sur 5 des 17 fichiers
# 2024 (fmli241x, 242, 243, 244, 251), soit 28% des lignes de la periode.
# A reevaluer plus tard si une solution est trouvee.
PERIOD_2_YEARS = ["2021", "2022", "2023", "2024"]

# ---------------------------------------------------------------------------
# LE PERIMETRE DE COLONNES REVALIDE, FAMILLE PAR FAMILLE
# ---------------------------------------------------------------------------

FAMILLE_1_IDENTIFIANTS = ["NEWID", "CUID", "FINLWT21"]

# Les deux versions du revenu gardees en parallele (decision du jour) :
# FINATXEM (net, la reference methodologique la plus solide) et FINCBTXM
# (brut, filet de securite / comparaison), en attendant de trancher.
FAMILLE_2_REVENU = ["FINATXEM", "FINCBTXM"]

FAMILLE_2BIS_EPARGNE = ["LIQUIDX", "LIQUDYRX", "IRAX", "IRAYRX"]

# FOODCQ/FOODPQ existent bien sur toute la periode retenue (2017-2020 et
# 2021-2023, verifie explicitement) : le detour par FDHOME/FDAWAY
# n'etait necessaire que pour les 5 fichiers 2024 desormais exclus.
EXPENDITURE_BASES = [
    "TOTEXP", "FOOD", "HOUS", "TRANS", "HEALTH", "APPAR", "ENTERT",
    "EDUCA", "ALCBEV", "TOBACC", "PERSCA", "READ", "MISC", "CASHCO",
    "LIFINS", "RETPEN",
]
FAMILLE_3_DEPENSES = [f"{base}{suffixe}" for base in EXPENDITURE_BASES for suffixe in ("CQ", "PQ")]

FAMILLE_4_DEMOGRAPHIE = [
    "FAM_SIZE", "PERSLT18", "PERSOT64", "AGE_REF", "VEHQ",
    "FAM_TYPE", "CUTENURE", "REGION", "BLS_URBN", "EDUC_REF",
]

FAMILLE_5_TECHNIQUE = ["STATE"]

COLONNES_RETENUES = (
    FAMILLE_1_IDENTIFIANTS + FAMILLE_2_REVENU + FAMILLE_2BIS_EPARGNE
    + FAMILLE_3_DEPENSES + FAMILLE_4_DEMOGRAPHIE + FAMILLE_5_TECHNIQUE
)


# ---------------------------------------------------------------------------
# CHARGEMENT (meme logique de recherche de fichiers que le reste du projet)
# ---------------------------------------------------------------------------

def find_survey_files(year_folder: Path) -> list[Path]:
    """Trouve tous les fichiers fmli*.csv dans un dossier, insensible a la casse."""
    pattern = "[Ff][Mm][Ll][Ii]*.csv"
    return sorted(year_folder.glob(pattern))


def load_selected_columns_for_years(years: list[str], data_root: Path = DATA_ROOT) -> pd.DataFrame:
    """Charge les fichiers Interview bruts pour une liste d'annees, en ne
    gardant QUE les colonnes du perimetre revalide (COLONNES_RETENUES).
    Si une colonne attendue manque dans un fichier (schema instable
    d'une annee a l'autre, deja documente en EDA), elle est simplement
    absente pour ce fichier - signalee, pas une erreur bloquante.
    """
    frames = []
    for year in years:
        year_2digits = year[-2:]
        year_folder = data_root / f"intrvw{year_2digits}"
        if not year_folder.exists():
            print(f"[ATTENTION] Dossier introuvable, annee ignoree : {year_folder}")
            continue

        files = find_survey_files(year_folder)
        if not files:
            print(f"[ATTENTION] Aucun fichier fmli*.csv trouve dans {year_folder}")
            continue

        for f in files:
            try:
                df = pd.read_csv(f, low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(f, low_memory=False, encoding="latin-1")

            colonnes_presentes = [c for c in COLONNES_RETENUES if c in df.columns]
            colonnes_absentes = [c for c in COLONNES_RETENUES if c not in df.columns]
            if colonnes_absentes:
                print(f"  [ATTENTION] {f.name} : {len(colonnes_absentes)} colonne(s) du perimetre absente(s) "
                      f"({colonnes_absentes})")

            df_selection = df[colonnes_presentes].copy()
            df_selection["SOURCE_YEAR"] = year
            df_selection["SOURCE_FILE"] = f.name
            frames.append(df_selection)
            print(f"Charge : {f}  ({len(df_selection)} lignes, {len(colonnes_presentes)}/{len(COLONNES_RETENUES)} "
                  f"colonnes du perimetre retenues)")

    if not frames:
        raise FileNotFoundError("Aucun fichier charge. Verifiez la structure du dossier 'data/'.")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    print(f"\nTotal charge : {len(combined)} lignes, {combined.shape[1]} colonnes "
          f"(sur {len(COLONNES_RETENUES) + 2} attendues, +2 pour SOURCE_YEAR/SOURCE_FILE)")
    return combined


# ---------------------------------------------------------------------------
# EXECUTION PRINCIPALE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Perimetre retenu : {len(COLONNES_RETENUES)} colonnes")
    print(f"  Famille 1 (identifiants/poids) : {len(FAMILLE_1_IDENTIFIANTS)}")
    print(f"  Famille 2 (revenu)              : {len(FAMILLE_2_REVENU)}")
    print(f"  Famille 2bis (epargne)          : {len(FAMILLE_2BIS_EPARGNE)}")
    print(f"  Famille 3 (depenses)            : {len(FAMILLE_3_DEPENSES)}")
    print(f"  Famille 4 (demographie)         : {len(FAMILLE_4_DEMOGRAPHIE)}")
    print(f"  Famille 5 (technique)           : {len(FAMILLE_5_TECHNIQUE)}")

    print(f"\n{'=' * 70}\nCHARGEMENT PERIODE 1 (2017-2020)\n{'=' * 70}")
    df_p1 = load_selected_columns_for_years(PERIOD_1_YEARS)
    output_p1 = OUTPUT_DIR / "interview_period1_raw_selected.csv"
    df_p1.to_csv(output_p1, index=False)
    print(f"Sauvegarde : {output_p1}")

    print(f"\n{'=' * 70}\nCHARGEMENT PERIODE 2 (2021-2024)\n{'=' * 70}")
    df_p2 = load_selected_columns_for_years(PERIOD_2_YEARS)
    output_p2 = OUTPUT_DIR / "interview_period2_raw_selected.csv"
    df_p2.to_csv(output_p2, index=False)
    print(f"Sauvegarde : {output_p2}")

    print(f"\n{'=' * 70}")
    print("Termine. Deux fichiers prets pour une nouvelle EDA et un nouveau nettoyage :")
    print(f"  {output_p1}")
    print(f"  {output_p2}")