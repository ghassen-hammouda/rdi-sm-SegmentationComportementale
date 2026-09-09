"""
Preparation de l'espace de variables pour le clustering (US-09), a partir
de poverty_targets_period{1,2}.csv.

Etapes, dans l'ordre :
1. Exclusion des foyers a FINATXEM <= 0 (revenu net negatif ou nul,
   0.26% des foyers period1, 0.15% period2 - profil economique
   atypique, cf. diagnostic prealable, ratio_logement_revenu_net non
   calculable pour ces foyers)
   Puis exclusion des foyers a TOTEXP_FULL_ANNUEL == 0 (0.02% period1,
   0 period2 - tous les part_* deviennent NaN par construction dans
   build_business_features_v2.py, division par NaN volontaire).
2. Separation explicite de l'espace en 3 groupes de colonnes :
   - IDENTIFIANT (CUID) : conserve a part, jamais dans le clustering
   - FEATURES (l'espace de clustering a proprement parler)
   - RESERVE (revenu, statuts de pauvrete, colonnes techniques) :
     jamais vues par le clustering, sauvegardees a part pour la
     validation externe apres coup
3. Encodage one-hot des categorielles (REGION, FAM_TYPE, CUTENURE),
   drop_first=False (toutes les modalites gardees)
4. Recodage binaire de BLS_URBN (1/2 -> 1/0, meme logique que
   ecodage_categoriel_v2.py)
5. Standardisation (RobustScaler) de TOUT l'espace de features, y
   compris les colonnes one-hot - choix valide avec l'encadrant/le
   projet, pour que le poids de chaque variable dans le calcul de
   distance ne depende pas de son type d'origine.

EDUC_REF (ordinale) n'est PAS one-hot encodee - l'ordre croissant
(deja verifie dans le projet) est une information utile pour le
clustering, le one-hot la detruirait.
"""

import pandas as pd
from pathlib import Path
from sklearn.preprocessing import RobustScaler

INPUT_DIR = Path("outputs_v2")
OUTPUT_DIR = Path("outputs_v2")

FILES = {
    "period1": "poverty_targets_period1.csv",
    "period2": "poverty_targets_period2.csv",
}

ID_COLUMN = "CUID"
INCOME_COLUMN_FOR_EXCLUSION = "FINATXEM"

EXPENDITURE_BASES = [
    "TOTEXP", "FOOD", "HOUS", "TRANS", "HEALTH", "APPAR", "ENTERT",
    "EDUCA", "ALCBEV", "TOBACC", "PERSCA", "READ", "MISC", "CASHCO",
    "LIFINS", "RETPEN",
]

# --- Colonnes de l'espace de clustering (FEATURES) ---
PART_COLUMNS = [f"part_{base.lower()}" for base in EXPENDITURE_BASES if base != "TOTEXP"]

CONTINUOUS_FEATURES = PART_COLUMNS + [
    "totexp_par_equivalent_adulte",
    "ratio_logement_revenu_net",
    "FAM_SIZE",
    "PERSLT18",
    "PERSOT64",
    "AGE_REF",
    "VEHQ",
    "EDUC_REF",  # ordinale, gardee numerique, standardisee comme les autres continues
]

BINARY_FEATURE = "BLS_URBN"

ONEHOT_FEATURES = ["REGION", "FAM_TYPE", "CUTENURE"]

# --- Colonnes reservees (jamais vues par le clustering, gardees pour la validation) ---
RESERVED_COLUMNS = [
    "FINATXEM", "FINCBTXM", "FINATXEM_ESTIME_FLAG",
    "statut_pauvrete_revenu_net", "statut_pauvrete_consommation",
    "REVENU_NET_EQUIVALISE", "CONSOMMATION_EQUIVALISEE", "SEUIL_PAUVRETE",
    "ECHELLE_EQUIVALENCE", "STATE", "FINLWT21", "NB_TRIMESTRES",
]


def exclude_negative_or_null_income(df: pd.DataFrame, period_label: str) -> pd.DataFrame:
    before = len(df)
    mask_valide = df[INCOME_COLUMN_FOR_EXCLUSION] > 0
    n_exclus = int((~mask_valide).sum())
    df_filtre = df.loc[mask_valide].copy()
    print(f"[{period_label}] Foyers exclus (FINATXEM <= 0) : {n_exclus} "
          f"({n_exclus/before*100:.2f}% des {before} foyers) -> {len(df_filtre)} foyers retenus")
    return df_filtre.reset_index(drop=True)


def exclude_null_total_expenditure(df: pd.DataFrame, period_label: str) -> pd.DataFrame:
    """Foyers a TOTEXP_FULL_ANNUEL == 0 : tous les part_* deviennent NaN
    par construction dans build_business_features_v2.py (division par
    NaN volontaire, pour eviter une division par zero). Volume marginal
    (7 foyers period1, 0 period2 observes), exclusion avec tracabilite,
    meme logique que l'exclusion FINATXEM <= 0."""
    before = len(df)
    mask_valide = df["TOTEXP_FULL_ANNUEL"] != 0
    n_exclus = int((~mask_valide).sum())
    df_filtre = df.loc[mask_valide].copy()
    print(f"[{period_label}] Foyers exclus (TOTEXP_FULL_ANNUEL == 0, part_* NaN) : {n_exclus} "
          f"({n_exclus/before*100:.2f}% des {before} foyers) -> {len(df_filtre)} foyers retenus")
    return df_filtre.reset_index(drop=True)


def clip_extreme_values(df: pd.DataFrame, period_label: str) -> pd.DataFrame:
    """Ecretage au 99e percentile de totexp_par_equivalent_adulte, AVANT
    la standardisation - meme convention que ratio_logement_revenu_net
    dans build_business_features_v2.py (deja ecrete au 99e percentile).

    Motif : diagnostic PCA + dendrogramme sur echantillon (US-10 v2) a
    montre qu'en period1, quelques foyers avec une valeur extreme sur
    cette variable dominaient completement le calcul de distance du
    clustering (86% de variance expliquee par 2 axes PCA, contre 33%
    en period2 - signe d'un phenomene domine par des outliers plutot
    qu'une vraie structure multidimensionnelle). Le RobustScaler ne
    borne pas les valeurs extremes, seulement l'echelle - un ecretage
    explicite est necessaire en amont."""
    df = df.copy()
    plafond = df["totexp_par_equivalent_adulte"].quantile(0.99)
    n_ecretes = int((df["totexp_par_equivalent_adulte"] > plafond).sum())
    df["totexp_par_equivalent_adulte"] = df["totexp_par_equivalent_adulte"].clip(upper=plafond)
    print(f"[{period_label}] Ecretage totexp_par_equivalent_adulte au 99e percentile : "
          f"plafond={plafond:,.0f} $, {n_ecretes} foyers ecretes "
          f"({n_ecretes/len(df)*100:.2f}%)")
    return df


def encode_binary(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df[column] = df[column].map({1: 1, 2: 0})
    n_missing = int(df[column].isna().sum())
    if n_missing > 0:
        print(f"[ATTENTION] {n_missing} valeur(s) de {column} hors {{1,2}} apres recodage, a examiner.")
    return df


def build_clustering_space(df: pd.DataFrame, period_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retourne (features_scaled, reserved) : deux DataFrame alignes sur le
    meme index, l'un pour le clustering, l'autre garde de cote pour la
    validation externe apres coup. L'identifiant CUID est conserve dans
    les deux, pour pouvoir reassembler les resultats plus tard."""
    df = df.copy()

    missing_continuous = [c for c in CONTINUOUS_FEATURES if c not in df.columns]
    if missing_continuous:
        print(f"[ATTENTION] Variables continues attendues absentes : {missing_continuous}")
    existing_continuous = [c for c in CONTINUOUS_FEATURES if c in df.columns]

    df = encode_binary(df, BINARY_FEATURE)

    existing_onehot = [c for c in ONEHOT_FEATURES if c in df.columns]
    missing_onehot = [c for c in ONEHOT_FEATURES if c not in df.columns]
    if missing_onehot:
        print(f"[ATTENTION] Colonnes one-hot attendues absentes : {missing_onehot}")

    n_avant = df.shape[1]
    df = pd.get_dummies(df, columns=existing_onehot, prefix=existing_onehot, dtype=int, drop_first=False)
    n_apres = df.shape[1]
    print(f"[{period_label}] One-hot applique sur {existing_onehot} (drop_first=False) : "
          f"{n_avant} -> {n_apres} colonnes ({n_apres - n_avant} colonnes creees)")

    onehot_generated_cols = [c for c in df.columns
                              if any(c.startswith(f"{prefix}_") for prefix in existing_onehot)]

    feature_columns = existing_continuous + [BINARY_FEATURE] + onehot_generated_cols
    print(f"[{period_label}] Espace de clustering final : {len(feature_columns)} variables "
          f"({len(existing_continuous)} continues + 1 binaire + {len(onehot_generated_cols)} one-hot)")

    scaler = RobustScaler()
    features_scaled = pd.DataFrame(
        scaler.fit_transform(df[feature_columns]),
        columns=feature_columns,
        index=df.index,
    )
    features_scaled.insert(0, ID_COLUMN, df[ID_COLUMN].values)
    print(f"[{period_label}] RobustScaler applique sur l'ensemble des {len(feature_columns)} variables "
          f"(continues ET one-hot).")

    existing_reserved = [c for c in RESERVED_COLUMNS if c in df.columns]
    missing_reserved = [c for c in RESERVED_COLUMNS if c not in df.columns]
    if missing_reserved:
        print(f"[ATTENTION] Colonnes reservees attendues absentes : {missing_reserved}")
    reserved = df[[ID_COLUMN] + existing_reserved].copy()

    return features_scaled, reserved


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    for period_label, filename in FILES.items():
        print(f"\n{'=' * 70}\nPREPARATION ESPACE DE CLUSTERING - {period_label.upper()}\n{'=' * 70}")

        filepath = INPUT_DIR / filename
        df = pd.read_csv(filepath, low_memory=False)
        print(f"Charge : {filepath} ({len(df)} foyers, {df.shape[1]} colonnes)")

        df = exclude_negative_or_null_income(df, period_label)
        df = exclude_null_total_expenditure(df, period_label)
        df = clip_extreme_values(df, period_label)

        features_scaled, reserved = build_clustering_space(df, period_label)

        output_features = OUTPUT_DIR / f"clustering_features_{period_label}.csv"
        output_reserved = OUTPUT_DIR / f"clustering_reserved_{period_label}.csv"
        features_scaled.to_csv(output_features, index=False)
        reserved.to_csv(output_reserved, index=False)

        print(f"\nSauvegarde : {output_features} ({len(features_scaled)} foyers, "
              f"{features_scaled.shape[1] - 1} variables de clustering)")
        print(f"Sauvegarde : {output_reserved} ({len(reserved)} foyers, "
              f"{reserved.shape[1] - 1} variables reservees a la validation)")

    print(f"\n{'=' * 70}")
    print("Termine. Fichiers dans outputs_v2/ :")
    print("  clustering_features_period{1,2}.csv  -> espace pret pour le clustering (CUID + variables standardisees)")
    print("  clustering_reserved_period{1,2}.csv  -> CUID + revenu/statuts de pauvrete, pour la validation apres coup")
    print("  Les deux fichiers partagent le meme CUID, dans le meme ordre, pour pouvoir")
    print("  reassembler les resultats du clustering avec les cibles de pauvrete ensuite.")