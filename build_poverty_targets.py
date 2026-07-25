"""
US-06 / US-08 : Construction du dataset par foyer et des cibles de pauvrete

Ce script transforme les donnees Interview nettoyees (US-05, granularite
foyer-trimestre) en un dataset a granularite FOYER (une ligne par CUID),
puis construit deux cibles de pauvrete comparables (revenu et consommation),
et termine par une exploration du nouveau dataset (manquantes, doublons,
types) - exactement comme on l'a fait pour les donnees brutes en US-04,
mais applique cette fois au dataset final de modelisation.

Etapes, dans l'ordre decide :
1. Chargement des fichiers nettoyes (US-05)
2. Reconstruction _FULL (CQ+PQ) pour tous les postes de depense retenus
3. Agregation par foyer (CUID) : nombre de trimestres, sommes annualisees
   (proratisees si moins de 4 trimestres), demographie (derniere valeur
   connue), revenu (moyenne des declarations)
4. Echelle d'equivalence OCDE modifiee, a partir de FAM_SIZE et PERSLT18
5. Seuil de pauvrete = mediane(revenu equivalise) / 2, calcule separement
   par periode
6. Deux cibles : statut_pauvrete_revenu et statut_pauvrete_consommation,
   comparees au MEME seuil
7. Exploration du nouveau dataset : nulls, doublons, types, repartition
   des classes

Perimetre : Interview uniquement (Diary laisse de cote). P1 et P2 traites
et sauvegardes separement (deux seuils distincts).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob

INPUT_DIR = Path("outputs_clean")
OUTPUT_DIR = Path("outputs_poverty")
OUTPUT_DIR.mkdir(exist_ok=True)

# Postes de depense retenus (verifies presents en EDA), a annualiser au
# meme titre que TOTEXP / FOOD / HOUS.
EXPENDITURE_BASES = [
    "TOTEXP", "FOOD", "HOUS", "TRANS", "HEALTH", "APPAR", "ENTERT",
    "EDUCA", "ALCBEV", "TOBACC", "PERSCA", "READ", "MISC", "CASHCO",
    "LIFINS", "RETPEN",
]

# Variables demographiques : on garde la DERNIERE valeur connue du foyer
# (celle de son interview la plus recente), car ce sont des attributs du
# foyer plutot que des flux a additionner.
DEMOGRAPHIC_COLUMNS = [
    "FAM_SIZE", "PERSLT18", "PERSOT64", "FAM_TYPE", "AGE_REF",
    "EDUC_REF", "CUTENURE", "VEHQ", "REGION", "BLS_URBN",
]

# Revenu : moyenne des declarations du foyer sur ses interviews
# disponibles (FINCBTXM/FINATXEM sont deja des mesures annuelles
# glissantes a chaque interview ; la moyenne lisse les petites
# variations de declaration d'un trimestre a l'autre plutot que de
# trancher arbitrairement pour la derniere valeur).
# Deux versions gardees en parallele : brut (convention Orshansky/HHS,
# utilisee historiquement) et net (apres impots, plus proche de ce
# qu'un foyer a reellement disponible - cf. echange sur la pertinence
# du brut vs net).
INCOME_COLUMN_BRUT = "FINCBTXM"
INCOME_COLUMN_NET = "FINATXEM"


# ---------------------------------------------------------------------------
# 1. CHARGEMENT DU FICHIER NETTOYE LE PLUS RECENT (US-05)
# ---------------------------------------------------------------------------

def find_latest_clean_file(period_label: str) -> Path:
    """Trouve le fichier interview_{period}_clean_v*.csv le plus recent
    (le plus grand horodatage), pour ne pas avoir a mettre a jour un nom
    de fichier fige a chaque nouvelle execution de US-05.
    """
    pattern = str(INPUT_DIR / f"interview_{period_label}_clean_v*.csv")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"Aucun fichier trouve pour le motif : {pattern}")
    return Path(candidates[-1])


# ---------------------------------------------------------------------------
# 2. RECONSTRUCTION _FULL POUR TOUS LES POSTES DE DEPENSE
# ---------------------------------------------------------------------------

def build_full_quarter_expenditure(df: pd.DataFrame, base_name: str) -> pd.DataFrame:
    cq_col, pq_col = f"{base_name}CQ", f"{base_name}PQ"
    missing = [c for c in (cq_col, pq_col) if c not in df.columns]
    if missing:
        print(f"[ATTENTION] Colonne(s) manquante(s) pour {base_name} : {missing}, ignore.")
        return df
    full_col = f"{base_name}_FULL"
    df[full_col] = df[cq_col].fillna(0) + df[pq_col].fillna(0)
    return df


# ---------------------------------------------------------------------------
# 3. AGREGATION PAR FOYER (CUID)
# ---------------------------------------------------------------------------

def aggregate_by_household(df: pd.DataFrame) -> pd.DataFrame:
    """Transforme un DataFrame foyer-trimestre en un DataFrame une-ligne
    par foyer (CUID), avec :
    - le nombre de trimestres observes
    - les depenses _FULL annualisees (somme des trimestres x 4/nb_trimestres)
    - les variables demographiques : derniere valeur connue du foyer
    - le revenu : moyenne des declarations du foyer
    """
    full_cols = [f"{base}_FULL" for base in EXPENDITURE_BASES if f"{base}_FULL" in df.columns]

    nb_trimestres = df.groupby("CUID").size().rename("NB_TRIMESTRES")

    sums = df.groupby("CUID")[full_cols].sum()
    annualized = sums.multiply(4 / nb_trimestres, axis=0)
    annualized.columns = [f"{c}_ANNUEL" for c in annualized.columns]

    demo_cols_present = [c for c in DEMOGRAPHIC_COLUMNS if c in df.columns]
    demo = df.sort_values("SOURCE_YEAR").groupby("CUID")[demo_cols_present].last()

    income = df.groupby("CUID")[[INCOME_COLUMN_BRUT, INCOME_COLUMN_NET]].mean()

    household_df = pd.concat([nb_trimestres, annualized, demo, income], axis=1).reset_index()
    print(f"Agregation par foyer : {len(df)} lignes foyer-trimestre -> {len(household_df)} foyers")
    return household_df


# ---------------------------------------------------------------------------
# 3bis. ECRETAGE DES VALEURS ABERRANTES SUR LES DEPENSES ANNUALISEES
# ---------------------------------------------------------------------------

def cap_annualized_outliers(df: pd.DataFrame, period_label: str,
                             iqr_multiplier: float = 3.0, z_threshold: float = 3.0) -> pd.DataFrame:
    """Applique le meme ecretage (IQR x3 + z-score > 3) que celui deja
    utilise en US-05, mais etendu ici a TOUTES les depenses annualisees
    (16 postes), pas seulement les 4 variables coeur (TOTEXP/FOOD/HOUS/
    FINCBTXM) deja traitees au niveau trimestriel en US-05.

    Necessaire car l'annualisation (x jusqu'a 4 pour un foyer a un seul
    trimestre) peut amplifier une valeur trimestrielle deja elevee sur
    les 12 postes qui n'avaient jamais ete ecretes ; laisser ces valeurs
    non traitees peut faire dominer artificiellement le calcul de la PCA
    (une colonne a variance demesuree ecrase toutes les autres, meme
    apres RobustScaler qui n'egalise pas les variances comme le ferait
    StandardScaler).
    """
    annual_cols = [f"{base}_FULL_ANNUEL" for base in EXPENDITURE_BASES
                   if f"{base}_FULL_ANNUEL" in df.columns]

    print(f"[{period_label}] Ecretage des valeurs aberrantes sur {len(annual_cols)} postes annualises :")
    for col in annual_cols:
        s = df[col]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low_iqr, high_iqr = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr

        mean, std = s.mean(), s.std()
        is_outlier_z = (s - mean).abs() > z_threshold * std if std > 0 else pd.Series(False, index=s.index)
        is_outlier_iqr = (s < low_iqr) | (s > high_iqr)
        is_outlier = is_outlier_iqr | is_outlier_z
        n_outliers = int(is_outlier.sum())

        df[f"{col}_CAPPED_FLAG"] = is_outlier
        df[col] = s.clip(lower=low_iqr, upper=high_iqr)

        if n_outliers > 0:
            print(f"    {col} : {n_outliers} valeur(s) ecretee(s) [{low_iqr:,.2f} ; {high_iqr:,.2f}]")

    return df


# ---------------------------------------------------------------------------
# 4. ECHELLE D'EQUIVALENCE (OCDE MODIFIEE)
# ---------------------------------------------------------------------------

def compute_equivalence_scale(df: pd.DataFrame) -> pd.Series:
    """Echelle OCDE modifiee : 1.0 pour le premier adulte, 0.5 par adulte
    supplementaire, 0.3 par enfant (PERSLT18).
    """
    nb_adultes = (df["FAM_SIZE"] - df["PERSLT18"]).clip(lower=1)
    nb_adultes_suppl = (nb_adultes - 1).clip(lower=0)
    nb_enfants = df["PERSLT18"]
    return 1.0 + 0.5 * nb_adultes_suppl + 0.3 * nb_enfants


# ---------------------------------------------------------------------------
# 5-6. SEUIL DE PAUVRETE ET DEUX CIBLES
# ---------------------------------------------------------------------------

def discordance_breakdown(df: pd.DataFrame, col_a: str, col_b: str,
                           label_a: str, label_b: str) -> None:
    """Detaille un taux d'accord global en ses deux sens de desaccord :
    combien de foyers sont pauvres selon A mais pas selon B, et
    inversement. Un taux d'accord global peut cacher deux effets qui se
    compensent partiellement (des foyers basculent dans un sens,
    d'autres dans l'autre) - ce detail montre lequel des deux sens
    domine reellement, plutot qu'un seul pourcentage global.
    """
    total = len(df)
    accord_pauvre = int(((df[col_a] == 1) & (df[col_b] == 1)).sum())
    accord_non_pauvre = int(((df[col_a] == 0) & (df[col_b] == 0)).sum())
    a_seul = int(((df[col_a] == 1) & (df[col_b] == 0)).sum())
    b_seul = int(((df[col_a] == 0) & (df[col_b] == 1)).sum())

    print(f"    Accord (pauvre des deux cotes)     : {accord_pauvre} ({accord_pauvre/total*100:.2f}%)")
    print(f"    Accord (non pauvre des deux cotes) : {accord_non_pauvre} ({accord_non_pauvre/total*100:.2f}%)")
    print(f"    Pauvre selon {label_a} seulement : {a_seul} ({a_seul/total*100:.2f}%)")
    print(f"    Pauvre selon {label_b} seulement : {b_seul} ({b_seul/total*100:.2f}%)")


def build_poverty_targets(df: pd.DataFrame, period_label: str) -> pd.DataFrame:
    """Calcule l'echelle d'equivalence, les revenus (brut et net) et la
    consommation equivalises, le seuil (mediane du revenu BRUT
    equivalise / 2, propre a cette periode - reference historique
    Orshansky/HHS), puis TROIS cibles binaires comparees au MEME seuil :
    revenu brut, revenu net, consommation.

    Le seuil reste ancre sur le revenu brut (convention officielle) meme
    pour la comparaison avec le revenu net : l'objectif est d'isoler
    l'effet du choix "brut vs net" sur la classification, pas de le
    melanger avec un changement de seuil en plus.
    """
    df["ECHELLE_EQUIVALENCE"] = compute_equivalence_scale(df)

    df["REVENU_BRUT_EQUIVALISE"] = df[INCOME_COLUMN_BRUT] / df["ECHELLE_EQUIVALENCE"]
    df["REVENU_NET_EQUIVALISE"] = df[INCOME_COLUMN_NET] / df["ECHELLE_EQUIVALENCE"]
    df["CONSOMMATION_EQUIVALISEE"] = df["TOTEXP_FULL_ANNUEL"] / df["ECHELLE_EQUIVALENCE"]

    mediane_revenu_brut_equivalise = df["REVENU_BRUT_EQUIVALISE"].median()
    seuil = mediane_revenu_brut_equivalise / 2
    print(f"[{period_label}] Mediane du revenu BRUT equivalise : {mediane_revenu_brut_equivalise:,.2f} $")
    print(f"[{period_label}] Seuil de pauvrete retenu (mediane brut / 2) : {seuil:,.2f} $")

    df["SEUIL_PAUVRETE"] = seuil
    df["statut_pauvrete_revenu_brut"] = (df["REVENU_BRUT_EQUIVALISE"] < seuil).astype(int)
    df["statut_pauvrete_revenu_net"] = (df["REVENU_NET_EQUIVALISE"] < seuil).astype(int)
    df["statut_pauvrete_consommation"] = (df["CONSOMMATION_EQUIVALISEE"] < seuil).astype(int)

    taux_brut = df["statut_pauvrete_revenu_brut"].mean() * 100
    taux_net = df["statut_pauvrete_revenu_net"].mean() * 100
    taux_conso = df["statut_pauvrete_consommation"].mean() * 100
    print(f"[{period_label}] Taux de pauvrete (revenu brut)  : {taux_brut:.2f}%")
    print(f"[{period_label}] Taux de pauvrete (revenu net)   : {taux_net:.2f}%")
    print(f"[{period_label}] Taux de pauvrete (consommation) : {taux_conso:.2f}%")

    print(f"\n[{period_label}] Detail revenu brut vs revenu net :")
    discordance_breakdown(df, "statut_pauvrete_revenu_brut", "statut_pauvrete_revenu_net", "brut", "net")

    print(f"\n[{period_label}] Detail revenu brut vs consommation :")
    discordance_breakdown(df, "statut_pauvrete_revenu_brut", "statut_pauvrete_consommation", "brut", "consommation")

    print(f"\n[{period_label}] Detail revenu net vs consommation :")
    discordance_breakdown(df, "statut_pauvrete_revenu_net", "statut_pauvrete_consommation", "net", "consommation")

    return df


# ---------------------------------------------------------------------------
# US-06 : FEATURES METIER (part du budget, ratio logement/revenu,
#         depense par tete via l'echelle d'equivalence)
# ---------------------------------------------------------------------------

def add_business_features(df: pd.DataFrame, period_label: str) -> pd.DataFrame:
    """Construit les features comportementales a partir des colonnes
    annualisees deja calculees, pour TOUS les postes de depense retenus
    (choix valide : tout construire, filtrer ensuite en US-12 plutot que
    de pre-selectionner maintenant) :

    - part_{poste} = poids du poste dans le budget total du foyer
    - {poste}_par_equivalent_adulte = depense par tete, ajustee a la
      taille du foyer via l'echelle d'equivalence deja calculee
    - ratio_logement_revenu_brut/net = indicateur de vulnerabilite au
      logement (au-dela de 30% du revenu = "surcharge de logement"
      selon les standards HUD)

    part_food correspond directement a l'indicateur de la loi d'Engel
    (plus elevee chez les foyers modestes) : aucune colonne separee
    n'est necessaire, part_food EST cette feature.
    """
    total_budget = df["TOTEXP_FULL_ANNUEL"].replace(0, np.nan)

    for base in EXPENDITURE_BASES:
        if base == "TOTEXP":
            continue
        col = f"{base}_FULL_ANNUEL"
        if col not in df.columns:
            print(f"[ATTENTION] {col} absente, features ignorees pour {base}.")
            continue
        df[f"part_{base.lower()}"] = df[col] / total_budget
        df[f"{base.lower()}_par_equivalent_adulte"] = df[col] / df["ECHELLE_EQUIVALENCE"]

    # Depense totale par equivalent-adulte : deja calculee (alias explicite
    # pour la retrouver facilement avec les autres "_par_equivalent_adulte").
    df["totexp_par_equivalent_adulte"] = df["CONSOMMATION_EQUIVALISEE"]

    # Ratio logement / revenu : diagnostic prealable sur les revenus
    # nuls ou negatifs (le ratio n'a pas de sens dans ces cas-la).
    for label, income_col in [("brut", INCOME_COLUMN_BRUT), ("net", INCOME_COLUMN_NET)]:
        nb_non_positif = int((df[income_col] <= 0).sum())
        if nb_non_positif > 0:
            print(f"[ATTENTION] {nb_non_positif} foyer(s) avec revenu {label} <= 0 "
                  f"-> ratio_logement_revenu_{label} sera NaN ou negatif pour ces foyers, a traiter avant modelisation.")
        revenu_positif = df[income_col].where(df[income_col] > 0, np.nan)
        df[f"ratio_logement_revenu_{label}"] = df["HOUS_FULL_ANNUEL"] / revenu_positif

    print(f"[{period_label}] Features metier construites : "
          f"{len([c for c in df.columns if c.startswith(('part_', 'ratio_logement')) or c.endswith('par_equivalent_adulte')])} colonnes")

    df = clean_business_features(df, period_label)

    return df


def clean_business_features(df: pd.DataFrame, period_label: str) -> pd.DataFrame:
    """Corrige deux problemes de qualite reperes sur les features US-06
    fraichement construites :

    1. Les parts de budget (part_*) peuvent sortir de [0,1] : soit un
       montant de depense legerement negatif (herite de l'ecretage IQR
       /z-score de l'US-05, qui autorise des bornes basses negatives),
       soit un budget total (TOTEXP_FULL_ANNUEL) proche de zero qui fait
       exploser le ratio. Une part de budget est bornee par definition
       entre 0 et 1 : on ecrete a cette borne logique plutot que de
       laisser des valeurs qui n'ont pas de sens economique.

    2. Les ratios logement/revenu peuvent exploser quand le revenu est
       tres proche de zero (sans etre <= 0, deja gere a part) : un
       foyer avec 5 $ de revenu et une depense logement normale produit
       un ratio de plusieurs milliers, qui n'apporte aucune information
       utile et ecraserait toute analyse statistique dessus. Ecretage
       au 99e percentile (borne haute) et a 0 (borne basse), meme
       logique que l'ecretage IQR/z-score deja applique en US-05.
    """
    part_cols = [c for c in df.columns if c.startswith("part_")]
    for col in part_cols:
        n_invalid = int(((df[col] < 0) | (df[col] > 1)).sum())
        if n_invalid > 0:
            print(f"{col} : {n_invalid} valeur(s) hors de [0,1] -> ecretees a la borne logique.")
        df[col] = df[col].clip(lower=0, upper=1)

    for col in ["ratio_logement_revenu_brut", "ratio_logement_revenu_net"]:
        if col not in df.columns or df[col].dropna().empty:
            continue
        upper_bound = df[col].quantile(0.99)
        n_capped_haut = int((df[col] > upper_bound).sum())
        n_capped_bas = int((df[col] < 0).sum())
        if n_capped_haut > 0:
            print(f"{col} : {n_capped_haut} valeur(s) au-dela du 99e percentile ({upper_bound:.2f}) -> ecretees.")
        if n_capped_bas > 0:
            print(f"{col} : {n_capped_bas} valeur(s) negative(s) -> ramenees a 0.")
        df[col] = df[col].clip(lower=0, upper=upper_bound)

    return df


# ---------------------------------------------------------------------------
# 8. EXPLORATION DU NOUVEAU DATASET (nulls, doublons, types)
# ---------------------------------------------------------------------------

def explore_new_dataset(df: pd.DataFrame, period_label: str) -> None:
    print(f"\n{'=' * 70}\nEXPLORATION DU DATASET FOYER - {period_label}\n{'=' * 70}")

    print(f"\n--- [1] Volumetrie ---")
    print(f"{len(df)} foyers, {df.shape[1]} colonnes")

    print(f"\n--- [2] Doublons sur CUID ---")
    nb_doublons = df["CUID"].duplicated().sum()
    print(f"CUID dupliques : {nb_doublons} (attendu : 0, une ligne par foyer)")

    print(f"\n--- [3] Types de colonnes ---")
    print(df.dtypes.astype(str).value_counts())

    print(f"\n--- [4] Valeurs manquantes ---")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("Aucune valeur manquante detectee.")
    else:
        missing_pct = (missing / len(df) * 100).round(2)
        print(pd.DataFrame({"nb_manquant": missing, "pct_manquant": missing_pct}))

    print(f"\n--- [5] Repartition du nombre de trimestres observes par foyer ---")
    print(df["NB_TRIMESTRES"].value_counts().sort_index())

    print(f"\n--- [6] Repartition des classes (cibles de pauvrete) ---")
    for target in ["statut_pauvrete_revenu_brut", "statut_pauvrete_revenu_net", "statut_pauvrete_consommation"]:
        print(f"{target} :")
        print(df[target].value_counts(normalize=True).round(4) * 100)

    print(f"\n--- [7] Aperçu des features metier (US-06) ---")
    part_cols = [c for c in df.columns if c.startswith("part_")]
    if part_cols:
        print(df[part_cols].describe().T[["mean", "min", "max"]])
    ratio_cols = [c for c in df.columns if c.startswith("ratio_logement_revenu")]
    if ratio_cols:
        print(df[ratio_cols].describe().T[["mean", "min", "max"]])


# ---------------------------------------------------------------------------
# PIPELINE COMPLET, PAR PERIODE
# ---------------------------------------------------------------------------

def build_poverty_dataset_for_period(period_label: str) -> pd.DataFrame:
    print(f"\n{'#' * 70}\n# {period_label.upper()}\n{'#' * 70}")

    filepath = find_latest_clean_file(period_label)
    print(f"Chargement : {filepath}")
    df = pd.read_csv(filepath, low_memory=False)

    print("\n--- Reconstruction des postes de depense (_FULL = CQ + PQ) ---")
    for base in EXPENDITURE_BASES:
        df = build_full_quarter_expenditure(df, base)

    required_cols = ["CUID", "SOURCE_YEAR", INCOME_COLUMN_BRUT, INCOME_COLUMN_NET] + DEMOGRAPHIC_COLUMNS
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise KeyError(f"Colonnes requises absentes du fichier nettoye : {missing_required}")

    print("\n--- Agregation par foyer ---")
    household_df = aggregate_by_household(df)

    print("\n--- Ecretage des valeurs aberrantes (depenses annualisees) ---")
    household_df = cap_annualized_outliers(household_df, period_label)

    print("\n--- Construction des cibles de pauvrete ---")
    household_df = build_poverty_targets(household_df, period_label)

    print("\n--- Construction des features metier (US-06) ---")
    household_df = add_business_features(household_df, period_label)

    explore_new_dataset(household_df, period_label)

    output_path = OUTPUT_DIR / f"poverty_dataset_{period_label}.csv"
    household_df.to_csv(output_path, index=False)
    print(f"\nDataset foyer sauvegarde : {output_path} ({len(household_df)} foyers, {household_df.shape[1]} colonnes)")

    return household_df


# ---------------------------------------------------------------------------
# EXECUTION PRINCIPALE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    df_p1 = build_poverty_dataset_for_period("period1")
    df_p2 = build_poverty_dataset_for_period("period2")

    print(f"\n{'=' * 70}")
    print("RECAPITULATIF")
    print(f"{'=' * 70}")
    print(f"Periode 1 : {len(df_p1)} foyers")
    print(f"Periode 2 : {len(df_p2)} foyers")