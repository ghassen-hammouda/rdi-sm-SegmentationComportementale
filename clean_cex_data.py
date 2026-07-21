"""
US-05 : Nettoyage des donnees CEX (Interview Survey FMLI + Diary Survey FMLD)

Ce script applique les corrections identifiees et priorisees dans le
rapport d'audit US-04, dans l'ordre de priorite retenu, pour les DEUX
composantes du CEX (Interview et Diary) :

1. Dedoublonnage des trimestres/vagues publies deux fois (paires
   fmli181/181x, fmli191/191x cote Interview), en conservant la version
   la plus recente ("x")
2. Reconstitution des depenses trimestrielles completes (CQ + PQ) cote
   Interview, deja validee en US-04 (fonction reutilisee depuis
   load_and_audit_fmli.py). Sans objet cote Diary (FOODTOT est deja une
   variable agregee, pas de decoupage CQ/PQ dans cette enquete).
3. Suppression des colonnes structurellement vides (100% manquant sur
   TOUTE la periode ET sur CHAQUE annee prise individuellement), apres
   verification explicite qu'aucune annee ne fait exception
4. Signalement (sans suppression) des valeurs negatives sur les variables
   de depense et de revenu, pour examen manuel avant toute decision
   de suppression definitive
5. Strategie d'imputation par variable, choisie selon le taux de manquant
   et le type de donnee : mediane (numerique, < 5%), mode (categoriel,
   < 5%), mediane (numerique, 5-95% - KNN evalue et ecarte pour raison
   de performance, cf. impute_knn et son commentaire), mode (categoriel,
   5-95%), suppression (>= 95%, hors variables coeur d'analyse)
6. Detection des valeurs aberrantes par IQR ET z-score, traitees par
   ecretage (winsorization) sur les variables numeriques cles, avec
   signalement des lignes modifiees

Le resultat est sauvegarde dans des fichiers CSV nettoyes et versionnes
(horodates), distincts des donnees brutes, pour ne jamais ecraser la
source originale ni une execution precedente.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.impute import KNNImputer

# Reutilisation du travail deja fait en US-04, plutot que de dupliquer
# le code de chargement des donnees.
from load_and_audit_fmli import (
    load_survey_for_years,
    build_full_quarter_expenditure,
    missing_rate_by_group,
    missing_rate_report,
    PERIOD_1_YEARS,
    PERIOD_2_YEARS,
    ID_COLUMNS,
)

OUTPUT_DIR = Path("outputs_clean")

# Horodatage utilise pour versionner les fichiers de sortie : chaque
# execution produit des fichiers distincts plutot que d'ecraser
# silencieusement la version precedente.
VERSION_TAG = datetime.now().strftime("%Y%m%d")

# Seuils de classification des variables selon leur taux de manquant,
# pour decider automatiquement de la strategie d'imputation a appliquer.
LOW_MISSING_THRESHOLD_PCT = 5.0

# Variables de depense a reconstituer en CQ+PQ (Interview uniquement, cf.
# docstring ci-dessus : Diary ne suit pas ce decoupage).
EXPENDITURE_BASES = ["TOTEXP", "FOOD", "HOUS"]

# Variables reellement utilisees dans l'analyse principale (segmentation
# comportementale). Sert a verifier que les valeurs manquantes ne posent
# pas de probleme la ou ca compte vraiment, avant de documenter le reste
# sans y toucher.
CORE_ANALYSIS_COLUMNS_INTERVIEW = [
    "TOTEXP_FULL", "FOOD_FULL", "HOUS_FULL", "FINCBTXM",
    "REGION", "BLS_URBN", "FINLWT21", "INTERI",
]
CORE_ANALYSIS_COLUMNS_DIARY = ["FOODTOT", "FINLWT21"]

# Seuil au-dela duquel une variable (hors variables coeur) est jugee trop
# peu renseignee pour etre exploitable en modelisation. Documentee, non
# supprimee : on ne prend pas de decision destructive sans examen manuel.
HIGH_MISSING_THRESHOLD_PCT = 95.0

# Variables Interview candidates a la suppression (US-04, rapport section
# "taux de valeurs manquantes") : DESFLG/INT_PHON/INT_HOME sont vides a
# 100% sur la Periode 1 (elles disparaissent ensuite du schema en P2 -
# un mecanisme different, deja documente en US-04, pas traite ici).
# OTRINCBX est vide a 100% sur LES DEUX periodes 2017-2024 : c'est le
# seul cas de "colonne structurellement vide sur toute la duree" au sens
# strict, elle avait ete omise par erreur dans la premiere version de ce
# script alors qu'elle correspond exactement au critere de l'US-05.
COLUMNS_TO_DROP_KNOWN_EMPTY_INTERVIEW = ["DESFLG", "INT_PHON", "INT_HOME", "OTRINCBX"]

# Diary : MARI_AL1 est la seule variable identifiee en US-04 comme vide a
# 100% sur les deux periodes (rapport US-04, taux de manquant Diary).
COLUMNS_TO_DROP_KNOWN_EMPTY_DIARY = ["MARI_AL1"]

# Variables sur lesquelles on signale les valeurs negatives sans les
# supprimer automatiquement (a examiner manuellement avant toute action).
COLUMNS_TO_FLAG_NEGATIVE_INTERVIEW = ["TOTEXP_FULL", "HOUS_FULL", "FINCBTXM"]
COLUMNS_TO_FLAG_NEGATIVE_DIARY = ["FOODTOT"]

# Variables numeriques cles sur lesquelles l'ecretage des valeurs
# aberrantes (IQR + z-score) est applique explicitement (cf.
# detect_and_cap_outliers pour la justification du perimetre restreint).
COLUMNS_TO_CAP_OUTLIERS_INTERVIEW = ["TOTEXP_FULL", "FOOD_FULL", "HOUS_FULL", "FINCBTXM"]
COLUMNS_TO_CAP_OUTLIERS_DIARY = ["FOODTOT"]


# ---------------------------------------------------------------------------
# 0. VERIFICATION AVANT SUPPRESSION : la colonne est-elle vide sur CHAQUE
#    annee de la periode, pas seulement en moyenne cumulee ?
# ---------------------------------------------------------------------------

def verify_fully_empty_by_year(df: pd.DataFrame, columns: list[str],
                                tolerance_pct: float = 99.9) -> list[str]:
    """Avant de supprimer une colonne jugee 'structurellement vide' en
    US-04, verifie qu'elle est bien vide sur CHAQUE annee de la periode,
    pas seulement en moyenne cumulee sur l'ensemble des annees. Une
    colonne pourrait etre vide sur 2017-2019 mais renseignee en 2020 par
    exemple : la supprimer sans ce controle ferait perdre de la vraie
    donnee sans qu'on s'en rende compte.

    Retourne la liste des colonnes confirmees vides sur toutes les
    annees (donc sures a supprimer). Celles qui ne le sont pas sont
    signalees en detail et exclues de la liste retournee, pour examen
    manuel avant toute suppression.
    """
    confirmed = []
    for col in columns:
        if col not in df.columns:
            print(f"[ATTENTION] {col} absente du DataFrame, verification ignoree.")
            continue

        by_year = missing_rate_by_group(df, col, "SOURCE_YEAR")
        min_missing = by_year["taux_manquant_pct"].min()

        if min_missing >= tolerance_pct:
            confirmed.append(col)
        else:
            print(f"[ATTENTION] {col} n'est PAS vide sur toutes les annees "
                  f"(minimum observe = {min_missing:.2f}%) -> NON supprimee, "
                  f"a examiner manuellement avant toute decision :")
            print(by_year)

    return confirmed


# ---------------------------------------------------------------------------
# 1. DEDUPLICATION DES VAGUES PUBLIEES DEUX FOIS
# ---------------------------------------------------------------------------

def deduplicate_republished_quarters(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les doublons de NEWID en conservant la version la plus
    recente de chaque vague republiee (fichier dont le nom contient un
    'x', ex: fmli181x.csv, publie apres fmli181.csv).

    Principe : pour chaque NEWID en double, on trie les lignes pour que
    la version "x" (str contenant 'x' dans SOURCE_FILE) passe en dernier,
    puis on garde uniquement la derniere occurrence de chaque NEWID.

    Cote Diary, aucun fichier "x" n'a ete observe en US-04 (0 doublon
    NEWID detecte) : cette fonction agit alors comme un filet de securite
    sans effet, plutot qu'un traitement specifique a ajouter separement.
    """
    before = len(df)

    is_x_version = df["SOURCE_FILE"].str.contains("x", case=False, na=False)
    df_sorted = df.assign(_is_x=is_x_version).sort_values("_is_x")
    df_dedup = df_sorted.drop_duplicates(subset="NEWID", keep="last").drop(columns="_is_x")

    after = len(df_dedup)
    print(f"Dedoublonnage : {before} lignes -> {after} lignes ({before - after} doublons retires)")
    return df_dedup.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. SUPPRESSION DES COLONNES STRUCTURELLEMENT VIDES (apres verification)
# ---------------------------------------------------------------------------

def drop_known_empty_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Retire les colonnes confirmees entierement vides sur toutes les
    annees par verify_fully_empty_by_year. On les retire plutot que de
    les imputer, faute de toute valeur reelle permettant une imputation
    fiable.
    """
    existing = [c for c in columns if c in df.columns]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        print(f"[ATTENTION] Colonnes deja absentes, rien a retirer : {missing}")
    if existing:
        df = df.drop(columns=existing)
        print(f"Colonnes retirees (structurellement vides, confirmees par annee) : {existing}")
    return df


# ---------------------------------------------------------------------------
# 3. SIGNALEMENT DES VALEURS NEGATIVES (SANS SUPPRESSION AUTOMATIQUE)
# ---------------------------------------------------------------------------

def flag_negative_values(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Ajoute, pour chaque variable listee, une colonne booleenne
    '{variable}_NEGATIVE_FLAG' signalant les valeurs negatives, sans les
    modifier ni les supprimer. Objectif : permettre un examen manuel
    (ex: distinguer un remboursement legitime d'une erreur de saisie)
    avant de decider d'un traitement definitif, conformement a la
    recommandation du rapport US-04.
    """
    for col in columns:
        if col not in df.columns:
            print(f"[ATTENTION] {col} absente, signalement ignore.")
            continue
        flag_col = f"{col}_NEGATIVE_FLAG"
        df[flag_col] = df[col] < 0
        nb_negative = df[flag_col].sum()
        pct_negative = df[flag_col].mean() * 100
        print(f"{col} : {nb_negative} valeurs negatives signalees ({pct_negative:.2f}%)")
    return df


# ---------------------------------------------------------------------------
# 3bis. POLITIQUE POUR LES VALEURS MANQUANTES RESTANTES (non 100%)
# ---------------------------------------------------------------------------

def summarize_core_variables_missingness(df: pd.DataFrame, core_cols: list[str],
                                          label: str) -> pd.DataFrame:
    """Verifie le taux de manquant des variables reellement utilisees dans
    l'analyse principale, apres suppression des colonnes structurellement
    vides. Si l'une de ces variables coeur presentait un taux de manquant
    eleve, ce serait un vrai probleme a traiter en priorite avant toute
    modelisation ; sinon, ca confirme que le reste des variables tres
    incompletes (revenus detailles, etc.) ne concerne pas les variables
    utilisees ici et peut etre documente sans y toucher.
    """
    existing_cols = [c for c in core_cols if c in df.columns]
    missing_cols = [c for c in core_cols if c not in df.columns]
    if missing_cols:
        print(f"[ATTENTION] Variables coeur absentes du DataFrame : {missing_cols}")

    report = missing_rate_report(df, exclude_cols=ID_COLUMNS)
    core_report = report[report["variable"].isin(existing_cols)].reset_index(drop=True)

    print(f"\n--- Taux de manquant des variables coeur d'analyse ({label}) ---")
    print(core_report)

    high_risk = core_report[core_report["taux_manquant_pct"] > 5]
    if not high_risk.empty:
        print(f"[ATTENTION] {len(high_risk)} variable(s) coeur avec plus de 5% de manquant "
              f"-> a traiter en priorite avant toute modelisation :")
        print(high_risk)
    else:
        print("Toutes les variables coeur ont un taux de manquant negligeable (<= 5%).")

    return core_report


# ---------------------------------------------------------------------------
# 3ter. STRATEGIE D'IMPUTATION PAR VARIABLE (selon taux de manquant + type)
# ---------------------------------------------------------------------------

def classify_columns_by_missingness(df: pd.DataFrame, core_cols: list[str],
                                     low_threshold: float = LOW_MISSING_THRESHOLD_PCT,
                                     high_threshold: float = HIGH_MISSING_THRESHOLD_PCT) -> dict:
    """Classe chaque colonne (hors identifiants) dans une categorie de
    strategie d'imputation, selon deux criteres : le taux de manquant et
    le type de donnee. Politique retenue et justifiee :

    - 0% de manquant            -> rien a faire
    - ]0% ; 5%[, numerique      -> mediane (robuste aux distributions
                                    asymetriques observees en EDA, contrairement
                                    a la moyenne)
    - ]0% ; 5%[, categorielle   -> mode (valeur la plus frequente)
    - [5% ; 95%[, numerique     -> KNN (voir impute_knn pour la justification
                                    et l'adaptation necessaire au volume de donnees)
    - [5% ; 95%[, categorielle  -> mode (KNN non applicable directement a du
                                    categoriel non ordonne avec cette implementation)
    - >= 95%                    -> suppression (variable jugee inexploitable ;
                                    imputer au-dela de 95% reviendrait a fabriquer
                                    la donnee plutot qu'a la corriger)

    Les variables coeur d'analyse ne sont jamais supprimees par cette
    regle automatique, meme si elles depassaient 95% (verification de
    securite ; en pratique aucune ne s'en approche d'apres l'EDA).
    """
    report = missing_rate_report(df, exclude_cols=ID_COLUMNS)
    numeric_cols = set(df.select_dtypes(include=[np.number]).columns)

    zero = report[report["taux_manquant_pct"] == 0]["variable"].tolist()
    low = report[(report["taux_manquant_pct"] > 0) & (report["taux_manquant_pct"] < low_threshold)]["variable"].tolist()
    mid = report[(report["taux_manquant_pct"] >= low_threshold) & (report["taux_manquant_pct"] < high_threshold)]["variable"].tolist()
    high = [c for c in report[report["taux_manquant_pct"] >= high_threshold]["variable"].tolist() if c not in core_cols]

    classification = {
        "zero": zero,
        "low_numeric": [c for c in low if c in numeric_cols],
        "low_categorical": [c for c in low if c not in numeric_cols],
        "mid_numeric": [c for c in mid if c in numeric_cols],
        "mid_categorical": [c for c in mid if c not in numeric_cols],
        "high_to_drop": high,
    }

    print("Classification des colonnes par strategie d'imputation :")
    for key, cols in classification.items():
        print(f"  {key} : {len(cols)} colonne(s)")

    return classification


def impute_median(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Impute par la mediane les colonnes numeriques a faible taux de
    manquant (< 5%). La mediane est preferee a la moyenne car les
    distributions observees en EDA sont systematiquement asymetriques
    (moyenne toujours superieure a la mediane) : imputer avec la moyenne
    aurait introduit un biais vers le haut a chaque valeur imputee.
    """
    for col in columns:
        if col not in df.columns:
            continue
        n = int(df[col].isna().sum())
        if n == 0:
            continue
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"{col} : {n} valeur(s) imputee(s) par la mediane ({median_val:.2f})")
    return df


def impute_mode(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Impute par le mode (valeur la plus frequente) les colonnes
    categorielles a faible taux de manquant, ou les colonnes categorielles
    a taux de manquant intermediaire (le KNN classique ne s'applique pas
    directement a du categoriel non ordonne dans cette implementation).
    """
    for col in columns:
        if col not in df.columns:
            continue
        n = int(df[col].isna().sum())
        if n == 0:
            continue
        mode_series = df[col].mode(dropna=True)
        if mode_series.empty:
            print(f"[ATTENTION] {col} : aucun mode calculable (colonne entierement vide ?), ignoree.")
            continue
        mode_val = mode_series.iloc[0]
        df[col] = df[col].fillna(mode_val)
        print(f"{col} : {n} valeur(s) imputee(s) par le mode ({mode_val})")
    return df


def impute_knn(df: pd.DataFrame, columns: list[str], core_cols: list[str],
               sample_size: int = 8000, n_neighbors: int = 5, random_state: int = 42) -> pd.DataFrame:
    """NON APPELEE PAR DEFAUT dans apply_imputation_strategy (voir cette
    fonction pour la justification : benchmark a plus de 5 minutes de
    calcul pour un gain de precision non demontre a ce stade du projet ;
    la mediane est utilisee a la place pour la tranche 5-95%). Fonction
    conservee et fonctionnelle si vous souhaitez la reactiver plus tard,
    par exemple sur un sous-ensemble cible de variables plutot que sur
    toute la tranche 5-95%.

    Impute les colonnes numeriques a taux de manquant intermediaire
    (5-95%) par K-plus-proches-voisins.

    Justification du choix KNN plutot que la mediane pour cette tranche :
    contrairement a la mediane (qui impute la meme valeur constante pour
    tous les foyers, sans distinction), le KNN cherche des foyers dont le
    profil se ressemble sur d'autres variables numeriques disponibles et
    impute une valeur coherente avec ce profil - plus fidele qu'ecraser
    toute la variance d'une variable moyennement renseignee avec une
    seule valeur.

    Contrainte de calcul et adaptation retenue : un KNNImputer classique
    calcule les distances entre CHAQUE paire de lignes, sur TOUTES les
    colonnes fournies comme base de comparaison. Sur ~90 000 lignes,
    utiliser les ~700 colonnes numeriques disponibles comme base de
    distance est impraticable (des centaines de milliards d'operations).
    Deux adaptations sont donc appliquees : (1) la base de comparaison
    entre foyers est restreinte aux variables coeur d'analyse (core_cols)
    plutot qu'a toutes les colonnes numeriques - des foyers se ressemblent
    dejà bien sur revenu/depense/taille sans avoir besoin de comparer sur
    des centaines de variables de revenu detaille quasi vides ; (2)
    l'imputeur est calibre (fit) sur un echantillon aleatoire de reference
    (sample_size lignes) puis applique (transform) a l'ensemble du jeu de
    donnees, plutot que sur la totalite des lignes. Seules les valeurs
    reellement manquantes sont remplacees ; les valeurs deja observees ne
    sont jamais modifiees.
    """
    if not columns:
        print("Aucune colonne dans la tranche KNN (5%-95% de manquant, numerique).")
        return df

    numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
    # Base de comparaison : variables coeur (deja numeriques et propres a
    # ce stade) + les colonnes elles-memes a imputer, pour que chaque
    # variable participe au moins a sa propre estimation de voisinage.
    feature_cols = [c for c in core_cols if c in numeric_cols and c not in ID_COLUMNS]
    feature_cols += [c for c in columns if c not in feature_cols]

    print(f"Base de comparaison KNN : {len(feature_cols)} colonne(s) "
          f"(variables coeur + colonnes a imputer), echantillon de reference = {sample_size} lignes.")

    reference = df[feature_cols].sample(n=min(sample_size, len(df)), random_state=random_state)

    # Standardisation manuelle (via pandas, pas sklearn.StandardScaler qui
    # ne supporte pas les NaN) : les operations arithmetiques pandas
    # propagent naturellement les NaN sans erreur.
    ref_mean = reference.mean(skipna=True)
    ref_std = reference.std(skipna=True).replace(0, 1)

    reference_scaled = (reference - ref_mean) / ref_std
    full_scaled = (df[feature_cols] - ref_mean) / ref_std

    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputer.fit(reference_scaled.values)
    imputed_scaled = imputer.transform(full_scaled.values)

    imputed_scaled_df = pd.DataFrame(imputed_scaled, columns=feature_cols, index=df.index)
    imputed_original = imputed_scaled_df * ref_std + ref_mean

    for col in columns:
        if col not in feature_cols:
            continue
        missing_mask = df[col].isna()
        n = int(missing_mask.sum())
        if n == 0:
            continue
        df.loc[missing_mask, col] = imputed_original.loc[missing_mask, col]
        print(f"{col} : {n} valeur(s) imputee(s) par KNN (k={n_neighbors}, "
              f"reference={len(reference)} lignes)")

    return df


def drop_high_missing_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Supprime les colonnes classees >= 95% de manquant (hors variables
    coeur), apres qu'elles aient ete documentees. Contrairement a la
    premiere version de ce script (qui se contentait de les documenter
    sans y toucher), l'US-05 exige desormais un jeu de donnees propre :
    ces colonnes sont jugees inexploitables et sont donc retirees.
    """
    existing = [c for c in columns if c in df.columns]
    if existing:
        df = df.drop(columns=existing)
        print(f"{len(existing)} colonne(s) supprimee(s) (>= {HIGH_MISSING_THRESHOLD_PCT}% de manquant, "
              f"hors variables coeur).")
    return df


def apply_imputation_strategy(df: pd.DataFrame, core_cols: list[str], label: str) -> pd.DataFrame:
    """Applique la politique d'imputation complete a un DataFrame :
    classification, puis mediane / mode / KNN / suppression selon la
    categorie de chaque colonne. Fonction unique reutilisee pour
    Interview et Diary, afin de garantir un traitement strictement
    identique.
    """
    print(f"\n--- Classification et imputation ({label}) ---")
    classification = classify_columns_by_missingness(df, core_cols)

    df = impute_median(df, classification["low_numeric"])
    df = impute_mode(df, classification["low_categorical"])

    # KNN evalue et ecarte pour la tranche 5-95% numerique : benchmark sur
    # des dimensions comparables (94699 lignes x ~124 colonnes de base de
    # comparaison, echantillon de reference 8000 lignes) depasse 5 minutes
    # de calcul, disproportionne par rapport au gain de precision attendu
    # sur ce projet. La mediane est l'une des strategies explicitement
    # acceptees par l'US-05 (medianne / KNN / suppression) : retenue ici
    # par souci de performance, avec la meme justification de robustesse
    # aux distributions asymetriques que pour la tranche a faible manquant.
    df = impute_median(df, classification["mid_numeric"])
    df = impute_mode(df, classification["mid_categorical"])
    df = drop_high_missing_columns(df, classification["high_to_drop"])

    return df


# ---------------------------------------------------------------------------
# 3quater. VALEURS ABERRANTES : DETECTION (IQR + Z-SCORE) ET TRAITEMENT
# ---------------------------------------------------------------------------

def detect_and_cap_outliers(df: pd.DataFrame, columns: list[str],
                             iqr_multiplier: float = 3.0, z_threshold: float = 3.0) -> pd.DataFrame:
    """Detecte les valeurs aberrantes par deux methodes combinees (IQR
    avec un facteur 3, plus strict que le facteur 1.5 utilise en simple
    detection lors de l'EDA, et z-score avec seuil 3), puis les ecrete
    (winsorization) a la borne la plus proche plutot que de supprimer la
    ligne entiere - ceci preserve le foyer dans le jeu de donnees tout en
    neutralisant l'effet demesure de la valeur extreme.

    Une colonne '{variable}_CAPPED_FLAG' est ajoutee pour tracer les
    valeurs modifiees, dans la meme logique que le signalement des
    valeurs negatives.

    Perimetre volontairement restreint aux variables numeriques cles
    (deja etudiees en detail en EDA) plutot qu'applique aveuglement a
    toutes les colonnes numeriques : la plupart des ~700 autres colonnes
    numeriques sont des variables de revenu tres specifiques, presque
    entierement vides, ou deja supprimees a l'etape precedente - un
    ecretage systematique dessus n'aurait pas de sens economique.
    """
    for col in columns:
        if col not in df.columns:
            print(f"[ATTENTION] {col} absente, detection aberrantes ignoree.")
            continue

        s = df[col]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low_iqr, high_iqr = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr

        mean, std = s.mean(), s.std()
        if std and std > 0:
            z = (s - mean) / std
            is_outlier_z = z.abs() > z_threshold
        else:
            is_outlier_z = pd.Series(False, index=s.index)

        is_outlier_iqr = (s < low_iqr) | (s > high_iqr)
        is_outlier = is_outlier_iqr | is_outlier_z

        flag_col = f"{col}_CAPPED_FLAG"
        df[flag_col] = is_outlier
        n_outliers = int(is_outlier.sum())

        df[col] = s.clip(lower=low_iqr, upper=high_iqr)

        print(f"{col} : {n_outliers} valeur(s) aberrante(s) detectee(s) "
              f"(IQR x{iqr_multiplier} ou |z|>{z_threshold}), ecretee(s) aux bornes "
              f"[{low_iqr:.2f} ; {high_iqr:.2f}]")

    return df


# ---------------------------------------------------------------------------
# 4. PIPELINES DE NETTOYAGE COMPLETS
# ---------------------------------------------------------------------------

def clean_interview_data(years: list[str], period_label: str) -> pd.DataFrame:
    print(f"\n========== Nettoyage Interview (FMLI) — {period_label} ==========")

    print("\n--- Chargement des donnees brutes ---")
    df = load_survey_for_years(years, survey="interview")

    print("\n--- Etape 1 : dedoublonnage des vagues republiees ---")
    df = deduplicate_republished_quarters(df)

    print("\n--- Etape 2 : reconstitution des depenses trimestrielles (CQ+PQ) ---")
    for base in EXPENDITURE_BASES:
        df = build_full_quarter_expenditure(df, base)

    print("\n--- Etape 3 : verification des colonnes candidates a la suppression ---")
    confirmed_empty = verify_fully_empty_by_year(df, COLUMNS_TO_DROP_KNOWN_EMPTY_INTERVIEW)
    df = drop_known_empty_columns(df, confirmed_empty)

    print("\n--- Etape 4 : signalement des valeurs negatives ---")
    df = flag_negative_values(df, COLUMNS_TO_FLAG_NEGATIVE_INTERVIEW)

    print("\n--- Etape 5 : verification des variables coeur (avant imputation) ---")
    summarize_core_variables_missingness(df, CORE_ANALYSIS_COLUMNS_INTERVIEW, period_label)

    print("\n--- Etape 6 : strategie d'imputation (mediane / mode / KNN / suppression) ---")
    df = apply_imputation_strategy(df, CORE_ANALYSIS_COLUMNS_INTERVIEW, period_label)

    print("\n--- Etape 7 : detection et traitement des valeurs aberrantes (IQR + z-score) ---")
    df = detect_and_cap_outliers(df, COLUMNS_TO_CAP_OUTLIERS_INTERVIEW)

    return df


def clean_diary_data(years: list[str], period_label: str) -> pd.DataFrame:
    print(f"\n========== Nettoyage Diary (FMLD) — {period_label} ==========")

    print("\n--- Chargement des donnees brutes ---")
    df = load_survey_for_years(years, survey="diary")

    print("\n--- Etape 1 : dedoublonnage des vagues republiees (filet de securite) ---")
    df = deduplicate_republished_quarters(df)

    print("\n--- Etape 2 : verification des colonnes candidates a la suppression ---")
    confirmed_empty = verify_fully_empty_by_year(df, COLUMNS_TO_DROP_KNOWN_EMPTY_DIARY)
    df = drop_known_empty_columns(df, confirmed_empty)

    print("\n--- Etape 3 : signalement des valeurs negatives ---")
    df = flag_negative_values(df, COLUMNS_TO_FLAG_NEGATIVE_DIARY)

    print("\n--- Etape 4 : verification des variables coeur (avant imputation) ---")
    summarize_core_variables_missingness(df, CORE_ANALYSIS_COLUMNS_DIARY, period_label)

    print("\n--- Etape 5 : strategie d'imputation (mediane / mode / KNN / suppression) ---")
    df = apply_imputation_strategy(df, CORE_ANALYSIS_COLUMNS_DIARY, period_label)

    print("\n--- Etape 6 : detection et traitement des valeurs aberrantes (IQR + z-score) ---")
    df = detect_and_cap_outliers(df, COLUMNS_TO_CAP_OUTLIERS_DIARY)

    return df


def save_clean_data(df: pd.DataFrame, filename: str) -> None:
    """Sauvegarde le jeu nettoye avec un horodatage de version dans le
    nom de fichier (VERSION_TAG), plutot que d'ecraser silencieusement
    une version precedente du meme fichier.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    versioned_filename = filename.replace(".csv", f"_v{VERSION_TAG}.csv")
    output_path = OUTPUT_DIR / versioned_filename
    df.to_csv(output_path, index=False)
    print(f"\nDonnees nettoyees sauvegardees : {output_path} ({len(df)} lignes, {df.shape[1]} colonnes)")


# ---------------------------------------------------------------------------
# 5. EXECUTION PRINCIPALE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ============================================================
    # INTERVIEW SURVEY (FMLI)
    # ============================================================
    df_int_p1_clean = clean_interview_data(PERIOD_1_YEARS, "Periode 1 (2017-2020)")
    save_clean_data(df_int_p1_clean, "interview_period1_clean.csv")

    df_int_p2_clean = clean_interview_data(PERIOD_2_YEARS, "Periode 2 (2021-2024)")
    save_clean_data(df_int_p2_clean, "interview_period2_clean.csv")

    # ============================================================
    # DIARY SURVEY (FMLD)
    # ============================================================
    df_dia_p1_clean = clean_diary_data(PERIOD_1_YEARS, "Periode 1 (2017-2020)")
    save_clean_data(df_dia_p1_clean, "diary_period1_clean.csv")

    df_dia_p2_clean = clean_diary_data(PERIOD_2_YEARS, "Periode 2 (2021-2024)")
    save_clean_data(df_dia_p2_clean, "diary_period2_clean.csv")

    print("\n========== Recapitulatif ==========")
    print(f"Interview Periode 1 : {len(df_int_p1_clean)} lignes apres nettoyage")
    print(f"Interview Periode 2 : {len(df_int_p2_clean)} lignes apres nettoyage")
    print(f"Diary Periode 1     : {len(df_dia_p1_clean)} lignes apres nettoyage")
    print(f"Diary Periode 2     : {len(df_dia_p2_clean)} lignes apres nettoyage")