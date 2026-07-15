"""
US-04 : Chargement et audit qualite / EDA de la Consumer Expenditure Survey (CEX)
Composantes : Interview Survey (table FMLI) ET Diary Survey (table FMLD)

Structure de dossiers attendue (celle utilisee dans le projet cex_project) :

data/
  intrvw17/  <- contenu decompresse de intrvw17.zip
  intrvw18/  <- contenu decompresse de intrvw18.zip
  ...
  intrvw24/  <- contenu decompresse de intrvw24.zip
  diary17/   <- contenu decompresse de diary17.zip
  diary18/   <- contenu decompresse de diary18.zip
  ...
  diary24/   <- contenu decompresse de diary24.zip

Chaque dossier Interview contient plusieurs fichiers fmli*.csv (un par trimestre).
Chaque dossier Diary contient plusieurs fichiers fmld*.csv (un par periode de collecte).
Le nom exact varie selon l'annee. Ce script recherche automatiquement tous les
fichiers commencant par "fmli" (Interview) ou "fmld" (Diary) dans chaque dossier.

IMPORTANT : Interview et Diary ne couvrent generalement pas les memes foyers.
Elles sont chargees et auditees separement dans ce script, pas fusionnees,
car une fusion ligne a ligne par foyer n'est pas methodologiquement valide
sans un traitement specifique (hors perimetre de US-04).
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. CONFIGURATION : a adapter a votre arborescence reelle une fois telecharge
# ---------------------------------------------------------------------------

DATA_ROOT = Path("data")

PERIOD_1_YEARS = ["2017", "2018", "2019", "2020"]
PERIOD_2_YEARS = ["2021", "2022", "2023", "2024"]

# Colonnes a exclure de toute analyse statistique (identifiants techniques)
ID_COLUMNS = ["NEWID", "CUID"]

# Colonne de ponderation
WEIGHT_COLUMN = "FINLWT21"

# ---------------------------------------------------------------------------
# 2. CHARGEMENT DES FICHIERS FMLI
# ---------------------------------------------------------------------------

def find_survey_files(year_folder: Path, prefix: str) -> list[Path]:
    """Trouve tous les fichiers {prefix}*.csv dans un dossier donne, insensible a la casse.
    prefix attendu : 'fmli' (Interview) ou 'fmld' (Diary).
    """
    pattern = "".join(f"[{c.upper()}{c.lower()}]" for c in prefix) + "*.csv"
    return sorted(year_folder.glob(pattern))


def load_survey_for_years(years: list[str], survey: str, data_root: Path = DATA_ROOT) -> pd.DataFrame:
    """Charge et concatene tous les fichiers d'une enquete (Interview ou Diary)
    pour une liste d'annees.

    survey : 'interview' (dossier intrvwYY, fichiers fmli*.csv)
             ou 'diary' (dossier diaryYY, fichiers fmld*.csv)

    Convention de dossiers : annee sur 2 chiffres, sans underscore,
    ex: intrvw17, intrvw18, ..., diary17, diary18, ...

    Ajoute une colonne SOURCE_YEAR pour tracer l'origine de chaque ligne,
    utile pour comparer les deux periodes plus tard.
    """
    folder_prefix = "intrvw" if survey == "interview" else "diary"
    file_prefix = "fmli" if survey == "interview" else "fmld"

    frames = []
    for year in years:
        year_2digits = year[-2:]  # ex: "2017" -> "17"
        year_folder = data_root / f"{folder_prefix}{year_2digits}"
        if not year_folder.exists():
            print(f"[ATTENTION] Dossier introuvable, annee ignoree : {year_folder}")
            continue

        files = find_survey_files(year_folder, file_prefix)
        if not files:
            print(f"[ATTENTION] Aucun fichier {file_prefix}*.csv trouve dans {year_folder}")
            continue

        for f in files:
            try:
                df = pd.read_csv(f, low_memory=False)
            except UnicodeDecodeError:
                # certains fichiers CEX sont encodes en latin-1 plutot qu'en utf-8
                df = pd.read_csv(f, low_memory=False, encoding="latin-1")
            df["SOURCE_YEAR"] = year
            df["SOURCE_FILE"] = f.name
            frames.append(df)
            print(f"Charge : {f}  ({len(df)} lignes, {df.shape[1]} colonnes)")

    if not frames:
        raise FileNotFoundError(
            "Aucun fichier FMLI charge. Verifiez la structure du dossier 'data/'."
        )

    combined = pd.concat(frames, ignore_index=True, sort=False)
    print(f"\nTotal charge : {len(combined)} lignes, {combined.shape[1]} colonnes")
    return combined


# ---------------------------------------------------------------------------
# 3. VOLUMETRIE ET GRANULARITE (US-03 / premiere partie de US-04)
# ---------------------------------------------------------------------------

def summarize_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Nombre de foyers (lignes) par annee source."""
    return df.groupby("SOURCE_YEAR").size().rename("nb_lignes").reset_index()


# ---------------------------------------------------------------------------
# 4. TAUX DE VALEURS MANQUANTES PAR VARIABLE
# ---------------------------------------------------------------------------

def missing_rate_report(df: pd.DataFrame, exclude_cols: list[str] = None) -> pd.DataFrame:
    """Calcule le taux de valeurs manquantes par colonne, trie du plus eleve au plus faible.

    Note : dans les fichiers CEX, une valeur manquante peut apparaitre comme NaN,
    mais aussi parfois comme un code negatif special (ex: -1) selon la variable.
    Ce rapport ne detecte que les NaN standards ; les codes speciaux sont a
    verifier separement via le dictionnaire de variables du BLS.
    """
    exclude_cols = exclude_cols or []
    cols = [c for c in df.columns if c not in exclude_cols]
    report = pd.DataFrame({
        "variable": cols,
        "taux_manquant_pct": [df[c].isna().mean() * 100 for c in cols],
        "type": [str(df[c].dtype) for c in cols],
    })
    return report.sort_values("taux_manquant_pct", ascending=False).reset_index(drop=True)


def missing_rate_by_group(df: pd.DataFrame, target_col: str, group_col: str) -> pd.DataFrame:
    """Croise le taux de manquant d'une variable cible avec une variable de groupe
    (ex: le revenu croise avec la region), pour verifier si la non-reponse
    est aleatoire ou biaisee vers certains profils.
    """
    return (
        df.groupby(group_col)[target_col]
        .apply(lambda s: s.isna().mean() * 100)
        .rename("taux_manquant_pct")
        .reset_index()
        .sort_values("taux_manquant_pct", ascending=False)
    )


# ---------------------------------------------------------------------------
# 5. STATISTIQUES DESCRIPTIVES, PONDEREES ET NON PONDEREES
# ---------------------------------------------------------------------------

def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna()
    if mask.sum() == 0:
        return np.nan
    return np.average(values[mask], weights=weights[mask])


def weighted_std(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna()
    if mask.sum() == 0:
        return np.nan
    avg = weighted_mean(values, weights)
    variance = np.average((values[mask] - avg) ** 2, weights=weights[mask])
    return np.sqrt(variance)


def descriptive_stats(df: pd.DataFrame, numeric_cols: list[str],
                       weight_col: str = WEIGHT_COLUMN) -> pd.DataFrame:
    """Produit les statistiques descriptives non ponderees ET ponderees,
    pour objectiver l'ecart entre les deux (point souleve dans l'US-04).
    """
    rows = []
    for col in numeric_cols:
        if col not in df.columns:
            continue
        s = df[col]
        rows.append({
            "variable": col,
            "moyenne_non_ponderee": s.mean(),
            "moyenne_ponderee": weighted_mean(s, df[weight_col]) if weight_col in df.columns else np.nan,
            "mediane": s.median(),
            "ecart_type_non_pondere": s.std(),
            "ecart_type_pondere": weighted_std(s, df[weight_col]) if weight_col in df.columns else np.nan,
            "min": s.min(),
            "max": s.max(),
        })
    return pd.DataFrame(rows)


def build_full_quarter_expenditure(df: pd.DataFrame, base_name: str) -> pd.DataFrame:
    """Reconstitue une depense trimestrielle complete en combinant les
    variables CQ (current quarter) et PQ (previous quarter), conformement
    a la convention documentee dans le CE PUMD Getting Started Guide du BLS :
    le suffixe CQ ne couvre qu'une partie du trimestre calendaire (du debut
    du trimestre de collecte jusqu'a la date d'interview), et PQ couvre le
    reste (jusqu'a la fin du trimestre calendaire precedent). Sommer CQ+PQ
    reconstitue une periode de reference complete, plutot que d'utiliser
    CQ seule qui sous-estime fortement la depense pour les interviews
    realisees tot dans le trimestre.

    Ajoute au DataFrame une nouvelle colonne nommee '{base_name}_FULL'.
    """
    cq_col = f"{base_name}CQ"
    pq_col = f"{base_name}PQ"
    missing = [c for c in (cq_col, pq_col) if c not in df.columns]
    if missing:
        print(f"[ATTENTION] Colonne(s) manquante(s) pour {base_name} : {missing}")
        return df

    full_col = f"{base_name}_FULL"
    df[full_col] = df[cq_col].fillna(0) + df[pq_col].fillna(0)
    print(f"Variable construite : {full_col} = {cq_col} + {pq_col}")
    return df


# ---------------------------------------------------------------------------
# 6. DETECTION D'ECRETAGE (TOPCODING)
# ---------------------------------------------------------------------------

def detect_topcoding(df: pd.DataFrame, col: str, top_share_threshold: float = 0.01) -> dict:
    """Signale un ecretage probable si la valeur maximale d'une colonne
    revient anormalement souvent (plus que top_share_threshold de l'echantillon),
    ce qui est la signature typique d'un plafonnement artificiel des valeurs extremes.
    """
    s = df[col].dropna()
    if s.empty:
        return {"variable": col, "max_value": np.nan, "share_at_max": np.nan, "suspect": False}
    max_val = s.max()
    share_at_max = (s == max_val).mean()
    return {
        "variable": col,
        "max_value": max_val,
        "share_at_max_pct": share_at_max * 100,
        "suspect_topcoding": share_at_max > top_share_threshold,
    }


# ---------------------------------------------------------------------------
# 7. CORRELATIONS
# ---------------------------------------------------------------------------

def correlation_matrix(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    existing = [c for c in numeric_cols if c in df.columns]
    return df[existing].corr()


# ---------------------------------------------------------------------------
# 7bis. RECHERCHE DE COLONNES PAR MOT-CLE (pour trouver les vrais noms
#       plutot que de les deviner, ex: le total des depenses dans FMLD)
# ---------------------------------------------------------------------------

def find_columns_by_keyword(df: pd.DataFrame, keyword: str) -> list[str]:
    """Retourne toutes les colonnes dont le nom contient le mot-cle donne
    (insensible a la casse). Utile pour trouver le vrai nom d'une variable
    sans avoir a deviner ou a ouvrir le dictionnaire xlsx pour chaque cas.
    """
    keyword = keyword.upper()
    return [c for c in df.columns if keyword in c.upper()]


def list_all_columns(df: pd.DataFrame, output_path: str = "outputs/all_columns.csv") -> pd.DataFrame:
    """Exporte la liste complete des colonnes d'un DataFrame (nom + type),
    pour pouvoir la parcourir dans Excel plutot que de deviner les noms
    de variables un par un.
    """
    cols_df = pd.DataFrame({
        "variable": df.columns,
        "type": [str(df[c].dtype) for c in df.columns],
    })
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    cols_df.to_csv(output_path, index=False)
    print(f"Liste complete des {len(cols_df)} colonnes sauvegardee : {output_path}")
    return cols_df


# ---------------------------------------------------------------------------
# 8. DETECTION DES DOUBLONS (ex: trimestres Q1/Q1x publies deux fois)
# ---------------------------------------------------------------------------

def detect_duplicate_newid(df: pd.DataFrame) -> pd.DataFrame:
    """Compte les NEWID qui apparaissent plus d'une fois dans le DataFrame.

    Un NEWID identifie normalement un foyer a un trimestre precis ; s'il
    apparait plusieurs fois, c'est le signe d'un doublon (par exemple un
    trimestre Q1 charge a la fois depuis le fichier de l'annee courante
    et depuis le fichier "x" de l'annee suivante, comme observe sur le CEX).
    """
    if "NEWID" not in df.columns:
        print("[ATTENTION] Colonne NEWID absente, impossible de verifier les doublons.")
        return pd.DataFrame()

    counts = df["NEWID"].value_counts()
    duplicated = counts[counts > 1].reset_index()
    duplicated.columns = ["NEWID", "nb_occurrences"]
    return duplicated


def duplicate_summary_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Pour chaque annee source, compte combien de NEWID sont dupliques,
    en croisant avec la colonne SOURCE_FILE pour voir quels fichiers
    se chevauchent (utile pour reperer precisement les paires Q1 / Q1x).
    """
    dup_ids = detect_duplicate_newid(df)["NEWID"] if "NEWID" in df.columns else []
    if len(dup_ids) == 0:
        return pd.DataFrame()
    subset = df[df["NEWID"].isin(dup_ids)]
    return subset.groupby(["SOURCE_YEAR", "SOURCE_FILE"]).size().reset_index(name="nb_lignes_dupliquees")


# ---------------------------------------------------------------------------
# 9. VISUALISATIONS (valeurs manquantes, doublons)
# ---------------------------------------------------------------------------

def plot_missing_rate(report_df: pd.DataFrame, top_n: int = 20,
                       title: str = "Taux de valeurs manquantes (top variables)",
                       output_path: str = "outputs/missing_rate.png") -> None:
    """Barre horizontale des variables avec le plus haut taux de valeurs
    manquantes. Sauvegarde l'image plutot que de l'afficher a l'ecran,
    pour fonctionner aussi bien en script qu'en notebook.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = report_df.head(top_n).sort_values("taux_manquant_pct")

    plt.figure(figsize=(9, max(4, 0.35 * len(top))))
    plt.barh(top["variable"], top["taux_manquant_pct"], color="#c0392b")
    plt.xlabel("Taux de valeurs manquantes (%)")
    plt.title(title)
    plt.tight_layout()

    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Graphique sauvegarde : {output_path}")


def plot_missing_heatmap(df: pd.DataFrame, cols: list[str],
                          title: str = "Carte des valeurs manquantes",
                          output_path: str = "outputs/missing_heatmap.png",
                          sample_rows: int = 500) -> None:
    """Carte visuelle (points rouges = manquant) sur un echantillon de lignes,
    pour voir d'un coup d'oeil si les manquants sont concentres sur certaines
    lignes/colonnes ou repartis au hasard.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    existing = [c for c in cols if c in df.columns]
    sample = df[existing].sample(min(sample_rows, len(df)), random_state=42)

    plt.figure(figsize=(9, 6))
    plt.imshow(sample.isna().T, aspect="auto", cmap="Reds", interpolation="none")
    plt.yticks(range(len(existing)), existing, fontsize=8)
    plt.xlabel(f"Echantillon de {len(sample)} lignes")
    plt.title(title)
    plt.tight_layout()

    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Graphique sauvegarde : {output_path}")


def plot_duplicate_summary(dup_by_year: pd.DataFrame,
                            title: str = "Lignes dupliquees par fichier source",
                            output_path: str = "outputs/duplicates_by_file.png") -> None:
    """Barres du nombre de lignes dupliquees par fichier source, pour
    visualiser precisement quels fichiers se chevauchent (typiquement
    les paires Q1 / Q1x du CEX).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if dup_by_year.empty:
        print("Aucun doublon a afficher.")
        return

    labels = dup_by_year["SOURCE_FILE"]
    plt.figure(figsize=(9, max(4, 0.35 * len(labels))))
    plt.barh(labels, dup_by_year["nb_lignes_dupliquees"], color="#e67e22")
    plt.xlabel("Nombre de lignes dupliquees (meme NEWID)")
    plt.title(title)
    plt.tight_layout()

    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Graphique sauvegarde : {output_path}")


def plot_distribution(df: pd.DataFrame, col: str, bins: int = 60,
                       clip_percentiles: tuple = (1, 99),
                       output_path: str = None) -> None:
    """Histogramme de distribution d'une variable numerique.

    clip_percentiles permet de recadrer l'axe des x entre le 1er et le 99e
    percentile par defaut, pour eviter qu'une poignee de valeurs extremes
    n'ecrase la lisibilite du graphique (frequent avec le revenu/depense).
    Les valeurs extremes ne sont pas supprimees des donnees, seulement de
    l'affichage.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if col not in df.columns:
        print(f"[ATTENTION] Colonne {col} absente, histogramme ignore.")
        return

    s = df[col].dropna()
    if s.empty:
        print(f"[ATTENTION] Colonne {col} entierement vide, histogramme ignore.")
        return

    lo, hi = np.percentile(s, clip_percentiles)

    plt.figure(figsize=(8, 5))
    plt.hist(s, bins=bins, range=(lo, hi), color="#2980b9", edgecolor="white")
    plt.axvline(s.mean(), color="#c0392b", linestyle="--", label=f"Moyenne = {s.mean():.0f}")
    plt.axvline(s.median(), color="#27ae60", linestyle="--", label=f"Mediane = {s.median():.0f}")
    plt.xlabel(col)
    plt.ylabel("Nombre de foyers")
    plt.title(f"Distribution de {col} (entre le {clip_percentiles[0]}e et le {clip_percentiles[1]}e percentile)")
    plt.legend()
    plt.tight_layout()

    output_path = output_path or f"outputs/distribution_{col}.png"
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Graphique sauvegarde : {output_path}")


# ---------------------------------------------------------------------------
# 10. FOYERS DISTINCTS (CUID) VS LIGNES (NEWID = foyer-trimestre)
# ---------------------------------------------------------------------------

def unique_households_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Compte le nombre de foyers distincts (CUID unique) par annee source,
    a distinguer du nombre de lignes (NEWID = foyer-trimestre). Un meme
    foyer peut apparaitre plusieurs fois dans l'annee s'il a ete interroge
    a plusieurs trimestres consecutifs.
    """
    if "CUID" not in df.columns:
        print("[ATTENTION] Colonne CUID absente, impossible de compter les foyers distincts.")
        return pd.DataFrame()

    lignes = df.groupby("SOURCE_YEAR").size().rename("nb_lignes")
    foyers = df.groupby("SOURCE_YEAR")["CUID"].nunique().rename("nb_foyers_distincts")
    return pd.concat([lignes, foyers], axis=1).reset_index()


# ---------------------------------------------------------------------------
# 11. VERIFICATION DE L'HYPOTHESE "BOUNDING INTERVIEW"
# ---------------------------------------------------------------------------

def check_bounding_interview_hypothesis(df: pd.DataFrame, expense_col: str,
                                         interview_num_col: str = "INTERI",
                                         zero_threshold: float = 1.0) -> pd.DataFrame:
    """Verifie si les lignes a depense quasi nulle sont concentrees sur un
    numero d'interview particulier (typiquement la premiere interview,
    dite 'bounding interview', qui sert de reference et n'est pas censee
    etre utilisee pour les statistiques de depense selon la methodologie CEX).

    Pour chaque valeur de INTERI, calcule : le nombre de lignes, le nombre
    et le pourcentage de lignes a depense quasi nulle, et la moyenne de
    depense en excluant ces zeros.
    """
    if interview_num_col not in df.columns:
        print(f"[ATTENTION] Colonne {interview_num_col} absente, hypothese non verifiable.")
        return pd.DataFrame()
    if expense_col not in df.columns:
        print(f"[ATTENTION] Colonne {expense_col} absente.")
        return pd.DataFrame()

    rows = []
    for interi_val, group in df.groupby(interview_num_col):
        s = group[expense_col]
        near_zero = (s.abs() < zero_threshold)
        rows.append({
            interview_num_col: interi_val,
            "nb_lignes": len(s),
            "nb_quasi_zero": int(near_zero.sum()),
            "pct_quasi_zero": round(near_zero.mean() * 100, 2),
            "moyenne_avec_zeros": round(s.mean(), 2),
            "moyenne_sans_zeros": round(s[~near_zero].mean(), 2) if (~near_zero).sum() > 0 else np.nan,
        })
    return pd.DataFrame(rows).sort_values(interview_num_col).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 8. EXECUTION PRINCIPALE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- A adapter : liste des variables de depense/revenu agregees que vous
    # retenez pour l'EDA. Ces noms sont EXACTS dans le dictionnaire du BLS
    # a verifier une fois le dictionnaire xlsx ouvert (les noms varient un peu
    # selon les annees, ex: TOTEXPCQ, FOODCQ, HOUSCQ, TRANSCQ, FINCBTAX, etc.)
    CANDIDATE_NUMERIC_COLS = [
        "TOTEXPCQ",   # depense totale du trimestre (exemple, a verifier)
        "FINCBTXM",   # revenu avant impot (exemple, a verifier)
        "FOODCQ",     # depense alimentation (exemple, a verifier)
        "HOUSCQ",     # depense logement (exemple, a verifier)
    ]

    # Variables candidates pour la Diary Survey (petites depenses frequentes).
    # FOODTOT est la seule variable de depense deja agregee confirmee dans FMLD
    # (verifie via find_columns_by_keyword et list_all_columns : Diary ne
    # contient pas de depense totale toutes categories, contrairement a FMLI).
    CANDIDATE_NUMERIC_COLS_DIARY = [
        "FOODTOT",
    ]

    print("========== INTERVIEW SURVEY (FMLI) ==========")

    print("\n=== Chargement Periode 1 (2017-2020) ===")
    df_int_p1 = load_survey_for_years(PERIOD_1_YEARS, survey="interview")

    print("\n=== Chargement Periode 2 (2021-2024) ===")
    df_int_p2 = load_survey_for_years(PERIOD_2_YEARS, survey="interview")

    print("\n=== Liste complete des colonnes Interview ===")
    all_cols_int = list_all_columns(df_int_p1, output_path="outputs/all_columns_interview.csv")

    print("\n=== Foyers distincts vs lignes (Interview) ===")
    print(unique_households_by_year(df_int_p1))
    print(unique_households_by_year(df_int_p2))

    print("\n=== Volumetrie Interview ===")
    print(summarize_volume(df_int_p1))
    print(summarize_volume(df_int_p2))

    print("\n=== Taux de valeurs manquantes Interview (Periode 1) ===")
    miss_int_p1 = missing_rate_report(df_int_p1, exclude_cols=ID_COLUMNS)
    print(miss_int_p1.head(20))

    print("\n=== Reconstitution des depenses trimestrielles completes (CQ+PQ) ===")
    for base in ["TOTEXP", "FOOD", "HOUS"]:
        df_int_p1 = build_full_quarter_expenditure(df_int_p1, base)
    CANDIDATE_NUMERIC_COLS_FULL = ["TOTEXP_FULL", "FINCBTXM", "FOOD_FULL", "HOUS_FULL"]

    print("\n=== Statistiques descriptives Interview (Periode 1, CQ+PQ) ===")
    stats_int_p1_full = descriptive_stats(df_int_p1, CANDIDATE_NUMERIC_COLS_FULL)
    print(stats_int_p1_full)

    print("\n=== Detection d'ecretage Interview (Periode 1, CQ+PQ) ===")
    for col in CANDIDATE_NUMERIC_COLS_FULL:
        if col in df_int_p1.columns:
            print(detect_topcoding(df_int_p1, col))

    print("\n=== Detection des doublons Interview (Periode 1) ===")
    dup_int_p1 = detect_duplicate_newid(df_int_p1)
    print(f"Nombre de NEWID dupliques : {len(dup_int_p1)}")
    dup_by_file_p1 = duplicate_summary_by_year(df_int_p1)
    print(dup_by_file_p1)

    print("\n=== Generation des graphiques (Periode 1) ===")
    plot_missing_rate(miss_int_p1, top_n=20,
                       title="Interview - Taux de valeurs manquantes (Periode 1)",
                       output_path="outputs/missing_rate_interview_period1.png")
    plot_missing_heatmap(df_int_p1, CANDIDATE_NUMERIC_COLS,
                          title="Interview - Carte des manquants (Periode 1)",
                          output_path="outputs/missing_heatmap_interview_period1.png")
    plot_duplicate_summary(dup_by_file_p1,
                            title="Interview - Doublons par fichier (Periode 1)",
                            output_path="outputs/duplicates_interview_period1.png")
    for col in CANDIDATE_NUMERIC_COLS_FULL:
        plot_distribution(df_int_p1, col,
                           output_path=f"outputs/distribution_interview_{col}.png")

    print("\n=== Verification hypothese 'bounding interview' (Periode 1, CQ+PQ) ===")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    for col in CANDIDATE_NUMERIC_COLS_FULL:
        print(f"\n--- {col} par numero d'interview (INTERI) ---")
        print(check_bounding_interview_hypothesis(df_int_p1, col))

    print("\n=== Matrice de correlation Interview (Periode 1, CQ+PQ) ===")
    print(correlation_matrix(df_int_p1, CANDIDATE_NUMERIC_COLS_FULL))

    print("\n\n========== DIARY SURVEY (FMLD) ==========")

    print("\n=== Chargement Periode 1 (2017-2020) ===")
    df_dia_p1 = load_survey_for_years(PERIOD_1_YEARS, survey="diary")

    print("\n=== Chargement Periode 2 (2021-2024) ===")
    df_dia_p2 = load_survey_for_years(PERIOD_2_YEARS, survey="diary")

    print("\n=== Liste complete des colonnes Diary ===")
    all_cols_dia = list_all_columns(df_dia_p1, output_path="outputs/all_columns_diary.csv")

    print("\n=== Volumetrie Diary ===")
    print(summarize_volume(df_dia_p1))
    print(summarize_volume(df_dia_p2))

    print("\n=== Taux de valeurs manquantes Diary (Periode 1) ===")
    miss_dia_p1 = missing_rate_report(df_dia_p1, exclude_cols=ID_COLUMNS)
    print(miss_dia_p1.head(20))

    print("\n=== Recherche du nom reel de la depense totale dans Diary ===")
    print("Colonnes contenant 'TOTAL' :", find_columns_by_keyword(df_dia_p1, "TOTAL"))
    print("Colonnes contenant 'EXP' :", find_columns_by_keyword(df_dia_p1, "EXP"))

    print("\n=== Statistiques descriptives Diary (Periode 1) ===")
    stats_dia_p1 = descriptive_stats(df_dia_p1, CANDIDATE_NUMERIC_COLS_DIARY)
    print(stats_dia_p1)

    print("\n=== Foyers distincts vs lignes (Diary) ===")
    print(unique_households_by_year(df_dia_p1))
    print(unique_households_by_year(df_dia_p2))

    for col in CANDIDATE_NUMERIC_COLS_DIARY:
        plot_distribution(df_dia_p1, col,
                           output_path=f"outputs/distribution_diary_{col}.png")

    # Sauvegarde des rapports pour reprise ulterieure
    Path("outputs").mkdir(exist_ok=True)
    miss_int_p1.to_csv("outputs/missing_rate_interview_period1.csv", index=False)
    stats_int_p1_full.to_csv("outputs/descriptive_stats_interview_period1.csv", index=False)
    miss_dia_p1.to_csv("outputs/missing_rate_diary_period1.csv", index=False)
    stats_dia_p1.to_csv("outputs/descriptive_stats_diary_period1.csv", index=False)
    print("\nRapports sauvegardes dans outputs/")