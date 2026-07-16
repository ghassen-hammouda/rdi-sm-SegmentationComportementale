"""
US-04 : Chargement et audit qualite / EDA de la Consumer Expenditure Survey (CEX)
Composantes : Interview Survey (table FMLI) ET Diary Survey (table FMLD)

Perimetre de ce script : AUDIT / EDA UNIQUEMENT.
Le nettoyage (traitement des manquants, des aberrants, des doublons) fait
l'objet d'une autre User Story et n'est pas fait ici. Ce script se contente
de DETECTER et DOCUMENTER les problemes, pas de les corriger.

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

# Codes numeriques speciaux de "manquant" frequemment utilises dans le CEX
# (a confirmer variable par variable dans le dictionnaire BLS : la
# signification exacte d'un code peut varier d'une variable a l'autre).
SPECIAL_MISSING_CODES = [-1, -2]


# ---------------------------------------------------------------------------
# 2. CHARGEMENT DES FICHIERS FMLI / FMLD
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
            "Aucun fichier charge. Verifiez la structure du dossier 'data/'."
        )

    combined = pd.concat(frames, ignore_index=True, sort=False)
    print(f"\nTotal charge : {len(combined)} lignes, {combined.shape[1]} colonnes")
    return combined


# ---------------------------------------------------------------------------
# 2bis. COHERENCE DU SCHEMA ENTRE ANNEES
# ---------------------------------------------------------------------------

def schema_consistency_report(years: list[str], survey: str, data_root: Path = DATA_ROOT) -> pd.DataFrame:
    """Pour chaque annee, lit uniquement l'entete du premier fichier trouve
    (pas tout le fichier) et compare les colonnes d'une annee a l'autre.

    C'est une verification importante et souvent oubliee sur le CEX : le BLS
    renomme ou retire des variables d'une annee sur l'autre (ex: variables
    revisees suite a la refonte du questionnaire en 2015, nouvelles variables
    ajoutees a partir de certaines annees). Sans cette verification,
    pd.concat(sort=False) cree silencieusement des colonnes remplies de NaN
    pour les annees qui ne les ont pas, ce qui peut etre confondu a tort avec
    de la non-reponse.
    """
    folder_prefix = "intrvw" if survey == "interview" else "diary"
    file_prefix = "fmli" if survey == "interview" else "fmld"

    cols_by_year = {}
    for year in years:
        year_2digits = year[-2:]
        year_folder = data_root / f"{folder_prefix}{year_2digits}"
        files = find_survey_files(year_folder, file_prefix) if year_folder.exists() else []
        if not files:
            continue
        header = pd.read_csv(files[0], nrows=0).columns.tolist()
        cols_by_year[year] = set(header)

    if not cols_by_year:
        print("[ATTENTION] Aucune annee disponible pour verifier le schema.")
        return pd.DataFrame()

    all_cols = set.union(*cols_by_year.values())
    common_cols = set.intersection(*cols_by_year.values())

    rows = []
    for col in sorted(all_cols):
        rows.append({
            "variable": col,
            "presente_dans_toutes_les_annees": col in common_cols,
            "annees_presentes": ",".join(y for y in cols_by_year if col in cols_by_year[y]),
            "annees_absentes": ",".join(y for y in cols_by_year if col not in cols_by_year[y]),
        })
    report = pd.DataFrame(rows).sort_values(
        "presente_dans_toutes_les_annees").reset_index(drop=True)

    print(f"Colonnes communes a toutes les annees : {len(common_cols)}/{len(all_cols)}")
    n_variables_instables = (~report["presente_dans_toutes_les_annees"]).sum()
    if n_variables_instables > 0:
        print(f"[ATTENTION] {n_variables_instables} variable(s) absente(s) sur au moins une annee "
              f"-> a verifier avant toute concatenation entre annees.")
    return report


# ---------------------------------------------------------------------------
# 3. VOLUMETRIE ET GRANULARITE
# ---------------------------------------------------------------------------

def summarize_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Nombre de foyers (lignes) par annee source."""
    return df.groupby("SOURCE_YEAR").size().rename("nb_lignes").reset_index()


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
# 4. TAUX DE VALEURS MANQUANTES PAR VARIABLE
# ---------------------------------------------------------------------------

def missing_rate_report(df: pd.DataFrame, exclude_cols: list[str] = None) -> pd.DataFrame:
    """Calcule le taux de valeurs manquantes (NaN standard) par colonne,
    trie du plus eleve au plus faible.

    Note : les codes numeriques speciaux (ex: -1, -2) ne sont pas comptes
    ici comme manquants ; voir special_missing_code_report pour ceux-la.
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


def special_missing_code_report(df: pd.DataFrame, cols: list[str],
                                 codes: list = SPECIAL_MISSING_CODES) -> pd.DataFrame:
    """Complete missing_rate_report en verifiant si des codes numeriques
    speciaux (souvent -1 = non applicable, -2 = refus/ne sait pas selon
    les variables du CEX) sont utilises a la place d'un NaN standard.
    Detection uniquement : la signification exacte de chaque code est a
    confirmer au cas par cas dans le dictionnaire de variables du BLS
    avant tout traitement (US suivante).
    """
    rows = []
    for col in cols:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        s = df[col]
        for code in codes:
            share = (s == code).mean() * 100
            if share > 0:
                rows.append({"variable": col, "code_special": code, "pct_lignes": round(share, 2)})
    result = pd.DataFrame(rows)
    if result.empty:
        print("Aucun code special detecte parmi les colonnes verifiees.")
    return result


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
    pour objectiver l'ecart entre les deux.
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
# 5bis. VERIFICATIONS DE COHERENCE LOGIQUE (sanity checks, detection seule)
# ---------------------------------------------------------------------------

def logical_consistency_checks(df: pd.DataFrame, expense_cols: list[str],
                                weight_col: str = WEIGHT_COLUMN) -> pd.DataFrame:
    """Verifie des regles de bon sens qui, si violees, signalent un probleme
    de qualite ou une variable mal comprise. Detection uniquement, aucune
    correction n'est appliquee ici :
    - la ponderation doit etre strictement positive
    - une depense (CQ, PQ, FULL) ne doit normalement pas etre negative
      (contrairement au revenu, qui peut legitimement etre negatif, ex:
      travailleur independant en deficit)
    """
    rows = []

    if weight_col in df.columns:
        w = df[weight_col]
        rows.append({"controle": f"{weight_col} <= 0", "nb_lignes_concernees": int((w <= 0).sum())})
        rows.append({"controle": f"{weight_col} manquant", "nb_lignes_concernees": int(w.isna().sum())})
    else:
        print(f"[ATTENTION] Colonne de ponderation {weight_col} absente.")

    for col in expense_cols:
        if col not in df.columns:
            continue
        s = df[col]
        rows.append({"controle": f"{col} < 0", "nb_lignes_concernees": int((s < 0).sum())})

    return pd.DataFrame(rows)


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
        return {"variable": col, "max_value": np.nan, "share_at_max_pct": np.nan, "suspect_topcoding": False}
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


def plot_correlation_heatmap(corr_df: pd.DataFrame,
                              title: str = "Matrice de correlation",
                              output_path: str = "outputs/correlation_heatmap.png") -> None:
    """Version graphique de correlation_matrix : plus rapide a lire qu'un
    tableau de chiffres, notamment pour comparer visuellement P1 et P2
    d'un coup d'oeil plutot que de comparer les tableaux valeur par valeur.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if corr_df.empty:
        print("Matrice de correlation vide, graphique ignore.")
        return

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr_df.values, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_yticks(range(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_df.columns)

    # annote chaque case avec la valeur, pour ne pas se fier uniquement a la couleur
    for i in range(len(corr_df.columns)):
        for j in range(len(corr_df.columns)):
            ax.text(j, i, f"{corr_df.values[i, j]:.2f}", ha="center", va="center", color="black")

    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Coefficient de correlation")
    plt.tight_layout()

    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Graphique sauvegarde : {output_path}")


# ---------------------------------------------------------------------------
# 7bis. RECHERCHE DE COLONNES PAR MOT-CLE
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

    Un NEWID identifie normalement un foyer a un trimestre precis (Interview)
    ou a une periode de collecte (Diary) ; s'il apparait plusieurs fois,
    c'est le signe d'un doublon (par exemple un trimestre Q1 charge a la
    fois depuis le fichier de l'annee courante et depuis le fichier "x" de
    l'annee suivante, comme observe sur le CEX).
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
# 9. VISUALISATIONS
# ---------------------------------------------------------------------------

def plot_missing_rate(report_df: pd.DataFrame, top_n: int = 20,
                       title: str = "Taux de valeurs manquantes (top variables)",
                       output_path: str = "outputs/missing_rate.png") -> None:
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

    clip_percentiles recadre uniquement l'AFFICHAGE entre le 1er et le 99e
    percentile par defaut (les valeurs extremes ne sont pas supprimees des
    donnees, seulement de la vue graphique).
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
# 10. VERIFICATION DE L'HYPOTHESE "BOUNDING INTERVIEW"
# ---------------------------------------------------------------------------

def check_bounding_interview_hypothesis(df: pd.DataFrame, expense_col: str,
                                         interview_num_col: str = "INTERI",
                                         zero_threshold: float = 1.0) -> pd.DataFrame:
    """Verifie si les lignes a depense quasi nulle sont concentrees sur un
    numero d'interview particulier (typiquement la premiere interview,
    dite 'bounding interview', qui sert de reference et n'est pas censee
    etre utilisee pour les statistiques de depense selon la methodologie CEX).
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
# 11. VARIABLES CATEGORIELLES (repartition ponderee)
# ---------------------------------------------------------------------------

def weighted_value_counts(df: pd.DataFrame, col: str, weight_col: str = WEIGHT_COLUMN) -> pd.DataFrame:
    """Repartition ponderee des modalites d'une variable categorielle
    (ex: REGION, BLS_URBN). Un value_counts() non pondere sous-estime ou
    surestime la vraie repartition dans la population, puisque chaque
    foyer represente un nombre different de foyers reels via son poids.
    """
    if col not in df.columns or weight_col not in df.columns:
        print(f"[ATTENTION] {col} ou {weight_col} absent(e), repartition ignoree.")
        return pd.DataFrame()
    grouped = df.groupby(col)[weight_col].sum()
    total = grouped.sum()
    result = (grouped / total * 100).rename("pct_pondere").reset_index()
    return result.sort_values("pct_pondere", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 12. PIPELINE COMMUN INTERVIEW (applique a l'identique sur P1 et P2)
# ---------------------------------------------------------------------------

def run_interview_audit(df: pd.DataFrame, label: str,
                         full_bases: list[str] = ("TOTEXP", "FOOD", "HOUS")) -> dict:
    """Execute la meme chaine d'audit (reconstruction CQ+PQ, stats
    descriptives, ecretage, doublons, coherence logique) sur un DataFrame
    Interview donne. Centralise ici pour garantir que la Periode 1 et la
    Periode 2 recoivent exactement le meme traitement, condition necessaire
    pour pouvoir ensuite les comparer.
    """
    print(f"\n--- Audit Interview : {label} ---")

    for base in full_bases:
        df = build_full_quarter_expenditure(df, base)
    full_cols = [f"{b}_FULL" for b in full_bases] + ["FINCBTXM"]
    full_cols = [c for c in full_cols if c in df.columns]

    stats = descriptive_stats(df, full_cols)
    print(stats)

    topcoding = pd.DataFrame([detect_topcoding(df, c) for c in full_cols])
    print(topcoding)

    dup = detect_duplicate_newid(df)
    print(f"NEWID dupliques : {len(dup)}")

    checks = logical_consistency_checks(df, [c for c in full_cols if c != "FINCBTXM"])
    print(checks)

    return {
        "df": df,
        "stats": stats,
        "topcoding": topcoding,
        "duplicates": dup,
        "checks": checks,
        "full_cols": full_cols,
    }


def run_diary_audit(df: pd.DataFrame, label: str,
                     numeric_cols: list[str] = ("FOODTOT",)) -> dict:
    """Equivalent de run_interview_audit pour la Diary Survey : stats
    descriptives, doublons NEWID, coherence logique. Meme logique de
    traitement identique entre P1 et P2 pour permettre la comparaison.
    """
    print(f"\n--- Audit Diary : {label} ---")

    numeric_cols = [c for c in numeric_cols if c in df.columns]
    stats = descriptive_stats(df, numeric_cols)
    print(stats)

    dup = detect_duplicate_newid(df)
    print(f"NEWID dupliques : {len(dup)}")

    checks = logical_consistency_checks(df, numeric_cols)
    print(checks)

    return {"df": df, "stats": stats, "duplicates": dup, "checks": checks}


# ---------------------------------------------------------------------------
# 13. COMPARAISON PERIODE 1 VS PERIODE 2
# ---------------------------------------------------------------------------

def check_variable_availability(df_p1: pd.DataFrame, df_p2: pd.DataFrame,
                                 variables: list[str] = None) -> pd.DataFrame:
    """Pour une liste de variables (ou toutes les variables des deux
    DataFrames si non precisee), indique si chacune existe dans la
    Periode 1, la Periode 2, ou les deux.

    A appeler AVANT toute comparaison chiffree entre les deux periodes :
    un taux de manquant ou une moyenne qui "augmente" entre P1 et P2 n'a
    de sens que si la variable existe reellement des deux cotes. Sinon
    l'ecart affiche ne reflete qu'une disparition/apparition de colonne
    dans le schema du CEX (frequent d'une annee sur l'autre, cf.
    schema_consistency_report), pas un vrai changement de comportement
    des foyers ou de qualite de collecte.
    """
    if variables is None:
        variables = sorted(set(df_p1.columns) | set(df_p2.columns))

    rows = []
    for var in variables:
        in_p1 = var in df_p1.columns
        in_p2 = var in df_p2.columns
        rows.append({
            "variable": var,
            "present_p1": in_p1,
            "present_p2": in_p2,
            "comparable_entre_periodes": in_p1 and in_p2,
        })
    report = pd.DataFrame(rows)

    n_non_comparables = (~report["comparable_entre_periodes"]).sum()
    if n_non_comparables > 0:
        print(f"[ATTENTION] {n_non_comparables} variable(s) non presente(s) sur les deux periodes "
              f"-> a exclure de toute comparaison chiffree P1/P2 (voir colonne comparable_entre_periodes).")
    else:
        print("Toutes les variables verifiees sont presentes sur les deux periodes.")
    return report


def compare_periods(stats_p1: pd.DataFrame, stats_p2: pd.DataFrame) -> pd.DataFrame:
    """Fusionne les statistiques descriptives des deux periodes pour
    calculer l'ecart entre elles. Le decoupage en deux periodes n'a
    d'interet que si on objective ensuite si la moyenne, la mediane ou le
    taux de manquant ont evolue entre 2017-2020 et 2021-2024 (ex: effet
    inflation, changement de methodologie de collecte, COVID).

    Un merge interne (defaut de pd.merge) exclut silencieusement les
    variables absentes d'une des deux periodes : un message est affiche
    pour lister ces exclusions plutot que de les laisser disparaitre sans
    trace.
    """
    if stats_p1.empty or stats_p2.empty:
        return pd.DataFrame()

    vars_p1 = set(stats_p1["variable"])
    vars_p2 = set(stats_p2["variable"])
    only_p1 = vars_p1 - vars_p2
    only_p2 = vars_p2 - vars_p1
    if only_p1 or only_p2:
        print(f"[ATTENTION] Variables exclues de la comparaison (absentes d'une des deux periodes) : "
              f"presentes seulement en P1={sorted(only_p1)}, seulement en P2={sorted(only_p2)}")

    merged = stats_p1.merge(stats_p2, on="variable", suffixes=("_p1", "_p2"))
    merged["ecart_moyenne_ponderee_pct"] = (
        (merged["moyenne_ponderee_p2"] - merged["moyenne_ponderee_p1"])
        / merged["moyenne_ponderee_p1"] * 100
    )
    merged["ecart_mediane_pct"] = (
        (merged["mediane_p2"] - merged["mediane_p1"]) / merged["mediane_p1"] * 100
    )
    return merged


def compare_missing_rates(miss_p1: pd.DataFrame, miss_p2: pd.DataFrame,
                           df_p1: pd.DataFrame = None, df_p2: pd.DataFrame = None) -> pd.DataFrame:
    """Compare le taux de manquant d'une meme variable entre les deux
    periodes, pour reperer les variables dont la qualite de collecte
    s'est degradee ou amelioree.

    Si df_p1 et df_p2 sont fournis, ajoute une colonne
    'comparable_entre_periodes' : quand elle vaut False, l'ecart affiche
    n'est pas un vrai changement de non-reponse mais un artefact de
    disparition/apparition de colonne dans le schema (ex: DESFLG,
    INT_PHON, INT_HOME, retirees du questionnaire a partir de 2021).
    """
    if miss_p1.empty or miss_p2.empty:
        return pd.DataFrame()
    merged = miss_p1.merge(miss_p2, on="variable", suffixes=("_p1", "_p2"), how="outer")
    merged["ecart_pts"] = merged["taux_manquant_pct_p2"] - merged["taux_manquant_pct_p1"]

    if df_p1 is not None and df_p2 is not None:
        merged["comparable_entre_periodes"] = merged["variable"].apply(
            lambda v: (v in df_p1.columns) and (v in df_p2.columns)
        )

    return merged.sort_values("ecart_pts", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 14. EXECUTION PRINCIPALE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    Path("outputs").mkdir(exist_ok=True)

    # Variables candidates deja aggregees pour Diary. FOODTOT est la seule
    # variable de depense agregee confirmee dans FMLD (verifie via
    # find_columns_by_keyword et list_all_columns : Diary ne contient pas
    # de depense totale toutes categories, contrairement a FMLI).
    CANDIDATE_NUMERIC_COLS_DIARY = ["FOODTOT"]

    # ============================================================
    # INTERVIEW SURVEY (FMLI)
    # ============================================================
    print("========== INTERVIEW SURVEY (FMLI) ==========")

    print("\n=== Coherence du schema entre annees (Interview) ===")
    schema_int = schema_consistency_report(PERIOD_1_YEARS + PERIOD_2_YEARS, survey="interview")
    schema_int.to_csv("outputs/schema_consistency_interview.csv", index=False)

    print("\n=== Chargement Periode 1 (2017-2020) ===")
    df_int_p1 = load_survey_for_years(PERIOD_1_YEARS, survey="interview")

    print("\n=== Chargement Periode 2 (2021-2024) ===")
    df_int_p2 = load_survey_for_years(PERIOD_2_YEARS, survey="interview")

    list_all_columns(df_int_p1, output_path="outputs/all_columns_interview.csv")

    print("\n=== Foyers distincts vs lignes (Interview) ===")
    print(unique_households_by_year(df_int_p1))
    print(unique_households_by_year(df_int_p2))

    print("\n=== Volumetrie Interview ===")
    print(summarize_volume(df_int_p1))
    print(summarize_volume(df_int_p2))

    print("\n=== Taux de valeurs manquantes Interview P1 / P2 ===")
    miss_int_p1 = missing_rate_report(df_int_p1, exclude_cols=ID_COLUMNS)
    miss_int_p2 = missing_rate_report(df_int_p2, exclude_cols=ID_COLUMNS)
    print(miss_int_p1.head(20))
    print(miss_int_p2.head(20))

    print("\n=== Codes speciaux de manquant (Interview P1) ===")
    special_codes_int_p1 = special_missing_code_report(
        df_int_p1, ["FINCBTXM", "TOTEXPCQ", "FOODCQ", "HOUSCQ"])
    print(special_codes_int_p1)

    print("\n=== Audit Interview Periode 1 (pipeline commun) ===")
    res_int_p1 = run_interview_audit(df_int_p1, "Periode 1 (2017-2020)")
    df_int_p1 = res_int_p1["df"]

    print("\n=== Audit Interview Periode 2 (pipeline commun) ===")
    res_int_p2 = run_interview_audit(df_int_p2, "Periode 2 (2021-2024)")
    df_int_p2 = res_int_p2["df"]

    print("\n=== Disponibilite des variables entre les deux periodes (Interview) ===")
    availability_int = check_variable_availability(df_int_p1, df_int_p2)
    availability_int.to_csv("outputs/variable_availability_interview.csv", index=False)
    print(availability_int[~availability_int["comparable_entre_periodes"]].head(20))

    print("\n=== Comparaison statistique Periode 1 vs Periode 2 (Interview) ===")
    comparison_int = compare_periods(res_int_p1["stats"], res_int_p2["stats"])
    print(comparison_int)
    comparison_int.to_csv("outputs/comparison_periods_interview.csv", index=False)

    print("\n=== Comparaison des taux de manquant Periode 1 vs Periode 2 (Interview) ===")
    comparison_missing_int = compare_missing_rates(miss_int_p1, miss_int_p2, df_int_p1, df_int_p2)
    print(comparison_missing_int.head(20))
    comparison_missing_int.to_csv("outputs/comparison_missing_interview.csv", index=False)

    print("\n=== Repartition ponderee de variables categorielles (Interview P1) ===")
    for cat_col in ["REGION", "BLS_URBN"]:
        if cat_col in df_int_p1.columns:
            print(f"\n--- {cat_col} ---")
            print(weighted_value_counts(df_int_p1, cat_col))

    print("\n=== Verification hypothese 'bounding interview' (Periode 1) ===")
    for col in res_int_p1["full_cols"]:
        print(f"\n--- {col} par numero d'interview (INTERI) ---")
        print(check_bounding_interview_hypothesis(df_int_p1, col))

    print("\n=== Matrice de correlation Interview (Periode 1) ===")
    corr_int_p1 = correlation_matrix(df_int_p1, res_int_p1["full_cols"])
    print(corr_int_p1)
    plot_correlation_heatmap(corr_int_p1,
                              title="Interview - Correlation entre variables (Periode 1)",
                              output_path="outputs/correlation_interview_period1.png")

    print("\n=== Matrice de correlation Interview (Periode 2) ===")
    corr_int_p2 = correlation_matrix(df_int_p2, res_int_p2["full_cols"])
    print(corr_int_p2)
    plot_correlation_heatmap(corr_int_p2,
                              title="Interview - Correlation entre variables (Periode 2)",
                              output_path="outputs/correlation_interview_period2.png")

    print("\n=== Ecart de correlation Periode 1 vs Periode 2 (Interview) ===")
    common_corr_cols = [c for c in corr_int_p1.columns if c in corr_int_p2.columns]
    corr_diff = corr_int_p2.loc[common_corr_cols, common_corr_cols] - corr_int_p1.loc[common_corr_cols, common_corr_cols]
    print(corr_diff)

    print("\n=== Generation des graphiques Interview (Periode 1) ===")
    plot_missing_rate(miss_int_p1, top_n=20,
                       title="Interview - Taux de valeurs manquantes (Periode 1)",
                       output_path="outputs/missing_rate_interview_period1.png")
    plot_missing_heatmap(df_int_p1, res_int_p1["full_cols"],
                          title="Interview - Carte des manquants (Periode 1)",
                          output_path="outputs/missing_heatmap_interview_period1.png")
    dup_by_file_p1 = duplicate_summary_by_year(df_int_p1)
    plot_duplicate_summary(dup_by_file_p1,
                            title="Interview - Doublons par fichier (Periode 1)",
                            output_path="outputs/duplicates_interview_period1.png")
    for col in res_int_p1["full_cols"]:
        plot_distribution(df_int_p1, col, output_path=f"outputs/distribution_interview_{col}_period1.png")

    print("\n=== Generation des graphiques Interview (Periode 2) ===")
    plot_missing_rate(miss_int_p2, top_n=20,
                       title="Interview - Taux de valeurs manquantes (Periode 2)",
                       output_path="outputs/missing_rate_interview_period2.png")
    plot_missing_heatmap(df_int_p2, res_int_p2["full_cols"],
                          title="Interview - Carte des manquants (Periode 2)",
                          output_path="outputs/missing_heatmap_interview_period2.png")
    dup_by_file_p2 = duplicate_summary_by_year(df_int_p2)
    plot_duplicate_summary(dup_by_file_p2,
                            title="Interview - Doublons par fichier (Periode 2)",
                            output_path="outputs/duplicates_interview_period2.png")
    for col in res_int_p2["full_cols"]:
        plot_distribution(df_int_p2, col, output_path=f"outputs/distribution_interview_{col}_period2.png")

    # ============================================================
    # DIARY SURVEY (FMLD)
    # ============================================================
    print("\n\n========== DIARY SURVEY (FMLD) ==========")

    print("\n=== Coherence du schema entre annees (Diary) ===")
    schema_dia = schema_consistency_report(PERIOD_1_YEARS + PERIOD_2_YEARS, survey="diary")
    schema_dia.to_csv("outputs/schema_consistency_diary.csv", index=False)

    print("\n=== Chargement Periode 1 (2017-2020) ===")
    df_dia_p1 = load_survey_for_years(PERIOD_1_YEARS, survey="diary")

    print("\n=== Chargement Periode 2 (2021-2024) ===")
    df_dia_p2 = load_survey_for_years(PERIOD_2_YEARS, survey="diary")

    list_all_columns(df_dia_p1, output_path="outputs/all_columns_diary.csv")

    print("\n=== Recherche du nom reel de la depense totale dans Diary ===")
    print("Colonnes contenant 'TOTAL' :", find_columns_by_keyword(df_dia_p1, "TOTAL"))
    print("Colonnes contenant 'EXP' :", find_columns_by_keyword(df_dia_p1, "EXP"))

    print("\n=== Volumetrie Diary ===")
    print(summarize_volume(df_dia_p1))
    print(summarize_volume(df_dia_p2))

    print("\n=== Foyers distincts vs lignes (Diary) ===")
    print(unique_households_by_year(df_dia_p1))
    print(unique_households_by_year(df_dia_p2))

    print("\n=== Taux de valeurs manquantes Diary P1 / P2 ===")
    miss_dia_p1 = missing_rate_report(df_dia_p1, exclude_cols=ID_COLUMNS)
    miss_dia_p2 = missing_rate_report(df_dia_p2, exclude_cols=ID_COLUMNS)
    print(miss_dia_p1.head(20))
    print(miss_dia_p2.head(20))

    print("\n=== Audit Diary Periode 1 (pipeline commun) ===")
    res_dia_p1 = run_diary_audit(df_dia_p1, "Periode 1 (2017-2020)", CANDIDATE_NUMERIC_COLS_DIARY)

    print("\n=== Audit Diary Periode 2 (pipeline commun) ===")
    res_dia_p2 = run_diary_audit(df_dia_p2, "Periode 2 (2021-2024)", CANDIDATE_NUMERIC_COLS_DIARY)

    print("\n=== Disponibilite des variables entre les deux periodes (Diary) ===")
    availability_dia = check_variable_availability(df_dia_p1, df_dia_p2)
    availability_dia.to_csv("outputs/variable_availability_diary.csv", index=False)
    print(availability_dia[~availability_dia["comparable_entre_periodes"]].head(20))

    print("\n=== Comparaison statistique Periode 1 vs Periode 2 (Diary) ===")
    comparison_dia = compare_periods(res_dia_p1["stats"], res_dia_p2["stats"])
    print(comparison_dia)
    comparison_dia.to_csv("outputs/comparison_periods_diary.csv", index=False)

    print("\n=== Generation des graphiques Diary (Periode 1) ===")
    plot_missing_rate(miss_dia_p1, top_n=20,
                       title="Diary - Taux de valeurs manquantes (Periode 1)",
                       output_path="outputs/missing_rate_diary_period1.png")
    plot_missing_heatmap(df_dia_p1, CANDIDATE_NUMERIC_COLS_DIARY,
                          title="Diary - Carte des manquants (Periode 1)",
                          output_path="outputs/missing_heatmap_diary_period1.png")
    dup_by_file_dia_p1 = duplicate_summary_by_year(df_dia_p1)
    plot_duplicate_summary(dup_by_file_dia_p1,
                            title="Diary - Doublons par fichier (Periode 1)",
                            output_path="outputs/duplicates_diary_period1.png")
    for col in CANDIDATE_NUMERIC_COLS_DIARY:
        plot_distribution(df_dia_p1, col, output_path=f"outputs/distribution_diary_{col}_period1.png")

    print("\n=== Generation des graphiques Diary (Periode 2) ===")
    plot_missing_rate(miss_dia_p2, top_n=20,
                       title="Diary - Taux de valeurs manquantes (Periode 2)",
                       output_path="outputs/missing_rate_diary_period2.png")
    plot_missing_heatmap(df_dia_p2, CANDIDATE_NUMERIC_COLS_DIARY,
                          title="Diary - Carte des manquants (Periode 2)",
                          output_path="outputs/missing_heatmap_diary_period2.png")
    dup_by_file_dia_p2 = duplicate_summary_by_year(df_dia_p2)
    plot_duplicate_summary(dup_by_file_dia_p2,
                            title="Diary - Doublons par fichier (Periode 2)",
                            output_path="outputs/duplicates_diary_period2.png")
    for col in CANDIDATE_NUMERIC_COLS_DIARY:
        plot_distribution(df_dia_p2, col, output_path=f"outputs/distribution_diary_{col}_period2.png")

    # ============================================================
    # SAUVEGARDE DES RAPPORTS
    # ============================================================
    miss_int_p1.to_csv("outputs/missing_rate_interview_period1.csv", index=False)
    miss_int_p2.to_csv("outputs/missing_rate_interview_period2.csv", index=False)
    res_int_p1["stats"].to_csv("outputs/descriptive_stats_interview_period1.csv", index=False)
    res_int_p2["stats"].to_csv("outputs/descriptive_stats_interview_period2.csv", index=False)
    miss_dia_p1.to_csv("outputs/missing_rate_diary_period1.csv", index=False)
    miss_dia_p2.to_csv("outputs/missing_rate_diary_period2.csv", index=False)
    res_dia_p1["stats"].to_csv("outputs/descriptive_stats_diary_period1.csv", index=False)
    res_dia_p2["stats"].to_csv("outputs/descriptive_stats_diary_period2.csv", index=False)
    print("\nRapports sauvegardes dans outputs/")