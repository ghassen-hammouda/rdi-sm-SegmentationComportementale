"""
US-07 : Encodage, mise a l'echelle et reduction de dimension

Ce script prepare les datasets par foyer (poverty_dataset_period*.csv,
issus de US-06/US-08) pour l'exploitation par des algorithmes bases sur
la distance (clustering) ou par des modeles de classification :

1. Separation identifiants / cible / base des cibles (fuite) / features
2. Encodage des variables categorielles :
   - One-hot : REGION, FAM_TYPE, CUTENURE (pas d'ordre logique)
   - Ordinal (inchange) : EDUC_REF (deja croissant, verifie en amont)
   - Binaire recode 0/1 : BLS_URBN (1=urbain -> 1, 2=rural -> 0)
3. Mise a l'echelle : RobustScaler (median/IQR) sur les variables
   numeriques continues, coherent avec les distributions asymetriques
   deja observees tout au long du projet (moyenne toujours superieure
   a la mediane sur les variables de depense/revenu)
4. PCA : evaluation de la variance expliquee cumulee, sans imposer un
   nombre de composantes a l'avance

Perimetre : Interview uniquement, P1 et P2 traites separement et
sauvegardes separement (coherent avec les decisions US-06/US-08).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

INPUT_DIR = Path("outputs_poverty")
OUTPUT_DIR = Path("outputs_features")
OUTPUT_DIR.mkdir(exist_ok=True)

FILES = {
    "period1": "poverty_dataset_period1.csv",
    "period2": "poverty_dataset_period2.csv",
}

# ---------------------------------------------------------------------------
# DEFINITION DES ROLES DE CHAQUE COLONNE
# ---------------------------------------------------------------------------

# Identifiant technique, ni feature ni cible.
ID_COLUMN = "CUID"

# Cibles de pauvrete : jamais utilisees comme feature.
TARGET_COLUMNS = [
    "statut_pauvrete_revenu_brut",
    "statut_pauvrete_revenu_net",
    "statut_pauvrete_consommation",
]

# Variables ayant servi a CONSTRUIRE les cibles : exclues des features
# pour eviter la fuite de donnees (le modele ne doit pas "retrouver" la
# regle qui a servi a fabriquer l'etiquette qu'on lui demande de predire).
# TOTEXP_FULL_ANNUEL et son equivalent par tete sont ici au meme titre
# que le revenu, puisqu'ils determinent directement statut_pauvrete_consommation.
LEAKAGE_COLUMNS = [
    "FINCBTXM", "FINATXEM",
    "REVENU_BRUT_EQUIVALISE", "REVENU_NET_EQUIVALISE", "CONSOMMATION_EQUIVALISEE",
    "SEUIL_PAUVRETE",
    "TOTEXP_FULL_ANNUEL", "totexp_par_equivalent_adulte",
    "ECHELLE_EQUIVALENCE",  # redondante : calculee uniquement a partir de FAM_SIZE/PERSLT18, deja gardees comme features
]

# Variables categorielles nominales (pas d'ordre logique) -> one-hot
ONEHOT_COLUMNS = ["REGION", "FAM_TYPE", "CUTENURE"]

# Variable ordinale (ordre croissant deja verifie sur vos donnees : 0,10-16) -> inchangee
ORDINAL_COLUMNS = ["EDUC_REF"]

# Variable binaire -> recodee 0/1, pas de scaling (deja comparable)
BINARY_COLUMN = "BLS_URBN"


def get_numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    """Toutes les colonnes numeriques du dataset, moins l'identifiant,
    les cibles, les variables de fuite, la variable binaire, les
    variables categorielles (traitees a part), et les colonnes de
    tracabilite d'ecretage (_CAPPED_FLAG, booleennes, pas des features
    comportementales). Le reste = features numeriques continues.
    """
    excluded = {ID_COLUMN, BINARY_COLUMN, *TARGET_COLUMNS, *LEAKAGE_COLUMNS, *ONEHOT_COLUMNS}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if not c.endswith("_CAPPED_FLAG")]
    return [c for c in numeric_cols if c not in excluded]


# ---------------------------------------------------------------------------
# ENCODAGE
# ---------------------------------------------------------------------------

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Applique le one-hot sur les variables nominales et recode la
    variable binaire. EDUC_REF (ordinale) n'est pas modifiee : son ordre
    numerique croissant deja verifie sert directement l'algorithme.
    """
    df = df.copy()

    df = pd.get_dummies(df, columns=ONEHOT_COLUMNS, prefix=ONEHOT_COLUMNS, dtype=int)

    df[BINARY_COLUMN] = df[BINARY_COLUMN].map({1: 1, 2: 0})
    n_missing = df[BINARY_COLUMN].isna().sum()
    if n_missing > 0:
        print(f"[ATTENTION] {n_missing} valeur(s) de {BINARY_COLUMN} hors {{1,2}}, a examiner.")

    return df


# ---------------------------------------------------------------------------
# TRAITEMENT DES NaN RESIDUELS (heritage des divisions US-06 : revenu ou
# budget total nul/negatif pour une poignee de foyers)
# ---------------------------------------------------------------------------

def impute_remaining_nan(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Certaines features derivees (ratio_logement_revenu_*, part_*)
    peuvent contenir des NaN residuels : consequence de divisions par un
    revenu ou un budget total nul/negatif pour une poignee de foyers
    (deja signale explicitement lors de la construction des features en
    US-06, mais jamais corrige a ce stade). RobustScaler et PCA
    n'acceptent aucun NaN : on les impute ici par la mediane, la meme
    strategie que celle retenue pour toutes les variables numeriques a
    faible taux de manquant depuis l'US-05.
    """
    df = df.copy()
    nan_counts = df[numeric_cols].isna().sum()
    nan_counts = nan_counts[nan_counts > 0]

    if nan_counts.empty:
        print("Aucun NaN residuel detecte sur les features numeriques.")
        return df

    print(f"NaN residuels detectes sur {len(nan_counts)} colonne(s), imputation par la mediane :")
    for col, n in nan_counts.items():
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"  {col} : {n} valeur(s) imputee(s) (mediane = {median_val:.4f})")

    return df


# ---------------------------------------------------------------------------
# MISE A L'ECHELLE
# ---------------------------------------------------------------------------

def scale_numeric_features(df: pd.DataFrame, numeric_cols: list[str]) -> tuple[pd.DataFrame, RobustScaler]:
    """RobustScaler (base sur la mediane et l'IQR, pas la moyenne et
    l'ecart-type) : coherent avec les distributions asymetriques
    observees sur l'ensemble du projet, moins sensible aux valeurs
    encore extremes malgre l'ecretage deja applique en US-05/US-06.
    """
    scaler = RobustScaler()
    df = df.copy()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df, scaler


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def evaluate_pca(X: pd.DataFrame, period_label: str) -> pd.DataFrame:
    """Ajuste une PCA complete (autant de composantes que de variables)
    pour observer la variance expliquee cumulee, sans imposer de nombre
    de composantes a l'avance. Affiche le nombre de composantes
    necessaires pour conserver 90% et 95% de la variance.
    """
    pca = PCA()
    pca.fit(X)

    variance_ratio = pca.explained_variance_ratio_
    cumulative = np.cumsum(variance_ratio)

    n_90 = int(np.argmax(cumulative >= 0.90) + 1)
    n_95 = int(np.argmax(cumulative >= 0.95) + 1)

    print(f"[{period_label}] Nombre total de features en entree : {X.shape[1]}")
    print(f"[{period_label}] Composantes necessaires pour 90% de variance expliquee : {n_90}")
    print(f"[{period_label}] Composantes necessaires pour 95% de variance expliquee : {n_95}")

    report = pd.DataFrame({
        "composante": range(1, len(variance_ratio) + 1),
        "variance_expliquee": variance_ratio,
        "variance_cumulee": cumulative,
    })
    return report


def plot_pca_variance(report: pd.DataFrame, period_label: str, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(report["composante"], report["variance_cumulee"] * 100, marker="o", markersize=3, color="#2980b9")
    ax.axhline(90, color="#e67e22", linestyle="--", label="90% de variance")
    ax.axhline(95, color="#c0392b", linestyle="--", label="95% de variance")
    ax.set_xlabel("Nombre de composantes")
    ax.set_ylabel("Variance expliquee cumulee (%)")
    ax.set_title(f"PCA - Variance expliquee cumulee ({period_label})")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Graphique sauvegarde : {output_path}")


# ---------------------------------------------------------------------------
# PIPELINE PAR PERIODE
# ---------------------------------------------------------------------------

def process_period(period_label: str, filename: str) -> None:
    print(f"\n{'=' * 70}\nUS-07 - {period_label.upper()}\n{'=' * 70}")

    filepath = INPUT_DIR / filename
    df = pd.read_csv(filepath)
    print(f"Charge : {filepath} ({len(df)} foyers, {df.shape[1]} colonnes)")

    numeric_cols = get_numeric_feature_columns(df)
    print(f"\n{len(numeric_cols)} features numeriques continues identifiees (a mettre a l'echelle).")
    print(f"Colonnes exclues (fuite) : {LEAKAGE_COLUMNS}")

    print("\n--- Encodage des variables categorielles ---")
    df_encoded = encode_categorical(df)
    new_onehot_cols = [c for c in df_encoded.columns
                        if any(c.startswith(f"{base}_") for base in ONEHOT_COLUMNS)]
    print(f"Colonnes one-hot creees : {len(new_onehot_cols)} ({new_onehot_cols})")

    print("\n--- Traitement des NaN residuels (heritage US-06) ---")
    df_encoded = impute_remaining_nan(df_encoded, numeric_cols)

    print("\n--- Mise a l'echelle (RobustScaler) ---")
    df_scaled, scaler = scale_numeric_features(df_encoded, numeric_cols)

    feature_cols = numeric_cols + ORDINAL_COLUMNS + [BINARY_COLUMN] + new_onehot_cols
    X = df_scaled[feature_cols]
    print(f"Matrice de features finale : {X.shape[0]} foyers x {X.shape[1]} colonnes")

    print("\n--- Evaluation PCA ---")
    pca_report = evaluate_pca(X, period_label)
    pca_report.to_csv(OUTPUT_DIR / f"pca_variance_{period_label}.csv", index=False)
    plot_pca_variance(pca_report, period_label, OUTPUT_DIR / f"pca_variance_{period_label}.png")

    # Sauvegarde : features pretes a l'emploi (CUID + cibles gardees a
    # part, jamais melangees aux features, pour reference/verification).
    output_df = pd.concat([df[[ID_COLUMN] + TARGET_COLUMNS].reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    output_path = OUTPUT_DIR / f"features_{period_label}.csv"
    output_df.to_csv(output_path, index=False)
    print(f"\nFeatures encodees et mises a l'echelle sauvegardees : {output_path} "
          f"({output_df.shape[0]} lignes, {output_df.shape[1]} colonnes)")


if __name__ == "__main__":
    for period_label, filename in FILES.items():
        process_period(period_label, filename)

    print(f"\n{'=' * 70}")
    print("Termine. Fichiers dans outputs_features/ :")
    print("  features_period1.csv / features_period2.csv  -> features pretes (CUID + cibles a part)")
    print("  pca_variance_period1.csv / _period2.csv       -> variance expliquee, composante par composante")
    print("  pca_variance_period1.png / _period2.png       -> graphique de la variance cumulee")