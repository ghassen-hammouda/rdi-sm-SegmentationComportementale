"""
Regression lineaire estimant FINATXEM (revenu net) pour les foyers 2024
concernes par le changement structurel du questionnaire BLS, a partir
de FINCBTXM et des variables demographiques deja encodees/mises a
l'echelle (US-07 sur ce nouveau perimetre).

Formule retenue (validee etape par etape dans la conversation) :
FINATXEM = f(FINCBTXM, FINCBTXM^2, PERSLT18, PERSOT64, FAM_TYPE, AGE_REF,
             REGION, STATE, CUTENURE, annee)

Protocole :
1. Separer les foyers ou FINATXEM est connu (2021-2023) des foyers ou
   elle manque (2024, ~23176 foyers)
2. Verifier la comparabilite de la ponderation entre les deux groupes
   (recommandation de l'encadrant)
3. Entrainer une regression lineaire sur un sous-ensemble d'entrainement,
   evaluer sa qualite sur un sous-ensemble de validation MIS DE COTE
   (jamais vu par le modele) avant de l'appliquer aux foyers 2024
4. Appliquer le modele aux foyers 2024, tracer chaque valeur estimee
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

INPUT_DIR = Path("outputs_v2")
OUTPUT_DIR = Path("outputs_v2")

FILENAME = "interview_period2_scaled.csv"
WEIGHT_COLUMN = "FINLWT21"
RANDOM_STATE = 42


def get_predictor_columns(df: pd.DataFrame) -> list[str]:
    """Assemble la liste des colonnes predictives selon la formule
    validee : variables numeriques deja mises a l'echelle + toutes les
    colonnes one-hot issues de FAM_TYPE, REGION, STATE, CUTENURE et
    SOURCE_YEAR (l'annee, traitee comme categorielle - les regles
    fiscales changent de facon discrete, pas continue).
    """
    numeric_predictors = ["FINCBTXM_SCALED", "FINCBTXM_SCALED_SQ",
                           "PERSLT18_SCALED", "PERSOT64_SCALED", "AGE_REF_SCALED"]
    onehot_prefixes = ("FAM_TYPE_", "REGION_", "STATE_", "CUTENURE_", "SOURCE_YEAR_")

    existing_numeric = [c for c in numeric_predictors if c in df.columns]
    missing_numeric = [c for c in numeric_predictors if c not in df.columns]
    if missing_numeric:
        print(f"[ATTENTION] Predicteurs numeriques manquants : {missing_numeric}")

    onehot_cols = [c for c in df.columns if c.startswith(onehot_prefixes)]

    predictors = existing_numeric + onehot_cols
    print(f"{len(predictors)} variables predictives assemblees "
          f"({len(existing_numeric)} numeriques + {len(onehot_cols)} one-hot).")
    return predictors


def check_weight_comparability(df: pd.DataFrame, train_mask: pd.Series, target_mask: pd.Series,
                                weight_col: str = WEIGHT_COLUMN) -> None:
    if weight_col not in df.columns:
        print(f"[ATTENTION] {weight_col} absente, verification de ponderation impossible.")
        return

    poids_train_moyen = df.loc[train_mask, weight_col].mean()
    poids_train_median = df.loc[train_mask, weight_col].median()
    poids_target_moyen = df.loc[target_mask, weight_col].mean()
    poids_target_median = df.loc[target_mask, weight_col].median()

    ecart_moyenne = abs(poids_target_moyen - poids_train_moyen) / poids_train_moyen * 100
    ecart_mediane = abs(poids_target_median - poids_train_median) / poids_train_median * 100

    print(f"\n--- Verification de comparabilite de la ponderation ({weight_col}) ---")
    print(f"Foyers d'entrainement (2021-2023) : poids moyen={poids_train_moyen:,.1f}, median={poids_train_median:,.1f}")
    print(f"Foyers a estimer (2024)            : poids moyen={poids_target_moyen:,.1f}, median={poids_target_median:,.1f}")
    print(f"Ecart relatif (moyenne) : {ecart_moyenne:.1f}%  |  Ecart relatif (mediane) : {ecart_mediane:.1f}%")

    seuil = 15.0
    if ecart_moyenne < seuil and ecart_mediane < seuil:
        print(f"Verdict : COMPARABLE (< {seuil}%) -> l'estimation par regression est defendable.")
    else:
        print(f"Verdict : ECART NOTABLE (>= {seuil}%) -> a documenter comme limite dans le rapport.")


def train_and_evaluate(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Entraine sur 80% des foyers connus, evalue sur les 20% restants
    JAMAIS vus par le modele pendant l'entrainement - condition
    necessaire pour savoir si le modele generalise, pas juste s'il
    recite les donnees qu'il a deja vues.
    """
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE
    )

    model = LinearRegression()
    model.fit(X_fit, y_fit)

    y_pred_val = model.predict(X_val)
    r2 = r2_score(y_val, y_pred_val)
    mae = mean_absolute_error(y_val, y_pred_val)

    print(f"\n--- Evaluation du modele (sur les 20% de validation, jamais vus a l'entrainement) ---")
    print(f"Foyers d'entrainement : {len(X_fit)}  |  Foyers de validation : {len(X_val)}")
    print(f"R2 (variance expliquee)      : {r2:.4f}")
    print(f"Erreur absolue moyenne (MAE) : {mae:,.2f} $")

    # Reentraine sur 100% des foyers connus pour la version finale utilisee en prediction
    model_final = LinearRegression()
    model_final.fit(X_train, y_train)
    print(f"\nModele final reentraine sur l'ensemble des {len(X_train)} foyers ou FINATXEM est connu.")

    return model_final


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    filepath = INPUT_DIR / FILENAME
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Charge : {filepath} ({len(df)} lignes, {df.shape[1]} colonnes)")

    train_mask = df["FINATXEM"].notna()
    target_mask = df["FINATXEM"].isna()
    print(f"\nFoyers avec FINATXEM connu (entrainement) : {train_mask.sum()}")
    print(f"Foyers avec FINATXEM manquant (a estimer)  : {target_mask.sum()}")

    check_weight_comparability(df, train_mask, target_mask)

    predictors = get_predictor_columns(df)

    X_train = df.loc[train_mask, predictors]
    y_train = df.loc[train_mask, "FINATXEM"]

    model = train_and_evaluate(X_train, y_train)

    X_target = df.loc[target_mask, predictors]
    predictions = model.predict(X_target)

    df["FINATXEM_ESTIME_FLAG"] = target_mask
    df.loc[target_mask, "FINATXEM"] = predictions

    print(f"\n--- Resultat de l'estimation ---")
    print(f"FINATXEM manquant apres regression : {df['FINATXEM'].isna().sum()}")
    print(f"Valeurs estimees (apercu) :")
    print(df.loc[df["FINATXEM_ESTIME_FLAG"], ["FINCBTXM", "FINATXEM"]].describe())

    output_path = OUTPUT_DIR / "interview_period2_finatxem_regression.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSauvegarde : {output_path}")
    print(f"{df['FINATXEM_ESTIME_FLAG'].sum()} valeurs marquees comme estimees par regression.")