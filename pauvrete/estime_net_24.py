"""
Estimation de FINATXEM (revenu net) manquant sur les fichiers 2024, par
tranches de revenu (quantiles) plutot que par regression lineaire globale
ou imputation par mediane simple.

Pourquoi les tranches plutot qu'une regression lineaire classique :
une regression FINATXEM = a*FINCBTXM + b capture la tendance generale
(coherente avec la forte correlation observee, ~0.97), mais LISSE le
phenomene qui nous interesse le plus : l'effet des credits d'impot
(EITC, Child Tax Credit) qui peut faire depasser le net sur le brut,
UNIQUEMENT chez les foyers a faible revenu. Une seule droite pour tout
le monde gommerait cet effet localise.

A la place, on decoupe les foyers 2021-2023 (ou les DEUX revenus sont
connus) en tranches de revenu brut, on calcule le VRAI ratio net/brut
observe DANS CHAQUE TRANCHE, puis on applique ce ratio specifique aux
foyers 2024 selon leur propre tranche de revenu brut.

Chaque valeur estimee est tracee via FINATXEM_ESTIME_FLAG, pour ne
jamais confondre une vraie declaration avec une estimation statistique.
"""

import pandas as pd
import numpy as np
from pathlib import Path

INPUT_DIR = Path("outputs_v2")
OUTPUT_DIR = Path("outputs_v2")

FILENAME = "interview_period2_full.csv"
N_BINS = 20


def build_bin_ratios(df_train: pd.DataFrame, n_bins: int) -> tuple[np.ndarray, pd.DataFrame]:
    """A partir des foyers ou FINATXEM et FINCBTXM sont TOUS LES DEUX
    connus (2021-2023), decoupe en n_bins tranches de revenu brut et
    calcule le ratio net/brut moyen observe dans chaque tranche.
    """
    train = df_train.dropna(subset=["FINATXEM", "FINCBTXM"]).copy()
    print(f"Foyers utilises pour calibrer les tranches (revenu net ET brut connus) : {len(train)}")

    _, bin_edges = pd.qcut(train["FINCBTXM"], q=n_bins, retbins=True, duplicates="drop")
    train["tranche"] = pd.cut(train["FINCBTXM"], bins=bin_edges, include_lowest=True)

    bin_stats = train.groupby("tranche", observed=True).agg(
        nb_foyers=("FINCBTXM", "size"),
        moyenne_brut=("FINCBTXM", "mean"),
        moyenne_net=("FINATXEM", "mean"),
    )
    bin_stats["ratio_net_brut"] = bin_stats["moyenne_net"] / bin_stats["moyenne_brut"]

    print("\nRatio net/brut observe, par tranche de revenu brut :")
    print(bin_stats.round(3))

    return bin_edges, bin_stats


def estimate_missing_finatxem(df: pd.DataFrame, bin_edges: np.ndarray, bin_stats: pd.DataFrame) -> pd.DataFrame:
    """Applique, a chaque foyer 2024 sans FINATXEM, le ratio net/brut de
    sa propre tranche de revenu brut (calibree sur 2021-2023).
    """
    df = df.copy()
    df["FINATXEM_ESTIME_FLAG"] = df["FINATXEM"].isna() & df["FINCBTXM"].notna()

    n_a_estimer = int(df["FINATXEM_ESTIME_FLAG"].sum())
    print(f"\nFoyers a estimer (FINATXEM manquant, FINCBTXM present) : {n_a_estimer}")
    if n_a_estimer == 0:
        return df

    # On ramene le revenu brut des foyers a estimer dans la plage connue
    # des tranches (evite les valeurs NaN pour un revenu hors plage,
    # extrapolation impossible avec cette methode par tranches).
    brut_clippe = df.loc[df["FINATXEM_ESTIME_FLAG"], "FINCBTXM"].clip(
        lower=bin_edges[0], upper=bin_edges[-1]
    )
    tranche_assignee = pd.cut(brut_clippe, bins=bin_edges, include_lowest=True)

    ratio_par_tranche = bin_stats["ratio_net_brut"]
    ratio_applique = tranche_assignee.map(ratio_par_tranche)
    ratio_applique = pd.to_numeric(ratio_applique, errors="coerce")

    valeurs_estimees = df.loc[df["FINATXEM_ESTIME_FLAG"], "FINCBTXM"].to_numpy() * ratio_applique.to_numpy()
    df.loc[df["FINATXEM_ESTIME_FLAG"], "FINATXEM"] = valeurs_estimees

    print(f"Estimation terminee : {n_a_estimer} valeurs de FINATXEM completees.")
    print(f"Apercu des valeurs estimees (5 premieres) :")
    print(df.loc[df["FINATXEM_ESTIME_FLAG"], ["FINCBTXM", "FINATXEM"]].head())

    return df


def check_weight_comparability(df: pd.DataFrame, weight_col: str = "FINLWT21") -> bool:
    """Verifie, avant d'appliquer l'estimation, que les foyers utilises
    pour calibrer les ratios (2021-2023, revenu net ET brut connus) ont
    une ponderation comparable a celle des foyers a estimer (2024,
    revenu net manquant). Recommandation de l'encadrant : l'estimation
    par ratio n'est defendable que si les deux groupes representent des
    populations statistiquement comparables, pas juste des chiffres
    isoles pris au hasard.
    """
    calibration = df.dropna(subset=["FINATXEM", "FINCBTXM"])
    a_estimer = df[df["FINATXEM"].isna() & df["FINCBTXM"].notna()]

    if weight_col not in df.columns or a_estimer.empty:
        print("[ATTENTION] Verification de ponderation impossible (colonne absente ou rien a estimer).")
        return True

    poids_calib_moyen = calibration[weight_col].mean()
    poids_calib_median = calibration[weight_col].median()
    poids_estim_moyen = a_estimer[weight_col].mean()
    poids_estim_median = a_estimer[weight_col].median()

    ecart_relatif_moyenne = abs(poids_estim_moyen - poids_calib_moyen) / poids_calib_moyen * 100
    ecart_relatif_mediane = abs(poids_estim_median - poids_calib_median) / poids_calib_median * 100

    print(f"\n--- Verification de comparabilite de la ponderation ({weight_col}) ---")
    print(f"Foyers de calibration (2021-2023, net+brut connus) : {len(calibration)} foyers")
    print(f"  Poids moyen : {poids_calib_moyen:,.1f}  |  Poids median : {poids_calib_median:,.1f}")
    print(f"Foyers a estimer (2024, net manquant)               : {len(a_estimer)} foyers")
    print(f"  Poids moyen : {poids_estim_moyen:,.1f}  |  Poids median : {poids_estim_median:,.1f}")
    print(f"Ecart relatif (moyenne)  : {ecart_relatif_moyenne:.1f}%")
    print(f"Ecart relatif (mediane)  : {ecart_relatif_mediane:.1f}%")

    seuil_alerte = 15.0  # ecart relatif au-dela duquel on documente une reserve
    if ecart_relatif_moyenne < seuil_alerte and ecart_relatif_mediane < seuil_alerte:
        print(f"Verdict : COMPARABLE (ecarts < {seuil_alerte}%) -> l'estimation par tranches est defendable.")
        return True
    else:
        print(f"Verdict : ECART NOTABLE (>= {seuil_alerte}%) -> les deux groupes ne representent pas exactement "
              f"la meme population. L'estimation reste faite, mais cette limite doit etre documentee "
              f"explicitement dans le rapport.")
        return False


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    filepath = INPUT_DIR / FILENAME
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Charge : {filepath} ({len(df)} lignes)")
    print(f"FINATXEM manquant avant estimation : {df['FINATXEM'].isna().sum()} "
          f"({df['FINATXEM'].isna().mean()*100:.2f}%)")

    bin_edges, bin_stats = build_bin_ratios(df, N_BINS)

    ponderation_comparable = check_weight_comparability(df)

    df = estimate_missing_finatxem(df, bin_edges, bin_stats)

    print(f"\nFINATXEM manquant apres estimation : {df['FINATXEM'].isna().sum()} "
          f"(devrait etre proche de 0, sauf foyers sans FINCBTXM non plus)")

    bin_stats.to_csv(OUTPUT_DIR / "ratios_net_brut_par_tranche.csv")
    output_path = OUTPUT_DIR / "interview_period2_full_finatxem_estime.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSauvegarde : {output_path}")
    print(f"Colonne FINATXEM_ESTIME_FLAG ajoutee : {df['FINATXEM_ESTIME_FLAG'].sum()} valeurs marquees comme estimees.")