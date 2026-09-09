"""
Verification : la regression Ridge preserve-t-elle l'effet EITC chez les
foyers a faible revenu, ou le lisse-t-elle trop (a cause de la penalite
de regularisation) ?

Principe : on compare, pour les foyers dans la meme tranche de revenu
brut (les plus bas), le ratio net/brut REELLEMENT DECLARE (2021-2023,
FINATXEM connue) au ratio PREDIT par la regression (2024, FINATXEM
estimee). Si les deux ratios sont proches, la regression preserve bien
l'effet EITC observe dans les vraies donnees. Si le ratio predit est
nettement plus bas, la regularisation Ridge lisse trop l'effet.
"""

import pandas as pd
from pathlib import Path

INPUT_DIR = Path("outputs_v2")
FILENAME = "interview_period2_finatxem_regression.csv"

# Meme tranche basse que celle trouvee lors de l'approche par tranches
# (methode abandonnee en faveur de la regression, mais utile ici comme
# reference de comparaison) : les revenus bruts les plus faibles.
SEUIL_TRANCHE_BASSE = 8131  # borne haute de la 1ere tranche (deja calculee precedemment)


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    filepath = INPUT_DIR / FILENAME
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Charge : {filepath} ({len(df)} lignes)")

    if "FINATXEM_ESTIME_FLAG" not in df.columns:
        raise KeyError("FINATXEM_ESTIME_FLAG absente - relancer regression_finatxem_2024.py d'abord.")

    df_declare = df[~df["FINATXEM_ESTIME_FLAG"]].copy()
    df_estime = df[df["FINATXEM_ESTIME_FLAG"]].copy()

    print(f"\nFoyers avec FINATXEM reellement declaree (2021-2023) : {len(df_declare)}")
    print(f"Foyers avec FINATXEM estimee par regression (2024)    : {len(df_estime)}")

    # Restreindre aux foyers a revenu brut positif et dans la tranche basse
    declare_bas = df_declare[(df_declare["FINCBTXM"] > 0) & (df_declare["FINCBTXM"] <= SEUIL_TRANCHE_BASSE)]
    estime_bas = df_estime[(df_estime["FINCBTXM"] > 0) & (df_estime["FINCBTXM"] <= SEUIL_TRANCHE_BASSE)]

    print(f"\n--- Tranche basse : revenu brut entre 0 et {SEUIL_TRANCHE_BASSE:,} $ ---")
    print(f"Foyers declares dans cette tranche (2021-2023) : {len(declare_bas)}")
    print(f"Foyers estimes dans cette tranche (2024)         : {len(estime_bas)}")

    if len(declare_bas) > 0:
        ratio_declare = (declare_bas["FINATXEM"] / declare_bas["FINCBTXM"]).mean()
        print(f"\nRatio net/brut REELLEMENT DECLARE (2021-2023, tranche basse) : {ratio_declare:.3f}")
    else:
        ratio_declare = None
        print("\n[ATTENTION] Aucun foyer declare dans cette tranche.")

    if len(estime_bas) > 0:
        ratio_estime = (estime_bas["FINATXEM"] / estime_bas["FINCBTXM"]).mean()
        print(f"Ratio net/brut PREDIT PAR LA REGRESSION (2024, tranche basse) : {ratio_estime:.3f}")
    else:
        ratio_estime = None
        print("[ATTENTION] Aucun foyer estime dans cette tranche.")

    if ratio_declare is not None and ratio_estime is not None:
        ecart = abs(ratio_estime - ratio_declare)
        ecart_relatif = ecart / ratio_declare * 100
        print(f"\nEcart absolu  : {ecart:.3f}")
        print(f"Ecart relatif : {ecart_relatif:.1f}%")

        if ecart_relatif < 10:
            print("Verdict : la regression PRESERVE BIEN l'effet observe dans les donnees reelles "
                  "sur cette tranche de revenu (ecart < 10%).")
        elif ecart_relatif < 25:
            print("Verdict : ecart MODERE - la regression attenue partiellement l'effet, "
                  "a mentionner comme limite dans le rapport.")
        else:
            print("Verdict : ecart IMPORTANT - la regularisation Ridge lisse significativement "
                  "l'effet EITC sur cette tranche, limite methodologique a documenter clairement.")