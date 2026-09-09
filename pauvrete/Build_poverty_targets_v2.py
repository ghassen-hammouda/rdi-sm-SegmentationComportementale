"""
Seuil et cibles de pauvrete (US-08), sur le nouveau perimetre.

MODIFICATION : le seuil de pauvrete est desormais calcule sur la mediane
du revenu NET equivalise (FINATXEM), et non plus sur le revenu brut.
C'est la convention OCDE standard de la pauvrete relative (le point de
depart de la mesure OCDE est le revenu disponible, c'est-a-dire net,
pas brut). Le revenu brut (FINCBTXM) n'est plus utilise dans les
cibles de pauvrete - la cible statut_pauvrete_revenu_brut est retiree.

Methodologie :
- Echelle d'equivalence OCDE modifiee (deja calculee en US-06)
- Seuil = mediane(revenu NET equivalise) / 2, calcule separement par
  periode (convention OCDE : 50% de la mediane du revenu disponible)
- Deux cibles construites, comparees au MEME seuil :
  statut_pauvrete_revenu_net, statut_pauvrete_consommation
"""

import pandas as pd
from pathlib import Path

INPUT_DIR = Path("outputs_v2")
OUTPUT_DIR = Path("outputs_v2")

FILES = {
    "period1": "household_features_period1.csv",
    "period2": "household_features_period2.csv",
}


def discordance_breakdown(df: pd.DataFrame, col_a: str, col_b: str, label_a: str, label_b: str) -> None:
    total = len(df)
    both = int(((df[col_a] == 1) & (df[col_b] == 1)).sum())
    neither = int(((df[col_a] == 0) & (df[col_b] == 0)).sum())
    a_only = int(((df[col_a] == 1) & (df[col_b] == 0)).sum())
    b_only = int(((df[col_a] == 0) & (df[col_b] == 1)).sum())

    print(f"    Pauvre des deux cotes ({label_a} et {label_b}) : {both} ({both/total*100:.2f}%)")
    print(f"    Non pauvre des deux cotes                      : {neither} ({neither/total*100:.2f}%)")
    print(f"    Pauvre {label_a} seulement                     : {a_only} ({a_only/total*100:.2f}%)")
    print(f"    Pauvre {label_b} seulement                     : {b_only} ({b_only/total*100:.2f}%)")


def build_poverty_targets(df: pd.DataFrame, period_label: str) -> pd.DataFrame:
    df = df.copy()

    df["REVENU_NET_EQUIVALISE"] = df["FINATXEM"] / df["ECHELLE_EQUIVALENCE"]
    df["CONSOMMATION_EQUIVALISEE"] = df["TOTEXP_FULL_ANNUEL"] / df["ECHELLE_EQUIVALENCE"]

    mediane_net = df["REVENU_NET_EQUIVALISE"].median()
    seuil = mediane_net / 2
    print(f"[{period_label}] Mediane du revenu NET equivalise : {mediane_net:,.2f} $")
    print(f"[{period_label}] Seuil de pauvrete retenu (mediane net / 2, convention OCDE) : {seuil:,.2f} $")

    df["SEUIL_PAUVRETE"] = seuil
    df["statut_pauvrete_revenu_net"] = (df["REVENU_NET_EQUIVALISE"] < seuil).astype(int)
    df["statut_pauvrete_consommation"] = (df["CONSOMMATION_EQUIVALISEE"] < seuil).astype(int)

    taux_net = df["statut_pauvrete_revenu_net"].mean() * 100
    taux_conso = df["statut_pauvrete_consommation"].mean() * 100
    print(f"[{period_label}] Taux de pauvrete (revenu net)   : {taux_net:.2f}%")
    print(f"[{period_label}] Taux de pauvrete (consommation) : {taux_conso:.2f}%")

    print(f"\n[{period_label}] Detail revenu net vs consommation :")
    discordance_breakdown(df, "statut_pauvrete_revenu_net", "statut_pauvrete_consommation", "net", "consommation")

    return df


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    for period_label, filename in FILES.items():
        print(f"\n{'=' * 70}\nSEUIL ET CIBLES DE PAUVRETE - {period_label.upper()}\n{'=' * 70}")

        filepath = INPUT_DIR / filename
        df = pd.read_csv(filepath, low_memory=False)
        print(f"Charge : {filepath} ({len(df)} foyers, {df.shape[1]} colonnes)")

        df = build_poverty_targets(df, period_label)

        output_path = OUTPUT_DIR / f"poverty_targets_{period_label}.csv"
        df.to_csv(output_path, index=False)
        print(f"\nSauvegarde : {output_path} ({len(df)} foyers, {df.shape[1]} colonnes)")

    print(f"\n{'=' * 70}")
    print("Termine. Fichiers dans outputs_v2/ :")
    print("  poverty_targets_period1.csv")
    print("  poverty_targets_period2.csv")
    print("  -> 2 cibles de pauvrete construites (net, consommation), seuil ancre")
    print("     sur le revenu net (convention OCDE), pretes pour le clustering.")