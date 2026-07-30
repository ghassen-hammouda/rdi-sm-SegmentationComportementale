import pandas as pd
from pathlib import Path

INPUT_DIR = Path("outputs_v2")
OUTPUT_DIR = Path("outputs_eda_v2")
OUTPUT_DIR.mkdir(exist_ok=True)

FILES = {
    "period1": "interview_period1_full.csv",
    "period2": "interview_period2_full.csv",
}

# Variables à croiser pour juger de la cohérence
CHECK_COLS = ["FINATXEM", "FINCBTXM", "TOTEXP_FULL", "HOUS_FULL", "TRANS_FULL"]

def analyse_extremes(df, col="FINATXEM", n=20):
    top = df.nlargest(n, col)
    bottom = df.nsmallest(n, col)
    extremes = pd.concat([top, bottom])
    cols_present = [c for c in CHECK_COLS if c in df.columns]
    return extremes[cols_present]

for period, fname in FILES.items():
    df = pd.read_csv(INPUT_DIR / fname)
    print(f"\n=== {period} — cohérence FINATXEM vs autres variables ===")

    result = analyse_extremes(df, "FINATXEM")
    print(result.to_string())

    result.to_csv(OUTPUT_DIR / f"coherence_FINATXEM_{period}.csv")