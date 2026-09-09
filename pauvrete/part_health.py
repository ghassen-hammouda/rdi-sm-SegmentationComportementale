import pandas as pd
df = pd.read_csv("outputs_v2/household_period1.csv")

health_negatif = df[df["HEALTH_FULL_ANNUEL"] < 0]
print(f"{len(health_negatif)} foyers avec HEALTH_FULL_ANNUEL negatif")
print(health_negatif[["NB_TRIMESTRES", "TOTEXP_FULL_ANNUEL", "HEALTH_FULL_ANNUEL"]].head(10))