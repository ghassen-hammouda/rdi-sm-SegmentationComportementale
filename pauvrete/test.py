import pandas as pd
df = pd.read_csv("outputs_v2/interview_period2_finatxem_regression.csv")
negatifs = df[(df["FINATXEM_ESTIME_FLAG"]) & (df["FINATXEM"] < 0)]
print(f"{len(negatifs)} foyers avec un revenu net estime negatif")
print(negatifs["FINCBTXM"].describe())
