import pandas as pd
from pathlib import Path

df = pd.read_csv(Path("outputs_v2") / "interview_period2_scaled.csv", low_memory=False)

train_mask = df["FINATXEM"].notna()

# Nombre et statistiques des foyers avec FINATXEM reellement negatif en entrainement (2021-2023)
negatifs_entrainement = df.loc[train_mask & (df["FINATXEM"] < 0), "FINATXEM"]

print(f"{len(negatifs_entrainement)} foyers avec FINATXEM reellement negatif chez les foyers d'entrainement (2021-2023)")
print(f"sur un total de {train_mask.sum()} foyers d'entrainement "
      f"({len(negatifs_entrainement) / train_mask.sum() * 100:.2f}%)")

if len(negatifs_entrainement) > 0:
    print("\nStatistiques de ces valeurs negatives observees :")
    print(negatifs_entrainement.describe())