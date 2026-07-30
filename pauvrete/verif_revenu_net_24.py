import pandas as pd

for annee_dossier, fichier in [("intrvw17", "fmli212.csv"), ("intrvw22", "fmli222.csv"), ("intrvw23", "fmli232.csv")]:
    header = pd.read_csv(f"data/{annee_dossier}/{fichier}", nrows=0).columns.tolist()
    print(annee_dossier, ":", "FOODCQ" in header, "FOODPQ" in header)