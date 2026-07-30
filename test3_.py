import pandas as pd

header = pd.read_csv("data/intrvw17/fmli171x.csv", nrows=0).columns.tolist()

postes = ["TOTEXP", "FOOD", "HOUS", "TRANS", "HEALTH", "APPAR", "ENTERT",
          "EDUCA", "ALCBEV", "TOBACC", "PERSCA", "READ", "MISC", "CASHCO",
          "LIFINS", "RETPEN"]

for poste in postes:
    for suffixe in ["CQ", "PQ"]:
        nom = f"{poste}{suffixe}"
        matches = [c for c in header if c.startswith(nom)]
        print(f"{nom} : {matches}")