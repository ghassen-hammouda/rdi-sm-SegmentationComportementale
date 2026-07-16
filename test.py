import pandas as pd
from pathlib import Path

header = pd.read_csv("data/intrvw21/fmli212.csv", nrows=0).columns.tolist()
print("DESFLG" in header, "INT_PHON" in header, "INT_HOME" in header)