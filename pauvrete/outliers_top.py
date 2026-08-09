from pathlib import Path
fichiers = list(Path("data/intrvw24").glob("*.csv"))
print(len(fichiers), "fichiers trouves")
for f in fichiers:
    print(f.name)