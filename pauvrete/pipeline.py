"""
Orchestrateur du pipeline (US-23), style MLOps leger.

Permet d'executer CHAQUE ETAPE SEPAREMENT (utile pour deboguer, ou
pour ne relancer qu'un morceau apres une modification), ou bien TOUT
LE PIPELINE d'un coup, dans le bon ordre, avec logs clairs de ce qui
a ete execute, ce qui a reussi, ce qui a echoue.

IMPORTANT : la liste ETAPES ci-dessous reference les noms de scripts
tels qu'utilises dans ce projet. Si tes fichiers locaux portent des
noms differents (ex. build_poverty_targets.py au lieu de
Build_poverty_targets_v2.py), adapte la colonne "script" du dictionnaire
ETAPES pour qu'elle corresponde exactement aux noms presents sur ta
machine.

Usage :
    py pipeline.py --list                  -> liste les etapes disponibles
    py pipeline.py --step nettoyage        -> execute une seule etape
    py pipeline.py --all                   -> execute tout, dans l'ordre
    py pipeline.py --from clustering       -> execute a partir d'une etape donnee
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# ----------------------------------------------------------------------
# Definition des etapes du pipeline, DANS L'ORDRE D'EXECUTION.
# Cle = nom court utilise en ligne de commande.
# "script" = nom du fichier .py a executer (A ADAPTER a tes noms locaux).
# "description" = ce que fait l'etape, pour --list.
# ----------------------------------------------------------------------
ETAPES = {
    "selection_colonnes": {
        "script": "pauvrete_data.py",
        "description": "Selection des colonnes utiles depuis les fichiers bruts fmli*.csv",
    },
    "depenses_completes": {
        "script": "Build full expenditures.py",
        "description": "Reconstitution du trimestre complet (CQ + PQ)",
    },
    "nettoyage": {
        "script": "clean_v2.py",
        "description": "Dedoublonnage, colonnes incompletes, imputation, ecretage outliers",
    },
    "encodage": {
        "script": "ecodage_categoriel_v2.py",
        "description": "One-hot des categorielles (preparation regression)",
    },
    "scaling": {
        "script": "scaler_v2.py",
        "description": "RobustScaler (preparation regression)",
    },
    "regression_finatxem": {
        "script": "regression_finatxem_2024.py",
        "description": "Estimation du revenu net manquant en 2024 (regression lineaire)",
    },
    "fusion_revenu": {
        "script": "merge_net.py",
        "description": "Fusion du revenu net complete avec le fichier propre",
    },
    "agregation_foyer": {
        "script": "aggregate_by_household_v2.py",
        "description": "Passage de foyer-trimestre a un foyer = une ligne",
    },
    "features_metier": {
        "script": "build_business_features_v2.py",
        "description": "Echelle d'equivalence, parts de budget, ratios",
    },
    "cibles_pauvrete": {
        "script": "Build_poverty_targets_v2.py",
        "description": "Seuil de pauvrete et cibles (revenu net, consommation)",
    },
    "preparation_clustering": {
        "script": "build_clustering_target_v2.py",
        "description": "Exclusions, encodage, standardisation pour le clustering",
    },
    "benchmark_clustering": {
        "script": "benchmark_clustering_f.py",
        "description": "Benchmark K-Means / GMM / CAH sur echantillon",
    },
    "clustering_final": {
        "script": "final_clustering_population_v2.py",
        "description": "K-Means final sur toute la population, k=3 a 6",
    },
    "correction_regles": {
        "script": "apply_rules.py",
        "description": "Correction par regles metier (charge locative, isolement/age)",
    },
    "test_robustesse": {
        "script": "test_final.py",
        "description": "Verification de stabilite (split aleatoire 50/50)",
    },
    "resume_effectifs": {
        "script": "final_output.py",
        "description": "Consolidation des effectifs avant/apres correction",
    },
}

ORDRE_ETAPES = list(ETAPES.keys())


def lister_etapes() -> None:
    print("\nEtapes disponibles, dans l'ordre d'execution :\n")
    for i, (nom, info) in enumerate(ETAPES.items(), start=1):
        print(f"  {i:2d}. {nom:25s} ({info['script']})")
        print(f"      {info['description']}")


def executer_etape(nom: str) -> bool:
    if nom not in ETAPES:
        print(f"[ERREUR] Etape inconnue : '{nom}'. Utilise --list pour voir les etapes disponibles.")
        return False

    script = ETAPES[nom]["script"]
    script_path = Path(script)

    if not script_path.exists():
        print(f"[ERREUR] Script introuvable : '{script}' (etape '{nom}'). "
              f"Verifie le nom dans le dictionnaire ETAPES ou le repertoire courant.")
        return False

    print(f"\n{'=' * 70}\nETAPE : {nom}  ->  {script}\n{'=' * 70}")
    debut = time.time()

    resultat = subprocess.run([sys.executable, str(script_path)], capture_output=False)

    duree = time.time() - debut
    if resultat.returncode == 0:
        print(f"\n[OK] '{nom}' termine en {duree:.1f}s")
        return True
    else:
        print(f"\n[ECHEC] '{nom}' a echoue (code {resultat.returncode}) apres {duree:.1f}s")
        return False


def executer_pipeline(etapes_a_executer: list[str]) -> None:
    resultats = {}
    debut_total = time.time()

    for nom in etapes_a_executer:
        succes = executer_etape(nom)
        resultats[nom] = succes
        if not succes:
            print(f"\n[ARRET] Le pipeline s'arrete a l'etape '{nom}' suite a l'echec.")
            break

    duree_totale = time.time() - debut_total

    print(f"\n{'=' * 70}\nRESUME DU PIPELINE\n{'=' * 70}")
    for nom, succes in resultats.items():
        statut = "OK" if succes else "ECHEC"
        print(f"  [{statut}] {nom}")
    print(f"\nDuree totale : {duree_totale:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrateur du pipeline de segmentation de pauvrete")
    parser.add_argument("--list", action="store_true", help="Liste les etapes disponibles")
    parser.add_argument("--step", type=str, help="Execute une seule etape (par son nom court)")
    parser.add_argument("--all", action="store_true", help="Execute tout le pipeline, dans l'ordre")
    parser.add_argument("--from", dest="depuis", type=str,
                         help="Execute le pipeline a partir d'une etape donnee (incluse)")

    args = parser.parse_args()

    if args.list:
        lister_etapes()
    elif args.step:
        executer_etape(args.step)
    elif args.all:
        executer_pipeline(ORDRE_ETAPES)
    elif args.depuis:
        if args.depuis not in ORDRE_ETAPES:
            print(f"[ERREUR] Etape de depart inconnue : '{args.depuis}'")
        else:
            idx = ORDRE_ETAPES.index(args.depuis)
            executer_pipeline(ORDRE_ETAPES[idx:])
    else:
        parser.print_help()
        print("\nExemples :")
        print("  py pipeline.py --list")
        print("  py pipeline.py --step clustering_final")
        print("  py pipeline.py --all")
        print("  py pipeline.py --from correction_regles")