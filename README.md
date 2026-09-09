# Segmentation comportementale de la pauvreté — CEX

Projet de clustering et de règles métier pour identifier les foyers pauvres à partir des données de la **Consumer Expenditure Survey (CEX)** du Bureau of Labor Statistics (BLS, États-Unis).

## Contexte

Deux périodes étudiées séparément, à cause d'un changement de questionnaire du BLS en 2024 :
- **Period1** : 2017-2020 (38 439 foyers après nettoyage)
- **Period2** : 2021-2024 (32 930 foyers après nettoyage)

## Objectif

Identifier les foyers pauvres via une approche en deux temps :
1. **Clustering** (K-Means) sur le comportement de consommation et la démographie — sans jamais voir le revenu ni le statut de pauvreté
2. **Règles métier** pour corriger les limites connues du clustering, construites et validées empiriquement

## Structure du dépôt

```
pauvrete/
├── pipeline.py                          # Orchestrateur (exécution étape par étape ou complète)
├── regles_pauvrete.py                   # Module partagé des règles métier (Règle 1, Règle 2)
├── streamlit_dashboard.py               # Tableau de bord de présentation (pipeline, résultats, calculateur)
│
├── pauvrete_data.py                     # 1. Sélection des colonnes brutes
├── Build full expenditures.py           # 2. Reconstitution du trimestre complet (CQ+PQ)
├── clean_v2.py                          # 3. Nettoyage (doublons, outliers, imputation)
├── ecodage_categoriel_v2.py             # 4. Encodage (préparation régression)
├── scaler_v2.py                         # 5. Standardisation (préparation régression)
├── regression_finatxem_2024.py          # 6. Estimation du revenu net manquant (2024)
├── merge_net.py                         # 7. Fusion du revenu net complété
├── aggregate_by_household_v2.py         # 8. Agrégation foyer-trimestre → foyer
├── build_business_features_v2.py        # 9. Échelle d'équivalence, parts de budget, ratios
├── Build_poverty_targets_v2.py          # 10. Seuil et cibles de pauvreté
├── build_clustering_target_v2.py        # 11. Préparation de l'espace de clustering
├── benchmark_clustering_f.py            # 12. Benchmark K-Means / GMM / CAH (échantillon)
├── final_clustering_population_v2.py    # 13. K-Means final sur toute la population
├── apply_rules.py                       # 14. Correction par règles métier
├── test_final.py                        # 15. Test de robustesse (split 50/50)
├── final_output.py                      # 16. Consolidation des effectifs
│
└── outputs_v2/                          # Résultats générés (exclu de Git, voir .gitignore)
```

## Installation

```bash
py -m pip install pandas numpy scikit-learn scipy matplotlib streamlit --break-system-packages
```

## Exécution du pipeline

Le pipeline peut être exécuté étape par étape (utile pour déboguer) ou en une fois.

```bash
# Lister toutes les étapes disponibles
py pipeline.py --list

# Exécuter une seule étape
py pipeline.py --step clustering_final

# Exécuter tout le pipeline, dans l'ordre
py pipeline.py --all

# Reprendre à partir d'une étape donnée (les précédentes déjà faites)
py pipeline.py --from correction_regles
```

## Lancer le tableau de bord de présentation

```bash
py -m streamlit run streamlit_dashboard.py
```

5 onglets : Pipeline (vue d'ensemble), Résultats du clustering, Règles de correction, Test de robustesse, Calculateur interactif.

## Méthodologie clé

### Échelle d'équivalence (convention OCDE modifiée)
```
échelle = 1 + 0.5 × (adultes supplémentaires) + 0.3 × (enfants)
```

### Seuil de pauvreté (convention OCDE)
```
seuil = médiane(revenu net équivalisé) / 2
```
Calculé séparément par période, sur le revenu **net** (revenu disponible), pas brut.

### Estimation du revenu net manquant (2024)
28% des foyers 2024 n'ont pas de revenu net déclaré (changement de questionnaire BLS). Estimé par régression linéaire (revenu brut + démographie), validée par split 80/20 :
- **R² = 0.9812**
- **MAE = 3 639 $**

### Principe du clustering
Le clustering (K-Means, benchmarké contre GMM et CAH) ne voit **jamais** le revenu ni les statuts de pauvreté. Ces variables sont réintroduites uniquement après coup, pour valider si les groupes trouvés à l'aveugle correspondent à de la vraie pauvreté.

### Règles de correction

**Règle 1 — Isolement et âge** (profil dominant en period1) :
```
FAM_TYPE == 8 (personne seule)
ET AGE_REF >= 65
ET part_retpen < 4%
ET totexp_par_equivalent_adulte < 22 000 $
```

**Règle 2 — Charge locative extrême** (profil dominant en period2, présent aussi en period1) :
```
ratio_logement_revenu_net > seuil
  Period1 : seuil = 2.0
  Period2 : seuil = 1.5 (abaissé après diagnostic, taux de faux positifs mesuré à 11.1%)
```

La Règle 2 est prioritaire sur la Règle 1 en cas de chevauchement.

## Résultats principaux

| | Period1 | Period2 |
|---|---|---|
| Foyers pauvres (revenu net) | 7 237 (18,77%) | 6 003 (18,20%) |
| Foyers pauvres (consommation) | 6 448 (16,73%) | 5 030 (15,25%) |
| Foyers "charge locative" (après correction) | 1 262 (3,28%) | 1 289 (3,91%) |

Comparé au taux officiel du Census Bureau (~11-12%) : écart expliqué par le seuil relatif OCDE (vs seuil absolu américain) et l'absence de pondération d'enquête (`FINLWT21` perdu en amont du pipeline).

## Limites connues

- **Pas de pondération d'enquête** : `FINLWT21` n'a pas été conservé lors de l'agrégation par foyer. Les résultats représentent l'échantillon CEX, pas exactement la population américaine pondérée.

- **Une partie de period2 (2024) a un revenu net estimé**, pas déclaré (voir `FINATXEM_ESTIME_FLAG`).

