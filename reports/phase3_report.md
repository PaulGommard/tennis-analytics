# Phase 3 — optimisation et calibration Elo

## Protocole figé

- Historique continu depuis 2000, sans réinitialisation.
- Sélection des hyperparamètres exclusivement sur 2019–2022.
- Test final exclusivement à partir de 2023, consulté après gel des choix.
- Critère principal : log loss ; départage par Brier score puis accuracy.
- Aucun modèle de ML avancé : Elo global, Elo surface, shrinkage, inactivité simple et Platt.

## Audit de chronologie

Les 3,757 éditions (79,299 matchs) ont leurs
matchs regroupés sur une seule date source. La nouvelle chronologie exploite donc uniquement
l’ordre structurel des tours, avec mise à jour après chaque tour complet. Davis Cup, labels
inconnus et dates présentant un joueur dans plusieurs éditions restent conservateurs. Un seul
couple joueur/date inter-éditions a été détecté.

Ordre : Q1 → Q2 → Q3 → Q4 → RR → ER → R128 → R64 → R32 → R16 → QF → SF → FINAL.
F et BR partagent le même batch FINAL. Aucun match Q1–Q4 n’est présent dans ce corpus.

### Impact isolé de la chronologie sur le test (K=32, scale=400)

| model | matches | accuracy | log_loss | brier |
| --- | --- | --- | --- | --- |
| Elo V1 date batch | 10,376 | 0.6399 | 0.6334 | 0.2214 |
| Elo round K32/400 | 10,376 | 0.6403 | 0.6323 | 0.2210 |

Les détails par surface, niveau et année sont dans `elo_chronology_impact.csv`; l’audit complet
est dans `round_chronology_audit.md` et `round_chronology_audit.json`.

## Sélection sur validation 2019–2022

- Elo global : K=20, scale=300.
- Shrinkage surface : C=75, avec `w = n / (n + C)` par joueur.
- Variante brute retenue : `surface_shrinkage`.
- Inactivité retenue : seuil 90 jours, force 0.1.
- Calibration de Platt : intercept=-0.130460, pente=1.040503.

### Comparaison des variantes brutes sur validation

| variant | validation_matches | validation_accuracy | validation_log_loss | validation_brier |
| --- | --- | --- | --- | --- |
| global | 9,766 | 0.6399 | 0.6264 | 0.2189 |
| surface_independent | 9,766 | 0.6324 | 0.6306 | 0.2208 |
| surface_shrinkage | 9,766 | 0.6432 | 0.6210 | 0.2167 |

### Cinq meilleures variantes d’inactivité sur validation

| threshold_days | strength | is_no_decay | validation_accuracy | validation_log_loss | validation_brier |
| --- | --- | --- | --- | --- | --- |
| 90 | 0.1 | 0 | 0.6452 | 0.6194 | 0.2159 |
| 180 | 0.2 | 0 | 0.6476 | 0.6194 | 0.2159 |
| 180 | 0.1 | 0 | 0.6450 | 0.6196 | 0.2161 |
| 90 | 0.05 | 0 | 0.6444 | 0.6198 | 0.2162 |
| 180 | 0.05 | 0 | 0.6448 | 0.6202 | 0.2163 |

Les grilles exhaustives mais volontairement bornées sont dans `elo_global_tuning.csv`,
`elo_surface_shrinkage_tuning.csv` et `elo_inactivity_tuning.csv`.

## Résultat final sur le test 2023+

| model | matches | accuracy | log_loss | brier |
| --- | --- | --- | --- | --- |
| ATP Ranking | 10,188 | 0.6387 | — | — |
| Elo V1 date batch | 10,376 | 0.6399 | 0.6334 | 0.2214 |
| Elo round K32/400 | 10,376 | 0.6403 | 0.6323 | 0.2210 |
| Elo global optimized | 10,376 | 0.6379 | 0.6311 | 0.2207 |
| Surface Elo V1 date batch | 10,324 | 0.6305 | 0.6392 | 0.2243 |
| Surface Elo independent optimized | 10,324 | 0.6280 | 0.6387 | 0.2242 |
| Surface Elo shrinkage | 10,376 | 0.6386 | 0.6282 | 0.2197 |
| Elo + inactivity best tested | 10,376 | 0.6414 | 0.6257 | 0.2187 |
| Elo final raw | 10,376 | 0.6414 | 0.6257 | 0.2187 |
| Elo final calibrated | 10,376 | 0.6368 | 0.6253 | 0.2186 |

### Comparaison appariée avec le classement ATP

Chaque paire conserve exactement le même dénominateur : rangs ATP disponibles et distincts,
puis exclusion des éventuelles probabilités Elo exactement égales à 0,5 pour la paire brute.

| model | matches | accuracy | log_loss | brier |
| --- | --- | --- | --- | --- |
| ATP Ranking (raw overlap) | 10,175 | 0.6388 | — | — |
| Elo final raw | 10,175 | 0.6420 | 0.6250 | 0.2184 |
| ATP Ranking (calibrated overlap) | 10,188 | 0.6387 | — | — |
| Elo final calibrated | 10,188 | 0.6394 | 0.6243 | 0.2181 |

### Résultats par surface

| segment | model | matches | accuracy | log_loss | brier |
| --- | --- | --- | --- | --- | --- |
| Hard | Elo global optimized | 6,019 | 0.6430 | 0.6255 | 0.2183 |
| Hard | Surface Elo independent optimized | 6,019 | 0.6396 | 0.6294 | 0.2203 |
| Hard | Surface Elo shrinkage | 6,019 | 0.6417 | 0.6244 | 0.2181 |
| Hard | Elo final raw | 6,019 | 0.6415 | 0.6225 | 0.2174 |
| Hard | Elo final calibrated | 6,019 | 0.6396 | 0.6221 | 0.2172 |
| Clay | Elo global optimized | 3,356 | 0.6226 | 0.6462 | 0.2272 |
| Clay | Surface Elo independent optimized | 3,356 | 0.6190 | 0.6500 | 0.2287 |
| Clay | Surface Elo shrinkage | 3,356 | 0.6283 | 0.6391 | 0.2243 |
| Clay | Elo final raw | 3,356 | 0.6364 | 0.6344 | 0.2224 |
| Clay | Elo final calibrated | 3,356 | 0.6284 | 0.6329 | 0.2218 |
| Grass | Elo global optimized | 949 | 0.6649 | 0.6104 | 0.2117 |
| Grass | Surface Elo independent optimized | 949 | 0.5860 | 0.6584 | 0.2334 |
| Grass | Surface Elo shrinkage | 949 | 0.6607 | 0.6116 | 0.2128 |
| Grass | Elo final raw | 949 | 0.6660 | 0.6125 | 0.2131 |
| Grass | Elo final calibrated | 949 | 0.6544 | 0.6149 | 0.2142 |

Les résultats complets par surface et niveau sont dans `phase3_test_segments.csv`.

## Calibration sur le test

| model | bin | matches | mean_probability | actual_win_rate | gap |
| --- | --- | --- | --- | --- | --- |
| Final raw | 50-55% | 2,295 | 0.5243 | 0.5237 | -0.0005 |
| Final raw | 55-60% | 1,890 | 0.5747 | 0.5698 | -0.0049 |
| Final raw | 60-65% | 1,704 | 0.6240 | 0.6156 | -0.0084 |
| Final raw | 65-70% | 1,369 | 0.6742 | 0.6589 | -0.0153 |
| Final raw | 70-75% | 1,064 | 0.7234 | 0.6917 | -0.0317 |
| Final raw | 75-80% | 864 | 0.7739 | 0.7778 | 0.0039 |
| Final raw | 80-90% | 900 | 0.8429 | 0.8400 | -0.0029 |
| Final raw | 90-100% | 256 | 0.9312 | 0.9336 | 0.0023 |
| Final calibrated | 50-55% | 2,194 | 0.5245 | 0.5091 | -0.0154 |
| Final calibrated | 55-60% | 1,945 | 0.5742 | 0.5599 | -0.0143 |
| Final calibrated | 60-65% | 1,616 | 0.6242 | 0.6077 | -0.0165 |
| Final calibrated | 65-70% | 1,362 | 0.6735 | 0.6564 | -0.0171 |
| Final calibrated | 70-75% | 1,083 | 0.7248 | 0.6814 | -0.0434 |
| Final calibrated | 75-80% | 872 | 0.7734 | 0.7764 | 0.0030 |
| Final calibrated | 80-90% | 1,010 | 0.8429 | 0.8287 | -0.0141 |
| Final calibrated | 90-100% | 294 | 0.9337 | 0.9286 | -0.0051 |

La calibration est apprise sur la seule validation puis appliquée au test. Les probabilités
calibrées historiques antérieures au test ne sont pas publiées dans le parquet afin d’éviter
toute interprétation walk-forward incorrecte.

## Artefact retenu

`data/processed/match_ratings_v2.parquet` contient les ratings pré-match, les compteurs de
surface antérieurs, les probabilités brutes et la probabilité Platt uniquement sur 2023+.
`elo_annual_walk_forward.csv` donne les diagnostics annuels des variantes de rating fixes ;
il exclut volontairement Platt, appris ultérieurement sur 2019–2022.

## Limites

- La date source reste la date de début du tournoi : l’ordre des tours est structurel, pas daté.
- Tous les matchs d’un même tour restent gelés dans un batch.
- Les formats ambigus sont volontairement sous-informatifs plutôt que reconstruits artificiellement.
- Le niveau ATP `A` ne permet pas de séparer fiablement ATP 250 et ATP 500.
- Une bonne log loss ne constitue pas à elle seule une preuve de rentabilité de paris.
