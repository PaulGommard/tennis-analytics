# Phase 4 — statistiques historiques et régression logistique

## A. Audit des statistiques

Le bloc service complet couvre 72,261 matchs
(91.12%); la durée valide couvre
70,523 matchs (88.93%). Le train 2000–2018 est
conservé : les premières années sont déjà proches de 87–90 %, et le manque principal vient de
la Davis Cup. Voir `stats_availability_audit.md`.

## B. Features créées

Le dataset contient 215 colonnes et 78,845 matchs. Les ratios
service/retour utilisent des numérateurs et opportunités cumulés, avec horizons carrière, 365j
et 90j, priors temporels, variantes surface, forme et charge 7/14/30j. Définitions complètes :
`docs/features_v2.md`.

Shrinkage sélectionné sur validation : points C=50, surface C=1000.

## C–D. Signal incrémental et ablation (validation 2019–2022)

| model | features | c_value | accuracy | log_loss | brier | roc_auc |
| --- | --- | --- | --- | --- | --- | --- |
| MODEL_5_ALL | 50 | 1.0000 | 0.6659 | 0.6031 | 0.2090 | 0.7321 |
| MODEL_RANKING_STATS_FORM | 34 | 0.0100 | 0.6594 | 0.6076 | 0.2108 | 0.7267 |
| MODEL_RANKING_SERVE_RETURN | 15 | 0.0100 | 0.6534 | 0.6124 | 0.2129 | 0.7202 |
| MODEL_1_ELO_RANKING | 3 | 0.0100 | 0.6534 | 0.6127 | 0.2130 | 0.7199 |
| MODEL_FATIGUE_ABLATION | 14 | 100.0000 | 0.6542 | 0.6141 | 0.2136 | 0.7191 |
| MODEL_4_PLUS_FORM_ROLLING | 32 | 0.0100 | 0.6531 | 0.6145 | 0.2138 | 0.7178 |
| MODEL_2_ELO_SERVE | 9 | 0.0100 | 0.6460 | 0.6186 | 0.2156 | 0.7118 |
| MODEL_3_ELO_SERVE_RETURN | 13 | 0.0100 | 0.6462 | 0.6187 | 0.2156 | 0.7118 |
| MODEL_0_ELO | 1 | 0.0100 | 0.6480 | 0.6188 | 0.2157 | 0.7117 |
| MODEL_RETURN_ABLATION | 5 | 0.0100 | 0.6466 | 0.6188 | 0.2157 | 0.7116 |

Le meilleur bloc est `MODEL_5_ALL`. Les variables corrélées rendent les coefficients individuels
conditionnels aux autres variables; l'ablation est donc la preuve principale d'apport.

Le service seul ne bat pas Elo sur validation. Les statistiques carrière service+retour
n'ajoutent qu'un gain modeste une fois le ranking connu (0,61269 → 0,61236). Le signal le plus
net apparaît avec les fenêtres de forme et de statistiques (→ 0,60756), puis avec la charge et
l'expérience dans le modèle complet (→ 0,60314).

## E. Comparaison finale — test 2023–2026

| model | matches | accuracy | log_loss | brier | roc_auc |
| --- | --- | --- | --- | --- | --- |
| Elo Phase 3 raw | 10,376 | 0.6413 | 0.6257 | 0.2187 | 0.7007 |
| Elo Phase 3 calibrated | 10,376 | 0.6368 | 0.6253 | 0.2186 | 0.7007 |
| ATP Ranking | 10,188 | 0.6387 | — | — | — |
| MODEL_0_ELO | 10,376 | 0.6393 | 0.6259 | 0.2188 | 0.7007 |
| MODEL_1_ELO_RANKING | 10,376 | 0.6453 | 0.6217 | 0.2167 | 0.7091 |
| MODEL_2_ELO_SERVE | 10,376 | 0.6375 | 0.6260 | 0.2188 | 0.7006 |
| MODEL_RETURN_ABLATION | 10,376 | 0.6397 | 0.6255 | 0.2186 | 0.7014 |
| MODEL_3_ELO_SERVE_RETURN | 10,376 | 0.6391 | 0.6256 | 0.2186 | 0.7013 |
| MODEL_RANKING_SERVE_RETURN | 10,376 | 0.6450 | 0.6214 | 0.2165 | 0.7098 |
| MODEL_4_PLUS_FORM_ROLLING | 10,376 | 0.6473 | 0.6218 | 0.2168 | 0.7079 |
| MODEL_RANKING_STATS_FORM | 10,376 | 0.6557 | 0.6167 | 0.2143 | 0.7173 |
| MODEL_FATIGUE_ABLATION | 10,376 | 0.6454 | 0.6233 | 0.2175 | 0.7058 |
| MODEL_5_ALL | 10,376 | 0.6572 | 0.6145 | 0.2134 | 0.7206 |
| MODEL_5_ALL CALIBRATED | 10,376 | 0.6537 | 0.6130 | 0.2129 | 0.7206 |

Comparaison appariée sur les matchs avec deux rangs ATP disponibles et distincts :

| model | matches | accuracy | log_loss | brier | roc_auc |
| --- | --- | --- | --- | --- | --- |
| ATP Ranking | 10,188 | 0.6387 | — | — | — |
| Logistic Phase 4 | 10,188 | 0.6570 | 0.6114 | 0.2122 | 0.7227 |

## F–G. Surfaces et années

| segment_type | segment | model | matches | accuracy | log_loss | brier | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| surface | Hard | Elo Phase 3 calibrated | 6,019 | 0.6396 | 0.6221 | 0.2172 | 0.7056 |
| surface | Hard | Logistic Phase 4 | 6,019 | 0.6556 | 0.6106 | 0.2120 | 0.7235 |
| surface | Clay | Elo Phase 3 calibrated | 3,356 | 0.6284 | 0.6329 | 0.2218 | 0.6887 |
| surface | Clay | Logistic Phase 4 | 3,356 | 0.6532 | 0.6156 | 0.2138 | 0.7169 |
| surface | Grass | Elo Phase 3 calibrated | 949 | 0.6544 | 0.6149 | 0.2142 | 0.7192 |
| surface | Grass | Logistic Phase 4 | 949 | 0.6512 | 0.6153 | 0.2141 | 0.7213 |
| year | 2023 | Elo Phase 3 calibrated | 2,964 | 0.6350 | 0.6283 | 0.2197 | 0.6986 |
| year | 2023 | Logistic Phase 4 | 2,964 | 0.6538 | 0.6119 | 0.2129 | 0.7211 |
| year | 2024 | Elo Phase 3 calibrated | 3,136 | 0.6368 | 0.6210 | 0.2171 | 0.7040 |
| year | 2024 | Logistic Phase 4 | 3,136 | 0.6527 | 0.6097 | 0.2115 | 0.7246 |
| year | 2025 | Elo Phase 3 calibrated | 2,839 | 0.6379 | 0.6287 | 0.2200 | 0.6955 |
| year | 2025 | Logistic Phase 4 | 2,839 | 0.6492 | 0.6211 | 0.2161 | 0.7120 |
| year | 2026 | Elo Phase 3 calibrated | 1,437 | 0.6381 | 0.6220 | 0.2168 | 0.7079 |
| year | 2026 | Logistic Phase 4 | 1,437 | 0.6646 | 0.6063 | 0.2098 | 0.7303 |

## H. Calibration

Calibration Platt retenue : True. Elle a été ajustée sur 2019–2022 uniquement,
après entraînement du modèle sur 2000–2018. Buckets complets : `phase4_calibration.csv`.

| model | matches | accuracy | log_loss | brier | roc_auc |
| --- | --- | --- | --- | --- | --- |
| Logistic raw | 9,766 | 0.6659 | 0.6031 | 0.2090 | 0.7321 |
| Logistic Platt | 9,766 | 0.6627 | 0.6020 | 0.2084 | 0.7321 |

## I. Coefficients standardisés

| feature | standardized_coefficient |
| --- | --- |
| elo_logit | 0.6804 |
| service_points_won_rate_surface_diff | 0.3920 |
| first_serve_won_rate_365d_diff | 0.3491 |
| sets_14d_diff | -0.3487 |
| rank_diff | 0.3427 |
| return_points_won_rate_surface_diff | 0.3214 |
| return_points_won_rate_career_diff | -0.3203 |
| service_points_won_rate_365d_diff | -0.2961 |
| games_30d_diff | -0.2633 |
| sets_30d_diff | 0.2593 |
| first_serve_won_rate_90d_diff | 0.2565 |
| first_serve_won_rate_career_diff | -0.2315 |
| games_14d_diff | 0.2236 |
| games_7d_diff | -0.2089 |
| service_points_won_rate_career_diff | -0.1932 |
| service_points_won_rate_90d_diff | -0.1916 |
| second_serve_won_rate_365d_diff | 0.1773 |
| win_rate_365d_diff | -0.1754 |
| sets_7d_diff | 0.1553 |
| rank_points_diff | 0.1501 |

Ils décrivent des associations conditionnelles, pas des effets causaux. Les ratios service et
retour sont fortement liés; leur importance doit être lue avec l'ablation.

## J. Protections anti-leakage

- snapshot de toutes les features avant le batch de round, puis mise à jour;
- stats, score et durée du match courant visibles seulement au batch suivant;
- priors circuit calculés uniquement avec l'historique antérieur;
- surface et fatigue strictement antérieures;
- imputer, scaler et régression ajustés sur 2000–2018 seulement;
- shrinkage, blocs, régularisation et calibration choisis sur 2019–2022;
- aucune fonction de sélection ne reçoit le masque test; l'évaluation test se trouve après le gel complet.

La date exacte des matchs n'étant pas disponible, la charge intra-tournoi suit l'ordre des
rounds et ne prétend pas reconstruire des jours calendaires inexistants.

L'artefact `data/processed/models/logistic_phase4.joblib` contient l'imputer, le scaler, la
régression, le calibrateur et l'ordre exact des 50 features.

## Journal d'ouverture du test

La table finale figée a été générée le 6 octobre 2026 à 00:21:33 CEST, après sélection sur
validation. Une exécution d'ingénierie préliminaire avait affiché le test juste avant l'ajout de
deux ablations diagnostiques demandées (`ranking+serve+return` et `ranking+stats+form`). Ces
ajouts n'ont modifié ni le shrinkage, ni le modèle final, ni C, ni la calibration; aucun choix
n'a été effectué à partir de leurs résultats test. Le code maintient le masque test hors de
toutes les fonctions de sélection.
