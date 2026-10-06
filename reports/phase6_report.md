# Phase 6 — modèles non linéaires pré-match

## Statut du benchmark

La période 2023–2026 est un **historical benchmark déjà consulté**, pas un test aveugle. Aucun
résultat de cette période n'entre dans les features, hyperparamètres, early stopping, choix de
variante, calibration ou sélection du modèle. Un nouveau holdout prospectif reste nécessaire.

## Protocole walk-forward

Cinq folds valident successivement 2018, 2019, 2020, 2021 et 2022; chaque entraînement utilise
strictement les années antérieures. La sélection minimise la moyenne de log loss, puis Brier,
ROC-AUC et accuracy. Les calibrateurs sont eux-mêmes évalués en walk-forward de 2019 à 2022 sur
des prédictions out-of-fold. Les artefacts finaux sont entraînés jusqu'au 31 décembre 2022.

La recherche comprend 18 combinaisons algorithme/variante/configuration et
90 entraînements arbres. Les paramètres et chaque fold sont
conservés dans `phase6_tuning.csv`.

À protocole de calibration identique sur 2019–2022, la Logistic Phase 5 refit atteint une log
loss de 0.60021, contre
0.60048 pour le meilleur arbre sélectionné. Ce différentiel
mesure le gain validation de la non-linéarité sans consulter 2023–2026.

| calibration | matches | accuracy | log_loss | brier | roc_auc |
| --- | --- | --- | --- | --- | --- |
| raw | 9,766 | 0.66988 | 0.60056 | 0.20786 | 0.73514 |
| platt | 9,766 | 0.66598 | 0.60021 | 0.20754 | 0.73497 |
| isotonic | 9,766 | 0.66762 | 0.60092 | 0.20784 | 0.73341 |

## Sélection des hyperparamètres

| algorithm | variant | config_id | mean_log_loss | mean_brier | mean_roc_auc | mean_accuracy | median_best_iteration | total_fit_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| catboost | full | cat_depth6_regularized | 0.60140 | 0.20799 | 0.73506 | 0.66556 | 734.00000 | 117.71528 |
| xgboost | full | xgb_d4_regularized | 0.60214 | 0.20832 | 0.73406 | 0.66335 | 585.00000 | 31.66110 |
| lightgbm | full | lgb_leaves7 | 0.60215 | 0.20833 | 0.73398 | 0.66351 | 749.00000 | 37.85124 |
| catboost | no_fatigue_proxy | cat_depth6_regularized | 0.60247 | 0.20845 | 0.73385 | 0.66356 | 698.00000 | 101.03066 |
| xgboost | no_fatigue_proxy | xgb_d4_regularized | 0.60316 | 0.20882 | 0.73269 | 0.66318 | 407.00000 | 23.70443 |
| lightgbm | no_fatigue_proxy | lgb_leaves31_regularized | 0.60333 | 0.20881 | 0.73284 | 0.66395 | 411.00000 | 26.48281 |

## Dataset complet vs sans proxy de fatigue

La variante `no_fatigue_proxy` retire exactement les 13 variables Phase 4 basées sur
`matches/sets/games/minutes` à 7/14/30 jours et `rest_days`. Elle conserve la charge de tournois
antérieurs Phase 5, qui exclut l'édition courante.

| algorithm | variant | calibration | accuracy | log_loss | brier | roc_auc |
| --- | --- | --- | --- | --- | --- | --- |
| catboost | full | platt | 0.66414 | 0.60048 | 0.20748 | 0.73567 |
| catboost | no_fatigue_proxy | platt | 0.66301 | 0.60163 | 0.20797 | 0.73430 |
| lightgbm | full | platt | 0.66291 | 0.60124 | 0.20780 | 0.73452 |
| lightgbm | no_fatigue_proxy | platt | 0.66394 | 0.60240 | 0.20827 | 0.73332 |
| xgboost | full | platt | 0.66496 | 0.60096 | 0.20767 | 0.73494 |
| xgboost | no_fatigue_proxy | platt | 0.66312 | 0.60180 | 0.20808 | 0.73367 |

La version complète gagne en log loss validation pour les trois familles; la pénalité de retrait
est catboost: +0.00115, lightgbm: +0.00116, xgboost: +0.00084. Le signal n'est donc pas un accident propre à un seul moteur,
mais les importances montrent qu'Elo, ranking, expérience et statistiques ajustées dominent les
proxies de progression. La variante complète reste expérimentale, pas causale.

Le candidat global choisi avant ouverture du benchmark est
`catboost/full` avec calibration
`platt`. Le meilleur candidat explicitement calibré est
`catboost/full` avec
`platt`.

## Calibration validation

| algorithm | variant | calibration | matches | accuracy | log_loss | brier | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| catboost | full | platt | 9,766 | 0.66414 | 0.60048 | 0.20748 | 0.73567 |
| catboost | full | raw | 9,766 | 0.66568 | 0.60059 | 0.20766 | 0.73589 |
| catboost | full | isotonic | 9,766 | 0.66650 | 0.60498 | 0.20780 | 0.73359 |
| catboost | no_fatigue_proxy | platt | 9,766 | 0.66301 | 0.60163 | 0.20797 | 0.73430 |
| catboost | no_fatigue_proxy | raw | 9,766 | 0.66312 | 0.60182 | 0.20819 | 0.73455 |
| catboost | no_fatigue_proxy | isotonic | 9,766 | 0.66742 | 0.60317 | 0.20821 | 0.73221 |
| lightgbm | full | platt | 9,766 | 0.66291 | 0.60124 | 0.20780 | 0.73452 |
| lightgbm | full | raw | 9,766 | 0.66404 | 0.60138 | 0.20803 | 0.73475 |
| lightgbm | full | isotonic | 9,766 | 0.66701 | 0.60412 | 0.20807 | 0.73245 |
| lightgbm | no_fatigue_proxy | platt | 9,766 | 0.66394 | 0.60240 | 0.20827 | 0.73332 |
| lightgbm | no_fatigue_proxy | raw | 9,766 | 0.66353 | 0.60270 | 0.20855 | 0.73354 |
| lightgbm | no_fatigue_proxy | isotonic | 9,766 | 0.66414 | 0.60676 | 0.20885 | 0.73078 |
| xgboost | full | platt | 9,766 | 0.66496 | 0.60096 | 0.20767 | 0.73494 |
| xgboost | full | raw | 9,766 | 0.66588 | 0.60114 | 0.20789 | 0.73519 |
| xgboost | full | isotonic | 9,766 | 0.66486 | 0.60610 | 0.20819 | 0.73206 |
| xgboost | no_fatigue_proxy | platt | 9,766 | 0.66312 | 0.60180 | 0.20808 | 0.73367 |
| xgboost | no_fatigue_proxy | raw | 9,766 | 0.66383 | 0.60203 | 0.20833 | 0.73390 |
| xgboost | no_fatigue_proxy | isotonic | 9,766 | 0.66230 | 0.60646 | 0.20857 | 0.73148 |

La courbe rétrospective est enregistrée dans `phase6_calibration.png`; les buckets détaillés
sont dans `phase6_calibration.csv`.

## Historical benchmark 2023–2026

| model | matches | accuracy | log_loss | brier | roc_auc |
| --- | --- | --- | --- | --- | --- |
| ATP Ranking | 10,188 | 0.63869 | — | — | — |
| Elo Phase 3 | 10,376 | 0.63676 | 0.62533 | 0.21861 | 0.70074 |
| Logistic Phase 5 | 10,376 | 0.65738 | 0.60988 | 0.21158 | 0.72428 |
| CatBoost | 10,376 | 0.65960 | 0.60914 | 0.21130 | 0.72529 |
| XGBoost | 10,376 | 0.66018 | 0.60897 | 0.21117 | 0.72572 |
| LightGBM | 10,376 | 0.66008 | 0.60990 | 0.21158 | 0.72447 |
| Best calibrated tree | 10,376 | 0.65960 | 0.60914 | 0.21130 | 0.72529 |

- meilleure accuracy : **XGBoost**;
- meilleure log loss : **XGBoost**;
- meilleur Brier : **XGBoost**;
- meilleur ROC-AUC : **XGBoost**.

Comparaison des six candidats arbres avec leur calibration sélectionnée :

| algorithm | variant | probability | matches | accuracy | log_loss | brier | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| catboost | full | platt | 10,376 | 0.65960 | 0.60914 | 0.21130 | 0.72529 |
| xgboost | full | platt | 10,376 | 0.66018 | 0.60897 | 0.21117 | 0.72572 |
| lightgbm | full | platt | 10,376 | 0.66008 | 0.60990 | 0.21158 | 0.72447 |
| catboost | no_fatigue_proxy | platt | 10,376 | 0.66018 | 0.60980 | 0.21158 | 0.72458 |
| xgboost | no_fatigue_proxy | platt | 10,376 | 0.66066 | 0.61022 | 0.21175 | 0.72379 |
| lightgbm | no_fatigue_proxy | platt | 10,376 | 0.66172 | 0.60981 | 0.21153 | 0.72463 |

## Gain réel contre Logistic Phase 5

- accuracy : +0.222 point;
- log loss : -0.00073;
- Brier : -0.00028;
- ROC-AUC : +0.00101.

## Stabilité par surface et année

| segment_type | segment | model | matches | accuracy | log_loss | brier | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| surface | Hard | Logistic Phase 5 | 6,019 | 0.65875 | 0.60755 | 0.21077 | 0.72674 |
| surface | Hard | CatBoost | 6,019 | 0.66074 | 0.60741 | 0.21054 | 0.72794 |
| surface | Hard | XGBoost | 6,019 | 0.65908 | 0.60869 | 0.21102 | 0.72666 |
| surface | Hard | LightGBM | 6,019 | 0.66057 | 0.60916 | 0.21125 | 0.72594 |
| surface | Hard | Best calibrated tree | 6,019 | 0.66074 | 0.60741 | 0.21054 | 0.72794 |
| surface | Clay | Logistic Phase 5 | 3,356 | 0.65435 | 0.61316 | 0.21265 | 0.72069 |
| surface | Clay | CatBoost | 3,356 | 0.65256 | 0.61361 | 0.21332 | 0.71796 |
| surface | Clay | XGBoost | 3,356 | 0.65673 | 0.61126 | 0.21227 | 0.72111 |
| surface | Clay | LightGBM | 3,356 | 0.65405 | 0.61275 | 0.21289 | 0.71904 |
| surface | Clay | Best calibrated tree | 3,356 | 0.65256 | 0.61361 | 0.21332 | 0.71796 |
| surface | Grass | Logistic Phase 5 | 949 | 0.66596 | 0.60870 | 0.21131 | 0.72536 |
| surface | Grass | CatBoost | 949 | 0.67229 | 0.60221 | 0.20861 | 0.73344 |
| surface | Grass | XGBoost | 949 | 0.67334 | 0.60062 | 0.20791 | 0.73492 |
| surface | Grass | LightGBM | 949 | 0.67334 | 0.60172 | 0.20842 | 0.73310 |
| surface | Grass | Best calibrated tree | 949 | 0.67229 | 0.60221 | 0.20861 | 0.73344 |
| year | 2023 | Logistic Phase 5 | 2,964 | 0.66228 | 0.60695 | 0.21066 | 0.72764 |
| year | 2023 | CatBoost | 2,964 | 0.66296 | 0.61142 | 0.21214 | 0.72396 |
| year | 2023 | XGBoost | 2,964 | 0.66835 | 0.61093 | 0.21195 | 0.72432 |
| year | 2023 | LightGBM | 2,964 | 0.66296 | 0.61274 | 0.21275 | 0.72215 |
| year | 2023 | Best calibrated tree | 2,964 | 0.66296 | 0.61142 | 0.21214 | 0.72396 |
| year | 2024 | Logistic Phase 5 | 3,136 | 0.65561 | 0.60745 | 0.21050 | 0.72760 |
| year | 2024 | CatBoost | 3,136 | 0.65593 | 0.60277 | 0.20888 | 0.73142 |
| year | 2024 | XGBoost | 3,136 | 0.65529 | 0.60277 | 0.20875 | 0.73187 |
| year | 2024 | LightGBM | 3,136 | 0.65912 | 0.60293 | 0.20887 | 0.73140 |
| year | 2024 | Best calibrated tree | 3,136 | 0.65593 | 0.60277 | 0.20888 | 0.73142 |
| year | 2025 | Logistic Phase 5 | 2,839 | 0.65551 | 0.61845 | 0.21498 | 0.71499 |
| year | 2025 | CatBoost | 2,839 | 0.65340 | 0.61796 | 0.21501 | 0.71491 |
| year | 2025 | XGBoost | 2,839 | 0.65657 | 0.61736 | 0.21461 | 0.71639 |
| year | 2025 | LightGBM | 2,839 | 0.65622 | 0.61809 | 0.21490 | 0.71565 |
| year | 2025 | Best calibrated tree | 2,839 | 0.65340 | 0.61796 | 0.21501 | 0.71491 |
| year | 2026 | Logistic Phase 5 | 1,437 | 0.65484 | 0.60427 | 0.20917 | 0.73099 |
| year | 2026 | CatBoost | 1,437 | 0.67293 | 0.60096 | 0.20753 | 0.73669 |
| year | 2026 | XGBoost | 1,437 | 0.66110 | 0.60186 | 0.20806 | 0.73476 |
| year | 2026 | LightGBM | 1,437 | 0.66388 | 0.60308 | 0.20851 | 0.73342 |
| year | 2026 | Best calibrated tree | 1,437 | 0.67293 | 0.60096 | 0.20753 | 0.73669 |

Grass reste le segment le plus incertain en raison de son faible volume. Les résultats ne sont
jamais utilisés pour modifier le modèle gelé.

## Niveau de tournoi

Les codes source distinguent fiablement `G` (Grand Slam) et `M` (Masters); tous les autres
niveaux sont regroupés sans inventer ATP 250/500.

| segment | model | matches | accuracy | log_loss | brier | roc_auc |
| --- | --- | --- | --- | --- | --- | --- |
| Grand Slam | Logistic Phase 5 | 1,767 | 0.74024 | 0.53185 | 0.17858 | 0.81005 |
| Grand Slam | CatBoost | 1,767 | 0.72835 | 0.53940 | 0.18121 | 0.80442 |
| Grand Slam | XGBoost | 1,767 | 0.72892 | 0.53711 | 0.18017 | 0.80727 |
| Grand Slam | LightGBM | 1,767 | 0.72949 | 0.53867 | 0.18088 | 0.80546 |
| Grand Slam | Best calibrated tree | 1,767 | 0.72835 | 0.53940 | 0.18121 | 0.80442 |
| Masters | Logistic Phase 5 | 2,580 | 0.64729 | 0.62703 | 0.21890 | 0.70380 |
| Masters | CatBoost | 2,580 | 0.64651 | 0.63055 | 0.22025 | 0.70080 |
| Masters | XGBoost | 2,580 | 0.64612 | 0.63000 | 0.22004 | 0.70122 |
| Masters | LightGBM | 2,580 | 0.64302 | 0.63092 | 0.22042 | 0.69975 |
| Masters | Best calibrated tree | 2,580 | 0.64651 | 0.63055 | 0.22025 | 0.70080 |
| Other | Logistic Phase 5 | 6,029 | 0.63742 | 0.62540 | 0.21813 | 0.70457 |
| Other | CatBoost | 6,029 | 0.64505 | 0.62042 | 0.21629 | 0.70997 |
| Other | XGBoost | 6,029 | 0.64604 | 0.62102 | 0.21646 | 0.70955 |
| Other | LightGBM | 6,029 | 0.64704 | 0.62178 | 0.21679 | 0.70855 |
| Other | Best calibrated tree | 6,029 | 0.64505 | 0.62042 | 0.21629 | 0.70997 |

## Confiance

| model | bin | matches | mean_probability | actual_win_rate |
| --- | --- | --- | --- | --- |
| Logistic Phase 5 | 50-55% | 1,746 | 0.52483 | 0.51145 |
| Logistic Phase 5 | 55-60% | 1,680 | 0.57496 | 0.55238 |
| Logistic Phase 5 | 60-70% | 2,985 | 0.64817 | 0.63551 |
| Logistic Phase 5 | 70-80% | 2,219 | 0.74712 | 0.72375 |
| Logistic Phase 5 | 80-90% | 1,300 | 0.84328 | 0.83462 |
| Logistic Phase 5 | 90-100% | 446 | 0.93342 | 0.92377 |
| Best tree calibrated | 50-55% | 1,637 | 0.52519 | 0.51680 |
| Best tree calibrated | 55-60% | 1,642 | 0.57497 | 0.56334 |
| Best tree calibrated | 60-70% | 2,962 | 0.64788 | 0.62492 |
| Best tree calibrated | 70-80% | 2,423 | 0.74708 | 0.72142 |
| Best tree calibrated | 80-90% | 1,336 | 0.84270 | 0.84057 |
| Best tree calibrated | 90-100% | 376 | 0.92603 | 0.93351 |

## Importance et SHAP

Les importances natives (gain pour XGBoost/LightGBM, `LossFunctionChange` pour CatBoost) et les
permutations sont calculées sur 2022 avec un modèle diagnostique entraîné jusqu'en 2021.

| algorithm | feature | native_gain | permutation_log_loss_increase | permutation_std |
| --- | --- | --- | --- | --- |
| catboost | elo_logit | 0.36817 | 0.03446 | 0.00333 |
| catboost | rank_diff | 0.21880 | 0.01202 | 0.00138 |
| catboost | surface_matches_diff | 0.09830 | 0.00571 | 0.00128 |
| catboost | age_years_diff | 0.09812 | 0.00566 | 0.00015 |
| catboost | return_above_expected_surface_diff | 0.07851 | 0.00436 | 0.00073 |
| catboost | rank_points_diff | 0.02090 | 0.00217 | 0.00048 |
| catboost | serve_above_expected_90d_diff | 0.03425 | 0.00197 | 0.00051 |
| catboost | serve_above_expected_surface_diff | 0.02980 | 0.00154 | 0.00041 |
| catboost | return_above_expected_90d_diff | 0.01141 | 0.00142 | 0.00086 |
| catboost | opponent_elo_mean_365d_diff | 0.02848 | 0.00120 | 0.00056 |
| catboost | career_matches_diff | 0.00327 | 0.00113 | 0.00022 |
| catboost | return_points_won_rate_90d_diff | 0.00853 | 0.00098 | 0.00036 |
| lightgbm | elo_logit | 0.50371 | 0.04844 | 0.00332 |
| lightgbm | rank_diff | 0.15464 | 0.01279 | 0.00149 |
| lightgbm | age_years_diff | 0.01748 | 0.00538 | 0.00032 |
| lightgbm | return_above_expected_surface_diff | 0.01333 | 0.00485 | 0.00070 |
| lightgbm | surface_matches_diff | 0.04905 | 0.00452 | 0.00109 |
| lightgbm | serve_above_expected_90d_diff | 0.01843 | 0.00268 | 0.00065 |
| lightgbm | return_above_expected_90d_diff | 0.00939 | 0.00220 | 0.00076 |
| lightgbm | serve_above_expected_surface_diff | 0.01616 | 0.00165 | 0.00018 |
| lightgbm | rank_points_diff | 0.04245 | 0.00105 | 0.00024 |
| lightgbm | opponent_elo_mean_365d_diff | 0.01620 | 0.00091 | 0.00037 |
| lightgbm | career_matches_diff | 0.00413 | 0.00066 | 0.00019 |
| lightgbm | rest_days_diff | 0.01254 | 0.00061 | 0.00047 |
| xgboost | elo_logit | 0.22516 | 0.04394 | 0.00326 |
| xgboost | rank_diff | 0.03757 | 0.01125 | 0.00141 |
| xgboost | age_years_diff | 0.01146 | 0.00529 | 0.00015 |
| xgboost | surface_matches_diff | 0.02770 | 0.00506 | 0.00106 |
| xgboost | return_above_expected_surface_diff | 0.01036 | 0.00457 | 0.00073 |
| xgboost | serve_above_expected_surface_diff | 0.01399 | 0.00232 | 0.00022 |
| xgboost | return_above_expected_90d_diff | 0.01091 | 0.00220 | 0.00087 |
| xgboost | serve_above_expected_90d_diff | 0.01516 | 0.00204 | 0.00067 |
| xgboost | rank_points_diff | 0.06997 | 0.00105 | 0.00038 |
| xgboost | return_points_won_rate_90d_diff | 0.00651 | 0.00096 | 0.00021 |
| xgboost | career_matches_diff | 0.00703 | 0.00094 | 0.00028 |
| xgboost | opponent_elo_mean_365d_diff | 0.01861 | 0.00087 | 0.00033 |

Top SHAP du meilleur arbre calibré, en valeur absolue moyenne sur un échantillon déterministe du
benchmark :

| feature | mean_abs_shap | mean_shap | std_shap |
| --- | --- | --- | --- |
| elo_logit | 0.29813 | 0.01910 | 0.38675 |
| surface_matches_diff | 0.17978 | 0.02965 | 0.19455 |
| rank_diff | 0.13085 | -0.02031 | 0.25470 |
| age_years_diff | 0.10595 | -0.01349 | 0.14782 |
| return_above_expected_surface_diff | 0.07761 | 0.00415 | 0.09730 |
| opponent_elo_mean_365d_diff | 0.07059 | 0.00207 | 0.07822 |
| serve_above_expected_surface_diff | 0.06680 | 0.00026 | 0.08665 |
| serve_above_expected_90d_diff | 0.05971 | 0.00271 | 0.08436 |
| return_above_expected_90d_diff | 0.05260 | 0.00541 | 0.05655 |
| rank_points_diff | 0.04904 | 0.00555 | 0.06995 |
| prior_tournament_games_30d_diff | 0.04286 | 0.00298 | 0.04637 |
| rest_days_diff | 0.04243 | 0.00079 | 0.07698 |
| opponent_rank_quality_mean_365d_diff | 0.03854 | 0.00017 | 0.04819 |
| return_points_won_rate_90d_diff | 0.02882 | -0.00143 | 0.03657 |
| service_points_won_rate_90d_diff | 0.02808 | 0.00211 | 0.03440 |
| win_rate_365d_diff | 0.02706 | -0.00046 | 0.03022 |
| opponent_elo_mean_90d_diff | 0.02680 | 0.00272 | 0.03241 |
| service_points_won_rate_surface_diff | 0.02423 | -0.00017 | 0.03039 |
| serve_above_expected_365d_diff | 0.02118 | 0.00168 | 0.03544 |
| matches_30d_diff | 0.02078 | 0.00143 | 0.02358 |

SHAP explique le modèle en unités de sortie interne/log-odds et non en causalité ni en points de
probabilité additifs. Les exemples individuels sont dans `phase6_shap_examples.csv`.

## Interactions apprises

| feature_1 | feature_2 | mean_abs_interaction |
| --- | --- | --- |
| elo_logit | rank_diff | 0.02041 |
| elo_logit | surface_matches_diff | 0.01954 |
| elo_logit | opponent_elo_mean_365d_diff | 0.01208 |
| elo_logit | prior_tournament_games_30d_diff | 0.01016 |
| elo_logit | rank_points_diff | 0.00872 |
| rank_diff | rank_points_diff | 0.00785 |
| career_matches_diff | surface_matches_diff | 0.00757 |
| surface_matches_diff | h2h_matches_before | 0.00748 |
| rank_diff | surface_matches_diff | 0.00688 |
| elo_logit | serve_above_expected_surface_diff | 0.00622 |
| elo_logit | serve_above_expected_90d_diff | 0.00595 |
| rank_diff | career_matches_diff | 0.00471 |
| rank_diff | rest_days_diff | 0.00446 |
| elo_logit | second_serve_won_rate_365d_diff | 0.00428 |
| rank_diff | prior_tournament_games_30d_diff | 0.00387 |
| elo_logit | win_rate_90d_diff | 0.00382 |
| rank_diff | opponent_elo_mean_365d_diff | 0.00378 |
| elo_logit | win_rate_365d_diff | 0.00346 |
| elo_logit | return_above_expected_surface_diff | 0.00322 |
| surface_matches_diff | age_years_diff | 0.00321 |

## Désaccord Logistic vs arbre

| matches | share_of_benchmark | logistic_accuracy | tree_accuracy | logistic_mean_confidence | tree_mean_confidence |
| --- | --- | --- | --- | --- | --- |
| 843 | 0.08125 | 0.48636 | 0.51364 | 0.53854 | 0.54727 |

Les features qui dominent les désaccords sont détaillées dans `phase6_disagreement_shap.csv`.

## Erreurs confiantes

238 erreurs dépassent 80 % de confiance. L'audit complet est dans
`phase6_confident_errors.md`. Les données ne permettent pas d'identifier fiablement les
blessures ou retours de blessure.

## Complexité et recommandation

Au regard de la performance probabiliste, de la stabilité, de la calibration, de
l'interprétabilité et du coût opérationnel, la recommandation est de **conserver Logistic Phase 5 comme moteur PRE-MATCH V1**.
La sélection technique de l'arbre reste fondée uniquement sur 2018–2022; le benchmark sert à
quantifier rétrospectivement sa stabilité. Aucun réseau neuronal, Transformer, modèle live ou
point-by-point n'est commencé.
