# Tournament Context Adapter V1 — rapport final

## Conclusion exécutive

**Recommandation : NE PAS ACTIVER EN PRODUCTION — conserver PRE-MATCH V1.**

L'adapter conservateur avec cap à 5 points améliore très légèrement la log loss agrégée
sur le test 2023–2026 (`0.609876 → 0.609692`), mais dégrade le Brier
(`0.211585 → 0.211623`) et surtout ne généralise pas de façon stable entre éditions : le
gain moyen par tournoi vaut `-0.001402`, avec IC 95 %
`[-0.002531, -0.000274]`. Cet intervalle est entièrement
négatif. La V1 répond donc à la question scientifique : un signal existe dans certains
segments, mais il n'est pas assez stable pour dépasser PRE-MATCH V1 en production.

## A. Dataset

- `78,845` matchs, `3,757` éditions, 2000–2026 ;
- `2,077` clés de tournois historiques normalisées ;
- couverture minutes `42.1%`, stats détaillées `43.2%` ;
- `0` violation du gel avant round ;
- test final entièrement hors échantillon : 2023–2026, `10,376` matchs.

La chronologie est structurelle et tous les matchs d'un round utilisent le même snapshot.
PRE-MATCH V1 est figé ; son empreinte et celle des sources sont dans
`models/tournament_adapter_v1/prematch_v1_freeze.json`.

## B. Player tournament state

L'état contient charge, scores, durée avec couverture, taux bruts et résidus par rapport à
l'attendu. Le shrinkage joueur choisi en validation est `C=75`.
Les plus gros coefficients standardisés sont descriptifs, pas causaux :

| feature | standardized_logit_contribution |
| --- | --- |
| best_of_5_x_logit_p_base | 0.093015 |
| tournament_speed_proxy_c20_x_style_speed_diff | 0.073058 |
| same_tournament_history_matches_diff | 0.063289 |
| level_grand_slam_x_logit_p_base | 0.052013 |
| break_over_expected_c75_diff | 0.050812 |
| return_points_won_rate_tournament_diff | 0.046891 |
| tournament_workload_environment_c20_x_games_played_diff | -0.046646 |
| same_tournament_wins_above_expected_c20_diff | 0.039030 |
| break_rate_tournament_diff | -0.036903 |
| retirements_diff | -0.035634 |
| sets_lost_diff | 0.034150 |
| surface_grass_x_logit_p_base | -0.032303 |
| matches_played_diff | -0.028862 |
| serve_over_expected_c75_diff | 0.028609 |
| round_R32_x_logit_p_base | -0.027884 |

## C. Tournament global state

Le bloc global seul (`T3`) n'améliore pas T0 en validation. Les conditions globales sont
orientées uniquement via des interactions de style ou de confiance, afin de ne jamais
favoriser arbitrairement A. Le shrinkage tournoi sélectionné est
`C=20`. Le proxy de vitesse n'est pas une mesure
physique de court speed.

## D. Performance vs expected

`T2_PERFORMANCE_VS_EXPECTED` (`0.599387`) est meilleur que les taux bruts seuls
`T2B_RAW_TOURNAMENT_STATS` (`0.599535`) en log loss validation. Les combiner n'améliore pas
le bloc résiduel seul. Le signal « au-delà de l'attendu » est donc préférable, mais son gain
reste exploratoire.

## E. Fatigue

Le workload seul ne bat pas T0. Ajouter les minutes apporte un gain validation minuscule
dans le modèle combiné (`T6B 0.599103` contre `T6 0.599125`), avec seulement 42 % de
couverture. La variante avec minutes est conservée pour la recherche, jamais interprétée
comme une mesure complète de fatigue.

## F. Ablation walk-forward 2019–2022

Chaque année est prédite par un modèle entraîné uniquement sur les années précédentes ; une
édition entière reste dans un seul split.

| model | features | matches | accuracy | log_loss | brier | gain_vs_T0 |
| --- | --- | --- | --- | --- | --- | --- |
| T6B_ALL_WITH_MINUTES | 45 | 9766 | 0.667213 | 0.599103 | 0.207337 | 0.000587 |
| T6_ALL_NO_MINUTES | 43 | 9766 | 0.667110 | 0.599125 | 0.207347 | 0.000566 |
| T2_PERFORMANCE_VS_EXPECTED | 6 | 9766 | 0.666803 | 0.599387 | 0.207295 | 0.000304 |
| T2C_RAW_PLUS_EXPECTED | 11 | 9766 | 0.667110 | 0.599529 | 0.207354 | 0.000162 |
| T2B_RAW_TOURNAMENT_STATS | 5 | 9766 | 0.667520 | 0.599535 | 0.207365 | 0.000156 |
| T0_PREMATCH_V1 | 0 | 9766 | 0.667725 | 0.599691 | 0.207379 | 0.000000 |
| T1B_WORKLOAD_WITH_MINUTES | 10 | 9766 | 0.667110 | 0.599715 | 0.207385 | -0.000024 |
| T5_SAME_TOURNAMENT_HISTORY | 2 | 9766 | 0.665472 | 0.599738 | 0.207531 | -0.000047 |
| T4_ROUND | 9 | 9766 | 0.667725 | 0.599771 | 0.207444 | -0.000081 |
| T1_WORKLOAD | 8 | 9766 | 0.667622 | 0.599785 | 0.207415 | -0.000095 |
| T3_TOURNAMENT_ENVIRONMENT | 8 | 9766 | 0.668032 | 0.599806 | 0.207428 | -0.000116 |

La régularisation sélectionnée est `100.0`. Le
meilleur modèle validation est combiné avec minutes, mais son avantage ne se confirme pas
de façon stable sur le test final.

## G. Performance par round — test 2023–2026

Un gain positif signifie une log loss plus faible que PRE-MATCH V1.

| segment | matches | base_log_loss | adapter_log_loss | gain_log_loss | base_brier | adapter_brier |
| --- | --- | --- | --- | --- | --- | --- |
| RR | 1042 | 0.643638 | 0.646889 | -0.003251 | 0.224921 | 0.225805 |
| R128 | 1568 | 0.611614 | 0.611307 | 0.000307 | 0.213891 | 0.214297 |
| R64 | 1531 | 0.599223 | 0.598367 | 0.000856 | 0.206986 | 0.206378 |
| R32 | 3002 | 0.604352 | 0.604353 | -0.000001 | 0.208974 | 0.209149 |
| R16 | 1698 | 0.603708 | 0.603692 | 0.000016 | 0.209145 | 0.209312 |
| QF | 859 | 0.613628 | 0.610004 | 0.003623 | 0.212237 | 0.211071 |
| SF | 449 | 0.621939 | 0.623013 | -0.001074 | 0.217113 | 0.217726 |
| F | 226 | 0.595647 | 0.591458 | 0.004188 | 0.204737 | 0.203123 |
| BR | 1 | 0.644233 | 0.744304 | -0.100070 | 0.225563 | 0.275557 |

QF et finale sont favorables, SF défavorable. Cette alternance et les petits effectifs des
rounds avancés ne permettent pas de valider l'hypothèse d'un gain croissant régulier.

### Selon le nombre de matchs déjà observés

| segment | matches | base_log_loss | adapter_log_loss | gain_log_loss | base_brier | adapter_brier |
| --- | --- | --- | --- | --- | --- | --- |
| 0-8 | 5166 | 0.623313 | 0.624255 | -0.000942 | 0.217736 | 0.218204 |
| 9-20 | 1857 | 0.628051 | 0.627778 | 0.000273 | 0.219419 | 0.219249 |
| 21-40 | 1714 | 0.612280 | 0.610473 | 0.001807 | 0.212310 | 0.211716 |
| 40+ | 1639 | 0.544415 | 0.542483 | 0.001932 | 0.182562 | 0.182143 |

Le bucket `0–8` se dégrade ; les buckets plus informés s'améliorent, mais cela ne suffit pas
à rendre le gain stable par tournoi.

## H. Surface, niveau et format

### Surface

| segment | matches | base_log_loss | adapter_log_loss | gain_log_loss | base_brier | adapter_brier |
| --- | --- | --- | --- | --- | --- | --- |
| Hard | 6019 | 0.607548 | 0.607431 | 0.000117 | 0.210771 | 0.210890 |
| Clay | 3356 | 0.613157 | 0.614284 | -0.001126 | 0.212653 | 0.213094 |
| Grass | 949 | 0.608702 | 0.603191 | 0.005511 | 0.211311 | 0.209366 |

### Niveau

| segment | matches | base_log_loss | adapter_log_loss | gain_log_loss | base_brier | adapter_brier |
| --- | --- | --- | --- | --- | --- | --- |
| Other | 6029 | 0.625400 | 0.626168 | -0.000769 | 0.218130 | 0.218358 |
| Grand Slam | 1767 | 0.531854 | 0.528756 | 0.003098 | 0.178575 | 0.178018 |
| Masters | 2580 | 0.627035 | 0.626622 | 0.000413 | 0.218898 | 0.218900 |

### BO3 / BO5

| segment | matches | base_log_loss | adapter_log_loss | gain_log_loss | base_brier | adapter_brier |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | 8568 | 0.625476 | 0.625885 | -0.000408 | 0.218174 | 0.218337 |
| 5 | 1808 | 0.535944 | 0.532955 | 0.002989 | 0.180360 | 0.179806 |

Le gazon et les Grands Chelems sont favorables sur ce test, la terre battue est défavorable.
Ces découpages sont des diagnostics, pas de nouvelles sélections de modèle.

## I. Corrections et erreurs

- correction absolue moyenne : `1.877` point de probabilité ;
- correction absolue maximale : `5.000` points ;
- cap sélectionné sur validation : `probability_0.05`.

Résultats des caps en validation :

| cap | matches | accuracy | log_loss | brier | auc |
| --- | --- | --- | --- | --- | --- |
| probability_0.05 | 9766 | 0.666189 | 0.598837 | 0.207229 | 0.735779 |
| none | 9766 | 0.667213 | 0.599103 | 0.207337 | 0.735684 |
| probability_0.10 | 9766 | 0.667213 | 0.599161 | 0.207365 | 0.735591 |
| logit_0.5 | 9766 | 0.667213 | 0.599254 | 0.207360 | 0.735651 |

| correction_bucket | matches | mean_abs_delta | base_log_loss | adjusted_log_loss |
| --- | --- | --- | --- | --- |
| 0-5pp | 9811 | 0.016969 | 0.612618 | 0.612566 |
| 5-10pp | 565 | 0.050000 | 0.562248 | 0.559781 |

Les 200 plus fortes corrections, avec résultat réel, sont conservées dans
`reports/tournament_adapter_largest_corrections.csv`.

## J. Calibration

| model | matches | accuracy | log_loss | brier | auc |
| --- | --- | --- | --- | --- | --- |
| raw | 5611 | 0.666370 | 0.594758 | 0.205591 | 0.738802 |
| temperature | 5611 | 0.666370 | 0.595430 | 0.205732 | 0.738802 |

La calibration température symétrique n'est pas retenue (`uses_temperature_calibration =
False`). Le contrôle PRE-MATCH recalibré n'est pas
retenu non plus ; l'amélioration/dégradation observée vient donc bien de l'adapter et du cap,
pas d'un simple intercept de recalibration.

### Comparaison finale

| model | matches | accuracy | log_loss | brier | auc |
| --- | --- | --- | --- | --- | --- |
| PREMATCH_V1 | 10376 | 0.657382 | 0.609876 | 0.211585 | 0.724285 |
| PREMATCH_V1_TEMPERATURE_CONTROL | 10376 | 0.657382 | 0.609876 | 0.211585 | 0.724285 |
| OFFSET_RAW | 10376 | 0.657864 | 0.609862 | 0.211726 | 0.723786 |
| OFFSET_FINAL | 10376 | 0.657672 | 0.609692 | 0.211623 | 0.724068 |
| FULL_LOGISTIC | 10376 | 0.656901 | 0.609934 | 0.211683 | 0.724296 |

L'offset respecte exactement `logit(p_final) = logit(p_base) + correction_context`. La
régression logistique complète, autorisée à réapprendre le socle, n'est pas meilleure.

## K. Replay et simulation

Exemple de chemin historique complet : **Us Open 2025 (atp:2025-560:2025-08-25)**.

| round | matches | observed_before | base_log_loss | adapter_log_loss | mean_abs_correction_pp |
| --- | --- | --- | --- | --- | --- |
| R128 | 64 | 0 | 0.549686 | 0.546314 | 3.354869 |
| R64 | 31 | 64 | 0.673980 | 0.712162 | 3.678181 |
| R32 | 16 | 95 | 0.576596 | 0.603076 | 3.749334 |
| R16 | 8 | 111 | 0.279570 | 0.257057 | 3.482975 |
| QF | 4 | 119 | 0.431286 | 0.410722 | 4.346087 |
| SF | 2 | 123 | 0.221729 | 0.236929 | 4.562506 |
| F | 1 | 125 | 0.836012 | 0.877952 | 1.780224 |

Ce tableau rejoue le chemin réellement observé round par round. L'API
`simulate_tournament` permet en plus des Monte Carlo reproductibles : mode pré-tournoi avec
PRE-MATCH V1 seul, ou mode live avec snapshot réel gelé. Elle ne fabrique jamais les stats
des tours futurs. Un ancien bracket contrefactuel complet demanderait des features
point-in-time pour des confrontations jamais jouées ; V1 refuse de les inventer.

## L. Recommandation finale

**Conserver PRE-MATCH V1 seul en production.** Garder Tournament Adapter V1 comme artefact
de recherche reproductible et comme API expérimentale désactivée. Les critères de succès ne
sont pas tous satisfaits : log loss globale légèrement meilleure, mais Brier moins bon et IC
clusterisé par édition négatif. Une V2 ne devrait être tentée qu'avec davantage de couverture
statistique, une validation pré-enregistrée et des interactions de conditions mieux motivées.

La prochaine phase logique, sans activer cet adapter, est **PRODUCT / WEB PLATFORM V1** en
s'appuyant sur les moteurs déjà validés. Aucun travail web n'est commencé ici.
