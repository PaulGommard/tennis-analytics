# Phase 5 — qualité des adversaires et matchups

## Protocole

Phase 4 est figée par empreintes SHA-256 dans `phase5_phase4_freeze.json`. Toutes les sélections
Phase 5 utilisent 2019–2022; les priors, blocs, C et calibration sont gelés avant l'évaluation
finale 2023–2026.

## A. Strength of schedule

Les moyennes adverses Elo et `-rank` sur 30/90/365 jours sont utiles : log loss validation
0.60314 → 0.60232. Elles sont
conservées. Une moyenne historique simple a été préférée à une pondération supplémentaire pour
limiter les degrés de liberté; les fenêtres portent déjà la récence.

## B. Adjusted form

`wins_above_expected`, calculé avec les probabilités Elo brutes pré-match, n'améliore que de
façon négligeable la log loss (0.60314 →
0.60312). Ce bloc est supprimé de V3 production.

## C. Opponent-adjusted stats

C'est le meilleur bloc individuel : 0.60314 →
0.60202. Pour le service, l'attendu est
`1 − taux de retour historique adverse`; pour le retour, `1 − taux de service historique
adverse`. Cette identité respecte les bornes et évite le double comptage qu'introduirait une
formule `moyenne ligue + force serveur − force retourneur`. Les résidus sont pondérés par les
opportunités, avec priors sélectionnés de 500 points et 100 points sur
surface.

## D. H2H

Le H2H global régularisé apporte un gain très faible (0.60314 →
0.60305) avec C=5; il est conservé dans la combinaison.
La variante surface est moins bonne (0.60309) et supprimée.

## E. Matchup

Le proxy service récent A contre retour récent B ne gagne rien
(0.60316, contre 0.60314). Il est supprimé.

## F. Surface interactions

Les interactions Elo, service, retour et résidus ajustés × Clay/Grass améliorent la validation
(0.60314 → 0.60245). Elles sont
conservées, Hard servant de référence. Les coefficients sont des écarts conditionnels et ne
doivent pas être lus isolément comme effets causaux.

## G. Audit fatigue

Les features Phase 4 sont marquées expérimentales car les rounds d'un tournoi partagent la
même date de début. Leur retrait dans une comparaison équitable donne
0.60273, contre
0.60078 avec elles. Elles restent donc comme
proxies prédictifs expérimentaux, sans interprétation causale de fatigue. La charge de tournois
antérieurs, qui exclut l'édition courante, améliore seule à
0.60261. Voir `fatigue_validity_audit.md`.

## H. Ablation study

Le bloc final choisi sur validation est `MODEL_H_REDUCED_ROBUST_FATIGUE`. L'ablation, et non les coefficients
corrélés pris isolément, constitue la preuve principale.

| model | features | c_value | validation_accuracy | validation_log_loss | validation_brier | validation_roc_auc |
| --- | --- | --- | --- | --- | --- | --- |
| MODEL_H_REDUCED_ROBUST_FATIGUE | 86 | 1.0000 | 0.6701 | 0.6008 | 0.2080 | 0.7352 |
| MODEL_H_CORE_ROBUST_FATIGUE | 84 | 100.0000 | 0.6676 | 0.6008 | 0.2080 | 0.7351 |
| MODEL_H_REDUCED | 78 | 0.0100 | 0.6673 | 0.6014 | 0.2082 | 0.7344 |
| MODEL_H_CORE | 76 | 0.0100 | 0.6671 | 0.6014 | 0.2082 | 0.7344 |
| MODEL_H_ALL | 94 | 1.0000 | 0.6678 | 0.6018 | 0.2084 | 0.7349 |
| MODEL_D_OPPONENT_ADJUSTED_STATS | 58 | 0.0100 | 0.6669 | 0.6020 | 0.2085 | 0.7336 |
| MODEL_B_STRENGTH_OF_SCHEDULE | 56 | 1.0000 | 0.6650 | 0.6023 | 0.2087 | 0.7329 |
| MODEL_G_SURFACE_INTERACTIONS | 62 | 0.1000 | 0.6663 | 0.6025 | 0.2087 | 0.7330 |
| MODEL_ROBUST_FATIGUE | 58 | 10.0000 | 0.6657 | 0.6026 | 0.2088 | 0.7327 |
| MODEL_H_CLEAN_REDUCED_ROBUST_FATIGUE | 73 | 0.0100 | 0.6651 | 0.6027 | 0.2088 | 0.7329 |
| MODEL_F_H2H | 52 | 0.1000 | 0.6655 | 0.6031 | 0.2090 | 0.7322 |
| MODEL_F2_H2H_SURFACE | 54 | 0.1000 | 0.6659 | 0.6031 | 0.2090 | 0.7321 |
| MODEL_C_ADJUSTED_FORM | 53 | 0.1000 | 0.6665 | 0.6031 | 0.2090 | 0.7322 |
| MODEL_A_PHASE4_EXACT | 50 | 1.0000 | 0.6659 | 0.6031 | 0.2090 | 0.7321 |
| MODEL_E_MATCHUP | 52 | 0.1000 | 0.6654 | 0.6032 | 0.2090 | 0.7321 |
| MODEL_H_CLEAN_FATIGUE | 81 | 0.0100 | 0.6641 | 0.6038 | 0.2092 | 0.7325 |
| MODEL_EXPERIENCE | 51 | 0.1000 | 0.6657 | 0.6042 | 0.2095 | 0.7315 |

## I. Modèle final Phase 5

| model | matches | accuracy | log_loss | brier | roc_auc |
| --- | --- | --- | --- | --- | --- |
| ATP Ranking | 10,188 | 0.6387 | — | — | — |
| Elo Phase 3 calibrated | 10,376 | 0.6368 | 0.6253 | 0.2186 | 0.7007 |
| Logistic Phase 4 calibrated | 10,376 | 0.6537 | 0.6130 | 0.2129 | 0.7206 |
| Logistic Phase 5 raw | 10,376 | 0.6586 | 0.6113 | 0.2120 | 0.7243 |
| Logistic Phase 5 final | 10,376 | 0.6574 | 0.6099 | 0.2116 | 0.7243 |

### Par surface et année

| segment_type | segment | model | matches | accuracy | log_loss | brier | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| surface | Hard | Elo Phase 3 | 6,019 | 0.6396 | 0.6221 | 0.2172 | 0.7056 |
| surface | Hard | Logistic Phase 4 | 6,019 | 0.6556 | 0.6106 | 0.2120 | 0.7235 |
| surface | Hard | Logistic Phase 5 | 6,019 | 0.6587 | 0.6075 | 0.2108 | 0.7267 |
| surface | Clay | Elo Phase 3 | 3,356 | 0.6284 | 0.6329 | 0.2218 | 0.6887 |
| surface | Clay | Logistic Phase 4 | 3,356 | 0.6532 | 0.6156 | 0.2138 | 0.7169 |
| surface | Clay | Logistic Phase 5 | 3,356 | 0.6544 | 0.6132 | 0.2127 | 0.7207 |
| surface | Grass | Elo Phase 3 | 949 | 0.6544 | 0.6149 | 0.2142 | 0.7192 |
| surface | Grass | Logistic Phase 4 | 949 | 0.6512 | 0.6153 | 0.2141 | 0.7213 |
| surface | Grass | Logistic Phase 5 | 949 | 0.6660 | 0.6087 | 0.2113 | 0.7254 |
| year | 2023 | Elo Phase 3 | 2,964 | 0.6350 | 0.6283 | 0.2197 | 0.6986 |
| year | 2023 | Logistic Phase 4 | 2,964 | 0.6538 | 0.6119 | 0.2129 | 0.7211 |
| year | 2023 | Logistic Phase 5 | 2,964 | 0.6623 | 0.6069 | 0.2107 | 0.7276 |
| year | 2024 | Elo Phase 3 | 3,136 | 0.6368 | 0.6210 | 0.2171 | 0.7040 |
| year | 2024 | Logistic Phase 4 | 3,136 | 0.6527 | 0.6097 | 0.2115 | 0.7246 |
| year | 2024 | Logistic Phase 5 | 3,136 | 0.6556 | 0.6074 | 0.2105 | 0.7276 |
| year | 2025 | Elo Phase 3 | 2,839 | 0.6379 | 0.6287 | 0.2200 | 0.6955 |
| year | 2025 | Logistic Phase 4 | 2,839 | 0.6492 | 0.6211 | 0.2161 | 0.7120 |
| year | 2025 | Logistic Phase 5 | 2,839 | 0.6555 | 0.6184 | 0.2150 | 0.7150 |
| year | 2026 | Elo Phase 3 | 1,437 | 0.6381 | 0.6220 | 0.2168 | 0.7079 |
| year | 2026 | Logistic Phase 4 | 1,437 | 0.6646 | 0.6063 | 0.2098 | 0.7303 |
| year | 2026 | Logistic Phase 5 | 1,437 | 0.6548 | 0.6043 | 0.2092 | 0.7310 |

## Calibration

Calibration Platt retenue : True.

| model | matches | accuracy | log_loss | brier | roc_auc |
| --- | --- | --- | --- | --- | --- |
| Phase 5 raw | 9,766 | 0.6701 | 0.6008 | 0.2080 | 0.7352 |
| Phase 5 Platt | 9,766 | 0.6677 | 0.5997 | 0.2074 | 0.7352 |

Buckets test : `phase5_calibration.csv`.

## Coefficients standardisés principaux

| feature | standardized_coefficient |
| --- | --- |
| matches_14d_diff | 0.6173 |
| elo_logit | 0.5050 |
| prior_tournament_matches_14d_diff | -0.5010 |
| prior_tournament_games_14d_diff | 0.4607 |
| matches_30d_diff | 0.3991 |
| service_points_won_rate_surface_diff | 0.3541 |
| rank_diff | 0.3479 |
| games_30d_diff | -0.3476 |
| prior_tournament_matches_30d_diff | -0.3376 |
| service_points_won_rate_career_diff | -0.2911 |
| return_points_won_rate_career_diff | -0.2873 |
| return_above_expected_365d_diff | 0.2642 |
| return_points_won_rate_surface_diff | 0.2227 |
| serve_above_expected_365d_diff | 0.2198 |
| games_14d_diff | -0.2135 |
| prior_tournament_sets_14d_diff | -0.1925 |
| win_rate_365d_diff | -0.1821 |
| sets_14d_diff | -0.1730 |
| serve_above_expected_90d_diff | 0.1415 |
| return_points_won_rate_365d_diff | -0.1412 |
| prior_tournament_sets_30d_diff | 0.1386 |
| sets_7d_diff | 0.1248 |
| win_rate_career_diff | 0.1219 |
| win_rate_90d_diff | -0.1201 |
| prior_tournament_games_30d_diff | 0.1197 |

Les coefficients sont conditionnels à des features corrélées et ne sont pas causaux.

### Nouvelles features retenues

| feature | standardized_coefficient |
| --- | --- |
| prior_tournament_matches_14d_diff | -0.5010 |
| prior_tournament_games_14d_diff | 0.4607 |
| prior_tournament_matches_30d_diff | -0.3376 |
| return_above_expected_365d_diff | 0.2642 |
| serve_above_expected_365d_diff | 0.2198 |
| prior_tournament_sets_14d_diff | -0.1925 |
| serve_above_expected_90d_diff | 0.1415 |
| prior_tournament_sets_30d_diff | 0.1386 |
| prior_tournament_games_30d_diff | 0.1197 |
| return_above_expected_90d_diff | 0.1011 |
| serve_above_expected_career_diff | -0.0918 |
| opponent_rank_quality_mean_365d_diff | 0.0725 |
| return_above_expected_career_diff | -0.0719 |
| service_points_won_rate_90d_diff_x_clay | -0.0705 |
| return_above_expected_surface_diff | 0.0688 |
| prior_tournament_minutes_30d_diff | -0.0623 |
| return_above_expected_90d_diff_x_grass | -0.0557 |
| prior_tournament_minutes_14d_diff | 0.0532 |
| return_points_won_rate_90d_diff_x_clay | 0.0501 |
| serve_above_expected_90d_diff_x_clay | 0.0494 |
| opponent_elo_mean_90d_diff | 0.0440 |
| h2h_matches_before | -0.0432 |
| h2h_edge_A | 0.0409 |
| opponent_rank_quality_mean_30d_diff | 0.0393 |
| surface_grass | 0.0337 |

## J. Comparaison Phase 4

Phase 5 calibrée gagne +0.37
point d'accuracy, -0.00308 de log
loss, -0.00131 de Brier et
+0.00368 de ROC-AUC.

Comparaison appariée sur les 10,188 matchs avec deux rangs ATP disponibles et
distincts :

| model | matches | accuracy | log_loss | brier | roc_auc |
| --- | --- | --- | --- | --- | --- |
| ATP Ranking | 10,188 | 0.6387 | — | — | — |
| Logistic Phase 4 | 10,188 | 0.6570 | 0.6114 | 0.2122 | 0.7227 |
| Logistic Phase 5 | 10,188 | 0.6602 | 0.6082 | 0.2108 | 0.7265 |

## Journal d'ouverture du test

Une exécution technique préliminaire a produit un résultat test avant la dernière réduction de
features. Les suppressions et variantes finales ont ensuite été décidées exclusivement sur les
métriques validation, sans utiliser ce résultat test. Le tableau ci-dessus provient de la
dernière exécution figée. Cette déviation au protocole idéal est consignée explicitement.

## Anti-leakage

- forces adverses, Elo et ranks sont les snapshots pré-match;
- expected wins utilise exclusivement `final_raw_expected_A` Phase 3 pré-match;
- résidus, H2H et H2H surface sont mis à jour après le batch complet;
- rolling et surface excluent le match courant et les résultats futurs;
- imputer, scaler et régression sont fit sur 2000–2018;
- priors, ablations, C et Platt sont choisis sur 2019–2022;
- les tables de tuning ne contiennent aucune métrique test.

## Verdict pour la Phase 6

V3 est suffisamment mature pour un benchmark non linéaire contrôlé. La Phase 6 devra toutefois
comparer systématiquement la version complète à une variante sans les anciennes features de
fatigue, afin qu'un modèle d'arbres ne sur-exploite pas leur proxy de round. Aucun modèle non
linéaire n'est entraîné dans cette phase.
