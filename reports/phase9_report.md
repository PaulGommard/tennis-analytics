# Phase 9 — Shot-level analytics V1

## Conclusion exécutive

Le shot-level apporte un signal prédictif incrémental robuste au-delà du prior recalibré et le modèle de transition bat l'unigramme. Une Phase 10 séquentielle est donc justifiée, sans avoir été lancée.

Les résultats restent associatifs, conditionnels au sous-ensemble MCP et ne doivent pas être lus causalement.

## A. Dataset shot-level

- 7,532 matchs, 1,264,752 points, 6,396,818 événements, 1,002 joueurs.
- Parsing: 98.48 %; hommes seulement dans le snapshot local.

## B. Qualité du parser

19,524 points exclus avec motifs auditables; 1 désaccord de vainqueur exclu. Voir `mcp_parser_audit.md`.

## C. Service

| serve_number | serve_direction | attempts | win_rate | ace_rate |
|---|---|---|---|---|
| 1.000000 | wide | 597227 | 0.646396 | 0.077386 |
| 1.000000 | t | 542966 | 0.645803 | 0.100286 |
| 2.000000 | wide | 168867 | 0.519486 | 0.008474 |
| 2.000000 | body | 157760 | 0.510427 | 0.000311 |
| 2.000000 | t | 130900 | 0.506578 | 0.015982 |
| 1.000000 | body | 113037 | 0.599556 | 0.001539 |
| 1.000000 | unknown | 245 | 0.620408 | 0.077551 |

Le côté est reconstruit par la parité du score avant le point (y compris tie-breaks):

| court_side | surface | serve_number | serve_direction | attempts | win_rate_posterior | ace_rate |
|---|---|---|---|---|---|---|
| deuce | Hard | 1 | t | 190439 | 0.654035 | 0.113359 |
| ad | Hard | 1 | wide | 187957 | 0.643837 | 0.080550 |
| deuce | Hard | 1 | wide | 184361 | 0.658538 | 0.092492 |
| ad | Hard | 1 | t | 155697 | 0.643877 | 0.101274 |
| ad | Clay | 1 | wide | 80358 | 0.617994 | 0.037184 |
| deuce | Clay | 1 | t | 73181 | 0.619666 | 0.065249 |
| ad | Hard | 2 | wide | 70540 | 0.515253 | 0.004494 |
| deuce | Clay | 1 | wide | 68838 | 0.624085 | 0.056597 |
| deuce | Hard | 2 | body | 59004 | 0.510524 | 0.000136 |
| deuce | Hard | 2 | t | 57775 | 0.500666 | 0.008273 |

## D. Serve + 1

Patterns avec support minimum 200; taux lissés Beta(1,1). Les meilleurs taux ne sont pas des effets causaux.

| serve | serve_plus_1 | attempts | win_rate_posterior | lift_vs_global |
|---|---|---|---|---|
| serve_1:wide | f:down_the_line | 55963 | 0.617100 | 0.081290 |
| serve_1:t | f:down_the_line | 52370 | 0.597017 | 0.061208 |
| serve_1:t | f:crosscourt | 50798 | 0.604705 | 0.068895 |
| serve_1:wide | f:crosscourt | 48591 | 0.627745 | 0.091935 |
| serve_1:wide | b:crosscourt | 26567 | 0.559524 | 0.023715 |
| serve_2:body | f:crosscourt | 22934 | 0.531174 | -0.004636 |
| serve_2:wide | f:crosscourt | 22612 | 0.553418 | 0.017608 |
| serve_2:body | f:down_the_line | 22583 | 0.514855 | -0.020955 |
| serve_2:body | b:crosscourt | 22422 | 0.484793 | -0.051017 |
| serve_2:wide | f:down_the_line | 21811 | 0.528309 | -0.007501 |

## E. Return + 1

| return | return_plus_1 | attempts | win_rate_posterior | lift_vs_global |
|---|---|---|---|---|
| b:middle | f:crosscourt | 28878 | 0.472576 | 0.041744 |
| b:middle | b:crosscourt | 28029 | 0.469016 | 0.038184 |
| f:middle | b:crosscourt | 23951 | 0.472676 | 0.041844 |
| f:middle | f:crosscourt | 21821 | 0.491637 | 0.060805 |
| b:middle | b:middle | 20964 | 0.425737 | -0.005095 |
| b:crosscourt | b:crosscourt | 20538 | 0.483057 | 0.052225 |
| b:crosscourt | f:crosscourt | 18719 | 0.509321 | 0.078489 |
| b:middle | f:down_the_line | 18476 | 0.460602 | 0.029770 |
| b:middle | f:middle | 18298 | 0.405082 | -0.025750 |
| f:middle | b:middle | 17392 | 0.421352 | -0.009480 |

## F. Rally

| surface | rally_length | attempts | win_rate_posterior |
|---|---|---|---|
| Hard | 0_4 | 540584 | 0.668315 |
| Hard | 5_9 | 181996 | 0.655595 |
| Hard | 10_plus | 69239 | 0.440953 |
| Clay | 0_4 | 189704 | 0.632863 |
| Clay | 5_9 | 82733 | 0.648492 |
| Clay | 10_plus | 34249 | 0.440104 |
| Grass | 0_4 | 126605 | 0.677640 |
| Grass | 5_9 | 31732 | 0.686110 |
| Grass | 10_plus | 7722 | 0.423744 |
| Unknown | 5_9 | 33 | 0.714286 |

La longueur finale sert uniquement aux statistiques descriptives. Les modèles ne voient que la longueur déjà observée.

## G–H. Fingerprints et clustering

251 joueurs passent les seuils; K-Means retient k=2. La similarité cosinus exclut ranking et niveau externe. Interprétation détaillée dans `player_style_clusters.md`.

Interactions descriptives entre style du serveur et style du relanceur (sans interprétation causale):

| server_style_cluster | receiver_style_cluster | surface | attempts | win_rate_posterior |
|---|---|---|---|---|
| 0 | 0 | Hard | 551376 | 0.644707 |
| 0 | 0 | Clay | 209113 | 0.618344 |
| 0 | 0 | Grass | 92056 | 0.666895 |
| 1 | 0 | Hard | 38003 | 0.655335 |
| 0 | 1 | Hard | 37038 | 0.665092 |
| 1 | 1 | Hard | 29869 | 0.663185 |
| 1 | 1 | Grass | 15437 | 0.680484 |
| 0 | 1 | Grass | 10852 | 0.670720 |
| 1 | 0 | Grass | 10710 | 0.680545 |
| 1 | 0 | Clay | 9658 | 0.611077 |

## I. Prédiction du point

Même sous-ensemble MCP, split temporel par match; un contrôle `PRIOR_RECALIBRATED` sépare l'effet des coups d'une simple recalibration du prior.

| checkpoint | model | points | matches | log_loss | brier | auc |
|---|---|---|---|---|---|---|
| 1 | POINT_MODEL_V2 | 283075 | 2087 | 0.664011 | 0.235583 | 0.550364 |
| 1 | PRIOR_RECALIBRATED | 283075 | 2087 | 0.662912 | 0.235097 | 0.550364 |
| 1 | SHOT_LOGISTIC_AFTER_1 | 283075 | 2087 | 0.662190 | 0.234641 | 0.565091 |
| 2 | POINT_MODEL_V2 | 225399 | 2087 | 0.709749 | 0.257617 | 0.543374 |
| 2 | PRIOR_RECALIBRATED | 225399 | 2087 | 0.687701 | 0.247289 | 0.543374 |
| 2 | SHOT_LOGISTIC_AFTER_2 | 225399 | 2087 | 0.682145 | 0.244537 | 0.579755 |
| 3 | POINT_MODEL_V2 | 170208 | 2087 | 0.688676 | 0.247466 | 0.543071 |
| 3 | PRIOR_RECALIBRATED | 170208 | 2087 | 0.679728 | 0.243342 | 0.543071 |
| 3 | SHOT_LOGISTIC_AFTER_3 | 170208 | 2087 | 0.677041 | 0.241948 | 0.571476 |
| 5 | POINT_MODEL_V2 | 99364 | 2087 | 0.705438 | 0.255577 | 0.541205 |
| 5 | PRIOR_RECALIBRATED | 99364 | 2087 | 0.687044 | 0.246963 | 0.541205 |
| 5 | SHOT_LOGISTIC_AFTER_5 | 99364 | 2087 | 0.686768 | 0.246642 | 0.562492 |

Effets de log loss agrégés par match (positif = amélioration):

| checkpoint | comparison | mean_match_log_loss_gain | ci95_low | ci95_high | share_matches_improved |
|---|---|---|---|---|---|
| 1 | shot_minus_point_v2 | 0.001135 | 0.000299 | 0.001970 | 0.576425 |
| 1 | shot_minus_recalibrated_prior | -0.000257 | -0.001083 | 0.000569 | 0.545759 |
| 2 | shot_minus_point_v2 | 0.027799 | 0.026381 | 0.029217 | 0.838524 |
| 2 | shot_minus_recalibrated_prior | 0.005231 | 0.004223 | 0.006239 | 0.645424 |
| 3 | shot_minus_point_v2 | 0.011586 | 0.010275 | 0.012897 | 0.690465 |
| 3 | shot_minus_recalibrated_prior | 0.002435 | 0.001325 | 0.003545 | 0.596550 |
| 5 | shot_minus_point_v2 | 0.019634 | 0.017463 | 0.021806 | 0.714423 |
| 5 | shot_minus_recalibrated_prior | 0.000360 | -0.001254 | 0.001973 | 0.565405 |

Les checkpoints après N frappes excluent les points déjà terminés: les chiffres mesurent l'information additionnelle parmi les échanges encore vivants, pas une métrique artificiellement parfaite sur aces/winners déjà observés.

## J. Next-shot prediction

| split | model | transitions | log_loss | top1_accuracy |
|---|---|---|---|---|
| validation | unigram | 1099090 | 2.567509 | 0.168495 |
| validation | markov_1 | 1099090 | 2.280890 | 0.271341 |
| validation | player_markov | 1099090 | 2.229082 | 0.281642 |
| test | unigram | 1293186 | 2.542406 | 0.170949 |
| test | markov_1 | 1293186 | 2.275209 | 0.280197 |
| test | player_markov | 1293186 | 2.251101 | 0.283844 |

Le modèle joueur utilise un shrinkage de 100 transitions vers le Markov global et un minimum de 200 observations joueur.

## K. Surface

| surface | points | server_win_rate | average_rally_length | serve_wide_rate | serve_body_rate | serve_t_rate |
|---|---|---|---|---|---|---|
| Hard | 791819 | 0.645511 | 3.984042 | 0.427518 | 0.201783 | 0.370270 |
| Clay | 306686 | 0.615555 | 4.554235 | 0.455900 | 0.191342 | 0.352191 |
| Grass | 166059 | 0.667456 | 3.238349 | 0.418375 | 0.200260 | 0.380690 |
| Unknown | 188 | 0.728723 | 3.175532 | 0.404255 | 0.234043 | 0.361702 |

## L. Pression

Analyse limitée aux 1,230,984 points raccordés aux flags historiques de POINT MODEL V2.

| pressure | points | server_win_rate | average_rally_length | serve_wide_rate | serve_plus_1_attack_rate |
|---|---|---|---|---|---|
| normal | 1068278 | 0.643285 | 3.961623 | 0.428887 | 0.096333 |
| break_point | 88296 | 0.612553 | 4.650471 | 0.478855 | 0.095103 |
| tiebreak | 37422 | 0.647747 | 4.437069 | 0.423307 | 0.110202 |
| set_point | 23860 | 0.647946 | 3.892959 | 0.461903 | 0.108082 |
| match_point | 13128 | 0.643053 | 4.117002 | 0.457267 | 0.100229 |

Les écarts décrivent des changements de choix observés; ils ne démontrent ni clutch persistant ni causalité.

## Décision Phase 10

Décision: **GO prudent**. Critères: gain shot-level robuste au-delà de la recalibration = True; Markov meilleur que l'unigramme = True. La Phase 10 n'est pas commencée.
