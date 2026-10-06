# Audit PRE-MATCH V1 — Jaime Faria vs Arthur Gea

Audit du snapshot original produit pour Rolex Shanghai Masters, Round 1, Hard, 7 octobre 2026.
L'artefact PRE-MATCH V1 n'a été ni modifié ni réentraîné.

## 1. Prediction reconstruction

La probabilité **72,138532 %** est exactement la sortie du modèle sérialisé pour le vecteur
original sauvegardé (`919bd7133952012b7fd63394fcdf0abea7c4741dae04d29f07f4682e8e725864`).

- raw logit : `1.102480682608339`
- raw probability : `0.750724622179149`
- calibration Platt : `sigmoid(-0.087419784562064 + 0.942205597659056 × raw_logit)`
- calibrated logit : `0.951343685902490`
- final probability : `0.721385323808729`
- valeur SQLite : `0.721385323808729`

Conclusion mathématique : aucune divergence d'inférence. Le passage de 75,072462 % brut à
72,138532 % vient uniquement de la calibration Platt.

## 2. Exact logit decomposition

`raw_logit = intercept + Σ contributions`

- intercept : `-0.009376872`
- somme positive : `+1.925219111`
- somme négative : `-0.813361556`
- somme des contributions : `+1.111857555`
- total : `+1.102480683`

### 15 contributions positives

| Feature | Valeur | z | Coefficient | Contribution |
|---|---:|---:|---:|---:|
| `win_rate_90d_diff` | -0.3388888888888888 | -3.047797 | -0.120097 | +0.366032 |
| `matches_30d_diff` | 2.0 | 0.577383 | 0.399087 | +0.230426 |
| `win_rate_365d_diff` | -0.13921568627450986 | -1.008559 | -0.182101 | +0.183660 |
| `service_points_won_rate_career_diff` | -0.021003750281678446 | -0.594632 | -0.291080 | +0.173085 |
| `double_fault_rate_90d_diff` | -0.07258258732832303 | -3.955406 | -0.031577 | +0.124901 |
| `service_points_won_rate_365d_diff` | -0.04554481923160514 | -1.100281 | -0.085553 | +0.094132 |
| `return_points_won_rate_365d_diff` | -0.02653274291994695 | -0.635766 | -0.141232 | +0.089790 |
| `age_years_diff` | 1.5 | -0.759525 | -0.102666 | +0.077978 |
| `sets_30d_diff` | 6.0 | 0.671483 | 0.114455 | +0.076854 |
| `games_14d_diff` | -18.0 | -0.317525 | -0.213545 | +0.067806 |
| `minutes_30d_diff` | 200.0 | 0.592767 | 0.106384 | +0.063061 |
| `second_serve_won_rate_90d_diff` | 0.06931886046784719 | 1.577451 | 0.030722 | +0.048462 |
| `ace_rate_365d_diff` | -0.024974348741609448 | -0.510778 | -0.094048 | +0.048038 |
| `return_points_won_rate_career_diff` | -0.005256789412126317 | -0.162308 | -0.287329 | +0.046636 |
| `break_points_saved_rate_career_diff` | -0.05035944005077497 | -1.070494 | -0.035668 | +0.038182 |

### 15 contributions négatives

| Feature | Valeur | z | Coefficient | Contribution |
|---|---:|---:|---:|---:|
| `games_30d_diff` | 59.0 | 0.667530 | -0.347640 | -0.232060 |
| `service_points_won_rate_surface_diff` | -0.006670042637277773 | -0.226148 | 0.354129 | -0.080085 |
| `win_rate_career_diff` | -0.06518282988871221 | -0.520165 | 0.121854 | -0.063384 |
| `first_serve_won_rate_365d_diff` | -0.04287029704589418 | -0.804479 | 0.069843 | -0.056187 |
| `return_points_won_rate_90d_diff` | -0.08278186831065992 | -1.831268 | 0.026622 | -0.048753 |
| `hold_rate_career_diff` | -0.049194320244966505 | -0.756212 | 0.062399 | -0.047187 |
| `second_serve_won_rate_365d_diff` | -0.04215267503202291 | -1.137458 | 0.036537 | -0.041559 |
| `minutes_7d_diff` | 301.0 | 1.983316 | -0.020905 | -0.041461 |
| `rank_diff` | -4.0 | -0.098958 | 0.347865 | -0.034424 |
| `first_serve_won_rate_90d_diff` | -0.043904809147455315 | -0.786371 | 0.042643 | -0.033533 |
| `return_points_won_rate_surface_diff` | -0.0036545429909280447 | -0.121689 | 0.222691 | -0.027099 |
| `minutes_14d_diff` | 200.0 | 0.889327 | -0.024373 | -0.021676 |
| `first_return_points_won_rate_career_diff` | -0.012670502950766882 | -0.346060 | 0.045940 | -0.015898 |
| `surface_matches_diff` | 18.0 | -0.227642 | 0.064999 | -0.014796 |
| `surface_grass` | 0.0 | -0.338943 | 0.033739 | -0.011436 |

Le détail des 86 lignes (valeur brute, valeur imputée, moyenne, écart-type, z-score,
coefficient et contribution) est dans `reports/faria_gea_contributions.csv`.

## 3. A/B symmetry

Toutes les features antisymétriques changent correctement de signe :
`True` (échecs : `[]`).

- P(Faria | Faria=A), original : `72.138532 %`
- P(Gea | Gea=A), vecteur inversé : `26.611627 %`
- somme : `0.987501596069`

La somme n'est pas exactement 1 mais 0,987502. Ce n'est pas une inversion runtime : le modèle
logistique appris a un intercept non nul, les moyennes du scaler ne sont pas toutes nulles et la
calibration Platt a elle-même un intercept non nul. PRE-MATCH V1 n'est donc pas strictement
antisymétrique par construction. L'écart d'environ 1,25 point est une propriété figée du modèle.

## 4. Runtime vs research parity

Snapshot historique contrôlé : `atp:2026-520:2026-05-25:198`.

- SHA-256 artefact research : `5c71a6ee95a5afb141f62bc9cf0df62a7267c2f2c005e6a2750a71adf53fbca7`
- SHA-256 artefact gelé : `5c71a6ee95a5afb141f62bc9cf0df62a7267c2f2c005e6a2750a71adf53fbca7`
- artefacts identiques : `True`
- probabilité brute research/runtime : `0.674778694276451` / `0.674778694276451`
- probabilité calibrée research/runtime : `0.645719490909705` / `0.645719490909705`
- différence absolue : `0.000e+00`

Le pipeline d'inférence, l'imputation, la standardisation et les contributions sont identiques.
La divergence trouvée se situait dans la construction prospective des features, pas dans
l'artefact sklearn.

## 5. Counter-intuitive coefficients

| Feature | Valeur Faria−Gea | z-score | Coefficient | Contribution |
|---|---:|---:|---:|---:|
| `win_rate_90d_diff` | -0.338889 | -3.047797 | -0.120097 | +0.366032 |
| `win_rate_365d_diff` | -0.139216 | -1.008559 | -0.182101 | +0.183660 |
| `service_points_won_rate_career_diff` | -0.021004 | -0.594632 | -0.291080 | +0.173085 |

`win_rate_90d_diff = A - B`; sa valeur négative signifie bien Faria < Gea. La contribution
devient positive parce que son z-score est négatif et que le coefficient conditionnel appris
est lui aussi négatif. Dans le train, ces variables ont pourtant une corrélation marginale
positive avec la victoire, mais elles sont fortement corrélées à Elo et entre elles
(corrélation win-rate 365d/Elo ≈ 0,871 ; 90d/365d ≈ 0,705). Il s'agit d'un effet de suppression
multivariée sous régularisation L2 (`C=1`), pas d'une inversion A/B. Le signe est néanmoins peu
interprétable causalement et devient dangereux avec des données runtime approximatives.

## 6. Workload/date audit

Dans le snapshot original, `minutes_7d_diff=301` provenait du dernier snapshot historique de
Faria du 25 mai : deux matchs antérieurs du même batch Roland-Garros totalisaient 151 + 150
minutes. Le snapshot de Gea avait zéro minute sur 7 jours. Ces valeurs vieilles de 135 jours
étaient mélangées aux compteurs ESPN recalculés au 7 octobre, d'où l'incohérence avec zéro match,
set et jeu sur 7 jours.

Après correction : `minutes_7d_diff=0` car aucun match n'est encodé dans la fenêtre ; les durées
14d/30d sont `NaN` car ESPN ne fournit pas les durées, puis l'imputer gelé applique sa médiane.

### Jaime Faria

**7d — 0 matchs**

| Date utilisée | Tournoi | Adversaire | Sets | Jeux | Durée | Source | Précision date |
|---|---|---|---:|---:|---:|---|---|
| — | Aucun | — | 0 | 0 | 0 | — | — |

**14d — 4 matchs**

| Date utilisée | Tournoi | Adversaire | Sets | Jeux | Durée | Source | Précision date |
|---|---|---|---:|---:|---:|---|---|
| 2026-09-27 | Kinoshita Group Japan Open Tennis Championships | Arthur Fery | 3 | 29 | N/A | espn_public | tournament_start |
| 2026-09-27 | Kinoshita Group Japan Open Tennis Championships | Jaume Munar | 2 | 14 | N/A | espn_public | tournament_start |
| 2026-09-27 | Kinoshita Group Japan Open Tennis Championships | Shintaro Mochizuki | 2 | 18 | N/A | espn_public | tournament_start |
| 2026-09-27 | Kinoshita Group Japan Open Tennis Championships | Adolfo Daniel Vallejo | 3 | 34 | N/A | espn_public | tournament_start |

**30d — 6 matchs**

| Date utilisée | Tournoi | Adversaire | Sets | Jeux | Durée | Source | Précision date |
|---|---|---|---:|---:|---:|---|---|
| 2026-09-27 | Kinoshita Group Japan Open Tennis Championships | Arthur Fery | 3 | 29 | N/A | espn_public | tournament_start |
| 2026-09-27 | Kinoshita Group Japan Open Tennis Championships | Jaume Munar | 2 | 14 | N/A | espn_public | tournament_start |
| 2026-09-27 | Kinoshita Group Japan Open Tennis Championships | Shintaro Mochizuki | 2 | 18 | N/A | espn_public | tournament_start |
| 2026-09-27 | Kinoshita Group Japan Open Tennis Championships | Adolfo Daniel Vallejo | 3 | 34 | N/A | espn_public | tournament_start |
| 2026-09-22 | AITO Hangzhou Open | Terence Atmane | 3 | 32 | N/A | espn_public | tournament_start |
| 2026-09-22 | AITO Hangzhou Open | Hugo Gaston | 4 | 45 | N/A | espn_public | tournament_start |

**90d — 20 matchs**

| Date utilisée | Tournoi | Adversaire | Sets | Jeux | Durée | Source | Précision date |
|---|---|---|---:|---:|---:|---|---|
| 2026-09-27 | Kinoshita Group Japan Open Tennis Championships | Arthur Fery | 3 | 29 | N/A | espn_public | tournament_start |
| 2026-09-27 | Kinoshita Group Japan Open Tennis Championships | Jaume Munar | 2 | 14 | N/A | espn_public | tournament_start |
| 2026-09-27 | Kinoshita Group Japan Open Tennis Championships | Shintaro Mochizuki | 2 | 18 | N/A | espn_public | tournament_start |
| 2026-09-27 | Kinoshita Group Japan Open Tennis Championships | Adolfo Daniel Vallejo | 3 | 34 | N/A | espn_public | tournament_start |
| 2026-09-22 | AITO Hangzhou Open | Terence Atmane | 3 | 32 | N/A | espn_public | tournament_start |
| 2026-09-22 | AITO Hangzhou Open | Hugo Gaston | 4 | 45 | N/A | espn_public | tournament_start |
| 2026-08-11 | Cincinnati Open | Jenson Brooksby | 2 | 17 | N/A | espn_public | tournament_start |
| 2026-08-11 | Cincinnati Open | Ben Shelton | 2 | 20 | N/A | espn_public | tournament_start |
| 2026-08-11 | Cincinnati Open | Adam Walton | 4 | 43 | N/A | espn_public | tournament_start |
| 2026-08-11 | Cincinnati Open | Lorenzo Musetti | 2 | 20 | N/A | espn_public | tournament_start |
| 2026-08-11 | Cincinnati Open | Nikoloz Basilashvili | 2 | 20 | N/A | espn_public | tournament_start |
| 2026-08-11 | Cincinnati Open | Yibing Wu | 2 | 21 | N/A | espn_public | tournament_start |
| 2026-08-01 | National Bank Open presented by Rogers | Titouan Droguet | 3 | 30 | N/A | espn_public | tournament_start |
| 2026-08-01 | National Bank Open presented by Rogers | Christopher O'Connell | 3 | 31 | N/A | espn_public | tournament_start |
| 2026-07-18 | Millennium Estoril Open | Botic Van De Zandschulp | 2 | 19 | N/A | espn_public | tournament_start |
| 2026-07-18 | Millennium Estoril Open | Gonzalo Bueno | 2 | 18 | N/A | espn_public | tournament_start |
| 2026-07-18 | Millennium Estoril Open | Luciano Darderi | 3 | 28 | N/A | espn_public | tournament_start |
| 2026-07-18 | Millennium Estoril Open | Joao Domingues | 3 | 17 | N/A | espn_public | tournament_start |
| 2026-07-11 | EFG Swiss Open Gstaad | Stan Wawrinka | 3 | 33 | N/A | espn_public | tournament_start |
| 2026-07-11 | EFG Swiss Open Gstaad | Casper Ruud | 4 | 39 | N/A | espn_public | tournament_start |

**365d — 29 matchs**

| Date utilisée | Tournoi | Adversaire | Sets | Jeux | Durée | Source | Précision date |
|---|---|---|---:|---:|---:|---|---|
| 2026-09-27 | Kinoshita Group Japan Open Tennis Championships | Arthur Fery | 3 | 29 | N/A | espn_public | tournament_start |
| 2026-09-27 | Kinoshita Group Japan Open Tennis Championships | Jaume Munar | 2 | 14 | N/A | espn_public | tournament_start |
| 2026-09-27 | Kinoshita Group Japan Open Tennis Championships | Shintaro Mochizuki | 2 | 18 | N/A | espn_public | tournament_start |
| 2026-09-27 | Kinoshita Group Japan Open Tennis Championships | Adolfo Daniel Vallejo | 3 | 34 | N/A | espn_public | tournament_start |
| 2026-09-22 | AITO Hangzhou Open | Terence Atmane | 3 | 32 | N/A | espn_public | tournament_start |
| 2026-09-22 | AITO Hangzhou Open | Hugo Gaston | 4 | 45 | N/A | espn_public | tournament_start |
| 2026-08-11 | Cincinnati Open | Jenson Brooksby | 2 | 17 | N/A | espn_public | tournament_start |
| 2026-08-11 | Cincinnati Open | Ben Shelton | 2 | 20 | N/A | espn_public | tournament_start |
| 2026-08-11 | Cincinnati Open | Adam Walton | 4 | 43 | N/A | espn_public | tournament_start |
| 2026-08-11 | Cincinnati Open | Lorenzo Musetti | 2 | 20 | N/A | espn_public | tournament_start |
| 2026-08-11 | Cincinnati Open | Nikoloz Basilashvili | 2 | 20 | N/A | espn_public | tournament_start |
| 2026-08-11 | Cincinnati Open | Yibing Wu | 2 | 21 | N/A | espn_public | tournament_start |
| 2026-08-01 | National Bank Open presented by Rogers | Titouan Droguet | 3 | 30 | N/A | espn_public | tournament_start |
| 2026-08-01 | National Bank Open presented by Rogers | Christopher O'Connell | 3 | 31 | N/A | espn_public | tournament_start |
| 2026-07-18 | Millennium Estoril Open | Botic Van De Zandschulp | 2 | 19 | N/A | espn_public | tournament_start |
| 2026-07-18 | Millennium Estoril Open | Gonzalo Bueno | 2 | 18 | N/A | espn_public | tournament_start |
| 2026-07-18 | Millennium Estoril Open | Luciano Darderi | 3 | 28 | N/A | espn_public | tournament_start |
| 2026-07-18 | Millennium Estoril Open | Joao Domingues | 3 | 17 | N/A | espn_public | tournament_start |
| 2026-07-11 | EFG Swiss Open Gstaad | Stan Wawrinka | 3 | 33 | N/A | espn_public | tournament_start |
| 2026-07-11 | EFG Swiss Open Gstaad | Casper Ruud | 4 | 39 | N/A | espn_public | tournament_start |
| 2026-05-25 | Roland Garros | Denis Shapovalov | 3 | 32 | 151.0 | internal_historical | match_day |
| 2026-05-25 | Roland Garros | Jan Lennard Struff | 3 | 33 | 150.0 | internal_historical | match_day |
| 2026-05-25 | Roland Garros | Frances Tiafoe | 5 | 51 | 240.0 | internal_historical | match_day |
| 2026-04-22 | Madrid Masters | Hubert Hurkacz | 2 | 18 | 73.0 | internal_historical | match_day |
| 2026-02-16 | Rio de Janeiro | Sebastian Baez | 2 | 19 | 95.0 | internal_historical | match_day |
| 2026-02-16 | Rio de Janeiro | Damir Dzumhur | 2 | 23 | 123.0 | internal_historical | match_day |
| 2026-02-16 | Rio de Janeiro | Tomas Martin Etcheverry | 2 | 23 | 123.0 | internal_historical | match_day |
| 2026-01-19 | Australian Open | Alexander Blockx | 4 | 37 | 148.0 | internal_historical | match_day |
| 2026-01-19 | Australian Open | Andrey Rublev | 4 | 41 | 170.0 | internal_historical | match_day |
### Arthur Gea

**7d — 0 matchs**

| Date utilisée | Tournoi | Adversaire | Sets | Jeux | Durée | Source | Précision date |
|---|---|---|---:|---:|---:|---|---|
| — | Aucun | — | 0 | 0 | 0 | — | — |

**14d — 4 matchs**

| Date utilisée | Tournoi | Adversaire | Sets | Jeux | Durée | Source | Précision date |
|---|---|---|---:|---:|---:|---|---|
| 2026-09-27 | China Open | Zhizhen Zhang | 2 | 19 | N/A | espn_public | tournament_start |
| 2026-09-27 | China Open | Hubert Hurkacz | 5 | 58 | N/A | espn_public | tournament_start |
| 2026-09-27 | China Open | Te Rigele | 2 | 17 | N/A | espn_public | tournament_start |
| 2026-09-27 | China Open | Tomas Machac | 2 | 19 | N/A | espn_public | tournament_start |

**30d — 4 matchs**

| Date utilisée | Tournoi | Adversaire | Sets | Jeux | Durée | Source | Précision date |
|---|---|---|---:|---:|---:|---|---|
| 2026-09-27 | China Open | Zhizhen Zhang | 2 | 19 | N/A | espn_public | tournament_start |
| 2026-09-27 | China Open | Hubert Hurkacz | 5 | 58 | N/A | espn_public | tournament_start |
| 2026-09-27 | China Open | Te Rigele | 2 | 17 | N/A | espn_public | tournament_start |
| 2026-09-27 | China Open | Tomas Machac | 2 | 19 | N/A | espn_public | tournament_start |

**90d — 9 matchs**

| Date utilisée | Tournoi | Adversaire | Sets | Jeux | Durée | Source | Précision date |
|---|---|---|---:|---:|---:|---|---|
| 2026-09-27 | China Open | Zhizhen Zhang | 2 | 19 | N/A | espn_public | tournament_start |
| 2026-09-27 | China Open | Hubert Hurkacz | 5 | 58 | N/A | espn_public | tournament_start |
| 2026-09-27 | China Open | Te Rigele | 2 | 17 | N/A | espn_public | tournament_start |
| 2026-09-27 | China Open | Tomas Machac | 2 | 19 | N/A | espn_public | tournament_start |
| 2026-07-25 | Mifel Tennis Open by Telcel Oppo | SoonWoo Kwon | 3 | 31 | N/A | espn_public | tournament_start |
| 2026-07-25 | Mifel Tennis Open by Telcel Oppo | Michael Zheng | 3 | 22 | N/A | espn_public | tournament_start |
| 2026-07-25 | Mifel Tennis Open by Telcel Oppo | Francisco Cerundolo | 3 | 29 | N/A | espn_public | tournament_start |
| 2026-07-25 | Mifel Tennis Open by Telcel Oppo | Chak Lam Coleman Wong | 2 | 22 | N/A | espn_public | tournament_start |
| 2026-07-25 | Mifel Tennis Open by Telcel Oppo | Denis Shapovalov | 2 | 19 | N/A | espn_public | tournament_start |

**365d — 16 matchs**

| Date utilisée | Tournoi | Adversaire | Sets | Jeux | Durée | Source | Précision date |
|---|---|---|---:|---:|---:|---|---|
| 2026-09-27 | China Open | Zhizhen Zhang | 2 | 19 | N/A | espn_public | tournament_start |
| 2026-09-27 | China Open | Hubert Hurkacz | 5 | 58 | N/A | espn_public | tournament_start |
| 2026-09-27 | China Open | Te Rigele | 2 | 17 | N/A | espn_public | tournament_start |
| 2026-09-27 | China Open | Tomas Machac | 2 | 19 | N/A | espn_public | tournament_start |
| 2026-07-25 | Mifel Tennis Open by Telcel Oppo | SoonWoo Kwon | 3 | 31 | N/A | espn_public | tournament_start |
| 2026-07-25 | Mifel Tennis Open by Telcel Oppo | Michael Zheng | 3 | 22 | N/A | espn_public | tournament_start |
| 2026-07-25 | Mifel Tennis Open by Telcel Oppo | Francisco Cerundolo | 3 | 29 | N/A | espn_public | tournament_start |
| 2026-07-25 | Mifel Tennis Open by Telcel Oppo | Chak Lam Coleman Wong | 2 | 22 | N/A | espn_public | tournament_start |
| 2026-07-25 | Mifel Tennis Open by Telcel Oppo | Denis Shapovalov | 2 | 19 | N/A | espn_public | tournament_start |
| 2026-05-25 | Roland Garros | Karen Khachanov | 3 | 28 | 158.0 | internal_historical | match_day |
| 2026-05-17 | Hamburg | Aleksandar Kovacevic | 2 | 21 | 101.0 | internal_historical | match_day |
| 2026-02-02 | Montpellier | Giovanni Mpetshi Perricard | 2 | 9 | 39.0 | internal_historical | match_day |
| 2026-02-02 | Montpellier | Tomas Machac | 2 | 18 | 75.0 | internal_historical | match_day |
| 2026-02-02 | Montpellier | Adrian Mannarino | 3 | 32 | 151.0 | internal_historical | match_day |
| 2026-01-19 | Australian Open | Jiri Lehecka | 3 | 37 | 161.0 | internal_historical | match_day |
| 2026-01-19 | Australian Open | Stan Wawrinka | 5 | 53 | 273.0 | internal_historical | match_day |

ESPN attribue la date de début du tournoi à chaque ligne : quatre matchs de Faria au Japon ont
la date 27 septembre, et quatre matchs de Gea à Pékin ont aussi la date 27 septembre. Pour la
requête du 7 octobre, les huit sont donc exclus de 7d, inclus dans 14d/30d et le repos vaut 10
jours pour les deux. Les jours réels des rounds ne sont pas disponibles dans cette page : l'effet
exact sur 7d et le repos ne peut pas être récupéré proprement, mais le biais encodé est quantifié
ici (0 match en 7d contre jusqu'à 4 par joueur si certains rounds ont eu lieu après le 30 septembre).

## 7. Null/stale feature handling

Le snapshot original contenait `28`
valeurs nulles. Elles arrivent comme `NaN` dans `SimpleImputer(strategy='median')`, sont remplacées
par la médiane apprise sur le train, puis standardisées par `StandardScaler`. Elles ne deviennent
donc pas automatiquement zéro : elles ont le z-score `(médiane - moyenne) / écart-type`.
Le runtime utilise exactement les mêmes objets sklearn que l'entraînement.

| Feature | Valeur corrigée | Source | Date | Groupe | Exactitude |
|---|---:|---|---|---|---|
| `elo_logit` | 0.0285583318 | last compatible historical snapshot or frozen imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `rank_diff` | -4 | ESPN ranking / local identity / target context | 2026-10-01 to 2026-10-07 | FRESH | exact under resolved identity |
| `rank_points_diff` | -44 | ESPN ranking / local identity / target context | 2026-10-01 to 2026-10-07 | FRESH | exact under resolved identity |
| `ace_rate_career_diff` | 0.00111633989 | last compatible historical snapshot or frozen imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `double_fault_rate_career_diff` | 0.00254717154 | last compatible historical snapshot or frozen imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `first_serve_in_rate_career_diff` | 0.00481359281 | last compatible historical snapshot or frozen imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `first_serve_won_rate_career_diff` | -0.0150013318 | last compatible historical snapshot or frozen imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `second_serve_won_rate_career_diff` | -0.0287192852 | last compatible historical snapshot or frozen imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `service_points_won_rate_career_diff` | -0.0210037503 | last compatible historical snapshot or frozen imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `break_points_saved_rate_career_diff` | -0.0503594401 | last compatible historical snapshot or frozen imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `hold_rate_career_diff` | -0.0491943202 | last compatible historical snapshot or frozen imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `return_points_won_rate_career_diff` | -0.00525678941 | last compatible historical snapshot or frozen imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `first_return_points_won_rate_career_diff` | -0.012670503 | last compatible historical snapshot or frozen imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `second_return_points_won_rate_career_diff` | 0.00737233626 | last compatible historical snapshot or frozen imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `break_points_converted_rate_career_diff` | -0.0339921159 | last compatible historical snapshot or frozen imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `ace_rate_365d_diff` | -0.0249743487 | last compatible historical snapshot or frozen imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `ace_rate_90d_diff` | -0.00428740114 | last compatible historical snapshot or frozen imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `double_fault_rate_365d_diff` | -0.00941681505 | last compatible historical snapshot or frozen imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `double_fault_rate_90d_diff` | -0.0725825873 | last compatible historical snapshot or frozen imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `first_serve_in_rate_365d_diff` | -3.52809506e-05 | last compatible historical snapshot or frozen imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `first_serve_in_rate_90d_diff` | 0.0366972466 | last compatible historical snapshot or frozen imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `first_serve_won_rate_365d_diff` | -0.042870297 | last compatible historical snapshot or frozen imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `first_serve_won_rate_90d_diff` | -0.0439048091 | last compatible historical snapshot or frozen imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `second_serve_won_rate_365d_diff` | -0.042152675 | last compatible historical snapshot or frozen imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `second_serve_won_rate_90d_diff` | 0.0693188605 | last compatible historical snapshot or frozen imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `service_points_won_rate_365d_diff` | -0.0455448192 | last compatible historical snapshot or frozen imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `service_points_won_rate_90d_diff` | 0.0209886202 | last compatible historical snapshot or frozen imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `return_points_won_rate_365d_diff` | -0.0265327429 | last compatible historical snapshot or frozen imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `return_points_won_rate_90d_diff` | -0.0827818683 | last compatible historical snapshot or frozen imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `service_points_won_rate_surface_diff` | -0.0458668434 | last compatible historical snapshot or frozen imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `return_points_won_rate_surface_diff` | -0.00103236641 | last compatible historical snapshot or frozen imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `win_rate_career_diff` | -0.209558824 | local history + ESPN results | through 2026-09-27 | PARTIALLY REFRESHED | tournament-level ESPN dates |
| `win_rate_365d_diff` | -0.135775862 | local history + ESPN results | through 2026-09-27 | PARTIALLY REFRESHED | tournament-level ESPN dates |
| `win_rate_90d_diff` | -0.338888889 | local history + ESPN results | through 2026-09-27 | PARTIALLY REFRESHED | tournament-level ESPN dates |
| `matches_7d_diff` | 0 | local history + ESPN results | through 2026-09-27 | PARTIALLY REFRESHED | tournament-level ESPN dates |
| `sets_7d_diff` | 0 | local history + ESPN results | through 2026-09-27 | PARTIALLY REFRESHED | tournament-level ESPN dates |
| `games_7d_diff` | 0 | local history + ESPN results | through 2026-09-27 | PARTIALLY REFRESHED | tournament-level ESPN dates |
| `minutes_7d_diff` | 0 | last compatible historical snapshot or frozen imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `matches_14d_diff` | 0 | local history + ESPN results | through 2026-09-27 | PARTIALLY REFRESHED | tournament-level ESPN dates |
| `sets_14d_diff` | -1 | local history + ESPN results | through 2026-09-27 | PARTIALLY REFRESHED | tournament-level ESPN dates |
| `games_14d_diff` | -18 | local history + ESPN results | through 2026-09-27 | PARTIALLY REFRESHED | tournament-level ESPN dates |
| `minutes_14d_diff` | NaN | last compatible historical snapshot or frozen imputer; missing → training median imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `matches_30d_diff` | 2 | local history + ESPN results | through 2026-09-27 | PARTIALLY REFRESHED | tournament-level ESPN dates |
| `sets_30d_diff` | 6 | local history + ESPN results | through 2026-09-27 | PARTIALLY REFRESHED | tournament-level ESPN dates |
| `games_30d_diff` | 59 | local history + ESPN results | through 2026-09-27 | PARTIALLY REFRESHED | tournament-level ESPN dates |
| `minutes_30d_diff` | NaN | last compatible historical snapshot or frozen imputer; missing → training median imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `rest_days_diff` | 0 | local history + ESPN results | through 2026-09-27 | PARTIALLY REFRESHED | tournament-level ESPN dates |
| `career_matches_diff` | 31 | local history + ESPN results | through 2026-09-27 | PARTIALLY REFRESHED | tournament-level ESPN dates |
| `surface_matches_diff` | 6 | local history + ESPN results | through 2026-09-27 | PARTIALLY REFRESHED | tournament-level ESPN dates |
| `age_years_diff` | 1.40999316 | ESPN ranking / local identity / target context | 2026-10-01 to 2026-10-07 | FRESH | exact under resolved identity |
| `opponent_elo_mean_30d_diff` | NaN | last compatible historical snapshot or frozen imputer; missing → training median imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `opponent_rank_quality_mean_30d_diff` | NaN | last compatible historical snapshot or frozen imputer; missing → training median imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `opponent_elo_mean_90d_diff` | NaN | last compatible historical snapshot or frozen imputer; missing → training median imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `opponent_rank_quality_mean_90d_diff` | NaN | last compatible historical snapshot or frozen imputer; missing → training median imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `opponent_elo_mean_365d_diff` | NaN | last compatible historical snapshot or frozen imputer; missing → training median imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `opponent_rank_quality_mean_365d_diff` | NaN | last compatible historical snapshot or frozen imputer; missing → training median imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `serve_above_expected_career_diff` | NaN | last compatible historical snapshot or frozen imputer; missing → training median imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `serve_above_expected_90d_diff` | NaN | last compatible historical snapshot or frozen imputer; missing → training median imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `serve_above_expected_365d_diff` | NaN | last compatible historical snapshot or frozen imputer; missing → training median imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `serve_above_expected_surface_diff` | NaN | last compatible historical snapshot or frozen imputer; missing → training median imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `return_above_expected_career_diff` | NaN | last compatible historical snapshot or frozen imputer; missing → training median imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `return_above_expected_90d_diff` | NaN | last compatible historical snapshot or frozen imputer; missing → training median imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `return_above_expected_365d_diff` | NaN | last compatible historical snapshot or frozen imputer; missing → training median imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `return_above_expected_surface_diff` | NaN | last compatible historical snapshot or frozen imputer; missing → training median imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `h2h_matches_before` | 0 | ESPN ranking / local identity / target context | 2026-10-01 to 2026-10-07 | FRESH | exact under resolved identity |
| `h2h_edge_A` | 0 | ESPN ranking / local identity / target context | 2026-10-01 to 2026-10-07 | FRESH | exact under resolved identity |
| `elo_logit_x_clay` | 0 | last compatible historical snapshot or frozen imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `service_points_won_rate_90d_diff_x_clay` | 0 | last compatible historical snapshot or frozen imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `return_points_won_rate_90d_diff_x_clay` | -0 | last compatible historical snapshot or frozen imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `serve_above_expected_90d_diff_x_clay` | NaN | last compatible historical snapshot or frozen imputer; missing → training median imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `return_above_expected_90d_diff_x_clay` | NaN | last compatible historical snapshot or frozen imputer; missing → training median imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `elo_logit_x_grass` | 0 | last compatible historical snapshot or frozen imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `service_points_won_rate_90d_diff_x_grass` | 0 | last compatible historical snapshot or frozen imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `return_points_won_rate_90d_diff_x_grass` | -0 | last compatible historical snapshot or frozen imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `serve_above_expected_90d_diff_x_grass` | NaN | last compatible historical snapshot or frozen imputer; missing → training median imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `return_above_expected_90d_diff_x_grass` | NaN | last compatible historical snapshot or frozen imputer; missing → training median imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `surface_clay` | 0 | ESPN ranking / local identity / target context | 2026-10-01 to 2026-10-07 | FRESH | exact under resolved identity |
| `surface_grass` | 0 | ESPN ranking / local identity / target context | 2026-10-01 to 2026-10-07 | FRESH | exact under resolved identity |
| `prior_tournament_matches_14d_diff` | 0 | local history + ESPN results | through 2026-09-27 | PARTIALLY REFRESHED | tournament-level ESPN dates |
| `prior_tournament_sets_14d_diff` | -1 | local history + ESPN results | through 2026-09-27 | PARTIALLY REFRESHED | tournament-level ESPN dates |
| `prior_tournament_games_14d_diff` | -18 | local history + ESPN results | through 2026-09-27 | PARTIALLY REFRESHED | tournament-level ESPN dates |
| `prior_tournament_minutes_14d_diff` | NaN | last compatible historical snapshot or frozen imputer; missing → training median imputer | 2026-05-25 or earlier | STALE | stale or unavailable |
| `prior_tournament_matches_30d_diff` | 2 | local history + ESPN results | through 2026-09-27 | PARTIALLY REFRESHED | tournament-level ESPN dates |
| `prior_tournament_sets_30d_diff` | 6 | local history + ESPN results | through 2026-09-27 | PARTIALLY REFRESHED | tournament-level ESPN dates |
| `prior_tournament_games_30d_diff` | 59 | local history + ESPN results | through 2026-09-27 | PARTIALLY REFRESHED | tournament-level ESPN dates |
| `prior_tournament_minutes_30d_diff` | NaN | last compatible historical snapshot or frozen imputer; missing → training median imputer | 2026-05-25 or earlier | STALE | stale or unavailable |

## 8. Bugs found

Trois bugs runtime ont été confirmés et corrigés, sans toucher au modèle :

1. **Workload hybride** : minutes 7/14/30 anciennes combinées à des comptes récents.
2. **Mauvaise surface** : les stats `*_surface` venaient du dernier match, même sur une autre
   surface. Pour Faria, un état terre battue était utilisé pour Shanghai Hard.
3. **Double comptage** : `career_matches` et parfois `surface_matches` additionnaient toute la
   page ESPN 2026 à un historique qui contenait déjà janvier–mai. Les taux 365d omettaient aussi
   octobre–décembre 2025.

La correction fusionne l'historique local jusqu'au 25 mai avec uniquement les résultats ESPN
postérieurs, sélectionne un snapshot historique compatible avec la surface cible, reconstruit
H2H et `prior_tournament_*`, recalcule l'âge, et transmet les durées indisponibles comme `NaN`.

## 9. Corrected prediction

- avant : **Faria 72.139 % / Gea 27.861 %**
- après : **Faria 59.511 % / Gea 40.489 %**
- raw probability corrigée : `0.622819366759761`
- final probability corrigée : `0.595109023374001`

Ablations diagnostiques du snapshot original — elles ne sont pas des prédictions valides :

| Variante | P(Faria) |
|---|---:|
| FULL | 72.139 % |
| without form block | 62.085 % |
| without workload block | 68.774 % |
| without serve/return block | 67.538 % |
| without ranking block | 72.822 % |
| without Elo block | 72.037 % |

Cas artificiels en espace standardisé :

| Cas | P(A) |
|---|---:|
| identical / all standardized differences zero | 47.595 % |
| ranking only (+1 SD rank and points for A) | 57.949 % |
| Elo only (+1 SD for A) | 59.376 % |
| recent form only (+1 SD on three win rates for A) | 43.385 % |

Le cas « recent form only » confirme le signe conditionnel contre-intuitif appris ; ce n'est pas
une erreur d'affichage.

## 10. Final conclusion

### MODEL CORRECT / RUNTIME FEATURES INCORRECT (corrigées)

**72,1 % était la sortie PRE-MATCH V1 mathématiquement correcte pour le vecteur enregistré, mais
ce vecteur n'était pas correctement reconstruit.** L'explication des signes était correcte en
log-odds bruts ; elle révélait de vrais coefficients conditionnels négatifs. Après correction du
runtime, le même modèle figé donne **59.5 % pour Faria et
40.5 % pour Gea**. La qualité reste `MEDIUM` à cause des
dates ESPN au niveau tournoi, des durées absentes et des statistiques service/retour historiques.
