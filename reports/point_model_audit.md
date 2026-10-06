# POINT MODEL V1 — audit

## Population et protocole

Le modèle principal porte sur les hommes et combine les adaptateurs Grand Chelem et MCP sans
effacer la provenance. Le split est effectué par date de match : entraînement jusqu'au
31 décembre 2018, validation 2019–2022 et test à partir du 1er janvier 2023. Comme toutes les
lignes d'un match ont la même date et le même `match_id`, aucun match ne traverse deux splits.

| Split | Points | Matchs |
|---|---:|---:|
| Train | 1 233 208 | 6 219 |
| Validation | 479 467 | 2 573 |
| Test | 449 886 | 2 564 |

Les taux service/retour carrière, surface et 365 jours sont des snapshots calculés uniquement
avec les matchs achevés avant la date du match. Les matchs du même jour sont tous transformés
avant toute mise à jour. PRE-MATCH V1 est lui aussi figé avant match.

## Sélection sur validation

La métrique principale est la log-loss. `P5_RECENT_SCORE` est retenu.

| Modèle | Contenu cumulatif | Log-loss validation | Brier | ROC-AUC |
|---|---|---:|---:|---:|
| P0_CONSTANT | taux serveur train | 0,652540 | 0,229973 | 0,5000 |
| P1_SERVE_RETURN | forces service/retour | 0,649753 | 0,228708 | 0,5432 |
| P2_SURFACE | + surface | 0,649745 | 0,228705 | 0,5434 |
| P3_PREMATCH | + PRE-MATCH V1 | 0,649034 | 0,228375 | 0,5488 |
| P4_SCORE | + état du score | 0,648357 | 0,228065 | 0,5535 |
| P5_RECENT | P3 + fenêtre 365 jours | 0,649009 | 0,228364 | 0,5489 |
| **P5_RECENT_SCORE** | P5 + état du score | **0,648333** | **0,228054** | **0,5536** |

La baseline « le serveur gagne toujours » a 64,15 % d'accuracy en validation, mais une
log-loss de 4,9526 : elle illustre pourquoi l'accuracy seule est trompeuse.

## Test gelé 2023–2026

| Population | Points | Accuracy | Log-loss | Brier | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Combiné hommes | 449 886 | 0,643763 | **0,646444** | **0,227142** | **0,558046** |
| MCP hommes | 340 443 | 0,643700 | 0,646290 | 0,227074 | 0,559051 |
| Grand Chelem hommes | 109 443 | 0,643796 | 0,646653 | 0,227234 | 0,557673 |

Les performances MCP et Grand Chelem sont proches. `source` n'est donc pas injecté comme
feature : les deux adaptateurs restent auditables, mais le modèle ne peut pas apprendre un
raccourci propre au fournisseur.

## Le score et les points importants

Sur validation, ajouter le bloc score à P5 améliore la log-loss point-pondérée de 0,000677.
L'analyse appariée par match donne un gain moyen de 0,001270, IC 95 %
[0,001125 ; 0,001414], avec 56,6 % des matchs améliorés. Sur le test gelé, les valeurs sont
0,000691 et 0,001304, IC 95 % [0,001143 ; 0,001465]. Le gain est donc reproductible mais
**petit**.

Les coefficients standardisés les plus liés au score sont `sets_diff_server` (+0,0551),
`games_diff_server` (+0,0457), `is_break_point` (-0,0256) et
`server_leading_in_game` (+0,0203). Les coefficients tie-break (-0,00065), deuce (-0,00140),
set point (+0,00126) et match point (-0,00277) sont proches de zéro. Cela ne démontre pas un
« clutch » individuel : le bloc score apporte un faible signal moyen après contrôle des forces,
mais les indicateurs les plus narratifs ne présentent pas ici un effet important.

## Surface et calibration

| Surface | Points test | Taux serveur / accuracy | Log-loss | Brier |
|---|---:|---:|---:|---:|
| Clay | 92 565 | 0,6191 | 0,660765 | 0,234060 |
| Grass | 94 746 | 0,6623 | 0,635583 | 0,221907 |
| Hard | 262 575 | 0,6458 | 0,645315 | 0,226593 |

La calibration est bonne dans les zones bien peuplées : 55–60 % prédit donne 57,93 % observé,
60–65 % donne 62,79 %, 65–70 % donne 67,57 % et 70–75 % donne 71,85 %. Il n'existe aucun point
test prédit au-dessus de 80 %, ce qui est cohérent avec l'incertitude intrinsèque d'un point.

Les tables reproductibles sont `point_model_comparison.csv`, `point_model_segments.csv`,
`point_model_calibration.csv`, `point_model_coefficients.csv` et
`point_model_score_effect.csv` dans ce répertoire.
