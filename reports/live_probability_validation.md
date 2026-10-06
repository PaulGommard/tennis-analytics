# LIVE TENNIS V1 — validation historique

## Périmètre

Le replay principal couvre 2 318 matchs masculins terminés normalement et 413 965 états avant
point du test 2023–2026 pour lesquels PRE-MATCH V1 est disponible : 1 856 matchs MCP et
462 matchs Grand Chelem. Cela représente 92,0 % des points masculins du test. Les abandons,
scores divergents et matchs incomplets sont exclus.

Pour chaque match, les probabilités de point au service de A et B issues de POINT MODEL V1
sont ajustées par un unique décalage symétrique de log-odds. Le décalage est résolu par
bisection pour que le Markov au premier point reproduise PRE-MATCH V1. Il est calculé sans
utiliser le résultat du match. Les probabilités jeu/set/match sont ensuite exactes, sans Monte
Carlo.

## Résultats tous états

| Cible | Accuracy | Log-loss | Brier | ROC-AUC |
|---|---:|---:|---:|---:|
| Prochain point | 0,6450 | 0,645612 | 0,226742 | 0,6720 |
| Jeu courant | 0,8086 | 0,425470 | 0,136345 | 0,8856 |
| Set courant | 0,7506 | 0,484549 | 0,162462 | 0,8418 |
| Match | 0,7690 | **0,456225** | **0,151584** | 0,8620 |

Sur les mêmes états, conserver PRE-MATCH V1 statique donne une log-loss match de 0,581350 et
un Brier de 0,199449. Cette comparaison mesure la valeur de l'état du score au fil du match ;
elle ne constitue pas une comparaison indépendante point par point puisque plusieurs états
proviennent du même match.

## Information acquise pendant le match

| Instant (une observation/match) | LIVE log-loss | PRE-MATCH statique | LIVE Brier |
|---|---:|---:|---:|
| Premier point | 0,555570 | 0,555570 | 0,188521 |
| Après 10 points | 0,538325 | 0,555570 | 0,181609 |
| Milieu du match | 0,446165 | 0,555570 | 0,144315 |
| Après un set gagné par A | 0,415549 | 0,551825 | 0,131525 |
| Après un set perdu par A | 0,417847 | 0,559554 | 0,134010 |

Au premier point, l'écart maximal observé avec PRE-MATCH vient uniquement de la précision
numérique de la bisection (les métriques sont égales à 1e-9 près). La probabilité devient
ensuite progressivement plus informative, comme attendu.

## Sources, surfaces et calibration

La log-loss match sur tous les états vaut 0,461225 pour MCP et 0,441795 pour Grand Chelem.
Elle vaut 0,491254 sur terre battue, 0,466473 sur gazon et 0,440704 sur dur. Ces différences
mélangent difficulté des populations, longueur des formats et distribution des états ; elles
ne doivent pas être interprétées comme un effet causal de la surface.

La calibration par décile est globalement ordonnée. Les extrêmes sont légèrement trop prudents
ou trop confiants : 0–10 % prédit donne 2,66 % observé, et 90–100 % donne 96,76 % observé.
Les zones centrales restent proches (70–80 % : 75,01 % observé pour 75,01 % prédit).

## Replays

La figure `live_replay_examples.png` trace l'évolution complète de P(A gagne le match) pour :

- finale Wimbledon 2023, Carlos Alcaraz–Novak Djokovic ;
- finale Wimbledon 2024, Carlos Alcaraz–Novak Djokovic ;
- finale US Open 2024, Jannik Sinner–Taylor Fritz.

Les valeurs point par point sont dans `live_replay_examples.csv`. Le replay exhaustif est dans
`data/processed/live_probabilities_test.parquet` et les métriques dans
`live_probability_validation.csv`.
