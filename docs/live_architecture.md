# Architecture live après Phase 7A

La Phase 7A ne produit aucun modèle live. Elle sépare quatre responsabilités :

1. `prematch` fournit la probabilité initiale figée PRE-MATCH V1.
2. `point_by_point` normalise les observations sans inventer les champs absents.
3. `scoring` reconstruit l'état déterministe du match et valide la séquence.
4. `live` accueillera, en Phase 7B seulement, le calcul probabiliste en cours de match.

## Baseline mathématique proposée

Le premier baseline sera un modèle Markov/programmation dynamique, pas un nouvel
algorithme supervisé opaque. Pour un état `s` (serveur, score du jeu, jeux, sets), et une
probabilité `p(s)` que le serveur gagne le point suivant :

`V(s) = p(s) V(next(s, server)) + (1 - p(s)) V(next(s, returner))`

avec `V=1` si A a gagné le match et `V=0` si B l'a gagné. Les états de deuce/tie-break
sont résolus analytiquement ou par itération jusqu'à convergence; les autres par mémoïsation.
La première version utilisera des probabilités de point au service pré-match estimées à
partir des historiques T−1, puis une mise à jour bornée si le protocole de validation la
justifie. Les règles de format (BO3/BO5, tie-break final) font partie de l'entrée.

## Garde-fous avant Phase 7B

- split temporel et par match, jamais par point;
- aucune statistique calculée avec les points futurs du match évalué;
- calibration évaluée par tournoi, surface, format et progression du score;
- comparaison obligatoire à PRE-MATCH V1 constant et au baseline Markov;
- journal immuable des probabilités réellement émises;
- aucune formation avant mapping joueurs, déduplication et validation du score source.
