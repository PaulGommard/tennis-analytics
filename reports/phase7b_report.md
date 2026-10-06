# Phase 7B — LIVE TENNIS V1

## A. Dataset

Les adaptateurs conservent MCP et Grand Chelem séparés jusqu'au schéma canonique. Sur
18 040 matchs bruts, 15 729 ont un score entièrement cohérent et 15 353 sont terminés et
éligibles, pour 2 724 048 points ML. MCP contribue 7 315 matchs terminés et Grand Chelem
8 038 (4 041 hommes, 3 997 femmes). La couverture PRE-MATCH V1 des matchs éligibles est
62,08 % globalement et 92,0 % des points masculins du test 2023–2026.

L'état est toujours capturé avant le point. MCP encode son score dans l'orientation
serveur/receveur avant point ; Grand Chelem expose un état après point orienté joueur 1/joueur
2. Ces conventions ont deux adaptateurs et deux tests distincts. Une divergence exclut le match
entier, sans correction silencieuse.

## B. Baseline serveur

Sur le test, le serveur gagne 64,38 % des points. Prédire systématiquement « serveur » conserve
cette accuracy mais produit une log-loss catastrophique de 4,9211. La probabilité constante
apprise sur train donne 0,651220 de log-loss et 0,229331 de Brier.

## C. Point Model

La régression logistique `P5_RECENT_SCORE` est figée comme POINT MODEL V1. Sur 449 886 points
test : accuracy 0,643763, log-loss 0,646444, Brier 0,227142, ROC-AUC 0,558046. Le choix a été
fait exclusivement sur validation. Les snapshots carrière/surface/365 jours ne voient que le
passé du match.

## D. Importance du score

Ajouter l'état du score au bloc récent améliore la log-loss de 0,000677 en validation et
0,000691 en test. L'IC 95 % apparié par match est positif, mais l'effet absolu reste faible.
Break point et différences de sets/jeux portent l'essentiel du petit gain ; tie-break, deuce,
set point et match point ont des coefficients très proches de zéro. Il n'y a donc pas de preuve
forte d'un « clutch » générique au-delà du contexte de score.

## E. Surface

Le taux serveur et la difficulté diffèrent : terre 61,91 % / log-loss 0,660765, gazon
66,23 % / 0,635583, dur 64,58 % / 0,645315. Les forces surface améliorent très peu seules,
mais sont conservées car elles structurent les probabilités au service et restent
interprétables.

## F. Game Model

La récursion exacte gère 0/15/30/40, deuce et avantage. Sur 413 965 états test alignés :
accuracy 0,8086, log-loss 0,425470, Brier 0,136345 et ROC-AUC 0,8856.

## G. Set Model

Le DP exact gère tie-break standard, tie-break final à 10 points, Wimbledon historique à
12–12 et sets finaux à l'avantage. Résultats : accuracy 0,7506, log-loss 0,484549,
Brier 0,162462, ROC-AUC 0,8418.

## H. Match Model

Le même moteur compose les sets en BO3 ou BO5. Résultats tous états : accuracy 0,7690,
log-loss 0,456225, Brier 0,151584, ROC-AUC 0,8620. La log-loss passe de 0,538325 après dix
points à 0,446165 au milieu du match et environ 0,416 juste après le premier set.

## I. Cohérence PRE-MATCH

Un décalage symétrique minimal des log-odds augmente la probabilité de point au service de A et
diminue celle de B, ou inversement. Une bisection impose qu'au premier point le Markov égale
PRE-MATCH V1. Sur le replay, les métriques de départ sont identiques à 1e-9 près. L'API expose
le décalage appliqué et accepte le premier serveur explicitement ; si celui-ci manque, elle
utilise le serveur courant avec un avertissement.

## J. Exemples complets

`reports/live_replay_examples.png` et `.csv` contiennent trois finales 2023–2024. La ligne
horizontale est le prior PRE-MATCH et la courbe suit P(A gagne) avant chaque point. Le Parquet
exhaustif conserve les 413 965 états évalués.

## K. Limites

- Grand Chelem utilise une date approximative de début de tournoi ; MCP fournit le jour.
- Les jointures d'identité sont exactes ; une identité ambiguë reste locale à la source.
- Ace et double faute sont fiables dans l'adaptateur Grand Chelem mais laissés absents dans le
  canonique MCP V1 ; le shot-level MCP n'est volontairement pas exploité.
- Le modèle principal est masculin. Les points Grand Chelem femmes sont normalisés, mais ne
  servent pas à la sélection de POINT MODEL V1.
- 2 283 matchs divergent, 376 sont incomplets/abandonnés et ne participent pas à l'évaluation
  principale. Les formats non représentables ne sont jamais devinés.
- Les états d'un même match sont corrélés. Les métriques « tous états » mesurent l'usage live ;
  les comparaisons par phase utilisent une seule observation par match.
- Les sources sont CC BY-NC-SA 4.0 et ne permettent pas un usage commercial sans permission.
- Le moteur suppose des probabilités de service fixes à l'échelle du Markov, hormis la
  prédiction score-aware du prochain point. Un modèle adaptatif intra-match appartient à une
  phase ultérieure.

## Décision

Phase 7B est figée avec POINT MODEL V1 + moteur exact LIVE TENNIS V1. Aucun Transformer,
deep learning, scraping live ou modèle shot-level n'a été commencé. La suite naturelle est une
Phase 8 consacrée à l'adaptation séquentielle et à la distinction entre momentum mesurable et
illusion statistique, après définition d'un protocole hors-échantillon spécifique.
