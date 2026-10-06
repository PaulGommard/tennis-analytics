# Audit de validité des features de fatigue

## Verdict

Les features Phase 4 `matches/sets/games/minutes_{7,14,30}d` sont **expérimentales** et ne
mesurent pas des fenêtres calendaires exactes. La source ATP associe tous les rounds à la date
de début du tournoi. Un round précédent du même tournoi est donc compté avec zéro jour d'écart.
Ces variables combinent charge réelle et structure/progression dans le tableau.

Les corrélations sont extrêmes à horizon identique : sets, games et minutes atteignent souvent
0,97–0,99. Les moyennes absolues varient fortement par round (`fatigue_by_round.csv`). Elles ne
révèlent pas directement un résultat futur, car le snapshot précède le batch, mais constituent
un proxy indirect de progression et non une mesure pure de récupération physique.

## Dates alternatives déjà présentes

- Match Charting Project : 7 565 dates journalières valides sur 7 566 lignes, de 1960 à 2026,
  mais sur un échantillon charté, non homogène et bien plus petit que les 78 845 matchs analysés.
- Slam point-by-point : 10 513 lignes de matchs, aucune colonne date dans les 49 fichiers.
- ATP principal : `tourney_date` seulement, documentée comme début de tournoi.

Ces dates partielles ne sont pas fusionnées dans V3 : cela créerait deux définitions de fatigue
selon la disponibilité de la source.

## Alternative Phase 5

Les nouvelles variables `prior_tournament_*_{14,30}d` excluent explicitement l'édition
courante. Elles mesurent la charge lors de tournois antérieurs dont la date de début tombe dans
la fenêtre. Elles restent approximatives, mais ne codent plus le round courant. Phase 5 compare
un modèle conservant exactement la fatigue Phase 4 et un modèle la remplaçant par cette variante.

## Confiance

| Famille | Confiance | Limite |
| --- | --- | --- |
| Phase 4 intra-tournoi 7/14/30j | faible | dates identiques, proxy de round |
| `rest_days` | moyenne-faible | écart entre débuts de tournois; zéro intra-tournoi |
| Phase 5 tournois antérieurs | moyenne | date de début, pas jour réel du match |
| sets/games parsés | moyenne-haute | comptes explicites; scores ambigus deviennent NaN |
| minutes | moyenne | bonne couverture hors Davis Cup/2015, valeurs aberrantes filtrées |
