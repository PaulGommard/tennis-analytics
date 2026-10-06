# Phase 7A — audit point par point

## Inventaire constaté

| Source | Fichiers points | Lignes | Matchs points | Période | Date quotidienne |
|---|---:|---:|---:|---|---|
| Grand Slam PBP | 49 | 1,922,136 | 10,508 | 2011–2024 | non |
| Match Charting Project hommes | 3 | 1,284,276 | 7,532 | 1960–2026 | oui |

Les deux sources sont sous **CC BY-NC-SA 4.0** : attribution obligatoire,
usage commercial interdit, partage dans les mêmes conditions.

## Qualité et limites

- Grand Slam : 6 variantes de schéma, 10,598
  lignes sentinelles/non-points, taux serveur valide 99.97%,
  taux gagnant valide 99.97%, 0
  matchs points sans métadonnée, 0 clés
  `(match_id, PointNumber)` dupliquées.
- Grand Slam : aucune date quotidienne fiable; l'année et le tournoi sont disponibles.
  Les identifiants joueurs ne sont renseignés en paire que dans
  6.19% des métadonnées. Les divisions homme/femme
  sont inférées du millier de `match_num` et doivent être confirmées avant normalisation.
- MCP : taux serveur valide 100.00%, taux gagnant valide
  100.00%, 2,111 clés
  `(match_id, Pt)` dupliquées. Les noms sont présents mais pas d'identifiants ATP stables.
- MCP : les aces, doubles fautes et types de coups sont encodés dans `1st`/`2nd`, pas
  fournis comme flags canoniques. Un parseur versionné sera nécessaire.
- Les abandons ne sont pas identifiables de façon homogène. Aucun entraînement live ne
  doit commencer avant résolution des doublons, mapping joueurs, ordre des points et règles
  de tie-break par époque/tournoi.

## Décision Phase 7A

Le volume est suffisant pour préparer un baseline live, mais la jointure avec la base ATP
et la normalisation doivent rester une étape dédiée et auditée. Ce rapport ne prétend pas
que la couverture est exhaustive.
