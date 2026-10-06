# Phase 7A — audit point par point

## Inventaire constaté

| Source | Fichiers points | Lignes | Matchs points | Période | Date quotidienne |
|---|---:|---:|---:|---|---|
| Grand Slam PBP | 49 | 1,922,136 | 10,508 | 2011–2024 | non |
| Match Charting Project hommes | 3 | 1,284,276 | 7,532 | 1960–2026 | oui |

Les deux sources sont sous **CC BY-NC-SA 4.0** : attribution obligatoire,
usage commercial interdit, partage dans les mêmes conditions.

## Couverture réelle

- **Grand Slam PBP** : 1,859 noms de joueurs bruts. La convention de
  `match_num` permet d'inférer 5,211
  matchs hommes, 5,048 femmes
  et laisse 254 non classés.
  Surfaces inférées du tournoi : 5,261 Hard,
  2,445 Clay et
  2,807 Grass. Le format BO3/BO5 n'est pas un champ
  source fiable de ces fichiers et ne doit pas être inventé.
- **MCP** : collection masculine uniquement, 1,003 noms bruts ;
  4,850 Hard,
  1,877 Clay et
  837 Grass. Les métadonnées déclarent
  5,202 BO3 et 2,329 BO5,
  avec 33 valeurs absentes et deux valeurs anormales.

## Formats et encodage

- Grand Slam : une ligne par point/sentinelle, `PointServer` et `PointWinner` dans {1,2},
  `SetNo`, `GameNo`, `P1GamesWon/P2GamesWon`, `P1Score/P2Score`. Ace et double faute sont des
  flags par joueur. Les scores numériques de tie-break apparaissent sur
  49,235 lignes ; la règle de tie-break du match n'est pas
  fournie uniformément.
- MCP : `Pt` ordonne les points ; `Set1/Set2`, `Gm1/Gm2`, `Pts` et `Gm#` portent le score ;
  `Svr`/`PtWinner` utilisent {1,2}. `TbSet` indique qu'un set doit finir au tie-break, pas que
  le point courant est nécessairement joué dans celui-ci. `1st`, `2nd` et `Notes` portent
  l'encodage coup par coup. Les fichiers présents n'exposent pas les colonnes dérivées
  `isAce`, `isDouble`, `TB?` mentionnées dans le dictionnaire amont : elles devront être
  parsées et vérifiées, jamais supposées.
- Abandons : Grand Slam fournit 20 `Retired`, 628 `Complete`, 3 `In Progress`, mais le statut
  manque pour 9 862 matchs. MCP n'a pas de champ abandon dédié.

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
  `(match_id, Pt)` dupliquées et 1 ligne de métadonnées décalée.
  Les noms sont présents mais pas d'identifiants ATP stables.
- MCP : les aces, doubles fautes et types de coups sont encodés dans `1st`/`2nd`, pas
  fournis comme flags canoniques. Un parseur versionné sera nécessaire.
- Les abandons ne sont pas identifiables de façon homogène. Aucun entraînement live ne
  doit commencer avant résolution des doublons, mapping joueurs, ordre des points et règles
  de tie-break par époque/tournoi.

## Décision Phase 7A

Le volume est suffisant pour préparer un baseline live, mais la jointure avec la base ATP
et la normalisation doivent rester une étape dédiée et auditée. Ce rapport ne prétend pas
que la couverture est exhaustive.
