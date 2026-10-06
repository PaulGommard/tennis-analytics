# Schéma canonique point par point — Phase 7A

Le grain est **une ligne par point réellement joué**. Les lignes sentinelles des sources
restent dans une zone brute et ne sont jamais présentées comme des points.

| Champ canonique | Type | Grand Slam PBP | Match Charting Project | Règle |
|---|---|---|---|---|
| `source` | string | constante | constante | provenance obligatoire |
| `source_match_id` | string | `match_id` | `match_id` | clé brute, jamais réécrite |
| `canonical_match_id` | string nullable | jointure à construire | jointure à construire | identifiant ATP après résolution |
| `point_index` | int | `PointNumber` | `Pt` | strictement croissant après audit |
| `date` | date nullable | absente | `Date` via matches | aucune date inventée |
| `year` | int | `year`/nom fichier | dérivé de `Date` | année source |
| `tournament` | string | `slam` | `Tournament` | libellé brut conservé séparément |
| `surface` | enum nullable | mapping tournoi | `Surface` | Hard/Clay/Grass/Carpet |
| `player_1_name`, `player_2_name` | string | metadata matches | metadata matches | noms source |
| `player_1_id`, `player_2_id` | string nullable | ids source rares | absents | mapping explicite seulement |
| `server` | int{1,2} | `PointServer` | `Svr` | null si invalide |
| `point_winner` | int{1,2} | `PointWinner` | `PtWinner` | null si invalide |
| `set_number` | int | `SetNo` | dérivé de `Set1/Set2` | contrôlé par reconstruction |
| `games_1`, `games_2` | int | `P1GamesWon/P2GamesWon` | `Gm1/Gm2` | score avant/après point à versionner |
| `point_score_1`, `point_score_2` | string | `P1Score/P2Score` | `Pts` à parser | accepte 0/15/30/40/AD ou TB numérique |
| `is_tiebreak` | bool nullable | dérivé | `TbSet` + score | validé par moteur de score |
| `serve_number` | int nullable | `ServeNumber` si présent | présence de `2nd` | pas d'inférence silencieuse |
| `ace`, `double_fault` | bool nullable | flags P1/P2 | à parser de `1st/2nd` | null ≠ faux |
| `rally_length` | int nullable | `Rally` | à parser du coup par coup | définition source documentée |
| `serve_speed_kmh` | float nullable | `Speed_KMH` | absent | bornes qualité requises |
| `raw_first`, `raw_second`, `raw_notes` | string nullable | absent | `1st/2nd/Notes` | encodage MCP intact |
| `source_file`, `source_row` | string/int | fichier/ligne | fichier/ligne | traçabilité |

Les flags de fin de jeu/set/match, break point et score reconstruits sont des **dérivés**.
Ils doivent porter une version de règles et être comparés au score source. Une divergence
reste une erreur d'audit, jamais une correction silencieuse.
