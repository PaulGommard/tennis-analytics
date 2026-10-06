# Audit du parseur MCP

Taux de parsing canonique: **98.48 %** (1,264,752/1,284,276).
Événements écrits: **6,396,818**. Points exclus: **19,524**.

## Motifs d'exclusion

| Motif | Points | Part des sources |
|---|---:|---:|
| `special_point` | 14,026 | 1.09 % |
| `serve_direction_missing` | 1,617 | 0.13 % |
| `error_without_terminal_marker` | 1,104 | 0.09 % |
| `trailing_after_terminal` | 797 | 0.06 % |
| `first_serve_not_fault` | 394 | 0.03 % |
| `incorrect_challenge` | 277 | 0.02 % |
| `unknown_shot_attribute:3` | 277 | 0.02 % |
| `unknown_shot_attribute:2` | 270 | 0.02 % |
| `unknown_shot_attribute:1` | 224 | 0.02 % |
| `unknown_shot_attribute:8` | 104 | 0.01 % |
| `error_code_on_winner` | 93 | 0.01 % |
| `missing_terminal` | 80 | 0.01 % |
| `unknown_shot_attribute:9` | 79 | 0.01 % |
| `unknown_shot_code:C` | 79 | 0.01 % |
| `unknown_shot_attribute:7` | 78 | 0.01 % |
| `unknown_shot_code:c` | 13 | 0.00 % |
| `active_serve_fault` | 11 | 0.00 % |
| `winner_mismatch` | 1 | 0.00 % |

Les codes `P/Q/R/S` expliquent l'essentiel de `special_point`: ils portent une issue de score mais aucune séquence observable. Les inconnues officielles (`0`, `q`, `e`) sont acceptées et marquées; les caractères non résolus restent des exclusions.

## Contrôles

- Désaccords entre le vainqueur reconstruit et `PtWinner`: 1 (exclus).
- Métadonnées de match absentes: 0.
- Événements contenant une inconnue explicite: 4,740.
- Services manquants, fins absentes, multiples fins et caractères après terminal sont rejetés.
- Les tests couvrent exemples officiels, second service, profondeur, positions, double faute, codes spéciaux et absence d'information future aux checkpoints.
