# Dictionnaire des nouvelles features V3

V3 hérite intégralement de V2 sans modifier ses colonnes. Paramètres sélectionnés sur
validation : prior résiduel points=500, prior résiduel
surface=100, prior H2H=5 matchs.

## Ajustement à l'adversaire

Pour un serveur A contre le retourneur B :

`expected_serve_A = 1 - historical_return_points_won_B`

`serve_residual_A = actual_service_points_won_A - expected_serve_A × service_points_A`

Le résidu retour utilise symétriquement `1 - historical_service_points_won_opponent`. Les
résidus sont agrégés par opportunités puis shrinkés vers zéro. Les variantes `90d`, `365d`,
`career` et `surface` sont strictement antérieures au batch.

## Strength of schedule et forme

- `opponent_elo_mean_*`: Elo adverse pré-match moyen dans la fenêtre.
- `opponent_rank_quality_mean_*`: moyenne de `-rank`; plus haut = adversaires plus forts.
- `wins_above_expected_*`: somme de `résultat - probabilité Elo pré-match`.

## H2H

`h2h_edge_A = (wins_A + 0.5C)/(matches + C) - 0.5`. Le H2H surface suit la même formule.
Les compteurs excluent toujours le match courant et restent gelés dans le batch.

## Matchup, surface, expérience et charge

- `serve_matchup_A = service_90d_A - return_90d_B`, symétriquement pour B.
- interactions explicites Elo/service/retour × Clay/Grass; Hard est la référence.
- `years_since_first_match`: ancienneté depuis la première apparition historique.
- `prior_tournament_*`: charge 14/30j excluant l'édition courante.

## Nouvelles colonnes

| Feature | Formule | Historique | Shrinkage | Sens | Limite |
| --- | --- | --- | --- | --- | --- |
| `opponent_elo_mean_30d_diff` | moyenne(Elo adverse pré-match) A − B | 30 jours, match courant exclu | aucun | positif = calendrier A plus fort | moyenne non pondérée; historique parfois absent |
| `opponent_rank_quality_mean_30d_diff` | moyenne(−rang ATP adverse) A − B | 30 jours, match courant exclu | aucun | positif = adversaires A mieux classés | rang non linéaire et parfois manquant |
| `opponent_elo_mean_90d_diff` | moyenne(Elo adverse pré-match) A − B | 90 jours, match courant exclu | aucun | positif = calendrier A plus fort | moyenne non pondérée; historique parfois absent |
| `opponent_rank_quality_mean_90d_diff` | moyenne(−rang ATP adverse) A − B | 90 jours, match courant exclu | aucun | positif = adversaires A mieux classés | rang non linéaire et parfois manquant |
| `opponent_elo_mean_365d_diff` | moyenne(Elo adverse pré-match) A − B | 365 jours, match courant exclu | aucun | positif = calendrier A plus fort | moyenne non pondérée; historique parfois absent |
| `opponent_rank_quality_mean_365d_diff` | moyenne(−rang ATP adverse) A − B | 365 jours, match courant exclu | aucun | positif = adversaires A mieux classés | rang non linéaire et parfois manquant |
| `serve_above_expected_career_diff` | résidu serve A − B; attendu=1 − retour historique adverse | career, strictement antérieur au batch | 500 opportunités vers 0 | positif = serve au-dessus de l'attendu | attente additive simple; disponibilité des stats |
| `serve_above_expected_90d_diff` | résidu serve A − B; attendu=1 − retour historique adverse | 90d, strictement antérieur au batch | 500 opportunités vers 0 | positif = serve au-dessus de l'attendu | attente additive simple; disponibilité des stats |
| `serve_above_expected_365d_diff` | résidu serve A − B; attendu=1 − retour historique adverse | 365d, strictement antérieur au batch | 500 opportunités vers 0 | positif = serve au-dessus de l'attendu | attente additive simple; disponibilité des stats |
| `serve_above_expected_surface_diff` | résidu serve A − B; attendu=1 − retour historique adverse | surface, strictement antérieur au batch | 100 opportunités vers 0 | positif = serve au-dessus de l'attendu | attente additive simple; disponibilité des stats |
| `return_above_expected_career_diff` | résidu return A − B; attendu=1 − service historique adverse | career, strictement antérieur au batch | 500 opportunités vers 0 | positif = return au-dessus de l'attendu | attente additive simple; disponibilité des stats |
| `return_above_expected_90d_diff` | résidu return A − B; attendu=1 − service historique adverse | 90d, strictement antérieur au batch | 500 opportunités vers 0 | positif = return au-dessus de l'attendu | attente additive simple; disponibilité des stats |
| `return_above_expected_365d_diff` | résidu return A − B; attendu=1 − service historique adverse | 365d, strictement antérieur au batch | 500 opportunités vers 0 | positif = return au-dessus de l'attendu | attente additive simple; disponibilité des stats |
| `return_above_expected_surface_diff` | résidu return A − B; attendu=1 − service historique adverse | surface, strictement antérieur au batch | 100 opportunités vers 0 | positif = return au-dessus de l'attendu | attente additive simple; disponibilité des stats |
| `h2h_matches_before` | nombre de confrontations A–B antérieures | carrière, match courant exclu | aucun | positif = davantage d'historique commun | ne donne pas seul le sens des victoires |
| `h2h_edge_A` | (wins_A + 0,5C)/(matches + C) − 0,5 | carrière, match courant exclu | C=5 matchs vers 50 % | positif = avantage historique A | petits échantillons et biais de sélection |
| `elo_logit_x_clay` | elo_logit × 1(surface=Clay) | identique à la variable de base | hérité de la variable de base | écart de coefficient Clay vs Hard | corrélée à la base; échantillon surface réduit |
| `service_points_won_rate_90d_diff_x_clay` | service_points_won_rate_90d_diff × 1(surface=Clay) | identique à la variable de base | hérité de la variable de base | écart de coefficient Clay vs Hard | corrélée à la base; échantillon surface réduit |
| `return_points_won_rate_90d_diff_x_clay` | return_points_won_rate_90d_diff × 1(surface=Clay) | identique à la variable de base | hérité de la variable de base | écart de coefficient Clay vs Hard | corrélée à la base; échantillon surface réduit |
| `serve_above_expected_90d_diff_x_clay` | serve_above_expected_90d_diff × 1(surface=Clay) | identique à la variable de base | hérité de la variable de base | écart de coefficient Clay vs Hard | corrélée à la base; échantillon surface réduit |
| `return_above_expected_90d_diff_x_clay` | return_above_expected_90d_diff × 1(surface=Clay) | identique à la variable de base | hérité de la variable de base | écart de coefficient Clay vs Hard | corrélée à la base; échantillon surface réduit |
| `elo_logit_x_grass` | elo_logit × 1(surface=Grass) | identique à la variable de base | hérité de la variable de base | écart de coefficient Grass vs Hard | corrélée à la base; échantillon surface réduit |
| `service_points_won_rate_90d_diff_x_grass` | service_points_won_rate_90d_diff × 1(surface=Grass) | identique à la variable de base | hérité de la variable de base | écart de coefficient Grass vs Hard | corrélée à la base; échantillon surface réduit |
| `return_points_won_rate_90d_diff_x_grass` | return_points_won_rate_90d_diff × 1(surface=Grass) | identique à la variable de base | hérité de la variable de base | écart de coefficient Grass vs Hard | corrélée à la base; échantillon surface réduit |
| `serve_above_expected_90d_diff_x_grass` | serve_above_expected_90d_diff × 1(surface=Grass) | identique à la variable de base | hérité de la variable de base | écart de coefficient Grass vs Hard | corrélée à la base; échantillon surface réduit |
| `return_above_expected_90d_diff_x_grass` | return_above_expected_90d_diff × 1(surface=Grass) | identique à la variable de base | hérité de la variable de base | écart de coefficient Grass vs Hard | corrélée à la base; échantillon surface réduit |
| `surface_clay` | 1(surface=Clay) | match courant (contexte connu avant match) | aucun | intercept Clay relatif à Hard | ne mesure pas une aptitude individuelle |
| `surface_grass` | 1(surface=Grass) | match courant (contexte connu avant match) | aucun | intercept Grass relatif à Hard | ne mesure pas une aptitude individuelle |
| `prior_tournament_matches_14d_diff` | charge matches A − B, édition courante exclue | tournois antérieurs commencés dans 14 jours | aucun | positif = charge récente A supérieure | dates de début de tournoi, pas dates journalières |
| `prior_tournament_sets_14d_diff` | charge sets A − B, édition courante exclue | tournois antérieurs commencés dans 14 jours | aucun | positif = charge récente A supérieure | dates de début de tournoi, pas dates journalières |
| `prior_tournament_games_14d_diff` | charge games A − B, édition courante exclue | tournois antérieurs commencés dans 14 jours | aucun | positif = charge récente A supérieure | dates de début de tournoi, pas dates journalières |
| `prior_tournament_minutes_14d_diff` | charge minutes A − B, édition courante exclue | tournois antérieurs commencés dans 14 jours | aucun | positif = charge récente A supérieure | dates de début de tournoi, pas dates journalières |
| `prior_tournament_matches_30d_diff` | charge matches A − B, édition courante exclue | tournois antérieurs commencés dans 30 jours | aucun | positif = charge récente A supérieure | dates de début de tournoi, pas dates journalières |
| `prior_tournament_sets_30d_diff` | charge sets A − B, édition courante exclue | tournois antérieurs commencés dans 30 jours | aucun | positif = charge récente A supérieure | dates de début de tournoi, pas dates journalières |
| `prior_tournament_games_30d_diff` | charge games A − B, édition courante exclue | tournois antérieurs commencés dans 30 jours | aucun | positif = charge récente A supérieure | dates de début de tournoi, pas dates journalières |
| `prior_tournament_minutes_30d_diff` | charge minutes A − B, édition courante exclue | tournois antérieurs commencés dans 30 jours | aucun | positif = charge récente A supérieure | dates de début de tournoi, pas dates journalières |
