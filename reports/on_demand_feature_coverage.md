# Couverture on-demand des 86 features PREMATCH V1

## Verdict

PREMATCH V1 peut être appelé sans modifier son artefact, mais le provider gratuit retenu ne
permet pas de reproduire exactement un snapshot de recherche actuel. Rang, points ATP, âge,
résultats, scores et charge en matchs/sets/jeux sont actualisables. Les statistiques de
service/retour et les durées ne le sont pas.

Stratégie V1 : mettre à jour uniquement les champs reconstructibles, conserver les derniers
champs de service/retour avec leur date explicite, et laisser l'imputer figé agir lorsqu'une
valeur n'a jamais existé. Aucun nouveau modèle n'est entraîné.

## Couverture détaillée

| Features (86 au total) | Source | Refresh | Exactitude | État V1 |
|---|---|---:|---|---|
| `rank_diff`, `rank_points_diff` | classement ESPN hebdomadaire | oui | snapshot hebdomadaire exact | frais |
| `age_years_diff` | dates de naissance locales + date cible | oui | exacte | frais |
| `surface_clay`, `surface_grass` | tournoi upcoming + mapping historique contrôlé | conditionnel | exacte si tournoi résolu | frais ou manquant |
| `elo_logit`, `elo_logit_x_clay`, `elo_logit_x_grass` | dernier snapshot historique compatible | non | Elo non rejoué sur tous les adversaires récents | stale |
| `win_rate_career_diff`, `win_rate_365d_diff`, `win_rate_90d_diff` | historique local + résultats ESPN post-cutoff | oui | dates ESPN au niveau tournoi | approximé |
| `matches_7d_diff`, `sets_7d_diff`, `games_7d_diff` | résultats + score ESPN | oui | dates au niveau tournoi | approximé |
| `matches_14d_diff`, `sets_14d_diff`, `games_14d_diff` | résultats + score ESPN | oui | dates au niveau tournoi | approximé |
| `matches_30d_diff`, `sets_30d_diff`, `games_30d_diff` | résultats + score ESPN | oui | dates au niveau tournoi | approximé |
| `minutes_7d_diff`, `minutes_14d_diff`, `minutes_30d_diff` | durée ESPN absente | partiel | 0 si aucun match ; sinon `NaN` si une durée manque | imputation explicite |
| `rest_days_diff` | dates de tournoi ESPN | oui | jour exact du match absent | approximé |
| `career_matches_diff`, `surface_matches_diff` | historique local + ESPN post-cutoff + surface résolue | oui | couverture/mapping ESPN dépendants | approximé |
| `opponent_elo_mean_30d_diff`, `opponent_rank_quality_mean_30d_diff` | dernier snapshot historique | non | adversaires récents non reconstruits | stale ou imputé |
| `opponent_elo_mean_90d_diff`, `opponent_rank_quality_mean_90d_diff` | dernier snapshot historique | non | adversaires récents non reconstruits | stale ou imputé |
| `opponent_elo_mean_365d_diff`, `opponent_rank_quality_mean_365d_diff` | dernier snapshot historique | non | adversaires récents non reconstruits | stale ou imputé |
| `h2h_matches_before`, `h2h_edge_A` | historique local + résultats ESPN | oui | seulement matchs résolus/dédupliqués | approximé |
| `prior_tournament_matches_14d_diff`, `prior_tournament_sets_14d_diff`, `prior_tournament_games_14d_diff` | résultats + score ESPN | oui | date de tournoi | approximé |
| `prior_tournament_matches_30d_diff`, `prior_tournament_sets_30d_diff`, `prior_tournament_games_30d_diff` | résultats + score ESPN | oui | date de tournoi | approximé |
| `prior_tournament_minutes_14d_diff`, `prior_tournament_minutes_30d_diff` | durée ESPN absente | non | transmise comme `NaN` | imputation explicite |
| `ace_rate_career_diff`, `ace_rate_365d_diff`, `ace_rate_90d_diff` | stats match absentes | non | — | stale |
| `double_fault_rate_career_diff`, `double_fault_rate_365d_diff`, `double_fault_rate_90d_diff` | stats match absentes | non | — | stale |
| `first_serve_in_rate_career_diff`, `first_serve_in_rate_365d_diff`, `first_serve_in_rate_90d_diff` | stats match absentes | non | — | stale |
| `first_serve_won_rate_career_diff`, `first_serve_won_rate_365d_diff`, `first_serve_won_rate_90d_diff` | stats match absentes | non | — | stale |
| `second_serve_won_rate_career_diff`, `second_serve_won_rate_365d_diff`, `second_serve_won_rate_90d_diff` | stats match absentes | non | — | stale |
| `service_points_won_rate_career_diff`, `service_points_won_rate_365d_diff`, `service_points_won_rate_90d_diff`, `service_points_won_rate_surface_diff` | dernier snapshot historique, compatible avec la surface cible pour `*_surface` | non | — | stale |
| `return_points_won_rate_career_diff`, `return_points_won_rate_365d_diff`, `return_points_won_rate_90d_diff`, `return_points_won_rate_surface_diff` | dernier snapshot historique, compatible avec la surface cible pour `*_surface` | non | — | stale |
| `break_points_saved_rate_career_diff`, `break_points_converted_rate_career_diff`, `hold_rate_career_diff` | stats match absentes | non | — | stale |
| `first_return_points_won_rate_career_diff`, `second_return_points_won_rate_career_diff` | stats match absentes | non | — | stale |
| `serve_above_expected_career_diff`, `serve_above_expected_90d_diff`, `serve_above_expected_365d_diff`, `serve_above_expected_surface_diff` | stats service/retour absentes | non | — | stale |
| `return_above_expected_career_diff`, `return_above_expected_90d_diff`, `return_above_expected_365d_diff`, `return_above_expected_surface_diff` | stats service/retour absentes | non | — | stale |
| `service_points_won_rate_90d_diff_x_clay`, `service_points_won_rate_90d_diff_x_grass` | taux stale × surface actuelle | partiel | composante taux ancienne | stale |
| `return_points_won_rate_90d_diff_x_clay`, `return_points_won_rate_90d_diff_x_grass` | taux stale × surface actuelle | partiel | composante taux ancienne | stale |
| `serve_above_expected_90d_diff_x_clay`, `serve_above_expected_90d_diff_x_grass` | résidu stale × surface | non | — | stale |
| `return_above_expected_90d_diff_x_clay`, `return_above_expected_90d_diff_x_grass` | résidu stale × surface | non | — | stale |

## Qualité exposée

- `HIGH` : réservé à un futur provider couvrant résultats, rangs, durées et statistiques de
  service/retour avec identités résolues. Impossible avec ESPN seul.
- `MEDIUM` : joueurs résolus, surface connue, ranking frais, résultats récents récupérés ;
  statistiques service/retour explicitement stale.
- `LOW` : joueur ambigu, surface inconnue, ranking absent, refresh incomplet ou cache expiré
  sans accès réseau.

Le score de qualité décrit uniquement la fraîcheur des données, jamais la confiance du modèle.
