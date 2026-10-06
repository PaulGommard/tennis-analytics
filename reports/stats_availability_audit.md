# Audit des statistiques ATP — Phase 4

## Schéma disponible

Une ligne `match_stats` existe par joueur et contient des comptes bruts, jamais des taux :

- `aces`: 72,262 matchs complets (91.13%)
- `double_faults`: 72,262 matchs complets (91.13%)
- `service_points`: 72,262 matchs complets (91.13%)
- `first_serve_in`: 72,262 matchs complets (91.13%)
- `first_serve_won`: 72,262 matchs complets (91.13%)
- `second_serve_won`: 72,262 matchs complets (91.13%)
- `service_games`: 72,262 matchs complets (91.13%)
- `break_points_saved`: 72,262 matchs complets (91.13%)
- `break_points_faced`: 72,262 matchs complets (91.13%)

Le bloc service complet est disponible pour 72,261 matchs
(91.12%). Une durée strictement positive existe pour
70,523 matchs (88.93%).

## Définitions fiables

- points de second service tentés = `service_points - first_serve_in`;
- points de service gagnés = `first_serve_won + second_serve_won`;
- points de retour gagnés = points de service adverses moins points de service gagnés adverses;
- breaks concédés = `break_points_faced - break_points_saved`;
- jeux de service tenus = `service_games - breaks_conceded`.

Les taux sont agrégés par somme des numérateurs divisée par somme des opportunités, jamais par
moyenne de pourcentages de matchs.

## Couverture temporelle

Les fichiers CSV joints donnent le détail par année, niveau et surface. La couverture service
est déjà comprise entre 86,9 % et 90,0 % sur 2000–2014, puis dépasse généralement 94 % depuis
2016. La rupture principale est institutionnelle : Davis Cup (`D`) n'a qu'environ 18 % de
couverture, tandis que A/G/M dépassent 98 %. La durée tombe à 46,3 % en 2015.

Décision : conserver le train 2000–2018 et apprendre l'imputation sur ce train. Un cutoff récent
ne corrigerait pas la faible couverture Davis Cup et supprimerait un historique utile.

## Anomalies filtrées localement

- negative_values: 0
- first_serve_in_above_service_points: 0
- first_serve_won_above_first_serve_in: 0
- second_serve_won_above_attempts: 1
- break_points_saved_above_faced: 0
- breaks_conceded_above_service_games: 5
- nonpositive_duration: 59
- duration_above_360_minutes: 6

Une anomalie invalide uniquement le ratio concerné, pas toutes les statistiques du match. Les
durées nulles/négatives et supérieures à 360 minutes sont exclues des charges en minutes.

## Limite de calendrier

La date est le début du tournoi. Les rounds fournissent un ordre structurel sûr, mais pas les
jours réels. La fatigue est donc mise à jour après chaque batch de round; les formats ambigus
restent en un seul batch. Aucun `consecutive_play_days` n'est inventé.
