# Audit des sources tennis gratuites — 6 octobre 2026

## Verdict

Pour ce MVP personnel, la meilleure source testée sans compte est le JSON public ESPN pour
les prochains matchs, les identifiants joueurs et le classement ATP. La page publique de
résultats d'un joueur fournit en une requête ses tournois, adversaires, rounds, résultats et
scores. Elle ne fournit pas les statistiques détaillées de service/retour.

Cette source est gratuite et répond aujourd'hui, mais son API n'est pas un produit public
contractuel : le schéma peut changer. Elle doit donc rester derrière `TennisDataProvider`,
avec cache, timeouts, budget de requêtes et erreurs explicites.

## Tests réels

| Source | Test effectué | Résultat | Décision |
|---|---|---|---|
| ESPN Site/Core JSON | scoreboard ATP, fiche joueur, event log, classement semaine 40 | HTTP 200, données ATP du jour présentes | provider V1 |
| ESPN page résultats joueur | Taylor Fritz, saison 2026 | HTTP 200, tournoi/round/adversaire/W-L/score | historique récent V1 |
| SofaScore JSON | calendrier tennis du 6 octobre | HTTP 403 | exclu, aucun contournement |
| ATP Tour | page scores courants | HTTP 403, challenge Cloudflare | exclu, aucun contournement |
| Live Tennis API | endpoint upcoming sans clé | HTTP 401 attendu | option future avec clé gratuite |
| Tennis API via RapidAPI | endpoint sans clé | DNS/clé requis ; documentation active | option future avec clé gratuite |
| API-Tennis.com | endpoint fixtures sans clé | HTTP 200 avec erreur d'authentification | essai 14 jours seulement, pas V1 |
| Tennis-data.co.uk | classeur 2026 historique | domaine garé / HTTP 404 | exclu |
| Jeff Sackmann original | CSV ATP 2026 | HTTP 404 | exclu ; miroir local reste figé |
| Ultimate Tennis Statistics | page publique | HTTP 200 mais dernière mise à jour 30-12-2024 | trop ancien |
| TheSportsDB | API v1 clé publique `123` | HTTP 200, couverture tennis insuffisamment ciblée | non retenu |

## Matrice fonctionnelle

| Source | Upcoming | Tournoi/round | Surface | Joueurs/IDs | Rankings | Résultats récents | Stats match | PBP | Auth / quota | Fiabilité |
|---|---|---|---|---|---|---|---|---|---|---|
| ESPN public | oui | oui | non dans le scoreboard testé | oui | rang + points hebdomadaires | oui | non (`statsSource=none`) | non | aucune / non publié | moyenne |
| Live Tennis API FREE | oui | oui | documentée | oui | limité selon plan | 20 appels histoire/mois | premium | premium | clé, 100/jour | bonne mais historique gratuit trop court |
| Tennis API RapidAPI FREE | oui | oui | documentée | oui | oui | oui | stats live premium selon route | premium | clé, 50/jour, 4/s | à valider avec une clé utilisateur |
| API-Tennis.com | oui | oui | documentée | oui | oui | oui | annoncées | annoncé | essai 14 jours puis 40 $/mois | non gratuite durablement |
| ATP/SofaScore | potentiellement | potentiellement | potentiellement | oui | oui | oui | variable | variable | accès bloqué pendant l'audit | inutilisable proprement |
| snapshot local | non | historique | oui | IDs ATP | historique | jusqu'au 25-05-2026 | oui partiellement | corpus séparé | aucune | excellente mais figée |

## Détails ESPN retenus

- Upcoming : `site.api.espn.com/.../tennis/atp/scoreboard?dates=YYYYMMDD`.
- Historique joueur : page publique `/tennis/player/results/_/id/{id}/year/{year}`.
- Classement : Core API `.../leagues/atp/rankings`, puis ressource de la semaine.
- Le scoreboard testé expose date/heure, tournoi, round, court, joueurs et IDs ESPN.
- Le classement testé est daté du 1er octobre 2026 et expose rang et points jusqu'au rang 150.
- Les statistiques détaillées d'une compétition testée retournent 404 et la compétition
  déclare `statsSource: none`.
- La surface doit venir d'un mapping historique de tournoi contrôlé ; sinon `unknown`.

Le provider filtrera strictement `Men's Singles`, les statuts futurs et les joueurs non-TBD.
Une réponse ESPN peut regrouper ATP et WTA dans un même événement : le nom de l'endpoint seul
ne constitue donc pas un filtre suffisant.

## Politique d'utilisation V1

- requêtes uniquement après action utilisateur ;
- cache local avec TTL configurable ;
- maximum six requêtes externes pour une prédiction ;
- timeout dix secondes, un seul retry ;
- User-Agent explicite ;
- aucun CAPTCHA, cookie de challenge ou mécanisme anti-bot contourné ;
- provenance conservée pour chaque snapshot ;
- aucune promesse de disponibilité : le dernier cache valide reste consultable.

## Conclusion

ESPN suffit pour construire la page **Prochains matchs** et rafraîchir résultats, scores,
classement et points ATP des deux joueurs. Il ne suffit pas pour rafraîchir exactement les
86 features : les statistiques de service/retour et les durées restent anciennes. Le niveau
de qualité maximal du provider sans clé est donc `MEDIUM`, pas `HIGH`.

Sources documentaires :

- https://github.com/pseudo-r/Public-ESPN-API/blob/main/docs/sports/tennis.md
- https://docs.livetennisapi.com/reference.html
- https://docs.tennis-api.com/getting-started
- https://api-tennis.com/
- https://www.thesportsdb.com/documentation
