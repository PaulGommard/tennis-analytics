# Tennis Prediction Platform V1 — architecture produit

## Décision V1

Le MVP est une application locale composée d'un backend FastAPI, d'un frontend sans build
et d'une base SQLite. Les artefacts ML existants restent dans `models/` et ne sont jamais
réentraînés par l'application.

```text
Parquets historiques / futur fournisseur live
                    │
             TennisDataProvider
                    │
       services de prédiction Python
                    │
        SQLite + cache en mémoire borné
                    │
               API FastAPI
                    │
         Frontend HTML / CSS / JS
```

Le Tournament Context Adapter V1 est enregistré comme `research_only_disabled`. Il n'est
appelé par aucun endpoint officiel.

## Audit des moteurs existants

| Besoin produit | Artefact/API réutilisé | Décision |
| --- | --- | --- |
| Pré-match | `models/prematch_v1/model.joblib` | PREMATCH_V1 officiel, sans modification |
| Ledger prospectif | `prematch.registry` | Contrat append-only repris dans SQLite |
| Point suivant | `models/point_v2/model.joblib` | POINT_V2 officiel |
| Game / set | moteur Markov + POINT_V2 | LIVE_V2 officiel |
| Match live | blend `models/live_v3/model.joblib` | LIVE_V3 officiel |
| Score | `scoring.engine` | règles déterministes conservées |
| Tournoi | `tournament.engine.simulate_tournament` | PREMATCH_V1 uniquement |
| Tactique | SHOT_V1 | emplacement futur, hors priorité MVP |

Le `LivePredictor` historique public utilisait encore POINT V1. V1 produit ne l'expose donc
pas comme moteur officiel : `product.live.HistoricalLiveService` applique directement le
contrat POINT V2 / LIVE V2 / LIVE V3 aux snapshots point-in-time déjà construits.

## Arborescence

```text
backend/
  app.py                    routes FastAPI et fichiers statiques
frontend/
  index.html                shell accessible
  styles.css                design responsive data/sport
  app.js                    HOME, TOURNAMENT, MATCH, LIVE, ADMIN
src/tennis_analytics/product/
  providers/base.py         interface TennisDataProvider
  providers/historical.py   Mock / Historical Provider
  prematch.py               inférence figée + feature-state cache
  live.py                   façade POINT V2 / LIVE V2 / LIVE V3
  tournament.py             orchestration des simulations
  storage.py                schéma SQLite et écritures immuables
  service.py                façade applicative
  cli.py                    lancement local
```

## Data provider

`TennisDataProvider` définit :

- `get_tournaments()` ;
- `get_tournament()` ;
- `get_matches()` ;
- `get_bracket()` ;
- `get_match()` ;
- `get_live_score()` ;
- `get_point_by_point()`.

L'implémentation historique lit les tables ATP et les snapshots `point_features_v2`. Les
identifiants ATP et point-by-point étant différents, le lien repose sur la paire de joueurs,
l'année, la surface et la proximité de date. Une limite de 35 jours empêche les associations
lointaines. La disponibilité du replay est explicitement exposée par match.

Un fournisseur live futur devra implémenter la même interface. Les services, l'API et le
frontend resteront inchangés.

## Endpoints

```text
GET  /api/health
GET  /api/home
GET  /api/tournaments
GET  /api/tournaments/{id}
GET  /api/tournaments/{id}/bracket
GET  /api/tournaments/{id}/predictions
GET  /api/matches/{id}
GET  /api/matches/{id}/prediction
GET  /api/matches/{id}/live?point=N
GET  /api/matches/{id}/live/history?through_point=N
GET  /api/players/{id}
GET  /api/models
GET  /api/admin/performance
GET  /api/replay
POST /api/replay/select
POST /api/replay/reset
POST /api/replay/advance-round
```

FastAPI publie également `/docs` pour le contrat OpenAPI.

## Prédiction PRE-MATCH et latence

Pour un match historique, l'application charge une seule fois les 86 features figées et
indexe les lignes par `match_id`. La probabilité servie est celle de
`tournament_context_v1.p_base`, déjà vérifiée bit-à-bit contre PRE-MATCH V1. Une requête ne
rejoue donc jamais 80 000 matchs.

Pour une confrontation contrefactuelle de simulation, un `Player State Store` en mémoire
reconstruit les différences depuis le dernier snapshot strictement antérieur au tournoi.
L'Elo surface shrinké est reconstruit avec les constantes officielles. Les features
pair-specific indisponibles passent par l'imputer figé du modèle. Ces probabilités utilisent
le pipeline PREMATCH_V1, mais sont identifiées comme simulations de state store ; elles ne
remplacent pas une prédiction prospective enregistrée.

Les caches sont bornés et indexés par joueur, date, surface ou point. Le démarrage local
charge les datasets en quelques secondes ; les prédictions historiques chaudes sont ensuite
nettement sous la seconde.

## Prédictions immuables

`match_predictions` contient :

- `prediction_id` déterministe ;
- `match_id` ;
- `prediction_created_at` ;
- `model_version` ;
- `probability_a`, `probability_b` ;
- `feature_snapshot_hash` ;
- payload JSON complet ;
- résultat éventuel.

Une collision avec une valeur différente lève une erreur. Une lecture ultérieure renvoie le
premier snapshot au lieu de recalculer silencieusement. Les états live sont uniques par
`(match_id, point_index, model_version)`. Les simulations tournoi sont uniques par état du
bracket, version et nombre de simulations.

## Schéma de stockage

SQLite matérialise les tables prévues pour une migration future vers PostgreSQL :

- `players` ;
- `tournaments` ;
- `tournament_entries` ;
- `matches` ;
- `match_scores` ;
- `match_predictions` ;
- `live_states` ;
- `tournament_predictions` ;
- `model_versions` ;
- `replay_sessions`.

Les parquets restent la source historique du MVP. SQLite porte l'état produit, les caches
durables et l'historique prospectif.

## Flux live

```text
point historique N (avant le point)
        │
        ├── POINT_V2 ──────────── P(next point)
        │
        ├── POINT_V2 + Markov ─── P(game), P(set) / LIVE_V2
        │
        └── blend V1/V2 + Markov  P(match) / LIVE_V3
                    │
             stockage immuable
                    │
         frontend + courbe P(match)
```

L'orientation des joueurs est normalisée : si la source point-by-point inverse A/B par
rapport au match ATP, scores, serveur, vainqueur du prochain point et probabilités sont tous
réorientés ensemble.

## Simulation tournoi

Le bracket initial et les branches restantes utilisent `simulate_tournament` avec
PREMATCH_V1. En replay, valider un round remplace les branches par les vrais survivants puis
relance Monte Carlo. Aucun état du Tournament Adapter ne modifie les probabilités.

Le state store est figé avant le tournoi pour éviter d'utiliser les features réelles des
rounds futurs. V1 met donc à jour le bracket, pas les statistiques contrefactuelles des
joueurs. Une intégration live future pourra mettre à jour le state store à chaque résultat
réel, avec un test de parité research/production avant activation.

## Frontend

- **HOME** : replay actif, archives disponibles, matchs point-by-point ;
- **TOURNAMENT** : bracket horizontal, statut des rounds, probabilités de progression ;
- **MATCH** : probabilité PREMATCH_V1, snapshot et contributions du modèle ;
- **LIVE MATCH** : score, quatre horizons officiels, lecture/pause/seek et courbe P(match) ;
- **PERFORMANCE** : registry et benchmarks séparés des futures métriques prospectives ;
- **TACTICAL ANALYSIS** : réservé à SHOT V1 dans une version ultérieure.

Toutes les probabilités sont libellées comme probabilités de modèle, jamais comme certitudes.

## Lancement local

```bash
.venv/bin/python scripts/run_product_mvp.py --reload
```

Puis ouvrir `http://127.0.0.1:8000`. La documentation API est disponible sur
`http://127.0.0.1:8000/docs`.

## Déploiement futur

La prochaine phase, **LIVE DATA INTEGRATION**, devra choisir un fournisseur pour schedules,
brackets, scores et point-by-point. Avant un déploiement public : PostgreSQL, worker
d'ingestion idempotent, WebSocket/SSE, authentification admin, observabilité, sauvegardes et
gestion des secrets. Aucun fournisseur payant ni déploiement distant n'est configuré en V1.
