# Phase 7A — PRE-MATCH V1 en production

Date de gel : 6 octobre 2026. Cette phase s'arrête avant tout entraînement live.

## A. PRE-MATCH V1

Le moteur officiel est `Logistic Phase 5 / MODEL_H_REDUCED_ROBUST_FATIGUE`.
L'artefact `models/prematch_v1/model.joblib` est une copie binaire contrôlée par SHA-256
de l'artefact de Phase 5. Ses 86 variables ordonnées, son imputation médiane, sa
standardisation, ses coefficients, sa calibration Platt et ses hyperparamètres sont figés.

- apprentissage : jusqu'au 31 décembre 2018 ;
- calibration : 2019–2022 ;
- test historique : 2023–25 mai 2026, 10 376 matchs ;
- accuracy 0,657382 ; log loss 0,609876 ; Brier 0,211585 ; ROC-AUC 0,724285.

`metadata.json`, `feature_schema.json` et `config_snapshot.json` rendent le paquet
autodescriptif. Le replay intégral a reconstruit 78 845 lignes × 324 colonnes, dont 244
numériques comparées, avec le même motif de valeurs manquantes et un écart maximal de 0.

## B. Inférence réelle

La commande suivante résout les joueurs, reconstruit Elo et les features uniquement depuis
les matchs de date strictement antérieure, applique le modèle figé et expose probabilités,
warnings, hash de snapshot et dix contributions principales :

```bash
python scripts/predict_match.py \
  --player-a "Carlos Alcaraz" --player-b "Jannik Sinner" \
  --surface clay --date 2026-10-20
```

`--snapshot` ajoute les 86 entrées du modèle. Un nom exact ou un identifiant ATP est accepté.
Un doublon de nom déclenche une erreur demandant l'identifiant. Un joueur inconnu est traité
comme un cold start avec warning ; les valeurs absentes restent absentes jusqu'à l'imputation
figée du modèle. Le ranking obsolète/absent, le faible historique et l'absence de données de
surface/statistiques sont signalés. Pour une date historique, `replay(match_id)` est la voie
exacte par batch ; l'interface libre utilise par prudence une coupure à la date.

L'implémentation de référence privilégie la reproductibilité : elle rejoue l'historique complet
et prend environ 44 secondes pour une requête postérieure à tout le dataset sur la machine de
développement. Un checkpoint d'état historique pourra réduire cette latence sans toucher au
modèle ni à son contrat, avant une exposition en service temps réel.

## C. Holdout prospectif

Le cutoff honnête est le **6 octobre 2026**. Le registre
`data/prospective/prospective_predictions.parquet` est append-only au niveau applicatif :
chaque ligne contient `match_id`, date, joueurs, surface, probabilités, version, hash exact des
features et horodatage. Une collision différente est refusée. Le résultat est ajouté par une
commande séparée et devient lui-même immuable ; aucune probabilité n'est recalculée.

```bash
python scripts/predict_match.py ... --record
python scripts/record_prospective_result.py PREDICTION_ID 1
python scripts/evaluate_prospective.py
```

L'évaluation ignore les résultats manquants et calcule effectif, accuracy, log loss, Brier,
ROC-AUC lorsque les deux classes existent, et calibration par déciles. Elle n'est reliée à
aucune sélection ou ré-optimisation du modèle.

## D. Données point par point

- Grand Slam PBP : 49 fichiers, 1 922 136 lignes, 10 508 matchs avec points,
  période 2011–2024, hommes et femmes inférés depuis `match_num`.
- Match Charting Project hommes : 3 fichiers, 1 284 276 points, 7 532 matchs,
  période 1960–2026.
- Total brut : 3 206 412 lignes et 18 040 identifiants-matchs de source, sans prétendre
  qu'ils sont disjoints entre collections.

Le détail par année, tournoi, surface, sexe inféré, serveur, score, ace/double faute,
tie-break, abandon et anomalies est dans `reports/point_by_point_data_audit.md` et son JSON.

## E. Limites de qualité

Grand Slam n'a pas de date quotidienne, possède six variantes de schéma et des identifiants
joueurs largement absents. MCP n'a pas d'identifiants ATP stables et encode les coups dans des
chaînes qui exigent un parseur versionné. La couverture n'est pas exhaustive ; les abandons et
règles de tie-break sont hétérogènes. On observe aussi des lignes sentinelles, quelques valeurs
serveur/gagnant invalides, 2 111 doublons de clé MCP et une ligne de métadonnées MCP décalée.

## F. Scoring engine

`TennisScore` reconstruit score du point, jeux, sets, serveur et état du match. Il couvre
0/15/30/40, deuce/avantage, breaks, tie-break avec alternance de service, BO3, BO5 et set final
à avantage optionnel. Les tests ciblés passent ; toute séquence après fin du match ou entrée
invalide est refusée explicitement.

## G. Phase 7B proposée — LIVE TENNIS MODEL

Commencer par un baseline Markov / programmation dynamique exact qui combine l'état de score,
le serveur et des probabilités de point service/retour, avec PRE-MATCH V1 comme prior. Il devra
produire `P(A gagne le prochain point)`, puis par récursion `P(game)`, `P(set)` et `P(match)`.
La validation devra être temporelle et groupée par match, comparée à la probabilité pré-match
constante et au baseline mathématique, avant toute approche ML plus complexe.

La Phase 7B n'est pas commencée.
