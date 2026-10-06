# Tournament Context V1 — dictionnaire des features

## Contrat et chronologie

`data/processed/tournament_context_v1.parquet` contient une ligne par match ATP éligible
entre 2000 et 2026. `p_base` est la sortie exacte de PRE-MATCH V1, qui est lu mais jamais
réentraîné ni modifié.

Chaque édition est rejouée selon l'ordre structurel des rounds. L'état est copié avant un
round, tous les matchs du round sont prédits avec cette même version, puis tous les résultats
du round sont intégrés. Une date intra-tournoi n'est donc jamais utilisée pour ordonner deux
matchs d'un même round.

Les colonnes `round_state_version` et `matches_observed_before_round` permettent d'auditer
ce contrat. Les qualifications et round robins précèdent le tableau principal selon
`Q1, Q2, Q3, Q4, RR, ER, R128, R64, R32, R16, QF, SF, F/BR`.

## Identifiants et cible

- `match_id`, `tournament_id`, `tournament_edition_id`, `tournament_name` : traçabilité ;
  le nom et l'identifiant ne sont jamais encodés dans le modèle.
- `date`, `year`, `surface`, `level`, `level_group`, `best_of`, `round`, `round_order` :
  contexte structurel.
- `player_A_id`, `player_B_id`, `player_A_win` : orientation et cible.
- `p_base`, `logit_p_base` : probabilité PRE-MATCH V1 figée et son logit.

## État joueur dans l'édition

Pour chaque côté `A_` et `B_`, avant le match :

- charge : `matches_played`, `sets_played`, `games_played`, `minutes_played`,
  `sets_lost`, `games_lost`, `tiebreaks`, `five_set_matches`, `retirements` ;
- couverture : `minutes_coverage` ; une durée n'est utilisée que si elle est dans
  l'intervalle plausible 1–360 minutes ;
- résultats : `wins`, `expected_wins`, `wins_above_expected_raw` ; l'espérance cumulée
  provient exclusivement des `p_base` des matchs précédents ;
- volumes et numérateurs service/retour : points gagnés, aces, doubles fautes, breaks et
  opportunités correspondantes ;
- taux bruts tournoi : points de service/retour gagnés, ace, double faute et break.

Les colonnes suffixées `_diff` sont toujours `A - B`.

## Performance au-delà de l'attendu

Pour le service et le retour, l'attente pré-match combine le taux historique du joueur et
le complément du taux historique adverse. Les attentes ace/double faute reposent sur le
taux carrière. L'attente de break d'un joueur repose sur le taux de hold carrière adverse.

Les résidus cumulés sont divisés par `volume + C`, avec
`C ∈ {10, 20, 40, 75, 100}` :

```text
rate_over_expected_C = somme(observé - attendu) / (volume + C)
wins_above_expected_C = (victoires - somme(p_base)) / (matchs + C)
```

Cela équivaut à un shrinkage vers zéro. Le `C` de production expérimentale est choisi
uniquement en validation walk-forward.

Les scores permettent de mesurer la charge observée, mais aucun modèle séparé suffisamment
robuste d'« expected games/sets » n'est introduit en V1 ; ces deux résidus sont donc omis
plutôt que définis arbitrairement.

## État global du tournoi

Avant chaque round, les matchs déjà achevés produisent :

- résidus service, ace, double faute et break par rapport aux attentes des joueurs présents ;
- `tournament_tiebreak_rate`, `tournament_straight_sets_rate` ;
- résidu d'upsets par rapport à `min(p_base, 1-p_base)` ;
- moyennes de jeux, sets et minutes, avec couverture explicite.

Les variables synthétiques shrinkées utilisent `C ∈ {20, 40, 75, 100, 150}` :

- `serve_environment` et `return_environment` ;
- `break_environment` ;
- `speed_proxy = serve_environment + ace_environment - break_environment` ;
- `upset_environment` ;
- `workload_environment`.

`speed_proxy` est un indicateur statistique, pas une mesure physique de vitesse du court.
La longueur moyenne des échanges est absente : aucune couverture point/shot homogène n'est
disponible pour l'ensemble du replay.

Pour préserver la symétrie A/B, les états globaux n'entrent pas seuls dans l'adapter. Ils
sont multipliés par une différence de style pré-match (`style_serve_diff`,
`style_return_diff`, etc.), par `logit_p_base` pour l'environnement d'upsets, ou par la
différence de charge. Ainsi, un même environnement ne favorise pas arbitrairement le joueur A.

## Contexte historique et structurel

- `same_tournament_history_matches_diff` et
  `same_tournament_wins_above_expected_c20_diff` sont figés au début de l'édition. La clé est
  un nom normalisé stable, car `tournament_id` contient l'année. Aucun win rate naïf ni
  one-hot de tournoi n'est utilisé.
- Les indicateurs de round, surface, niveau et BO5 sont interagis avec `logit_p_base`. Ils
  peuvent corriger une sur/sous-confiance segmentaire sans introduire un décalage arbitraire
  vers A ou B.
- L'historique surface est déjà présent dans PRE-MATCH V1 ; le bloc « même tournoi » mesure
  donc uniquement son apport incrémental.

## Valeurs manquantes et API

Dans l'adapter offset, une feature contextuelle manquante signifie « aucune preuve pour
corriger » et vaut zéro. Les taux bruts manquants ne sont jamais inventés. L'API
`predict_tournament_match` retourne `p_base`, `p_adjusted` et des contributions groupées en
espace logit. Les allocations en points de probabilité sont descriptives et non causales.

`simulate_tournament` possède deux modes :

- `pre_tournament` : PRE-MATCH V1 uniquement ;
- `live` : état réel gelé au moment de la simulation, sans fabriquer de statistiques pour
  les rounds simulés.

## Limites V1

- couverture minutes : environ 42 %, couverture des stats détaillées : environ 43 % ;
- pas d'indoor/outdoor fiable sur toute la période ;
- pas de rally length homogène ;
- le chemin contrefactuel complet d'un ancien bracket exigerait des features PRE-MATCH
  point-in-time pour des confrontations qui n'ont jamais eu lieu ; elles ne sont pas
  reconstruites artificiellement.
