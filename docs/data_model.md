# Normalized data model

The model deliberately separates match structure, participation, statistics and
outcome. It never defines the first player as the winner.

## Tables

### `players.parquet`

One row per canonical player. `player_id` is `atp:<source id>` when a positive
ATP identifier exists. Otherwise it is a deterministic `synthetic:<hash>` built
from normalized name, birth date and country. Synthetic identities are visible
in the audit and are never fuzzy-merged.

### `tournaments.parquet`

Stable tournament identity (`tournament_id`, source ID and name). A tournament
can have several yearly editions.

### `tournament_editions.parquet`

Edition-specific properties: start date, surface, level and draw size. This
avoids pretending that a tournament's surface or level can never change.

### `matches.parquet`

One source record per row. `winner_player_id` is an explicit outcome. The source
score is retained with `score_perspective = winner`. The ATP files supply the
tournament start date, not the precise date of every match; this is recorded as
`date_precision = tournament_start`.

`natural_match_key` comes from tournament ID, tournament start date and source
match number. If a source repeats this key, every row is preserved and later
occurrences receive a `:dupN` suffix in `match_id`; the audit reports the issue.

### `match_participants.parquet`

Exactly two expected rows per match. Slots A and B are assigned by sorted
canonical `player_id`, independently of who won. Rank, ranking points and age
belong here because they are observations at match time. `is_winner` is an
explicit outcome column and must be excluded from pre-match features later.

### `match_stats.parquet`

One expected row per match/player with service statistics. Long form avoids
winner/loser column families and is easy to pivot into A/B only after the split
is outcome-independent.

## Identity policy

1. Trust a positive source ATP player ID.
2. Use a deterministic synthetic ID only when it is absent.
3. Never fuzzy-match automatically.
4. Write conflicting names, source IDs or birth dates to
   `data/interim/identity_issues.parquet` for human review.

## Live point model

`data/processed/point_features_v1.parquet` contains one row per canonical point. The score,
server and all flags describe the state **before** that point; `A_wins_next_point` and
`server_won_point` are targets. Provenance remains explicit in `source` and `source_match_id`.

MCP source scores are server/receiver-oriented and describe the state before the point. Grand
Slam source scores are player-1/player-2-oriented and describe the state after the point. Each
adapter replays the point with `TennisScore`; a mismatch excludes the whole match. MCP dates
have day precision, whereas Grand Slam dates are marked `tournament_approximation`.

Historical rate features are snapshots before the match, updated only after all matches with
the same date have been transformed. `ace` and `double_fault` are nullable because V1 does not
decode MCP shot strings. Exact ATP normalized-name matches receive the ATP ID; unmatched names
receive deterministic source-local IDs and are never fuzzy-joined.

`data/processed/live_probabilities_test.parquet` is an evaluation artifact for 2023–2026. It
stores P(A wins next point/current game/current set/match) before each point, alongside the
frozen PRE-MATCH prior and the symmetric consistency shift.

## In-match adaptation model

`data/processed/point_features_v2.parquet` extends V1 without changing it. Every cumulative,
recent-window, game, previous-set and dynamic-strength field is evaluated before the current
point. Dynamic service and return rates shrink observations from the current match toward the
pre-match historical rate. Historical clutch fields use only earlier dates and strongly shrunk
residual performance under pressure.

The V2 live evaluation keeps the Phase 7B Markov engine unchanged. It updates only the two
point-on-serve probabilities. Full point/game evaluation is stored for every eligible state;
the more expensive set/match comparison uses every tenth point plus every set start, marked by
`live_sampled` in `data/processed/live_v2_probabilities_test.parquet`.

## Stabilized match horizon

`data/processed/live_v3_probabilities_test.parquet` contains the Phase 8C comparison. V3 blends
V1 and V2 point-on-serve probabilities using only the number of prior match points, then sends
the blended probabilities through the unchanged exact scoring engine. Raw and horizon-calibrated
columns remain separate; the selected artifact applies no posterior calibration because none
improved temporal validation. Configuration and calibrators are frozen in `models/live_v3/`.
