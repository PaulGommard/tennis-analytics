# Phase 2: temporal features and Elo baselines

## Time semantics

The ATP source only supplies `tourney_date`, normally the tournament's starting
Monday. It does not supply the exact day or time of each match. Consequently,
round order and `match_num` are not treated as timestamps.

All matches with the same date are one **frozen global batch**:

1. Read ratings and history as they stood before that date.
2. Emit every feature row in the batch.
3. Compute all result-dependent deltas without mutating state.
4. Apply the summed deltas and append results only after the complete batch.

Thus one tournament round cannot leak into another round carrying the same
source date, nor can the arbitrary row order leak across simultaneous events.
The cost is deliberately stale within-tournament ratings.

## A/B and target

The normalized layer already assigns slots by lexical canonical `player_id`.
This deterministic order is reused. It does not inspect winner, score, ranking
or row order. `player_A_win` is generated afterward from `winner_id`.

## Eligibility

Walkovers, defaults and abandoned/unfinished matches are retained in the full
timeline but excluded from ratings, history features and evaluation. Retirements
are retained because play occurred and the source declares an official winner.

## Elo V1

Configuration is in `config/elo.toml`:

- initial rating 1500;
- K = 32;
- logistic scale = 400;
- independent Hard, Clay, Grass and Carpet ratings.

Carpet is kept separate because the corpus has enough historical observations
and because treating it as Hard would silently change the surface definition.
Unknown surfaces have global Elo features but `NaN` surface Elo features and do
not update a surface system.

When a player has multiple matches in one frozen batch, every match sees the
same pre-batch rating. Its Elo deltas are summed and applied once at the batch
boundary. A new player starts at 1500. There is no inactivity adjustment in V1.

## Historical features

Career counts, surface counts, days since last match, and 30/90/365-day windows
contain prior date batches only. A win rate is `NaN` when its match count is
zero; it is never imputed as 0%. `days_since_last_match` is also `NaN` for a
debut.

The batch size is kept only in the diagnostic timeline, not in the modeling
feature table: deriving it from all source rows on a date could reveal
post-event information about completed or cancelled matches.

ATP rank is taken directly from the match participant row, where the source
defines it at `tourney_date`. No future lookup or interpolation is performed.
`rank_diff = rank_B - rank_A`, so a positive number favors A.

## Evaluation

The fixed temporal periods are:

- history: 2000–2018;
- validation: 2019–2022;
- final test: 2023 onward.

Ratings are computed continuously from 2000; the periods control reporting,
not state resets. There is no shuffled split. Elo accuracy excludes exact
rating ties because neither player is the strict favorite; log loss and Brier
include the corresponding 0.5 probability. Ranking accuracy requires two
non-missing, unequal ranks.

Source level `A` combines tour-level events and does not reliably distinguish
ATP 250 from ATP 500. Reports therefore retain the honest label “Other
tour-level” instead of inventing a category.

## Anti-leakage controls

Unit tests use a hand-computed three-match history and verify pre-match Elo,
same-date batching, rolling windows, chronological sorting, surface isolation,
current-target independence and future-result independence. The feature audit
also checks chronology, key uniqueness, Elo probabilities, rank differences,
non-negative inactivity and rolling count/rate identities on the real output.
