# Phase 3: Elo chronology, tuning and calibration

## Scope

Phase 3 remains an Elo-only stage. It does not train XGBoost, random forests,
neural networks or transformers. It improves the temporal semantics of Elo,
selects a small parameter grid, regularizes surface ratings, tests one simple
inactivity mechanism and calibrates the final probabilities.

The complete numerical result is in
[`reports/phase3_report.md`](../reports/phase3_report.md).

## Structural chronology

The ATP source gives one tournament-start date, not exact match dates. An audit
shows that every tournament edition consequently places all its matches on one
source date. Phase 2 froze the entire date. Phase 3 safely recovers only the
known structural ordering:

`Q1 → Q2 → Q3 → Q4 → RR → ER → R128 → R64 → R32 → R16 → QF → SF → FINAL`

All matches within a stage are one frozen batch. Ratings are read for every
match first; result-dependent deltas are summed and applied only after the
complete batch. `F` and `BR` share `FINAL` because both depend on the
semifinals, but not on each other.

The conservative fallbacks are:

- Davis Cup: one batch for the entire tournament edition;
- unknown round label: one batch for the entire edition;
- a player appearing in multiple editions on the same source date: one batch
  for the entire date.

The last rule is active for the sole observed cross-edition player/date overlap.
No qualifying labels Q1–Q4 occur in this main-tour corpus, but their order is
defined for future-compatible inputs.

## Temporal protocol

Ratings run continuously from 2000 without reset. Periods have distinct roles:

- 2000–2018: rating history only;
- 2019–2022: validation and all model selection;
- 2023 onward: final untouched test.

The global grid is deliberately bounded to 77 combinations:

- `K ∈ {8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 64}`;
- `scale ∈ {250, 300, 350, 400, 450, 500, 600}`.

Selection minimizes validation log loss, then Brier score, then maximizes
accuracy. Test columns are absent from every tuning table. Unit tests verify
that validation guards reject 2023+ rows and that selection ignores any
unrelated test metric column.

## Surface shrinkage

Independent surface Elo is noisy for players with little surface history,
especially on grass. For each player and match surface:

`w = n_surface / (n_surface + C)`

`rating_blended = w × rating_surface + (1 - w) × rating_global`

`C ∈ {5, 10, 20, 30, 50, 75, 100}` is selected on validation only. The two
players can have different weights. An unknown surface falls back exactly to
global Elo.

## Inactivity

The only tested inactivity rule regresses a returning player's global and all
surface ratings toward 1500 before the return match:

`rating_after_decay = 1500 + (rating_before_decay - 1500) × (1 - strength)`

Thresholds `{90, 180, 365}` days and strengths `{0.05, 0.10, 0.20}` are compared
with no decay on validation. The selected rule is kept only because it improves
validation log loss. It is applied once at the first batch after the threshold,
before emitting its pre-match rating.

## Calibration

Platt scaling fits an intercept and slope to the logit of the selected raw
probability using 2019–2022 only. The learned transform is evaluated on 2023+
without refitting. Isotonic calibration was intentionally omitted to avoid an
extra, more flexible choice when the simple monotonic transform suffices.

The processed parquet stores calibrated probabilities only on test rows.
Historical rows remain null in that column so they cannot be mistaken for
strict out-of-sample calibration predictions.

## Outputs

- `data/processed/match_ratings_v2.parquet`: pre-match ratings and probabilities;
- `reports/round_chronology_audit.{md,json}`: source-round audit and fallbacks;
- `reports/elo_global_tuning.csv`: global grid, validation metrics only;
- `reports/elo_surface_shrinkage_tuning.csv`: shrinkage grid;
- `reports/elo_inactivity_tuning.csv`: inactivity comparison;
- `reports/elo_phase3_calibration.csv`: favorite-probability bins on test;
- `reports/elo_annual_walk_forward.csv`: annual rating diagnostics;
- `reports/phase3_test_comparison.csv`: final test model table;
- `reports/phase3_atp_overlap.csv`: accuracy appariée sur les matchs avec rangs ATP comparables;
- `reports/phase3_test_segments.csv`: test results by surface and level.

The annual file is walk-forward in rating state, but rows before 2019 are
retrospective diagnostics because the fixed hyperparameters were chosen later.
Platt is excluded from that file to avoid presenting in-sample or backward
calibration as walk-forward evidence.

## Anti-leakage guarantees

Tests cover frozen same-round features, next-round updates, future-result
independence, conservative Davis Cup and overlap batches, pre-return inactivity
decay, validation-only guards and validation-only parameter selection. The
real-output audit additionally checks unique match identifiers, probability
bounds and the 2023 boundary for calibrated values.
