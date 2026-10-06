# Phase 5: opponent adjustment, schedule and matchup diagnostics

## Scope and frozen baselines

Phase 5 extends `match_features_v2.parquet` without modifying any Phase 3 or
Phase 4 artifact. SHA-256 hashes of the frozen Phase 4 dataset, model, report
and configuration are stored in `reports/phase5_phase4_freeze.json`.

The temporal split remains training through 2018, validation in 2019–2022 and
test from 2023. Only L2 logistic regression is used.

## Opponent-adjusted serve and return

For player A serving against B, the simple pre-match expectation is:

`expected serve win rate A = 1 - historical return win rate B`

For each match, the opportunity-weighted residual is:

`actual service points won - expected rate × service points`

Return is symmetric, using `1 - historical service win rate B`. Residual
numerators and point opportunities are accumulated separately. Their rates
are shrinked toward zero with 500 point opportunities. Surface residuals are
shrinked with 100 opportunities. Both strengths were selected on validation.

The formulation measures performance relative to what an opponent historically
allows. It deliberately avoids fitting a complex nested opponent model.

## Strength of schedule and adjusted form

Schedule uses mean pre-match opponent Elo and `-ATP rank` over 30, 90 and 365
days. Higher rank quality therefore means stronger opposition.

Adjusted form is the rolling sum of:

`actual win - Phase 3 pre-match raw win probability`

It was audited but excluded from the production model because its validation
gain was negligible after the Phase 4 form variables were known.

## H2H and surface

Global H2H uses:

`(wins A + 0.5 × C) / (matches + C) - 0.5`, with `C=5`.

Global H2H provides only a marginal positive validation gain. Surface H2H is
too sparse and is excluded. Explicit Clay and Grass interactions allow Elo,
recent service/return and adjusted residual coefficients to depart from Hard,
which is the reference surface.

## Matchup and feature pruning

The direct proxy `service A - return B` did not improve validation and is not
in production V3. Years since first match and adjusted wins were also removed.
Production V3 contains all 215 V2 columns plus the 36 new columns selected for
the final model.

## Fatigue caveat

ATP dates are tournament-start dates. Phase 4 workload therefore partially
encodes progression through rounds. Phase 5 adds workload from prior tournament
editions only, but retains the old workload because removing it materially
degraded validation. It must be interpreted as a predictive structural proxy,
not a causal fatigue estimate. See `reports/fatigue_validity_audit.md`.

Match Charting Project has 7,565 valid daily dates but only for a selected
charted sample. The downloaded Slam point-by-point match files have no daily
date column. Mixing these partial sources into the main workload definition was
rejected.

## Selection and interpretation

Residual priors, H2H prior, feature blocks, L2 C and Platt calibration are
chosen on validation by log loss, then Brier and accuracy. Standardized
coefficients remain conditional on correlated predictors; the ablation table
is the primary evidence of incremental value.

The anti-leakage tests use hand-calculated batches and verify opponent strength,
Elo expected wins, residuals, H2H, surface H2H, future independence and current
tournament exclusion.
