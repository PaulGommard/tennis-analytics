# Phase 4: historical match statistics and logistic regression

## Scientific question

Phase 4 asks whether strictly historical service, return, form and workload
information adds predictive signal after the frozen Phase 3 Elo baseline is
already known. Only L2 logistic regression is used.

## Data and time protocol

- training: 2000–2018;
- validation and every selection: 2019–2022;
- final test: 2023 onward, through the available 2026 snapshot.

The service block is complete for 91.12% of matches. Early-year coverage is
already close to 87–90%; missingness is concentrated in Davis Cup. Training
therefore starts in 2000. Median imputation and standardization are fit on the
training period only.

All snapshots use the safe Phase 3 round chronology. A match's statistics,
score and duration are applied only after the complete round batch. Circuit
priors, surface counts and workload windows therefore contain earlier batches
only. Ambiguous formats retain conservative whole-event batches.

The ATP source has tournament-start dates rather than exact match dates. Recent
windows include structurally earlier rounds but cannot reconstruct real playing
days. `consecutive_play_days` is deliberately absent.

## Statistical definitions

Rates are ratios of cumulative numerators and cumulative opportunities—not
averages of per-match percentages. Service points won are reconstructed as
first-serve points won plus second-serve points won. Return rates are the
complement of the opponent's service outcomes.

Point-rate shrinkage uses the historical circuit rate available before the
match. Surface service and return rates shrink toward the player's global rate.
The selected strengths are 50 point opportunities and 1000 surface
opportunities, selected on validation only.

Score workload counts explicit standard-set games. A bracketed match tie-break
counts as one set and zero standard games. Partial sets before `RET` count their
recorded games. Ambiguous scores produce missing workload rather than guessed
values. Durations outside 1–360 minutes are excluded.

## Models

The ablation begins with Elo alone, then independently adds ranking, career
serve, career return, rolling form, workload and experience. The final model
uses 50 difference features. Six L2 strengths are tried per block, with log
loss as the primary selection criterion.

Platt calibration is fitted on validation after the final block and L2 strength
are frozen. The serialized artifact retains the train-fitted imputer, scaler,
logistic model, calibrator and exact ordered feature list.

## Interpretation

Because career and rolling serve/return variables are correlated, standardized
coefficients are conditional and can change sign. Block ablation is the primary
evidence for incremental signal; coefficients are secondary diagnostics.

See `reports/phase4_report.md` for results, `docs/features_v2.md` for the feature
dictionary, and `tests/test_phase4.py` for hand-calculated anti-leakage cases.
