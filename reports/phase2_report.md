# Phase 2 — Temporal baseline report

## Dataset

- Eligible matches: 78,845
- Players: 2,789
- Coverage: 2000-01-03 to 2026-05-25
- Date batches: 1,209
- History through 2018-12-31: 58,703 matches
- Validation 2019-01-01–2022-12-31: 9,766 matches
- Test from 2023-01-01: 10,376 matches

## Elo configuration

- Initial rating: 1500
- K factor: 32
- Scale: 400
- Formula: `1 / (1 + 10 ** ((Elo_B - Elo_A) / scale))`
- New players: fixed initial rating; no inactivity decay
- Surface systems: Hard, Clay, Grass, Carpet; Unknown is not updated
- Carpet remains separate rather than being merged into Hard

## Test baselines

| Model | Accuracy | Accuracy n | Log loss | Brier | Probability n |
|---|---:|---:|---:|---:|---:|
| ATP Ranking | 63.87% | 10,188 | — | — | 0 |
| Global Elo | 63.99% | 10,341 | 0.6334 | 0.2214 | 10,376 |
| Surface Elo | 63.05% | 10,256 | 0.6392 | 0.2243 | 10,324 |

## Test accuracy by surface

| Surface | Model | Accuracy | n |
|---|---|---:|---:|
| Clay | ATP Ranking | 63.32% | 3,318 |
| Clay | Global Elo | 62.80% | 3,347 |
| Clay | Surface Elo | 62.07% | 3,330 |
| Grass | ATP Ranking | 63.06% | 934 |
| Grass | Global Elo | 66.17% | 949 |
| Grass | Surface Elo | 59.02% | 942 |
| Hard | ATP Ranking | 64.24% | 5,901 |
| Hard | Global Elo | 64.39% | 5,999 |
| Hard | Surface Elo | 64.22% | 5,984 |

## Test accuracy by event level

The source cannot reliably separate ATP 250 from ATP 500: both are level `A`.

| Level | Model | Accuracy | n |
|---|---|---:|---:|
| Other tour-level | ATP Ranking | 62.11% | 5,020 |
| Other tour-level | Global Elo | 63.07% | 5,034 |
| Other tour-level | Surface Elo | 61.75% | 5,025 |
| Davis Cup | ATP Ranking | 64.12% | 694 |
| Davis Cup | Global Elo | 60.44% | 824 |
| Davis Cup | Surface Elo | 59.65% | 751 |
| Tour Finals | ATP Ranking | 61.33% | 75 |
| Tour Finals | Global Elo | 53.33% | 75 |
| Tour Finals | Surface Elo | 61.33% | 75 |
| Grand Slam | ATP Ranking | 69.09% | 1,760 |
| Grand Slam | Global Elo | 70.42% | 1,765 |
| Grand Slam | Surface Elo | 68.76% | 1,764 |
| Masters | ATP Ranking | 63.51% | 2,579 |
| Masters | Global Elo | 62.67% | 2,580 |
| Masters | Surface Elo | 62.68% | 2,578 |
| Olympics | ATP Ranking | 73.33% | 60 |
| Olympics | Global Elo | 69.84% | 63 |
| Olympics | Surface Elo | 63.49% | 63 |

## Data quality

- Rows missing at least one rank: 1,850 (2.35%)
- Rows with unknown surface: 52
- Matches involving at least one player with zero prior matches: 3,279
- Player A win rate: 49.73% (A/B is outcome-independent)
- Rank convention: `rank_diff = rank_B - rank_A`; positive means A is better ranked
- Retirement results are retained; non-played/administrative outcomes are excluded

## Exclusions

- walkover: 423
- default: 20
- abandoned: 11

## Temporal safeguards

All matches sharing the same tournament-start date are evaluated from one frozen pre-date state. Ratings, career counts and rolling history are updated only after the complete date batch. No random split is used.

## Calibration

| Model | Bin | n | Mean predicted | Actual favorite win rate |
|---|---|---:|---:|---:|
| Global Elo | 50-55% | 1,720 | 52.49% | 52.15% |
| Global Elo | 55-60% | 1,559 | 57.46% | 54.59% |
| Global Elo | 60-65% | 1,478 | 62.45% | 58.46% |
| Global Elo | 65-70% | 1,373 | 67.44% | 62.64% |
| Global Elo | 70-75% | 1,214 | 72.50% | 66.72% |
| Global Elo | 75-80% | 1,046 | 77.41% | 72.85% |
| Global Elo | 80-90% | 1,430 | 84.51% | 77.27% |
| Global Elo | 90-100% | 521 | 93.56% | 89.83% |
| Surface Elo | 50-55% | 2,026 | 52.52% | 51.43% |
| Surface Elo | 55-60% | 1,727 | 57.46% | 54.14% |
| Surface Elo | 60-65% | 1,472 | 62.46% | 60.87% |
| Surface Elo | 65-70% | 1,350 | 67.41% | 63.56% |
| Surface Elo | 70-75% | 1,121 | 72.35% | 67.89% |
| Surface Elo | 75-80% | 879 | 77.40% | 69.28% |
| Surface Elo | 80-90% | 1,267 | 84.30% | 78.77% |
| Surface Elo | 90-100% | 414 | 93.45% | 88.65% |
