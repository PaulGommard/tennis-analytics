# Tennis Analytics

End-to-end research platform for ATP tennis analytics and probabilistic prediction. It combines
leakage-safe historical feature engineering, frozen pre-match and live models, tournament
simulation, shot-level analysis, and a local FastAPI web interface.

The project estimates probabilities; it does not claim certainty or guaranteed betting returns.

## What it does

- estimates pre-match win probabilities for historical or upcoming ATP matches;
- lists upcoming matches from public ESPN pages and refreshes one prediction on demand;
- estimates next-point, game, set, and match probabilities during a match;
- simulates tournament brackets and progression probabilities;
- analyses serve direction, returns, shot sequences, tactical fingerprints, and Markov
  transitions;
- preserves frozen artifacts, temporal splits, feature snapshots, and audit reports.

## Architecture

```text
Historical ATP data
        |
        v
Normalization -> leakage-safe features -> Elo / rankings / form / workload
        |                                      |
        +--------------------------------------+
                                               v
                                         PRE-MATCH V1
                                               |
                         +---------------------+---------------------+
                         v                                           v
              Upcoming match predictor                    Tournament engine
                         |                              P(R16/QF/SF/F/Champion)
                         v
                 FastAPI + local web UI

Point-by-point data -> POINT V2 -> exact scoring/Markov engine -> GAME / SET V2
                                           |
                                           v
                               MATCH V3 stabilized blend

Match Charting Project -> SHOT V1 -> patterns / transitions / player fingerprints
```

The application is local-first: FastAPI serves a build-free HTML/CSS/JavaScript frontend and
stores runtime state in an ignored SQLite database. No cron job or continuous ingestion process
is configured.

## Models and benchmark results

All reported test periods are temporal holdouts. Exact methodology is available in `reports/`.

| Component | Role | Frozen test result | Status |
| --- | --- | --- | --- |
| **PRE-MATCH V1** | Winner before the match; calibrated logistic regression, 86 features | 2023–2026: accuracy **0.657382**, log loss **0.609876**, Brier **0.211585**, ROC-AUC **0.724285** | Official baseline |
| **POINT V2** | Next-point probability with strongly shrunk in-match adaptation | log loss **0.645859**, Brier **0.226880**, ROC-AUC **0.561445** | Official point component |
| **GAME V2** | Current-game probability through the exact scoring engine | log loss **0.422358**, Brier **0.135284** | Official live component |
| **SET V2** | Current-set probability through the exact scoring engine | log loss **0.491959**, Brier **0.165630** | Official live component |
| **MATCH V3** | Stable/adaptive blend followed by the Markov engine | log loss **0.455247**, Brier **0.151185**, ROC-AUC **0.862860** | Official live match component |
| **Tournament Adapter V1** | Context correction during a tournament | final log loss 0.609692 vs 0.609876 baseline, but no robust overall gain | Research only, disabled |
| **SHOT V1** | Shot patterns and simple next-shot/point models | evaluated on the MCP subset; see Phase 9 report | Exploratory |

### PRE-MATCH V1

PRE-MATCH V1 is the official pre-match model. Its 86 inputs cover:

- Elo and surface-adjusted Elo;
- ATP rank and ranking points;
- career, 365-day, and 90-day form;
- serve and return performance;
- surface experience and opponent quality;
- recent workload, rest, and tournament workload;
- age, experience, and head-to-head context.

The serialized package includes median imputation, standardization, logistic coefficients, and
Platt calibration. The model artifact is versioned in `models/prematch_v1/`.

### Live engine

POINT V2 predicts the next point. The deterministic tennis scoring engine propagates this to
game and set probabilities. MATCH V3 progressively mixes stable and adaptive service estimates:

```text
w(n) = n / (n + 200)
p_service_V3 = (1 - w) * p_service_V1 + w * p_service_V2
```

This deliberately slow adaptation limits overreaction early in a match. The official horizon
mapping is POINT V2 -> GAME V2 -> SET V2 -> MATCH V3.

### Tournament engine

The tournament engine simulates the remaining bracket with PRE-MATCH V1 and reports progression
probabilities such as `P(R16)`, `P(QF)`, `P(SF)`, `P(Final)`, and `P(Champion)`.

Tournament Context Adapter V1 is retained for reproducibility but is **not** enabled in the
official application: its small historical gain was not robust across tournaments, surfaces,
and metrics.

### Shot analytics

SHOT V1 parses Match Charting Project sequences to study serve direction, return and serve+1
patterns, rally length, n-grams, Markov transitions, style clusters, and player fingerprints.
More complex sequence models are intentionally paused; current results are associative and only
cover the charted subset.

## Project structure

```text
tennis-analytics/
├── backend/                    # FastAPI routes
├── config/                     # versioned model and source configuration
├── data/                       # local datasets/runtime state; contents ignored
├── docs/                       # architecture, schemas, sources, methodology
├── frontend/                   # build-free local web interface
├── models/                     # small frozen model artifacts
├── reports/                    # research and audit results
├── scripts/                    # ingestion, training, evaluation, inference
├── src/tennis_analytics/
│   ├── ingestion/              # safe downloads and manifests
│   ├── preprocessing/          # normalization and parsing
│   ├── features/               # temporal and opponent-adjusted features
│   ├── ratings/                # Elo engines
│   ├── modeling/               # pre-match and point models
│   ├── scoring/                # deterministic tennis scoring
│   ├── live/                   # live probability propagation
│   ├── tournament/             # bracket simulation and adapter research
│   ├── shots/                  # shot-level analytics
│   └── product/                # local product services and persistence
└── tests/
```

## Requirements

- Python **3.11 or newer** (declared in `pyproject.toml`);
- internet access for data downloads and the optional upcoming-match refresh;
- enough local storage for research datasets (the complete working dataset is about 2.8 GB).

The frontend has no Node.js dependency or build step. `pyproject.toml` is the only dependency
source of truth.

## Quick start

```bash
git clone https://github.com/<USERNAME>/tennis-analytics.git
cd tennis-analytics

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The frozen models are included, but the third-party and generated datasets are not. Complete the
data setup below before requesting predictions or historical replays.

Start the local application:

```bash
tennis-product --reload
```

Equivalent repository script:

```bash
python scripts/run_product_mvp.py --reload
```

Open:

- application: <http://127.0.0.1:8000/>
- upcoming ATP matches: <http://127.0.0.1:8000/upcoming>
- interactive API documentation: <http://127.0.0.1:8000/docs>
- health endpoint: <http://127.0.0.1:8000/api/health>

## Data setup

Large raw and processed datasets are intentionally excluded from Git. The source collections are
third-party data under their own terms, and the generated Parquet files can be rebuilt.

### Minimum pre-match dataset

The following pipeline downloads ATP results and reconstructs the files needed for PRE-MATCH V1
and on-demand predictions:

```bash
python scripts/download_atp_data.py --start-year 2000 --end-year 2026
python scripts/build_normalized_data.py --start-year 2000 --end-year 2026
python scripts/build_features_v1.py
python scripts/run_phase3.py
python scripts/run_phase4.py
python scripts/run_phase5.py
```

These jobs are research workloads, not an instant application bootstrap. They write only beneath
ignored `data/` paths and generated report paths. The included `models/prematch_v1/model.joblib`
remains the official artifact; rebuilding Phase 5 creates a research artifact under
`data/processed/models/`.

### Point, live, and shot datasets

```bash
# Grand Slam point-by-point data
python scripts/download_external_data.py \
  --dataset slam-pointbypoint --start-year 2011 --end-year 2024

# Match Charting Project data and aggregate tables
python scripts/download_external_data.py --dataset mcp --include-mcp-stats

# Normalize point data, then build V1/V2 features
python scripts/audit_point_by_point.py
python scripts/build_point_features_v1.py
python scripts/build_point_features_v2.py
```

Use `python scripts/verify_manifest.py` to verify downloaded files against the local manifest.
See [`docs/data_sources.md`](docs/data_sources.md) and
[`docs/point_by_point_schema.md`](docs/point_by_point_schema.md) before redistributing any data.

## Using the web application

### Upcoming matches

`/upcoming` requests the next ATP singles matches from public ESPN pages. Merely opening the page
does not run model inference. Each match card links to `/matches/{match_id}` and has an explicit
**Refresh & calculate prediction** action.

### On-demand prediction flow

```text
User clicks refresh
        |
        v
Fetch rankings and recent public results for two players
        |
        v
Resolve identities against the historical ATP dataset
        |
        v
Refresh runtime state and construct the 86 PRE-MATCH V1 features
        |
        v
Frozen PRE-MATCH V1 inference and local SQLite snapshot
```

There is no scheduled background ingestion. Runtime HTTP responses and the SQLite ledger are
local, ignored by Git, and recreated as needed.

### Match details

`/matches/{match_id}` displays, when available:

- players, tournament, round, surface, rank, and Elo;
- recent form, workload, surface experience, and serve/return comparisons;
- PRE-MATCH V1 win probabilities and leading log-odds contributions;
- `HIGH`, `MEDIUM`, or `LOW` data-quality status and freshness warnings;
- the immutable feature hash and a collapsible snapshot of all 86 inputs.

The current free runtime path normally reports `MEDIUM` or `LOW`, not `HIGH`.

### Command-line prediction

After completing the minimum data setup:

```bash
python scripts/predict_match.py \
  --player-a "Jannik Sinner" \
  --player-b "Carlos Alcaraz" \
  --surface hard \
  --date 2026-10-10 \
  --snapshot
```

Add `--record` only when deliberately creating an immutable local prospective record.

## Data quality and runtime audit

Public runtime data has explicit limitations:

- ESPN may expose a tournament start date rather than the exact day of each match;
- recent match durations can be unavailable;
- serve/return snapshots can be older than rankings and result history;
- unresolved values pass through the frozen training-time median imputer;
- identity resolution is conservative and refuses ambiguous names.

A targeted runtime audit found and corrected hybrid workload values, use of a snapshot from the
wrong surface, and duplicate counting of ESPN matches. PRE-MATCH V1 itself was not changed or
retrained. See the [Faria–Gea prediction audit](reports/faria_gea_prediction_audit.md).

## Development and validation

```bash
pytest
ruff check .
```

Useful focused commands:

```bash
python scripts/verify_prematch_replay.py
python scripts/evaluate_live_v2.py
python scripts/run_phase8c.py
python scripts/run_tournament_adapter_v1.py
python scripts/run_phase9.py --reports-only
```

Some full research commands require the optional `phase6` dependencies:

```bash
python -m pip install -e '.[dev,phase6]'
```

## Data sources and credits

- Jeff Sackmann ATP results, currently consumed through an archival mirror;
- Jeff Sackmann Grand Slam point-by-point collection, through the documented archive;
- Tennis Abstract Match Charting Project;
- public ESPN pages for bounded, user-triggered upcoming-match refreshes.

The downloaded historical collections are documented locally as CC BY-NC-SA 4.0 and do not
permit unlicensed commercial use. This repository does not claim ownership of them. Full URLs,
snapshot notes, coverage, and caveats are in [`docs/data_sources.md`](docs/data_sources.md) and
`config/sources.toml`.

## Limitations

- This is an experimental research project, not production infrastructure.
- Historical training and evaluation currently focus on men's ATP data.
- Third-party coverage is incomplete and some source dates are approximate.
- A clean clone contains models and code but no large research dataset.
- The public ESPN parser can break if upstream HTML changes.
- Tournament simulations assume the specified bracket and model probabilities; they do not model
  injuries, withdrawals, weather, or every scheduling effect.
- Shot-level results are limited to charted matches and are not causal conclusions.
- Predictions are probabilities, not certainties or betting guarantees.

## Licence and disclaimer

Original source code is available under the [MIT License](LICENSE). Third-party datasets and
derived data remain subject to their respective licences; the MIT licence does not relicense
them.

> This project is intended for research and educational purposes. Predictions are probabilistic
> model outputs and should not be interpreted as guaranteed outcomes or financial advice.
