# Local data directory

Only this documentation, the source manifest, and directory placeholders belong in Git.
Downloaded and generated tennis datasets are intentionally ignored because the complete local
working set is large and its source material is governed by third-party licences.

| Path | Purpose | Git policy |
| --- | --- | --- |
| `raw/` | ATP source CSV files | ignored |
| `external/` | Grand Slam point-by-point and Match Charting Project files | ignored |
| `interim/` | identity and normalization diagnostics | ignored |
| `processed/` | normalized tables, feature matrices, and research artifacts | ignored |
| `runtime/` | HTTP cache for bounded public-data refreshes | ignored |
| `product/` | local SQLite state and prediction ledger | ignored |
| `prospective/` | local prospective prediction registry | ignored |

See the root README for reconstruction commands and `docs/data_sources.md` for provenance and
licensing details. Do not add ignored datasets with `git add -f` without a separate licence and
size review.
