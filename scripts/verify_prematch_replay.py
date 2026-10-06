#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis_analytics.features.opponent_adjusted import (
    build_opponent_history_base,
    materialize_phase5_features,
)
from tennis_analytics.features.stat_history import build_stat_history_base, materialize_features


def main() -> None:
    processed = ROOT / "data" / "processed"
    ratings = pd.read_parquet(processed / "match_ratings_v2.parquet")
    timeline = pd.read_parquet(processed / "analytics" / "match_timeline.parquet")
    matches = pd.read_parquet(processed / "atp" / "matches.parquet")
    stats = pd.read_parquet(processed / "atp" / "match_stats.parquet")
    expected = pd.read_parquet(processed / "match_features_v3.parquet")
    base4 = build_stat_history_base(ratings, timeline, matches, stats)
    rebuilt2 = materialize_features(base4, point_prior_strength=50.0, surface_prior_strength=1000.0)
    base5 = build_opponent_history_base(rebuilt2, ratings, timeline, matches, stats)
    rebuilt = materialize_phase5_features(
        rebuilt2,
        base5,
        adjusted_point_prior=500.0,
        adjusted_surface_prior=100.0,
        h2h_prior=5.0,
    )
    if list(rebuilt["match_id"]) != list(expected["match_id"]):
        raise AssertionError("Replay match order differs from frozen research dataset.")
    numeric = expected.select_dtypes(include="number").columns
    left = rebuilt[numeric].to_numpy(dtype=float)
    right = expected[numeric].to_numpy(dtype=float)
    nan_equal = np.array_equal(np.isnan(left), np.isnan(right))
    max_delta = float(np.nanmax(np.abs(left - right)))
    exact = nan_equal and np.allclose(left, right, rtol=0.0, atol=0.0, equal_nan=True)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "rows": len(rebuilt),
        "columns": len(rebuilt.columns),
        "numeric_columns_compared": len(numeric),
        "nan_pattern_equal": nan_equal,
        "max_absolute_delta": max_delta,
        "exact_match": exact,
        "temporal_statement": (
            "Every snapshot is read before its chronology batch is updated; target outcomes "
            "and same-batch outcomes are therefore unavailable at prediction time."
        ),
    }
    path = ROOT / "reports" / "phase7a_historical_replay.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not exact:
        raise SystemExit("Historical replay is not exact.")


if __name__ == "__main__":
    main()
