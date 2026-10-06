#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis_analytics.shots.analysis import run_descriptive_analysis
from tennis_analytics.shots.build import build_shot_datasets
from tennis_analytics.shots.modeling import (
    build_point_v2_predictions,
    enrich_checkpoints,
    evaluate_next_shot,
    train_shot_models,
)
from tennis_analytics.shots.reporting import write_audit_reports, write_phase9_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and evaluate Phase 9 shot-level V1")
    parser.add_argument(
        "--resume-analysis",
        action="store_true",
        help="Reuse canonical parquet files and rerun predictions, analyses and reports",
    )
    parser.add_argument(
        "--reports-only",
        action="store_true",
        help="Regenerate markdown reports from existing artifacts",
    )
    args = parser.parse_args()
    if args.reports_only:
        audit = write_audit_reports()
        summary = {
            "audit": audit,
            "descriptive": {
                "players_with_fingerprint": len(
                    pd.read_csv(ROOT / "reports/player_style_fingerprints.csv")
                ),
                "selected_clusters": pd.read_csv(ROOT / "reports/player_style_clusters.csv")[
                    "cluster"
                ].nunique(),
                "pressure_points_with_exact_flags": int(
                    pd.read_csv(ROOT / "reports/shot_pressure_comparison.csv")["points"].sum()
                ),
            },
        }
        print(json.dumps(write_phase9_report(summary), indent=2))
        return
    result: dict[str, object] = {}
    if not args.resume_analysis:
        result["build"] = build_shot_datasets()
    result["point_v2"] = build_point_v2_predictions()
    result["checkpoints"] = enrich_checkpoints()
    result["descriptive"] = run_descriptive_analysis()
    result["shot_model"] = train_shot_models()
    result["next_shot"] = evaluate_next_shot()
    result["audit"] = write_audit_reports()
    result["decision"] = write_phase9_report(result)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
