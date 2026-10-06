from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tennis_analytics.paths import PROCESSED_DIR, PROJECT_ROOT, REPORTS_DIR
from tennis_analytics.ratings.elo import EloConfig, expected_score


def audit_features(features: pd.DataFrame, config: EloConfig | None = None) -> dict[str, object]:
    config = config or EloConfig()
    dates = pd.to_datetime(features["date"], errors="coerce")
    expected = features.apply(
        lambda row: expected_score(float(row["elo_A"]), float(row["elo_B"]), config.scale), axis=1
    )
    checks: dict[str, int] = {
        "duplicate_match_ids": int(features["match_id"].duplicated(keep=False).sum()),
        "invalid_dates": int(dates.isna().sum()),
        "non_chronological_rows": int((dates.diff().dropna() < pd.Timedelta(0)).sum()),
        "target_outside_binary": int((~features["player_A_win"].isin([0, 1])).sum()),
        "global_probability_formula_mismatch": int(
            (~np.isclose(features["elo_expected_A"], expected, atol=1e-12)).sum()
        ),
        "rank_diff_mismatch": int(
            (
                features["rank_diff"].notna()
                & ~np.isclose(
                    features["rank_diff"], features["rank_B"] - features["rank_A"], equal_nan=True
                )
            ).sum()
        ),
        "negative_days_since_last_match": int(
            (
                (features["days_since_last_match_A"] < 0)
                | (features["days_since_last_match_B"] < 0)
            ).sum()
        ),
        "probability_outside_unit_interval": int(
            (
                (features["elo_expected_A"] < 0)
                | (features["elo_expected_A"] > 1)
                | (features["surface_elo_expected_A"].dropna() < 0).reindex(
                    features.index, fill_value=False
                )
                | (features["surface_elo_expected_A"].dropna() > 1).reindex(
                    features.index, fill_value=False
                )
            ).sum()
        ),
    }
    for slot in ("A", "B"):
        for days in (30, 90, 365):
            matches = features[f"matches_{days}d_{slot}"]
            wins = features[f"wins_{days}d_{slot}"]
            rates = features[f"win_rate_{days}d_{slot}"]
            checks[f"rolling_wins_exceed_matches_{days}d_{slot}"] = int((wins > matches).sum())
            checks[f"rolling_zero_history_not_nan_{days}d_{slot}"] = int(
                ((matches == 0) & rates.notna()).sum()
            )
            checks[f"rolling_rate_mismatch_{days}d_{slot}"] = int(
                ((matches > 0) & ~np.isclose(rates, wins / matches, equal_nan=True)).sum()
            )
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rows": len(features),
        "checks": checks,
        "passed": all(value == 0 for value in checks.values()),
    }


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Audit temporal feature invariants")
    command.add_argument(
        "--features", type=Path, default=PROCESSED_DIR / "match_features_v1.parquet"
    )
    command.add_argument("--report", type=Path, default=REPORTS_DIR / "feature_audit_v1.json")
    command.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "elo.toml")
    command.add_argument("--strict", action="store_true")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = audit_features(pd.read_parquet(args.features), EloConfig.from_toml(args.config))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 1 if args.strict and not result["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
