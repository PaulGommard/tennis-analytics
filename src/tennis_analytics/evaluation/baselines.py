from __future__ import annotations

import argparse
import json
import logging
import math
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tennis_analytics.paths import PROCESSED_DIR, PROJECT_ROOT, REPORTS_DIR
from tennis_analytics.ratings.elo import EloConfig

LOGGER = logging.getLogger(__name__)
MODEL_NAMES = ("ATP Ranking", "Global Elo", "Surface Elo")
LEVEL_LABELS = {
    "G": "Grand Slam",
    "M": "Masters",
    "A": "Other tour-level",
    "D": "Davis Cup",
    "F": "Tour Finals",
    "O": "Olympics",
}


def load_split_config(path: Path) -> dict[str, pd.Timestamp]:
    with path.open("rb") as stream:
        values = tomllib.load(stream)["evaluation"]
    return {key: pd.Timestamp(value) for key, value in values.items()}


def assign_periods(features: pd.DataFrame, split: dict[str, pd.Timestamp]) -> pd.Series:
    dates = pd.to_datetime(features["date"])
    conditions = [
        dates <= split["history_end"],
        dates.between(split["validation_start"], split["validation_end"]),
        dates >= split["test_start"],
    ]
    return pd.Series(
        np.select(conditions, ["history", "validation", "test"], default="unassigned"),
        index=features.index,
        dtype="string",
    )


def _probability_metrics(y: pd.Series, probability: pd.Series) -> tuple[float, float]:
    probability = probability.astype(float).clip(1e-15, 1 - 1e-15)
    y = y.astype(float)
    log_loss = -float((y * np.log(probability) + (1 - y) * np.log(1 - probability)).mean())
    brier = float(((probability - y) ** 2).mean())
    return log_loss, brier


def metrics_for(frame: pd.DataFrame, model: str) -> dict[str, Any]:
    y = frame["player_A_win"].astype(int)
    if model == "ATP Ranking":
        mask = (
            frame["rank_A"].notna() & frame["rank_B"].notna() & (frame["rank_A"] != frame["rank_B"])
        )
        prediction = frame.loc[mask, "rank_A"] < frame.loc[mask, "rank_B"]
        return {
            "model": model,
            "rows": len(frame),
            "accuracy_n": int(mask.sum()),
            "accuracy": float((prediction.astype(int) == y.loc[mask]).mean())
            if mask.any()
            else np.nan,
            "probability_n": 0,
            "log_loss": np.nan,
            "brier": np.nan,
        }
    if model == "Global Elo":
        probability_column = "elo_expected_A"
        difference_column = "elo_diff"
    elif model == "Surface Elo":
        probability_column = "surface_elo_expected_A"
        difference_column = "surface_elo_diff"
    else:
        raise ValueError(f"Unknown baseline: {model}")
    probability_mask = frame[probability_column].notna()
    accuracy_mask = probability_mask & frame[difference_column].ne(0)
    prediction = frame.loc[accuracy_mask, difference_column] > 0
    probability = frame.loc[probability_mask, probability_column]
    log_loss, brier = (
        _probability_metrics(y.loc[probability_mask], probability)
        if probability_mask.any()
        else (np.nan, np.nan)
    )
    return {
        "model": model,
        "rows": len(frame),
        "accuracy_n": int(accuracy_mask.sum()),
        "accuracy": (
            float((prediction.astype(int) == y.loc[accuracy_mask]).mean())
            if accuracy_mask.any()
            else np.nan
        ),
        "probability_n": int(probability_mask.sum()),
        "log_loss": log_loss,
        "brier": brier,
    }


def evaluate(features: pd.DataFrame, split: dict[str, pd.Timestamp]) -> pd.DataFrame:
    data = features.copy()
    data["period"] = assign_periods(data, split)
    rows: list[dict[str, Any]] = []
    periods = {"all": data}
    periods.update({period: group for period, group in data.groupby("period", sort=False)})
    for period, period_frame in periods.items():
        segments: list[tuple[str, str, pd.DataFrame]] = [("overall", "All", period_frame)]
        segments.extend(
            ("surface", str(surface), group)
            for surface, group in period_frame.groupby("surface", dropna=False, sort=True)
        )
        segments.extend(
            ("level", LEVEL_LABELS.get(str(level), str(level)), group)
            for level, group in period_frame.groupby("level", dropna=False, sort=True)
        )
        for segment_type, segment, frame in segments:
            for model in MODEL_NAMES:
                rows.append(
                    {
                        "period": period,
                        "segment_type": segment_type,
                        "segment": segment,
                        **metrics_for(frame, model),
                    }
                )
    return pd.DataFrame(rows)


def calibration_table(features: pd.DataFrame, period: str = "test") -> pd.DataFrame:
    data = features.copy()
    if "period" in data and period != "all":
        data = data.loc[data["period"] == period]
    edges = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.0000001]
    labels = ["50-55%", "55-60%", "60-65%", "65-70%", "70-75%", "75-80%", "80-90%", "90-100%"]
    rows: list[pd.DataFrame] = []
    for model, probability_column in (
        ("Global Elo", "elo_expected_A"),
        ("Surface Elo", "surface_elo_expected_A"),
    ):
        subset = data.loc[data[probability_column].notna()].copy()
        subset = subset.loc[subset[probability_column] != 0.5]
        probability = subset[probability_column].astype(float)
        subset["favorite_probability"] = np.maximum(probability, 1 - probability)
        subset["favorite_won"] = np.where(
            probability > 0.5,
            subset["player_A_win"],
            1 - subset["player_A_win"],
        )
        subset["bin"] = pd.cut(
            subset["favorite_probability"], edges, labels=labels, right=False, include_lowest=True
        )
        grouped = (
            subset.groupby("bin", observed=False)
            .agg(
                n=("favorite_won", "size"),
                mean_predicted_probability=("favorite_probability", "mean"),
                actual_favorite_win_rate=("favorite_won", "mean"),
            )
            .reset_index()
        )
        grouped.insert(0, "model", model)
        grouped.insert(0, "period", period)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def _safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{key: _safe(value) for key, value in row.items()} for row in frame.to_dict("records")]


def format_report(
    features: pd.DataFrame,
    timeline: pd.DataFrame,
    metrics: pd.DataFrame,
    calibration: pd.DataFrame,
    split: dict[str, pd.Timestamp],
    elo_config: EloConfig,
) -> str:
    test_overall = metrics.query("period == 'test' and segment_type == 'overall'")
    test_surfaces = metrics.query(
        "period == 'test' and segment_type == 'surface' and segment in ['Hard', 'Clay', 'Grass']"
    )
    test_levels = metrics.query("period == 'test' and segment_type == 'level'")
    exclusions = timeline["exclusion_reason"].value_counts(dropna=True)
    periods = assign_periods(features, split).value_counts()
    player_count = pd.concat([features.player_A_id, features.player_B_id]).nunique()
    new_player_matches = int(
        ((features["career_matches_A"] == 0) | (features["career_matches_B"] == 0)).sum()
    )
    rank_missing = int((features["rank_A"].isna() | features["rank_B"].isna()).sum())
    unknown_surface = int(features["surface"].isna().sum())
    lines = [
        "# Phase 2 — Temporal baseline report",
        "",
        "## Dataset",
        "",
        f"- Eligible matches: {len(features):,}",
        f"- Players: {player_count:,}",
        f"- Coverage: {features.date.min().date()} to {features.date.max().date()}",
        f"- Date batches: {features.date.nunique():,}",
        f"- History through {split['history_end'].date()}: {periods.get('history', 0):,} matches",
        f"- Validation {split['validation_start'].date()}–{split['validation_end'].date()}: {periods.get('validation', 0):,} matches",
        f"- Test from {split['test_start'].date()}: {periods.get('test', 0):,} matches",
        "",
        "## Elo configuration",
        "",
        f"- Initial rating: {elo_config.initial_rating:g}",
        f"- K factor: {elo_config.k_factor:g}",
        f"- Scale: {elo_config.scale:g}",
        "- Formula: `1 / (1 + 10 ** ((Elo_B - Elo_A) / scale))`",
        "- New players: fixed initial rating; no inactivity decay",
        f"- Surface systems: {', '.join(elo_config.surfaces)}; Unknown is not updated",
        "- Carpet remains separate rather than being merged into Hard",
        "",
        "## Test baselines",
        "",
        "| Model | Accuracy | Accuracy n | Log loss | Brier | Probability n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in test_overall.itertuples():
        log_loss = "—" if pd.isna(row.log_loss) else f"{row.log_loss:.4f}"
        brier = "—" if pd.isna(row.brier) else f"{row.brier:.4f}"
        lines.append(
            f"| {row.model} | {row.accuracy:.2%} | {row.accuracy_n:,} | "
            f"{log_loss} | {brier} | {row.probability_n:,} |"
        )
    lines.extend(
        (
            "",
            "## Test accuracy by surface",
            "",
            "| Surface | Model | Accuracy | n |",
            "|---|---|---:|---:|",
        )
    )
    for row in test_surfaces.itertuples():
        lines.append(f"| {row.segment} | {row.model} | {row.accuracy:.2%} | {row.accuracy_n:,} |")
    lines.extend(
        (
            "",
            "## Test accuracy by event level",
            "",
            "The source cannot reliably separate ATP 250 from ATP 500: both are level `A`.",
            "",
            "| Level | Model | Accuracy | n |",
            "|---|---|---:|---:|",
        )
    )
    for row in test_levels.itertuples():
        lines.append(f"| {row.segment} | {row.model} | {row.accuracy:.2%} | {row.accuracy_n:,} |")
    lines.extend(
        (
            "",
            "## Data quality",
            "",
            f"- Rows missing at least one rank: {rank_missing:,} ({rank_missing / len(features):.2%})",
            f"- Rows with unknown surface: {unknown_surface:,}",
            f"- Matches involving at least one player with zero prior matches: {new_player_matches:,}",
            f"- Player A win rate: {features['player_A_win'].mean():.2%} (A/B is outcome-independent)",
            "- Rank convention: `rank_diff = rank_B - rank_A`; positive means A is better ranked",
            "- Retirement results are retained; non-played/administrative outcomes are excluded",
            "",
            "## Exclusions",
            "",
        )
    )
    lines.extend(f"- {reason}: {count:,}" for reason, count in exclusions.items())
    lines.extend(
        (
            "",
            "## Temporal safeguards",
            "",
            "All matches sharing the same tournament-start date are evaluated from one frozen pre-date state. Ratings, career counts and rolling history are updated only after the complete date batch. No random split is used.",
            "",
            "## Calibration",
            "",
            "| Model | Bin | n | Mean predicted | Actual favorite win rate |",
            "|---|---|---:|---:|---:|",
        )
    )
    for row in calibration.itertuples():
        lines.append(
            f"| {row.model} | {row.bin} | {row.n:,} | "
            f"{row.mean_predicted_probability:.2%} | {row.actual_favorite_win_rate:.2%} |"
        )
    return "\n".join(lines) + "\n"


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Evaluate ranking and Elo baselines temporally")
    command.add_argument(
        "--features", type=Path, default=PROCESSED_DIR / "match_features_v1.parquet"
    )
    command.add_argument(
        "--timeline", type=Path, default=PROCESSED_DIR / "analytics" / "match_timeline.parquet"
    )
    command.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "elo.toml")
    command.add_argument("--report-dir", type=Path, default=REPORTS_DIR)
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    features = pd.read_parquet(args.features)
    timeline = pd.read_parquet(args.timeline)
    split = load_split_config(args.config)
    elo_config = EloConfig.from_toml(args.config)
    features["period"] = assign_periods(features, split)
    metrics = evaluate(features.drop(columns="period"), split)
    calibration = calibration_table(features, "test")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.report_dir / "baseline_metrics_v1.csv", index=False)
    calibration.to_csv(args.report_dir / "elo_calibration_v1.csv", index=False)
    report = format_report(features, timeline, metrics, calibration, split, elo_config)
    (args.report_dir / "phase2_report.md").write_text(report, encoding="utf-8")
    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "splits": {key: value.date().isoformat() for key, value in split.items()},
        "metrics": _records(metrics),
        "calibration": _records(calibration),
    }
    (args.report_dir / "baseline_evaluation_v1.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(report, end="")
    LOGGER.info("Evaluation reports written to %s", args.report_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
