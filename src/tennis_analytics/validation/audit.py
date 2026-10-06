from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from tennis_analytics.paths import PROCESSED_DIR, REPORTS_DIR
from tennis_analytics.validation.rules import (
    duplicate_primary_keys,
    invalid_rankings,
    invalid_scores,
    negative_statistics,
    self_matches,
    winner_not_in_participants,
)

LOGGER = logging.getLogger(__name__)


def _percentage(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0


def _counts(series: pd.Series) -> dict[str, int]:
    values = series.fillna("Unknown").astype(str).value_counts().sort_index()
    return {key: int(value) for key, value in values.items()}


def _json_value(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def load_tables(input_dir: Path) -> dict[str, pd.DataFrame]:
    required = (
        "players",
        "tournaments",
        "tournament_editions",
        "matches",
        "match_participants",
        "match_stats",
    )
    missing = [name for name in required if not (input_dir / f"{name}.parquet").exists()]
    if missing:
        raise FileNotFoundError(f"Missing normalized table(s): {', '.join(missing)}")
    return {name: pd.read_parquet(input_dir / f"{name}.parquet") for name in required}


def build_audit(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    players = tables["players"]
    matches = tables["matches"]
    participants = tables["match_participants"]
    stats = tables["match_stats"]
    editions = tables["tournament_editions"]

    match_context = matches.merge(
        editions[["tournament_edition_id", "surface", "level"]],
        on="tournament_edition_id",
        how="left",
        validate="many_to_one",
    )
    dates = pd.to_datetime(matches["date"], errors="coerce")
    years = dates.dt.year.astype("Int64")
    ranks_missing = int(participants["rank"].isna().sum())
    duration_missing = int(matches["minutes"].isna().sum())
    height_missing = int(players["height_cm"].isna().sum())
    synthetic_ids = int((participants["id_source"] == "synthetic").sum())

    service_presence = stats.groupby("match_id")["service_points"].apply(
        lambda values: len(values) == 2 and values.notna().all()
    )
    service_missing_matches = len(matches) - int(service_presence.sum())
    duplicate_natural_rows = int(matches.duplicated("natural_match_key", keep=False).sum())
    duplicate_natural_groups = int((matches.groupby("natural_match_key").size() > 1).sum())
    checks = {
        "duplicate_match_ids": duplicate_primary_keys(matches, ["match_id"]),
        "duplicate_natural_match_rows": duplicate_natural_rows,
        "duplicate_natural_match_groups": duplicate_natural_groups,
        "duplicate_player_ids": duplicate_primary_keys(players, ["player_id"]),
        "invalid_dates": int(dates.isna().sum()),
        "invalid_scores": invalid_scores(matches),
        "winner_not_in_participants": winner_not_in_participants(matches, participants),
        "self_matches": self_matches(participants),
        "negative_stat_values": negative_statistics(stats),
        "invalid_rankings": invalid_rankings(participants),
        "participant_rows_without_source_player_id": synthetic_ids,
        "matches_without_exactly_two_participants": int(
            (
                participants.groupby("match_id").size().reindex(matches["match_id"], fill_value=0)
                != 2
            ).sum()
        ),
    }
    hard_fail_keys = (
        "duplicate_match_ids",
        "duplicate_player_ids",
        "winner_not_in_participants",
        "self_matches",
        "negative_stat_values",
        "invalid_rankings",
        "matches_without_exactly_two_participants",
    )
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "coverage": {
            "start_date": _json_value(dates.min()),
            "end_date": _json_value(dates.max()),
            "date_precision": "tournament_start",
        },
        "counts": {
            "matches": len(matches),
            "players": len(players),
            "tournaments": len(tables["tournaments"]),
            "tournament_editions": len(editions),
        },
        "matches_by_year": _counts(years),
        "matches_by_surface": _counts(match_context["surface"]),
        "matches_by_level": _counts(match_context["level"]),
        "missing": {
            "participant_rank": {
                "count": ranks_missing,
                "percent": _percentage(ranks_missing, len(participants)),
                "denominator": "participant rows",
            },
            "match_duration": {
                "count": duration_missing,
                "percent": _percentage(duration_missing, len(matches)),
                "denominator": "matches",
            },
            "complete_service_stats": {
                "count": service_missing_matches,
                "percent": _percentage(service_missing_matches, len(matches)),
                "denominator": "matches lacking service_points for either participant",
            },
            "player_height": {
                "count": height_missing,
                "percent": _percentage(height_missing, len(players)),
                "denominator": "players",
            },
        },
        "checks": checks,
        "passed_hard_checks": all(checks[key] == 0 for key in hard_fail_keys),
    }


def format_text(audit: dict[str, Any]) -> str:
    coverage = audit["coverage"]
    counts = audit["counts"]
    missing = audit["missing"]
    lines = [
        "ATP MATCH DATA AUDIT",
        "",
        (
            f"Coverage: {coverage['start_date']} to {coverage['end_date']} "
            f"({coverage['date_precision']})"
        ),
        f"Matches: {counts['matches']:,}",
        f"Players: {counts['players']:,}",
        f"Tournaments: {counts['tournaments']:,}",
        "",
        "Matches by surface:",
    ]
    lines.extend(f"  {key}: {value:,}" for key, value in audit["matches_by_surface"].items())
    lines.extend(("", "Matches by tournament level:"))
    lines.extend(f"  {key}: {value:,}" for key, value in audit["matches_by_level"].items())
    lines.extend(("", "Missing values:"))
    for label, values in missing.items():
        lines.append(f"  {label}: {values['count']:,} ({values['percent']:.2f}%)")
    lines.extend(("", "Quality checks:"))
    lines.extend(f"  {key}: {value:,}" for key, value in audit["checks"].items())
    lines.extend(("", f"Hard checks passed: {'YES' if audit['passed_hard_checks'] else 'NO'}"))
    return "\n".join(lines) + "\n"


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Audit normalized ATP data")
    command.add_argument("--input-dir", type=Path, default=PROCESSED_DIR / "atp")
    command.add_argument("--report-dir", type=Path, default=REPORTS_DIR)
    command.add_argument("--strict", action="store_true", help="Exit 1 when a hard check fails")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    tables = load_tables(args.input_dir)
    audit = build_audit(tables)
    report = format_text(audit)
    print(report, end="")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "data_audit.txt").write_text(report, encoding="utf-8")
    (args.report_dir / "data_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    LOGGER.info("Reports written to %s", args.report_dir)
    return 1 if args.strict and not audit["passed_hard_checks"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
