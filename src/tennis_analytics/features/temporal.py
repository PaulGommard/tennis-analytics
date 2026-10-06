from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from tennis_analytics.features.timeline import _read_tables, build_timeline, write_timeline
from tennis_analytics.paths import PROCESSED_DIR, PROJECT_ROOT
from tennis_analytics.ratings.elo import EloConfig, expected_score, rating_delta

LOGGER = logging.getLogger(__name__)
WINDOWS = (30, 90, 365)


@dataclass
class PlayerState:
    global_elo: float
    surface_elo: dict[str, float] = field(default_factory=dict)
    career_matches: int = 0
    career_wins: int = 0
    surface_matches: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    surface_wins: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_match_date: pd.Timestamp | None = None
    recent_results: deque[tuple[pd.Timestamp, int]] = field(default_factory=deque)


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else np.nan


def _recent_features(state: PlayerState, date: pd.Timestamp) -> dict[str, float | int]:
    oldest = date - timedelta(days=max(WINDOWS))
    while state.recent_results and state.recent_results[0][0] < oldest:
        state.recent_results.popleft()
    result: dict[str, float | int] = {}
    for days in WINDOWS:
        cutoff = date - timedelta(days=days)
        outcomes = [win for previous_date, win in state.recent_results if previous_date >= cutoff]
        matches = len(outcomes)
        wins = sum(outcomes)
        result[f"matches_{days}d"] = matches
        result[f"wins_{days}d"] = wins
        result[f"win_rate_{days}d"] = _rate(wins, matches)
    return result


def _snapshot(
    state: PlayerState,
    *,
    date: pd.Timestamp,
    surface: str | None,
    config: EloConfig,
) -> dict[str, float | int]:
    recent = _recent_features(state, date)
    recognized_surface = surface in config.surfaces
    surface_matches = state.surface_matches[surface] if recognized_surface else 0
    surface_wins = state.surface_wins[surface] if recognized_surface else 0
    return {
        "elo": state.global_elo,
        "surface_elo": (
            state.surface_elo.get(surface, config.initial_rating) if recognized_surface else np.nan
        ),
        "career_matches": state.career_matches,
        "career_wins": state.career_wins,
        "days_since_last_match": (
            (date - state.last_match_date).days if state.last_match_date is not None else np.nan
        ),
        "career_matches_surface": surface_matches,
        "career_wins_surface": surface_wins,
        "career_win_rate_surface": _rate(surface_wins, surface_matches),
        **recent,
    }


def _state(states: dict[str, PlayerState], player_id: str, config: EloConfig) -> PlayerState:
    if player_id not in states:
        states[player_id] = PlayerState(global_elo=config.initial_rating)
    return states[player_id]


def build_temporal_features(
    timeline: pd.DataFrame,
    config: EloConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = config or EloConfig()
    eligible = timeline.loc[timeline["is_elo_eligible"]].copy()
    eligible["date"] = pd.to_datetime(eligible["date"], errors="raise")
    eligible = eligible.sort_values(["date", "match_id"], kind="stable", ignore_index=True)
    states: dict[str, PlayerState] = {}
    feature_rows: list[dict[str, object]] = []

    # No state mutation is allowed while a date batch is being featurized.
    for date, batch in eligible.groupby("date", sort=True):
        pending: list[tuple[pd.Series, dict[str, object]]] = []
        for _, match in batch.iterrows():
            player_a = str(match["player_A_id"])
            player_b = str(match["player_B_id"])
            surface = match["surface"] if pd.notna(match["surface"]) else None
            state_a = _state(states, player_a, config)
            state_b = _state(states, player_b, config)
            snapshot_a = _snapshot(state_a, date=date, surface=surface, config=config)
            snapshot_b = _snapshot(state_b, date=date, surface=surface, config=config)
            global_expected = expected_score(
                float(snapshot_a["elo"]), float(snapshot_b["elo"]), config.scale
            )
            if pd.notna(snapshot_a["surface_elo"]) and pd.notna(snapshot_b["surface_elo"]):
                surface_expected = expected_score(
                    float(snapshot_a["surface_elo"]),
                    float(snapshot_b["surface_elo"]),
                    config.scale,
                )
            else:
                surface_expected = np.nan
            rank_a = match["rank_A"]
            rank_b = match["rank_B"]
            record: dict[str, object] = {
                "match_id": match["match_id"],
                "date": date,
                "date_precision": match["date_precision"],
                "tournament": match["tournament"],
                "tournament_id": match["tournament_id"],
                "tournament_edition_id": match["tournament_edition_id"],
                "level": match["level"],
                "surface": surface,
                "round": match["round"],
                "player_A_id": player_a,
                "player_B_id": player_b,
                "player_A_name": match["player_A_name"],
                "player_B_name": match["player_B_name"],
                "player_A_win": int(match["player_A_win"]),
                "rank_A": rank_a,
                "rank_B": rank_b,
                # Positive means A has the better (smaller) ATP rank.
                "rank_diff": rank_b - rank_a if pd.notna(rank_a) and pd.notna(rank_b) else np.nan,
                "elo_A": snapshot_a["elo"],
                "elo_B": snapshot_b["elo"],
                "elo_diff": float(snapshot_a["elo"]) - float(snapshot_b["elo"]),
                "elo_expected_A": global_expected,
                "surface_elo_A": snapshot_a["surface_elo"],
                "surface_elo_B": snapshot_b["surface_elo"],
                "surface_elo_diff": (
                    float(snapshot_a["surface_elo"]) - float(snapshot_b["surface_elo"])
                    if pd.notna(snapshot_a["surface_elo"]) and pd.notna(snapshot_b["surface_elo"])
                    else np.nan
                ),
                "surface_elo_expected_A": surface_expected,
            }
            for suffix, snapshot in (("A", snapshot_a), ("B", snapshot_b)):
                for name, value in snapshot.items():
                    if name not in ("elo", "surface_elo"):
                        record[f"{name}_{suffix}"] = value
            feature_rows.append(record)
            pending.append((match, record))

        global_deltas: dict[str, float] = defaultdict(float)
        surface_deltas: dict[tuple[str, str], float] = defaultdict(float)
        for match, record in pending:
            player_a = str(match["player_A_id"])
            player_b = str(match["player_B_id"])
            result_a = int(match["player_A_win"])
            delta = rating_delta(
                float(record["elo_A"]),
                float(record["elo_B"]),
                result_a,
                k_factor=config.k_factor,
                scale=config.scale,
            )
            global_deltas[player_a] += delta
            global_deltas[player_b] -= delta
            surface = match["surface"] if pd.notna(match["surface"]) else None
            if surface in config.surfaces:
                surface_delta = rating_delta(
                    float(record["surface_elo_A"]),
                    float(record["surface_elo_B"]),
                    result_a,
                    k_factor=config.k_factor,
                    scale=config.scale,
                )
                surface_deltas[(player_a, surface)] += surface_delta
                surface_deltas[(player_b, surface)] -= surface_delta

        for player_id, delta in global_deltas.items():
            states[player_id].global_elo += delta
        for (player_id, surface), delta in surface_deltas.items():
            current = states[player_id].surface_elo.get(surface, config.initial_rating)
            states[player_id].surface_elo[surface] = current + delta

        # Results enter historical counters only after every row in the batch
        # has already captured its pre-match features.
        for match, _ in pending:
            surface = match["surface"] if pd.notna(match["surface"]) else None
            for slot in ("A", "B"):
                player_id = str(match[f"player_{slot}_id"])
                win = int(match["player_A_win"]) if slot == "A" else 1 - int(match["player_A_win"])
                state = states[player_id]
                state.career_matches += 1
                state.career_wins += win
                state.last_match_date = date
                state.recent_results.append((date, win))
                if surface in config.surfaces:
                    state.surface_matches[surface] += 1
                    state.surface_wins[surface] += win

    features = pd.DataFrame(feature_rows)
    ratings_rows = []
    for player_id, state in sorted(states.items()):
        row: dict[str, object] = {
            "player_id": player_id,
            "global_elo": state.global_elo,
            "career_matches": state.career_matches,
            "career_wins": state.career_wins,
            "last_match_date": state.last_match_date,
        }
        for surface in config.surfaces:
            key = surface.casefold()
            row[f"{key}_elo"] = state.surface_elo.get(surface, config.initial_rating)
            row[f"{key}_matches"] = state.surface_matches[surface]
        ratings_rows.append(row)
    return features, pd.DataFrame(ratings_rows)


def write_feature_outputs(
    features: pd.DataFrame,
    latest_ratings: pd.DataFrame,
    *,
    output_path: Path,
    ratings_path: Path,
    config: EloConfig,
) -> None:
    for frame, path in ((features, output_path), (latest_ratings, ratings_path)):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".parquet.tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    metadata = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rows": len(features),
        "players": int(pd.concat([features["player_A_id"], features["player_B_id"]]).nunique()),
        "coverage": {
            "start": features["date"].min().date().isoformat(),
            "end": features["date"].max().date().isoformat(),
        },
        "elo": {
            "initial_rating": config.initial_rating,
            "k_factor": config.k_factor,
            "scale": config.scale,
            "surfaces": config.surfaces,
        },
        "temporal_policy": (
            "All matches sharing a tournament-start date use the same pre-date state; "
            "rating and history deltas are applied only after the whole date batch."
        ),
        "rank_diff": "rank_B - rank_A; positive means A is better ranked",
        "missing_history_rates": "NaN when the corresponding match count is zero",
    }
    output_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Build leakage-safe Elo and history features")
    command.add_argument("--input-dir", type=Path, default=PROCESSED_DIR / "atp")
    command.add_argument(
        "--timeline", type=Path, default=PROCESSED_DIR / "analytics" / "match_timeline.parquet"
    )
    command.add_argument("--output", type=Path, default=PROCESSED_DIR / "match_features_v1.parquet")
    command.add_argument(
        "--ratings-output",
        type=Path,
        default=PROCESSED_DIR / "analytics" / "player_ratings_latest_v1.parquet",
    )
    command.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "elo.toml")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    timeline = build_timeline(_read_tables(args.input_dir))
    write_timeline(timeline, args.timeline)
    config = EloConfig.from_toml(args.config)
    features, latest_ratings = build_temporal_features(timeline, config)
    write_feature_outputs(
        features,
        latest_ratings,
        output_path=args.output,
        ratings_path=args.ratings_output,
        config=config,
    )
    LOGGER.info("Feature dataset written: %d rows", len(features))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
