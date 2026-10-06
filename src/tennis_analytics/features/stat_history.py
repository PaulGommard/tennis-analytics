from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tennis_analytics.features.score import parse_score_workload

CORE_METRICS = (
    "ace_rate",
    "double_fault_rate",
    "first_serve_in_rate",
    "first_serve_won_rate",
    "second_serve_won_rate",
    "service_points_won_rate",
    "return_points_won_rate",
)
CAREER_ONLY_METRICS = (
    "break_points_saved_rate",
    "hold_rate",
    "first_return_points_won_rate",
    "second_return_points_won_rate",
    "break_points_converted_rate",
)
ALL_METRICS = CORE_METRICS + CAREER_ONLY_METRICS
SURFACE_METRICS = ("service_points_won_rate", "return_points_won_rate")
RATE_HORIZONS = ("career", "365d", "90d")
FATIGUE_WINDOWS = (7, 14, 30)
STAT_COLUMNS = (
    "aces",
    "double_faults",
    "service_points",
    "first_serve_in",
    "first_serve_won",
    "second_serve_won",
    "service_games",
    "break_points_saved",
    "break_points_faced",
)


@dataclass(frozen=True)
class HistoricalEvent:
    day: int
    metrics: dict[str, tuple[float, float]]
    win: int
    sets: float | None
    games: float | None
    minutes: float | None


def _ratio_pair(numerator: Any, denominator: Any) -> tuple[float, float] | None:
    if pd.isna(numerator) or pd.isna(denominator):
        return None
    numerator = float(numerator)
    denominator = float(denominator)
    if denominator <= 0 or numerator < 0 or numerator > denominator:
        return None
    return numerator, denominator


def metric_contributions(
    own: dict[str, Any], opponent: dict[str, Any]
) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}

    def add(name: str, numerator: Any, denominator: Any) -> None:
        pair = _ratio_pair(numerator, denominator)
        if pair is not None:
            result[name] = pair

    service_points = own["service_points"]
    first_in = own["first_serve_in"]
    second_attempts = (
        service_points - first_in if pd.notna(service_points) and pd.notna(first_in) else np.nan
    )
    service_won = (
        own["first_serve_won"] + own["second_serve_won"]
        if pd.notna(own["first_serve_won"]) and pd.notna(own["second_serve_won"])
        else np.nan
    )
    add("ace_rate", own["aces"], service_points)
    add("double_fault_rate", own["double_faults"], service_points)
    add("first_serve_in_rate", first_in, service_points)
    add("first_serve_won_rate", own["first_serve_won"], first_in)
    add("second_serve_won_rate", own["second_serve_won"], second_attempts)
    add("service_points_won_rate", service_won, service_points)
    add("break_points_saved_rate", own["break_points_saved"], own["break_points_faced"])
    breaks_conceded = (
        own["break_points_faced"] - own["break_points_saved"]
        if pd.notna(own["break_points_faced"]) and pd.notna(own["break_points_saved"])
        else np.nan
    )
    held = own["service_games"] - breaks_conceded if pd.notna(own["service_games"]) else np.nan
    add("hold_rate", held, own["service_games"])

    opponent_service_points = opponent["service_points"]
    opponent_first_in = opponent["first_serve_in"]
    opponent_second_attempts = (
        opponent_service_points - opponent_first_in
        if pd.notna(opponent_service_points) and pd.notna(opponent_first_in)
        else np.nan
    )
    opponent_service_won = (
        opponent["first_serve_won"] + opponent["second_serve_won"]
        if pd.notna(opponent["first_serve_won"]) and pd.notna(opponent["second_serve_won"])
        else np.nan
    )
    add(
        "return_points_won_rate",
        opponent_service_points - opponent_service_won
        if pd.notna(opponent_service_points) and pd.notna(opponent_service_won)
        else np.nan,
        opponent_service_points,
    )
    add(
        "first_return_points_won_rate",
        opponent_first_in - opponent["first_serve_won"]
        if pd.notna(opponent_first_in) and pd.notna(opponent["first_serve_won"])
        else np.nan,
        opponent_first_in,
    )
    add(
        "second_return_points_won_rate",
        opponent_second_attempts - opponent["second_serve_won"]
        if pd.notna(opponent_second_attempts) and pd.notna(opponent["second_serve_won"])
        else np.nan,
        opponent_second_attempts,
    )
    add(
        "break_points_converted_rate",
        opponent["break_points_faced"] - opponent["break_points_saved"]
        if pd.notna(opponent["break_points_faced"]) and pd.notna(opponent["break_points_saved"])
        else np.nan,
        opponent["break_points_faced"],
    )
    return result


def _aligned_stats(data: pd.DataFrame, stats: pd.DataFrame, side: str) -> pd.DataFrame:
    indexed = stats.set_index(["match_id", "player_id"])
    keys = pd.MultiIndex.from_arrays(
        [data["match_id"], data[f"player_{side}_id"]], names=["match_id", "player_id"]
    )
    return indexed.reindex(keys)[list(STAT_COLUMNS)].reset_index(drop=True)


def build_stat_history_base(
    ratings: pd.DataFrame,
    timeline: pd.DataFrame,
    matches: pd.DataFrame,
    stats: pd.DataFrame,
) -> pd.DataFrame:
    """Build strictly pre-batch sufficient statistics for Phase 4 features."""
    timeline_columns = ["match_id", "age_years_A", "age_years_B", "rank_points_A", "rank_points_B"]
    data = (
        ratings.merge(timeline[timeline_columns], on="match_id", how="left", validate="one_to_one")
        .merge(
            matches[["match_id", "minutes", "score"]],
            on="match_id",
            how="left",
            validate="one_to_one",
        )
        .sort_values(["chronology_batch_id", "match_id"], kind="stable")
        .reset_index(drop=True)
    )
    a_stats = _aligned_stats(data, stats, "A")
    b_stats = _aligned_stats(data, stats, "B")
    count = len(data)
    arrays: dict[str, np.ndarray] = {}

    def column(name: str) -> np.ndarray:
        if name not in arrays:
            arrays[name] = np.full(count, np.nan, dtype=float)
        return arrays[name]

    career: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    surface_career: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(lambda: [0.0, 0.0])
    )
    events: dict[str, deque[HistoricalEvent]] = defaultdict(deque)
    wins: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    circuit: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])

    batch_values = data["chronology_batch_id"].to_numpy()
    starts = np.r_[0, np.flatnonzero(batch_values[1:] != batch_values[:-1]) + 1]
    ends = np.r_[starts[1:], count]
    days = pd.to_datetime(data["date"]).to_numpy(dtype="datetime64[D]").astype("int64")

    for start, end in zip(starts, ends, strict=True):
        day = int(days[start])
        for index in range(start, end):
            surface = data.at[index, "surface"]
            for side in ("A", "B"):
                player = str(data.at[index, f"player_{side}_id"])
                history = events[player]
                while history and day - history[0].day > 365:
                    history.popleft()
                for metric in ALL_METRICS:
                    num, den = career[player][metric]
                    column(f"{metric}_career_num_{side}")[index] = num
                    column(f"{metric}_career_den_{side}")[index] = den
                for horizon, max_days in (("365d", 365), ("90d", 90)):
                    relevant = [event for event in history if day - event.day <= max_days]
                    for metric in CORE_METRICS:
                        pairs = [
                            event.metrics[metric] for event in relevant if metric in event.metrics
                        ]
                        column(f"{metric}_{horizon}_num_{side}")[index] = sum(
                            pair[0] for pair in pairs
                        )
                        column(f"{metric}_{horizon}_den_{side}")[index] = sum(
                            pair[1] for pair in pairs
                        )
                for metric in SURFACE_METRICS:
                    num, den = surface_career[(player, str(surface))][metric]
                    column(f"{metric}_surface_num_{side}")[index] = num
                    column(f"{metric}_surface_den_{side}")[index] = den
                career_wins, career_matches = wins[player]
                column(f"win_rate_career_num_{side}")[index] = career_wins
                column(f"win_rate_career_den_{side}")[index] = career_matches
                for horizon in (365, 90):
                    relevant = [event for event in history if day - event.day <= horizon]
                    column(f"win_rate_{horizon}d_num_{side}")[index] = sum(
                        event.win for event in relevant
                    )
                    column(f"win_rate_{horizon}d_den_{side}")[index] = len(relevant)
                for window in FATIGUE_WINDOWS:
                    recent = [event for event in history if day - event.day <= window]
                    column(f"matches_{window}d_{side}")[index] = len(recent)
                    for measure in ("sets", "games", "minutes"):
                        values = [getattr(event, measure) for event in recent]
                        column(f"{measure}_{window}d_{side}")[index] = (
                            sum(float(value) for value in values if value is not None)
                            if all(value is not None for value in values)
                            else np.nan
                        )
            for metric in ALL_METRICS:
                num, den = circuit[metric]
                column(f"circuit_{metric}_prior")[index] = num / den if den else np.nan

        # Results and match statistics become visible only after the complete round batch.
        for index in range(start, end):
            workload = parse_score_workload(data.at[index, "score"])
            minutes_value = data.at[index, "minutes"]
            minutes = (
                float(minutes_value)
                if pd.notna(minutes_value) and 0 < float(minutes_value) <= 360
                else None
            )
            row_a = a_stats.iloc[index].to_dict()
            row_b = b_stats.iloc[index].to_dict()
            contributions = {
                "A": metric_contributions(row_a, row_b),
                "B": metric_contributions(row_b, row_a),
            }
            for side in ("A", "B"):
                player = str(data.at[index, f"player_{side}_id"])
                win = int(
                    data.at[index, "player_A_win"]
                    if side == "A"
                    else 1 - data.at[index, "player_A_win"]
                )
                for metric, (num, den) in contributions[side].items():
                    career[player][metric][0] += num
                    career[player][metric][1] += den
                    circuit[metric][0] += num
                    circuit[metric][1] += den
                    if metric in SURFACE_METRICS:
                        key = (player, str(data.at[index, "surface"]))
                        surface_career[key][metric][0] += num
                        surface_career[key][metric][1] += den
                wins[player][0] += win
                wins[player][1] += 1
                events[player].append(
                    HistoricalEvent(
                        day=day,
                        metrics=contributions[side],
                        win=win,
                        sets=float(workload.sets) if workload.sets is not None else None,
                        games=float(workload.games) if workload.games is not None else None,
                        minutes=minutes,
                    )
                )

    base_columns = [
        "match_id",
        "date",
        "surface",
        "level",
        "round",
        "player_A_id",
        "player_B_id",
        "player_A_win",
        "chronology_batch_id",
        "elo_diff",
        "final_raw_expected_A",
        "rank_A",
        "rank_B",
        "rank_diff",
        "career_matches_A",
        "career_matches_B",
        "surface_matches_A",
        "surface_matches_B",
        "days_since_last_match_A",
        "days_since_last_match_B",
        "age_years_A",
        "age_years_B",
        "rank_points_A",
        "rank_points_B",
    ]
    return pd.concat([data[base_columns].copy(), pd.DataFrame(arrays, index=data.index)], axis=1)


def _adjusted_rate(num: pd.Series, den: pd.Series, prior: pd.Series, strength: float) -> pd.Series:
    result = (num + strength * prior) / (den + strength)
    return result.where(prior.notna() & ((den > 0) | (strength > 0)))


def materialize_features(
    base: pd.DataFrame,
    *,
    point_prior_strength: float,
    surface_prior_strength: float,
) -> pd.DataFrame:
    identifiers = [
        "match_id",
        "date",
        "surface",
        "level",
        "round",
        "player_A_id",
        "player_B_id",
        "player_A_win",
    ]
    output: dict[str, pd.Series] = {name: base[name] for name in identifiers}
    probability = base["final_raw_expected_A"].clip(1e-6, 1 - 1e-6)
    output["elo_raw_probability"] = probability
    output["elo_logit"] = np.log(probability / (1 - probability))
    output["elo_diff"] = base["elo_diff"]
    output["surface_elo_blended_diff"] = 300.0 * np.log10(probability / (1 - probability))
    for name in ("rank_A", "rank_B", "rank_diff", "rank_points_A", "rank_points_B"):
        output[name] = base[name]
    output["rank_points_diff"] = base["rank_points_A"] - base["rank_points_B"]
    for name in ("career_matches", "surface_matches", "age_years"):
        output[f"{name}_A"] = base[f"{name}_A"]
        output[f"{name}_B"] = base[f"{name}_B"]
        output[f"{name}_diff"] = base[f"{name}_A"] - base[f"{name}_B"]
    output["rest_days_A"] = base["days_since_last_match_A"]
    output["rest_days_B"] = base["days_since_last_match_B"]
    output["rest_days_diff"] = output["rest_days_A"] - output["rest_days_B"]

    for metric in ALL_METRICS:
        horizons = RATE_HORIZONS if metric in CORE_METRICS else ("career",)
        prior = base[f"circuit_{metric}_prior"]
        strength = point_prior_strength if metric in CORE_METRICS else 20.0
        for horizon in horizons:
            for side in ("A", "B"):
                num = base[f"{metric}_{horizon}_num_{side}"]
                den = base[f"{metric}_{horizon}_den_{side}"]
                output[f"{metric}_{horizon}_{side}"] = _adjusted_rate(num, den, prior, strength)
                output[f"{metric}_{horizon}_opportunities_{side}"] = den
            output[f"{metric}_{horizon}_diff"] = (
                output[f"{metric}_{horizon}_A"] - output[f"{metric}_{horizon}_B"]
            )

    for metric in SURFACE_METRICS:
        global_a = output[f"{metric}_career_A"]
        global_b = output[f"{metric}_career_B"]
        for side, global_rate in (("A", global_a), ("B", global_b)):
            num = base[f"{metric}_surface_num_{side}"]
            den = base[f"{metric}_surface_den_{side}"]
            output[f"{metric}_surface_{side}"] = (num + surface_prior_strength * global_rate) / (
                den + surface_prior_strength
            )
            output[f"{metric}_surface_opportunities_{side}"] = den
        output[f"{metric}_surface_diff"] = (
            output[f"{metric}_surface_A"] - output[f"{metric}_surface_B"]
        )

    for horizon in RATE_HORIZONS:
        for side in ("A", "B"):
            num = base[f"win_rate_{horizon}_num_{side}"]
            den = base[f"win_rate_{horizon}_den_{side}"]
            output[f"win_rate_{horizon}_{side}"] = (num + 5.0) / (den + 10.0)
        output[f"win_rate_{horizon}_diff"] = (
            output[f"win_rate_{horizon}_A"] - output[f"win_rate_{horizon}_B"]
        )

    for window in FATIGUE_WINDOWS:
        for measure in ("matches", "sets", "games", "minutes"):
            for side in ("A", "B"):
                output[f"{measure}_{window}d_{side}"] = base[f"{measure}_{window}d_{side}"]
            output[f"{measure}_{window}d_diff"] = (
                output[f"{measure}_{window}d_A"] - output[f"{measure}_{window}d_B"]
            )
    return pd.DataFrame(output, index=base.index)


def write_feature_dataset(
    features: pd.DataFrame,
    path: Path,
    *,
    point_prior_strength: float,
    surface_prior_strength: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    features.to_parquet(temporary, index=False)
    temporary.replace(path)
    metadata = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rows": len(features),
        "columns": len(features.columns),
        "point_prior_strength": point_prior_strength,
        "surface_prior_strength": surface_prior_strength,
        "target": "player_A_win",
        "difference_convention": "A minus B except rank_diff, inherited as rank_B minus rank_A",
        "temporal_rule": "snapshot complete round batch, then update",
    }
    path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
