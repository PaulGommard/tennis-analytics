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
from tennis_analytics.features.stat_history import STAT_COLUMNS, metric_contributions

WINDOWS = (30, 90, 365)


@dataclass(frozen=True)
class OpponentEvent:
    day: int
    edition: str
    surface: str
    opponent_elo: float
    opponent_rank_quality: float | None
    win_above_expected: float
    serve_residual_numerator: float | None
    serve_opportunities: float | None
    return_residual_numerator: float | None
    return_opportunities: float | None
    sets: float | None
    games: float | None
    minutes: float | None


def _aligned_stats(data: pd.DataFrame, stats: pd.DataFrame, side: str) -> pd.DataFrame:
    indexed = stats.set_index(["match_id", "player_id"])
    keys = pd.MultiIndex.from_arrays(
        [data["match_id"], data[f"player_{side}_id"]], names=["match_id", "player_id"]
    )
    return indexed.reindex(keys)[list(STAT_COLUMNS)].reset_index(drop=True)


def _weighted_residual(
    contribution: tuple[float, float] | None, expected_rate: float
) -> tuple[float | None, float | None]:
    if contribution is None or not np.isfinite(expected_rate):
        return None, None
    numerator, denominator = contribution
    return numerator - expected_rate * denominator, denominator


def build_opponent_history_base(
    features_v2: pd.DataFrame,
    ratings: pd.DataFrame,
    timeline: pd.DataFrame,
    matches: pd.DataFrame,
    stats: pd.DataFrame,
) -> pd.DataFrame:
    context = timeline[["match_id", "tournament_edition_id"]]
    data = (
        features_v2.merge(
            ratings[
                [
                    "match_id",
                    "chronology_batch_id",
                    "elo_A",
                    "elo_B",
                    "final_raw_expected_A",
                ]
            ],
            on="match_id",
            how="left",
            validate="one_to_one",
        )
        .merge(context, on="match_id", how="left", validate="one_to_one")
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

    history: dict[str, deque[OpponentEvent]] = defaultdict(deque)
    career_residuals: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"serve": [0.0, 0.0], "return": [0.0, 0.0]}
    )
    surface_residuals: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"serve": [0.0, 0.0], "return": [0.0, 0.0]}
    )
    h2h: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"matches": 0.0, "first_wins": 0.0}
    )
    h2h_surface: dict[tuple[str, str, str], dict[str, float]] = defaultdict(
        lambda: {"matches": 0.0, "first_wins": 0.0}
    )
    first_day: dict[str, int] = {}
    circuit_service_wins = 0.0
    circuit_service_points = 0.0

    batch_values = data["chronology_batch_id"].to_numpy()
    starts = np.r_[0, np.flatnonzero(batch_values[1:] != batch_values[:-1]) + 1]
    ends = np.r_[starts[1:], count]
    days = pd.to_datetime(data["date"]).to_numpy(dtype="datetime64[D]").astype("int64")

    for start, end in zip(starts, ends, strict=True):
        day = int(days[start])
        for index in range(start, end):
            edition = str(data.at[index, "tournament_edition_id"])
            surface = str(data.at[index, "surface"])
            player_a = str(data.at[index, "player_A_id"])
            player_b = str(data.at[index, "player_B_id"])
            pair = tuple(sorted((player_a, player_b)))
            pair_state = h2h[pair]
            a_is_first = player_a == pair[0]
            a_wins = (
                pair_state["first_wins"]
                if a_is_first
                else pair_state["matches"] - pair_state["first_wins"]
            )
            column("h2h_matches_before")[index] = pair_state["matches"]
            column("h2h_wins_A_before")[index] = a_wins
            surface_state = h2h_surface[(pair[0], pair[1], surface)]
            surface_a_wins = (
                surface_state["first_wins"]
                if a_is_first
                else surface_state["matches"] - surface_state["first_wins"]
            )
            column("h2h_surface_matches_before")[index] = surface_state["matches"]
            column("h2h_surface_wins_A_before")[index] = surface_a_wins

            for side, player in (("A", player_a), ("B", player_b)):
                events = history[player]
                while events and day - events[0].day > 365:
                    events.popleft()
                for window in WINDOWS:
                    recent = [event for event in events if day - event.day <= window]
                    column(f"opponent_elo_mean_{window}d_{side}")[index] = (
                        np.mean([event.opponent_elo for event in recent]) if recent else np.nan
                    )
                    ranks = [
                        event.opponent_rank_quality
                        for event in recent
                        if event.opponent_rank_quality is not None
                    ]
                    column(f"opponent_rank_quality_mean_{window}d_{side}")[index] = (
                        np.mean(ranks) if ranks else np.nan
                    )
                    column(f"wins_above_expected_{window}d_{side}")[index] = sum(
                        event.win_above_expected for event in recent
                    )
                    for kind in ("serve", "return"):
                        numerator = sum(
                            float(getattr(event, f"{kind}_residual_numerator"))
                            for event in recent
                            if getattr(event, f"{kind}_residual_numerator") is not None
                        )
                        denominator = sum(
                            float(getattr(event, f"{kind}_opportunities"))
                            for event in recent
                            if getattr(event, f"{kind}_opportunities") is not None
                        )
                        column(f"{kind}_residual_{window}d_num_{side}")[index] = numerator
                        column(f"{kind}_residual_{window}d_den_{side}")[index] = denominator
                for kind in ("serve", "return"):
                    numerator, denominator = career_residuals[player][kind]
                    column(f"{kind}_residual_career_num_{side}")[index] = numerator
                    column(f"{kind}_residual_career_den_{side}")[index] = denominator
                    surface_num, surface_den = surface_residuals[(player, surface)][kind]
                    column(f"{kind}_residual_surface_num_{side}")[index] = surface_num
                    column(f"{kind}_residual_surface_den_{side}")[index] = surface_den
                column(f"years_since_first_match_{side}")[index] = (
                    (day - first_day[player]) / 365.25 if player in first_day else 0.0
                )
                for window in (14, 30):
                    prior_events = [
                        event
                        for event in events
                        if day - event.day <= window and event.edition != edition
                    ]
                    column(f"prior_tournament_matches_{window}d_{side}")[index] = len(prior_events)
                    for measure in ("sets", "games", "minutes"):
                        values = [getattr(event, measure) for event in prior_events]
                        column(f"prior_tournament_{measure}_{window}d_{side}")[index] = (
                            sum(float(value) for value in values if value is not None)
                            if all(value is not None for value in values)
                            else np.nan
                        )

        # Every statistic below becomes visible only after the complete round batch.
        for index in range(start, end):
            player_a = str(data.at[index, "player_A_id"])
            player_b = str(data.at[index, "player_B_id"])
            surface = str(data.at[index, "surface"])
            edition = str(data.at[index, "tournament_edition_id"])
            row_a = a_stats.iloc[index].to_dict()
            row_b = b_stats.iloc[index].to_dict()
            contributions = {
                "A": metric_contributions(row_a, row_b),
                "B": metric_contributions(row_b, row_a),
            }
            circuit_serve = (
                circuit_service_wins / circuit_service_points if circuit_service_points else 0.60
            )
            p_a = float(data.at[index, "final_raw_expected_A"])
            workload = parse_score_workload(data.at[index, "score"])
            minutes_value = data.at[index, "minutes"]
            minutes = (
                float(minutes_value)
                if pd.notna(minutes_value) and 0 < float(minutes_value) <= 360
                else None
            )
            for side, player, opponent, opponent_elo, opponent_rank, expected_win in (
                ("A", player_a, player_b, data.at[index, "elo_B"], data.at[index, "rank_B"], p_a),
                (
                    "B",
                    player_b,
                    player_a,
                    data.at[index, "elo_A"],
                    data.at[index, "rank_A"],
                    1 - p_a,
                ),
            ):
                opponent_return = data.at[
                    index, f"return_points_won_rate_career_{'B' if side == 'A' else 'A'}"
                ]
                opponent_service = data.at[
                    index, f"service_points_won_rate_career_{'B' if side == 'A' else 'A'}"
                ]
                expected_serve = (
                    1 - float(opponent_return) if pd.notna(opponent_return) else circuit_serve
                )
                expected_return = (
                    1 - float(opponent_service) if pd.notna(opponent_service) else 1 - circuit_serve
                )
                serve_pair = contributions[side].get("service_points_won_rate")
                return_pair = contributions[side].get("return_points_won_rate")
                serve_residual, serve_den = _weighted_residual(serve_pair, expected_serve)
                return_residual, return_den = _weighted_residual(return_pair, expected_return)
                win = int(
                    data.at[index, "player_A_win"]
                    if side == "A"
                    else 1 - data.at[index, "player_A_win"]
                )
                history[player].append(
                    OpponentEvent(
                        day=int(days[index]),
                        edition=edition,
                        surface=surface,
                        opponent_elo=float(opponent_elo),
                        opponent_rank_quality=(
                            -float(opponent_rank) if pd.notna(opponent_rank) else None
                        ),
                        win_above_expected=win - expected_win,
                        serve_residual_numerator=serve_residual,
                        serve_opportunities=serve_den,
                        return_residual_numerator=return_residual,
                        return_opportunities=return_den,
                        sets=float(workload.sets) if workload.sets is not None else None,
                        games=float(workload.games) if workload.games is not None else None,
                        minutes=minutes,
                    )
                )
                first_day.setdefault(player, int(days[index]))
                for kind, residual, denominator in (
                    ("serve", serve_residual, serve_den),
                    ("return", return_residual, return_den),
                ):
                    if residual is not None and denominator is not None:
                        career_residuals[player][kind][0] += residual
                        career_residuals[player][kind][1] += denominator
                        surface_residuals[(player, surface)][kind][0] += residual
                        surface_residuals[(player, surface)][kind][1] += denominator
            for side in ("A", "B"):
                pair_value = contributions[side].get("service_points_won_rate")
                if pair_value is not None:
                    circuit_service_wins += pair_value[0]
                    circuit_service_points += pair_value[1]
            pair = tuple(sorted((player_a, player_b)))
            first_won = int(
                (player_a == pair[0] and data.at[index, "player_A_win"] == 1)
                or (player_b == pair[0] and data.at[index, "player_A_win"] == 0)
            )
            h2h[pair]["matches"] += 1
            h2h[pair]["first_wins"] += first_won
            surface_key = (pair[0], pair[1], surface)
            h2h_surface[surface_key]["matches"] += 1
            h2h_surface[surface_key]["first_wins"] += first_won

    return pd.concat(
        [
            data[["match_id"]].copy(),
            pd.DataFrame(arrays, index=data.index),
        ],
        axis=1,
    )


def materialize_phase5_features(
    features_v2: pd.DataFrame,
    base: pd.DataFrame,
    *,
    adjusted_point_prior: float,
    adjusted_surface_prior: float,
    h2h_prior: float,
) -> pd.DataFrame:
    base = base.set_index("match_id").loc[features_v2["match_id"]].reset_index()
    additions: dict[str, pd.Series] = {}
    for window in WINDOWS:
        for name in (
            "opponent_elo_mean",
            "opponent_rank_quality_mean",
            "wins_above_expected",
        ):
            for side in ("A", "B"):
                additions[f"{name}_{window}d_{side}"] = base[f"{name}_{window}d_{side}"]
            additions[f"{name}_{window}d_diff"] = (
                base[f"{name}_{window}d_A"] - base[f"{name}_{window}d_B"]
            )
        for kind in ("serve", "return"):
            for side in ("A", "B"):
                additions[f"{kind}_above_expected_{window}d_{side}"] = base[
                    f"{kind}_residual_{window}d_num_{side}"
                ] / (base[f"{kind}_residual_{window}d_den_{side}"] + adjusted_point_prior)
            additions[f"{kind}_above_expected_{window}d_diff"] = (
                additions[f"{kind}_above_expected_{window}d_A"]
                - additions[f"{kind}_above_expected_{window}d_B"]
            )
    for horizon in ("career", "surface"):
        strength = adjusted_point_prior if horizon == "career" else adjusted_surface_prior
        for kind in ("serve", "return"):
            for side in ("A", "B"):
                additions[f"{kind}_above_expected_{horizon}_{side}"] = base[
                    f"{kind}_residual_{horizon}_num_{side}"
                ] / (base[f"{kind}_residual_{horizon}_den_{side}"] + strength)
            additions[f"{kind}_above_expected_{horizon}_diff"] = (
                additions[f"{kind}_above_expected_{horizon}_A"]
                - additions[f"{kind}_above_expected_{horizon}_B"]
            )
    additions["h2h_matches_before"] = base["h2h_matches_before"]
    additions["h2h_edge_A"] = (base["h2h_wins_A_before"] + 0.5 * h2h_prior) / (
        base["h2h_matches_before"] + h2h_prior
    ) - 0.5
    additions["h2h_surface_matches_before"] = base["h2h_surface_matches_before"]
    additions["h2h_surface_edge_A"] = (base["h2h_surface_wins_A_before"] + 0.5 * h2h_prior) / (
        base["h2h_surface_matches_before"] + h2h_prior
    ) - 0.5
    for side in ("A", "B"):
        additions[f"serve_matchup_{side}"] = (
            features_v2[f"service_points_won_rate_90d_{side}"]
            - features_v2[f"return_points_won_rate_90d_{'B' if side == 'A' else 'A'}"]
        )
        additions[f"years_since_first_match_{side}"] = base[f"years_since_first_match_{side}"]
    additions["serve_matchup_diff"] = additions["serve_matchup_A"] - additions["serve_matchup_B"]
    additions["years_since_first_match_diff"] = (
        additions["years_since_first_match_A"] - additions["years_since_first_match_B"]
    )
    for window in (14, 30):
        for measure in ("matches", "sets", "games", "minutes"):
            for side in ("A", "B"):
                additions[f"prior_tournament_{measure}_{window}d_{side}"] = base[
                    f"prior_tournament_{measure}_{window}d_{side}"
                ]
            additions[f"prior_tournament_{measure}_{window}d_diff"] = (
                additions[f"prior_tournament_{measure}_{window}d_A"]
                - additions[f"prior_tournament_{measure}_{window}d_B"]
            )

    surface = features_v2["surface"]
    for label in ("Hard", "Clay", "Grass"):
        indicator = (surface == label).astype(float)
        additions[f"surface_{label.casefold()}"] = indicator
        for feature in (
            "elo_logit",
            "service_points_won_rate_90d_diff",
            "return_points_won_rate_90d_diff",
        ):
            additions[f"{feature}_x_{label.casefold()}"] = features_v2[feature] * indicator
        additions[f"serve_above_expected_90d_diff_x_{label.casefold()}"] = (
            additions["serve_above_expected_90d_diff"] * indicator
        )
        additions[f"return_above_expected_90d_diff_x_{label.casefold()}"] = (
            additions["return_above_expected_90d_diff"] * indicator
        )
    return pd.concat([features_v2.copy(), pd.DataFrame(additions, index=features_v2.index)], axis=1)


def write_v3_metadata(features: pd.DataFrame, path: Path, selected: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    features.to_parquet(temporary, index=False)
    temporary.replace(path)
    metadata = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rows": len(features),
        "columns": len(features.columns),
        "inherits": "match_features_v2.parquet without modification",
        "selected": selected,
        "temporal_rule": "snapshot before full round batch, then update",
    }
    path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
