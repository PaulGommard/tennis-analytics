from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd

from tennis_analytics.features.score import parse_score_workload
from tennis_analytics.features.stat_history import STAT_COLUMNS, metric_contributions
from tennis_analytics.modeling.logistic import apply_platt
from tennis_analytics.paths import PROCESSED_DIR, PROJECT_ROOT

OUTPUT_PATH = PROCESSED_DIR / "tournament_context_v1.parquet"
AUDIT_PATH = PROCESSED_DIR / "tournament_context_v1.audit.json"
MODEL_DIR = PROJECT_ROOT / "models" / "tournament_adapter_v1"
ROUND_ORDER = {
    "Q1": 0,
    "Q2": 1,
    "Q3": 2,
    "Q4": 3,
    "RR": 4,
    "ER": 5,
    "R128": 6,
    "R64": 7,
    "R32": 8,
    "R16": 9,
    "QF": 10,
    "SF": 11,
    "F": 12,
    "BR": 12,
}
PLAYER_PRIORS = (10, 20, 40, 75, 100)
TOURNAMENT_PRIORS = (20, 40, 75, 100, 150)


def _slug(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _safe(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _valid_minutes(value: Any) -> float | None:
    number = _safe(value)
    return number if np.isfinite(number) and 0 < number <= 360 else None


@dataclass
class PlayerState:
    matches: int = 0
    wins: float = 0.0
    expected_wins: float = 0.0
    sets: float = 0.0
    games: float = 0.0
    minutes: float = 0.0
    minutes_matches: int = 0
    sets_lost: float = 0.0
    games_lost: float = 0.0
    tiebreaks: float = 0.0
    five_set_matches: float = 0.0
    retirements: float = 0.0
    serve_residual_num: float = 0.0
    serve_won: float = 0.0
    serve_points: float = 0.0
    return_residual_num: float = 0.0
    return_won: float = 0.0
    return_points: float = 0.0
    ace_residual_num: float = 0.0
    aces: float = 0.0
    ace_points: float = 0.0
    df_residual_num: float = 0.0
    double_faults: float = 0.0
    df_points: float = 0.0
    break_residual_num: float = 0.0
    breaks: float = 0.0
    break_games: float = 0.0


@dataclass
class GlobalState:
    matches: int = 0
    service_residual_num: float = 0.0
    service_points: float = 0.0
    ace_residual_num: float = 0.0
    ace_points: float = 0.0
    df_residual_num: float = 0.0
    df_points: float = 0.0
    break_residual_num: float = 0.0
    service_games: float = 0.0
    upset_residual: float = 0.0
    games: float = 0.0
    sets: float = 0.0
    minutes: float = 0.0
    minutes_matches: int = 0
    tiebreak_matches: int = 0
    straight_sets: int = 0
    score_matches: int = 0


@dataclass
class HistoricalVenueState:
    matches: int = 0
    residual: float = 0.0


def _freeze_inputs() -> dict[str, Any]:
    paths = [
        PROJECT_ROOT / "models" / "prematch_v1" / "model.joblib",
        PROJECT_ROOT / "models" / "prematch_v1" / "metadata.json",
        PROCESSED_DIR / "match_features_v3.parquet",
        PROCESSED_DIR / "analytics" / "match_timeline.parquet",
        PROCESSED_DIR / "atp" / "matches.parquet",
        PROCESSED_DIR / "atp" / "match_stats.parquet",
    ]
    records = [
        {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in paths
    ]
    payload = {
        "statement": "Tournament Adapter V1 reads but never modifies frozen PRE-MATCH V1.",
        "files": records,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "prematch_v1_freeze.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _stats_lookup(stats: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row.match_id), str(row.player_id)): {
            column: getattr(row, column) for column in STAT_COLUMNS
        }
        for row in stats.itertuples(index=False)
    }


def _expected_rate(own: Any, opponent_complement: Any, fallback: float) -> float:
    values = [value for value in (_safe(own), _safe(opponent_complement)) if np.isfinite(value)]
    return float(np.mean(values)) if values else fallback


def _pair(contributions: dict[str, tuple[float, float]], name: str) -> tuple[float, float] | None:
    value = contributions.get(name)
    if value is None or value[1] <= 0:
        return None
    return float(value[0]), float(value[1])


def _player_snapshot(state: PlayerState, prefix: str) -> dict[str, float]:
    result = {
        f"{prefix}_matches_played": float(state.matches),
        f"{prefix}_wins": state.wins,
        f"{prefix}_expected_wins": state.expected_wins,
        f"{prefix}_wins_above_expected_raw": state.wins - state.expected_wins,
        f"{prefix}_sets_played": state.sets,
        f"{prefix}_games_played": state.games,
        f"{prefix}_minutes_played": state.minutes if state.minutes_matches == state.matches else np.nan,
        f"{prefix}_minutes_coverage": state.minutes_matches / state.matches if state.matches else 0.0,
        f"{prefix}_sets_lost": state.sets_lost,
        f"{prefix}_games_lost": state.games_lost,
        f"{prefix}_tiebreaks": state.tiebreaks,
        f"{prefix}_five_set_matches": state.five_set_matches,
        f"{prefix}_retirements": state.retirements,
        f"{prefix}_serve_residual_num": state.serve_residual_num,
        f"{prefix}_service_points_won": state.serve_won,
        f"{prefix}_serve_points": state.serve_points,
        f"{prefix}_return_residual_num": state.return_residual_num,
        f"{prefix}_return_points_won": state.return_won,
        f"{prefix}_return_points": state.return_points,
        f"{prefix}_ace_residual_num": state.ace_residual_num,
        f"{prefix}_aces": state.aces,
        f"{prefix}_ace_points": state.ace_points,
        f"{prefix}_df_residual_num": state.df_residual_num,
        f"{prefix}_double_faults": state.double_faults,
        f"{prefix}_df_points": state.df_points,
        f"{prefix}_break_residual_num": state.break_residual_num,
        f"{prefix}_breaks": state.breaks,
        f"{prefix}_break_games": state.break_games,
        f"{prefix}_service_points_won_rate_tournament": (
            state.serve_won / state.serve_points if state.serve_points else np.nan
        ),
        f"{prefix}_return_points_won_rate_tournament": (
            state.return_won / state.return_points if state.return_points else np.nan
        ),
        f"{prefix}_ace_rate_tournament": (
            state.aces / state.ace_points if state.ace_points else np.nan
        ),
        f"{prefix}_double_fault_rate_tournament": (
            state.double_faults / state.df_points if state.df_points else np.nan
        ),
        f"{prefix}_break_rate_tournament": (
            state.breaks / state.break_games if state.break_games else np.nan
        ),
    }
    for prior in PLAYER_PRIORS:
        result[f"{prefix}_wins_above_expected_c{prior}"] = (
            (state.wins - state.expected_wins) / (state.matches + prior)
        )
        result[f"{prefix}_serve_over_expected_c{prior}"] = state.serve_residual_num / (
            state.serve_points + prior
        )
        result[f"{prefix}_return_over_expected_c{prior}"] = state.return_residual_num / (
            state.return_points + prior
        )
        result[f"{prefix}_ace_over_expected_c{prior}"] = state.ace_residual_num / (
            state.ace_points + prior
        )
        result[f"{prefix}_df_over_expected_c{prior}"] = state.df_residual_num / (
            state.df_points + prior
        )
        result[f"{prefix}_break_over_expected_c{prior}"] = state.break_residual_num / (
            state.break_games + prior
        )
    return result


def _global_snapshot(state: GlobalState) -> dict[str, float]:
    result = {
        "tournament_matches_observed": float(state.matches),
        "tournament_service_residual_num": state.service_residual_num,
        "tournament_service_points": state.service_points,
        "tournament_ace_residual_num": state.ace_residual_num,
        "tournament_ace_points": state.ace_points,
        "tournament_df_residual_num": state.df_residual_num,
        "tournament_df_points": state.df_points,
        "tournament_break_residual_num": state.break_residual_num,
        "tournament_service_games": state.service_games,
        "tournament_upset_residual_raw": state.upset_residual,
        "tournament_games_per_match": state.games / state.score_matches if state.score_matches else np.nan,
        "tournament_sets_per_match": state.sets / state.score_matches if state.score_matches else np.nan,
        "tournament_minutes_per_match": state.minutes / state.minutes_matches if state.minutes_matches else np.nan,
        "tournament_minutes_coverage": state.minutes_matches / state.matches if state.matches else 0.0,
        "tournament_tiebreak_rate": state.tiebreak_matches / state.score_matches if state.score_matches else 0.0,
        "tournament_straight_sets_rate": state.straight_sets / state.score_matches if state.score_matches else 0.0,
    }
    for prior in TOURNAMENT_PRIORS:
        serve = state.service_residual_num / (state.service_points + prior)
        ace = state.ace_residual_num / (state.ace_points + prior)
        double_fault = state.df_residual_num / (state.df_points + prior)
        breaks = state.break_residual_num / (state.service_games + prior)
        result[f"tournament_serve_environment_c{prior}"] = serve
        result[f"tournament_return_environment_c{prior}"] = -serve
        result[f"tournament_ace_environment_c{prior}"] = ace
        result[f"tournament_df_environment_c{prior}"] = double_fault
        result[f"tournament_break_environment_c{prior}"] = breaks
        result[f"tournament_speed_proxy_c{prior}"] = serve + ace - breaks
        result[f"tournament_upset_environment_c{prior}"] = state.upset_residual / (
            state.matches + prior
        )
        weight = state.matches / (state.matches + prior)
        result[f"tournament_workload_environment_c{prior}"] = weight * (
            state.games / state.score_matches if state.score_matches else 0.0
        )
    return result


def _score_details(score: Any, best_of: Any) -> dict[str, float | str | None]:
    workload = parse_score_workload(score)
    text = str(score).upper() if pd.notna(score) else ""
    set_scores = re.findall(r"(?<!\[)(\d{1,2})-(\d{1,2})(?:\(\d{1,2}\))?", text)
    winner_games = sum(int(left) for left, _ in set_scores) if set_scores else None
    loser_games = sum(int(right) for _, right in set_scores) if set_scores else None
    winner_sets = sum(int(left) > int(right) for left, right in set_scores) if set_scores else None
    loser_sets = sum(int(left) < int(right) for left, right in set_scores) if set_scores else None
    tiebreaks = len(re.findall(r"\(\d{1,2}\)", text)) + len(re.findall(r"\[\d+-\d+\]", text))
    completed_target = int(_safe(best_of, 3)) // 2 + 1
    return {
        "sets": float(workload.sets) if workload.sets is not None else None,
        "games": float(workload.games) if workload.games is not None else None,
        "winner_games": float(winner_games) if winner_games is not None else None,
        "loser_games": float(loser_games) if loser_games is not None else None,
        "winner_sets": float(winner_sets) if winner_sets is not None else None,
        "loser_sets": float(loser_sets) if loser_sets is not None else None,
        "tiebreaks": float(tiebreaks),
        "straight_sets": float(loser_sets == 0 and winner_sets == completed_target) if winner_sets is not None else None,
        "status": workload.status,
    }


def _round_flags(round_name: str) -> dict[str, float]:
    return {f"round_{label}": float(round_name == label) for label in ("R128", "R64", "R32", "R16", "QF", "SF", "F", "RR", "OTHER")}


def _level_group(level: str) -> str:
    return "Grand Slam" if level == "G" else "Masters" if level == "M" else "Other"


def build_tournament_replay() -> dict[str, Any]:
    """Replay every edition as frozen round batches before any adapter training."""
    _freeze_inputs()
    features = pd.read_parquet(PROCESSED_DIR / "match_features_v3.parquet")
    timeline = pd.read_parquet(PROCESSED_DIR / "analytics" / "match_timeline.parquet")
    matches = pd.read_parquet(PROCESSED_DIR / "atp" / "matches.parquet")
    stats = pd.read_parquet(PROCESSED_DIR / "atp" / "match_stats.parquet")
    artifact = joblib.load(PROJECT_ROOT / "models" / "prematch_v1" / "model.joblib")
    raw = artifact["pipeline"].predict_proba(features[artifact["feature_columns"]])[:, 1]
    p_base = apply_platt(artifact["calibrator"], raw) if artifact.get("calibrator") is not None else raw
    features = features.copy()
    features["p_base"] = p_base
    context = timeline[
        [
            "match_id",
            "tournament",
            "tournament_id",
            "tournament_edition_id",
            "score",
            "winner_id",
            "is_elo_eligible",
        ]
    ]
    match_context = matches[["match_id", "best_of", "minutes"]]
    data = (
        features.merge(context, on="match_id", how="left", validate="one_to_one")
        .merge(match_context, on="match_id", how="left", validate="one_to_one")
    )
    data = data.loc[data["is_elo_eligible"] & data["round"].isin(ROUND_ORDER)].copy()
    data["date"] = pd.to_datetime(data["date"])
    data["round_order"] = data["round"].map(ROUND_ORDER).astype(int)
    data["tournament_name_key"] = data["tournament"].map(_slug)
    data["level_group"] = data["level"].map(_level_group)
    data = data.sort_values(
        ["date", "tournament_edition_id", "round_order", "match_id"], kind="stable"
    ).reset_index(drop=True)
    stats_by_key = _stats_lookup(stats)
    venue_history: defaultdict[tuple[str, str], HistoricalVenueState] = defaultdict(
        HistoricalVenueState
    )
    output_rows: list[dict[str, Any]] = []
    frozen_round_violations = 0

    for (_, edition_id), edition in data.groupby(
        ["date", "tournament_edition_id"], sort=True, observed=True
    ):
        players: defaultdict[str, PlayerState] = defaultdict(PlayerState)
        global_state = GlobalState()
        venue_key = str(edition.iloc[0]["tournament_name_key"])
        start_history = {
            player: HistoricalVenueState(state.matches, state.residual)
            for (venue, player), state in venue_history.items()
            if venue == venue_key
        }
        for round_order, round_frame in edition.groupby("round_order", sort=True):
            before_count = global_state.matches
            pending: list[tuple[pd.Series, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
            for _, row in round_frame.iterrows():
                player_a = str(row["player_A_id"])
                player_b = str(row["player_B_id"])
                state_a = players[player_a]
                state_b = players[player_b]
                snapshot = {
                    "match_id": row["match_id"],
                    "date": row["date"],
                    "year": int(row["date"].year),
                    "tournament_id": row["tournament_id"],
                    "tournament_edition_id": edition_id,
                    "tournament_name": row["tournament"],
                    "tournament_name_key": venue_key,
                    "surface": row["surface"],
                    "level": row["level"],
                    "level_group": row["level_group"],
                    "best_of": row["best_of"],
                    "round": row["round"],
                    "round_order": round_order,
                    "player_A_id": player_a,
                    "player_B_id": player_b,
                    "player_A_win": int(row["player_A_win"]),
                    "p_base": float(row["p_base"]),
                    "logit_p_base": float(np.log(np.clip(row["p_base"], 1e-6, 1 - 1e-6) / np.clip(1 - row["p_base"], 1e-6, 1))),
                    "round_state_version": f"{edition_id}:before_order:{round_order}",
                    "matches_observed_before_round": before_count,
                    "surface_clay": float(row["surface"] == "Clay"),
                    "surface_grass": float(row["surface"] == "Grass"),
                    "level_grand_slam": float(row["level"] == "G"),
                    "level_masters": float(row["level"] == "M"),
                    "best_of_5": float(row["best_of"] == 5),
                    "style_serve_diff": _safe(
                        row.get("service_points_won_rate_career_diff"), 0.0
                    ),
                    "style_return_diff": _safe(
                        row.get("return_points_won_rate_career_diff"), 0.0
                    ),
                    "style_ace_diff": _safe(row.get("ace_rate_career_diff"), 0.0),
                    "style_df_diff": _safe(
                        row.get("double_fault_rate_career_diff"), 0.0
                    ),
                    "style_hold_diff": _safe(row.get("hold_rate_career_diff"), 0.0),
                    **_round_flags(str(row["round"])),
                    **_player_snapshot(state_a, "A"),
                    **_player_snapshot(state_b, "B"),
                    **_global_snapshot(global_state),
                }
                for side, player in (("A", player_a), ("B", player_b)):
                    history = start_history.get(player, HistoricalVenueState())
                    snapshot[f"{side}_same_tournament_history_matches"] = float(history.matches)
                    snapshot[f"{side}_same_tournament_wins_above_expected_c20"] = history.residual / (
                        history.matches + 20
                    )
                output_rows.append(snapshot)
                own_a = stats_by_key.get((str(row["match_id"]), player_a), {})
                own_b = stats_by_key.get((str(row["match_id"]), player_b), {})
                pending.append((row, own_a, own_b, _score_details(row["score"], row["best_of"])))
            if global_state.matches != before_count:
                frozen_round_violations += 1
            for row, own_a, own_b, score in pending:
                player_ids = {"A": str(row["player_A_id"]), "B": str(row["player_B_id"])}
                contributions = {
                    "A": metric_contributions(own_a, own_b) if own_a and own_b else {},
                    "B": metric_contributions(own_b, own_a) if own_a and own_b else {},
                }
                expected = {}
                for side, other in (("A", "B"), ("B", "A")):
                    expected[(side, "serve")] = _expected_rate(
                        row.get(f"service_points_won_rate_career_{side}"),
                        1 - _safe(row.get(f"return_points_won_rate_career_{other}")),
                        0.60,
                    )
                    expected[(side, "return")] = _expected_rate(
                        row.get(f"return_points_won_rate_career_{side}"),
                        1 - _safe(row.get(f"service_points_won_rate_career_{other}")),
                        0.40,
                    )
                    expected[(side, "ace")] = _safe(row.get(f"ace_rate_career_{side}"), 0.06)
                    expected[(side, "df")] = _safe(row.get(f"double_fault_rate_career_{side}"), 0.04)
                    # A player's break signal is a return-side statistic: compare the
                    # breaks won on the opponent's service games with the opponent's
                    # pre-match hold expectation.
                    expected[(side, "break")] = 1 - _safe(
                        row.get(f"hold_rate_career_{other}"), 0.80
                    )
                winner_side = "A" if int(row["player_A_win"]) else "B"
                workload_minutes = _valid_minutes(row["minutes"])
                for side, other in (("A", "B"), ("B", "A")):
                    state = players[player_ids[side]]
                    won = float(side == winner_side)
                    expected_win = float(row["p_base"] if side == "A" else 1 - row["p_base"])
                    state.matches += 1
                    state.wins += won
                    state.expected_wins += expected_win
                    if score["sets"] is not None:
                        state.sets += float(score["sets"])
                        state.games += float(score["games"] or 0)
                        state.sets_lost += float(score["loser_sets"] if won else score["winner_sets"])
                        state.games_lost += float(score["loser_games"] if won else score["winner_games"])
                        state.tiebreaks += float(score["tiebreaks"])
                    if workload_minutes is not None:
                        state.minutes += workload_minutes
                        state.minutes_matches += 1
                    state.five_set_matches += float(_safe(row["best_of"]) == 5 and (score["sets"] or 0) >= 5)
                    state.retirements += float(score["status"] == "retired")
                    serve_pair = _pair(contributions[side], "service_points_won_rate")
                    return_pair = _pair(contributions[side], "return_points_won_rate")
                    ace_pair = _pair(contributions[side], "ace_rate")
                    df_pair = _pair(contributions[side], "double_fault_rate")
                    for pair, metric, denominator_name in (
                        (serve_pair, "serve", "serve_points"),
                        (return_pair, "return", "return_points"),
                        (ace_pair, "ace", "ace_points"),
                        (df_pair, "df", "df_points"),
                    ):
                        if pair is not None:
                            numerator, denominator = pair
                            setattr(state, f"{metric}_residual_num", getattr(state, f"{metric}_residual_num") + numerator - expected[(side, metric)] * denominator)
                            numerator_name = {
                                "serve": "serve_won",
                                "return": "return_won",
                                "ace": "aces",
                                "df": "double_faults",
                            }[metric]
                            setattr(
                                state,
                                numerator_name,
                                getattr(state, numerator_name) + numerator,
                            )
                            setattr(state, denominator_name, getattr(state, denominator_name) + denominator)
                    opponent_stats = own_b if side == "A" else own_a
                    if (
                        opponent_stats
                        and pd.notna(opponent_stats.get("service_games"))
                        and pd.notna(opponent_stats.get("break_points_faced"))
                        and pd.notna(opponent_stats.get("break_points_saved"))
                    ):
                        service_games = float(opponent_stats["service_games"])
                        breaks = float(
                            opponent_stats["break_points_faced"]
                            - opponent_stats["break_points_saved"]
                        )
                        state.break_residual_num += breaks - expected[(side, "break")] * service_games
                        state.breaks += breaks
                        state.break_games += service_games
                    venue_state = venue_history[(venue_key, player_ids[side])]
                    venue_state.matches += 1
                    venue_state.residual += won - expected_win
                global_state.matches += 1
                favorite_lost = float((row["p_base"] >= 0.5) != bool(row["player_A_win"]))
                expected_upset = min(float(row["p_base"]), 1 - float(row["p_base"]))
                global_state.upset_residual += favorite_lost - expected_upset
                for side, own in (("A", own_a), ("B", own_b)):
                    for metric, pair_name, total_name, residual_name in (
                        ("serve", "service_points_won_rate", "service_points", "service_residual_num"),
                        ("ace", "ace_rate", "ace_points", "ace_residual_num"),
                        ("df", "double_fault_rate", "df_points", "df_residual_num"),
                    ):
                        pair = _pair(contributions[side], pair_name)
                        if pair is not None:
                            numerator, denominator = pair
                            setattr(global_state, residual_name, getattr(global_state, residual_name) + numerator - expected[(side, metric)] * denominator)
                            setattr(global_state, total_name, getattr(global_state, total_name) + denominator)
                    if own and pd.notna(own.get("service_games")) and pd.notna(own.get("break_points_faced")) and pd.notna(own.get("break_points_saved")):
                        service_games = float(own["service_games"])
                        breaks = float(own["break_points_faced"] - own["break_points_saved"])
                        expected_breaks_conceded = 1 - _safe(
                            row.get(f"hold_rate_career_{side}"), 0.80
                        )
                        global_state.break_residual_num += (
                            breaks - expected_breaks_conceded * service_games
                        )
                        global_state.service_games += service_games
                if score["sets"] is not None:
                    global_state.score_matches += 1
                    global_state.sets += float(score["sets"])
                    global_state.games += float(score["games"] or 0)
                    global_state.tiebreak_matches += int(float(score["tiebreaks"]) > 0)
                    global_state.straight_sets += int(bool(score["straight_sets"]))
                if workload_minutes is not None:
                    global_state.minutes += workload_minutes
                    global_state.minutes_matches += 1

    output = pd.DataFrame(output_rows)
    for name in (
        "matches_played",
        "sets_played",
        "games_played",
        "minutes_played",
        "sets_lost",
        "games_lost",
        "tiebreaks",
        "five_set_matches",
        "retirements",
    ):
        output[f"{name}_diff"] = output[f"A_{name}"] - output[f"B_{name}"]
    for prior in PLAYER_PRIORS:
        for name in (
            "wins_above_expected",
            "serve_over_expected",
            "return_over_expected",
            "ace_over_expected",
            "df_over_expected",
            "break_over_expected",
        ):
            output[f"{name}_c{prior}_diff"] = output[f"A_{name}_c{prior}"] - output[
                f"B_{name}_c{prior}"
            ]
    for name in (
        "service_points_won_rate_tournament",
        "return_points_won_rate_tournament",
        "ace_rate_tournament",
        "double_fault_rate_tournament",
        "break_rate_tournament",
    ):
        output[f"{name}_diff"] = output[f"A_{name}"] - output[f"B_{name}"]
    output["same_tournament_history_matches_diff"] = (
        output["A_same_tournament_history_matches"]
        - output["B_same_tournament_history_matches"]
    )
    output["same_tournament_wins_above_expected_c20_diff"] = (
        output["A_same_tournament_wins_above_expected_c20"]
        - output["B_same_tournament_wins_above_expected_c20"]
    )
    output["minutes_coverage_diff"] = (
        output["A_minutes_coverage"] - output["B_minutes_coverage"]
    )
    output["style_speed_diff"] = (
        output["style_serve_diff"]
        + output["style_ace_diff"]
        - output["style_return_diff"]
    )
    interactions: dict[str, pd.Series] = {}
    for prior in TOURNAMENT_PRIORS:
        interactions[f"tournament_serve_environment_c{prior}_x_style_serve_diff"] = (
            output[f"tournament_serve_environment_c{prior}"] * output["style_serve_diff"]
        )
        interactions[f"tournament_return_environment_c{prior}_x_style_return_diff"] = (
            output[f"tournament_return_environment_c{prior}"] * output["style_return_diff"]
        )
        interactions[f"tournament_ace_environment_c{prior}_x_style_ace_diff"] = (
            output[f"tournament_ace_environment_c{prior}"] * output["style_ace_diff"]
        )
        interactions[f"tournament_df_environment_c{prior}_x_style_df_diff"] = (
            output[f"tournament_df_environment_c{prior}"] * output["style_df_diff"]
        )
        interactions[f"tournament_break_environment_c{prior}_x_style_return_diff"] = (
            output[f"tournament_break_environment_c{prior}"] * output["style_return_diff"]
        )
        interactions[f"tournament_speed_proxy_c{prior}_x_style_speed_diff"] = (
            output[f"tournament_speed_proxy_c{prior}"] * output["style_speed_diff"]
        )
        interactions[f"tournament_upset_environment_c{prior}_x_logit_p_base"] = (
            output[f"tournament_upset_environment_c{prior}"] * output["logit_p_base"]
        )
        interactions[f"tournament_workload_environment_c{prior}_x_games_played_diff"] = (
            output[f"tournament_workload_environment_c{prior}"]
            * output["games_played_diff"]
        )
    for label in ("R128", "R64", "R32", "R16", "QF", "SF", "F", "RR", "OTHER"):
        interactions[f"round_{label}_x_logit_p_base"] = (
            output[f"round_{label}"] * output["logit_p_base"]
        )
    for name in (
        "surface_clay",
        "surface_grass",
        "level_grand_slam",
        "level_masters",
        "best_of_5",
    ):
        interactions[f"{name}_x_logit_p_base"] = output[name] * output["logit_p_base"]
    output = pd.concat([output, pd.DataFrame(interactions, index=output.index)], axis=1)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT_PATH, index=False)
    audit = {
        "rows": len(output),
        "editions": int(output["tournament_edition_id"].nunique()),
        "years": [int(output["year"].min()), int(output["year"].max())],
        "frozen_round_violations": frozen_round_violations,
        "first_match_player_state_zero": bool(
            (output.loc[output["A_matches_played"] == 0, "A_wins"] == 0).all()
            and (output.loc[output["B_matches_played"] == 0, "B_wins"] == 0).all()
        ),
        "base_probability_sha256": hashlib.sha256(
            output[["match_id", "p_base"]].to_csv(index=False).encode()
        ).hexdigest(),
        "minutes_coverage": float(
            pd.concat([output["A_minutes_coverage"], output["B_minutes_coverage"]]).mean()
        ),
        "stat_coverage": float(
            pd.concat([output["A_serve_points"], output["B_serve_points"]]).gt(0).mean()
        ),
    }
    AUDIT_PATH.write_text(json.dumps(audit, indent=2) + "\n")
    return audit
