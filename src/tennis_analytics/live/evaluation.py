from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from tennis_analytics.live.markov import (
    MatchFormat,
    align_serve_probabilities_to_prematch,
    clear_probability_caches,
    live_probabilities,
)
from tennis_analytics.modeling.point_model import MODEL_DIR
from tennis_analytics.paths import PROCESSED_DIR, REPORTS_DIR
from tennis_analytics.point_by_point.normalize import FEATURE_PATH

OUTPUT_PATH = PROCESSED_DIR / "live_probabilities_test.parquet"
METRICS_PATH = REPORTS_DIR / "live_probability_validation.csv"
REPLAY_PATH = REPORTS_DIR / "live_replay_examples.csv"


def match_format(rule_label: str, best_of: int) -> MatchFormat:
    if rule_label in {"advantage_final_set", "source_no_final_tb"}:
        return MatchFormat(best_of=int(best_of), final_set_tiebreak=False)
    if rule_label == "wimbledon_12_all":
        return MatchFormat(
            best_of=int(best_of),
            final_set_tiebreak=True,
            final_set_tiebreak_at=12,
            final_set_tiebreak_points=7,
        )
    if rule_label in {"australian_open_10pt", "grand_slam_unified_10pt"}:
        return MatchFormat(
            best_of=int(best_of),
            final_set_tiebreak=True,
            final_set_tiebreak_at=6,
            final_set_tiebreak_points=10,
        )
    return MatchFormat(best_of=int(best_of))


def _logit_shift(probability: np.ndarray | float, delta: float) -> np.ndarray:
    values = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(values / (1 - values)) + delta
    return 1 / (1 + np.exp(-logits))


def _metric_row(
    target: pd.Series | np.ndarray,
    probability: pd.Series | np.ndarray,
    *,
    target_name: str,
    scope: str,
    model: str = "LIVE_V1",
) -> dict[str, Any]:
    y = np.asarray(target, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    row: dict[str, Any] = {
        "scope": scope,
        "target": target_name,
        "model": model,
        "observations": len(y),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }
    row["roc_auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else math.nan
    return row


def _neutral_row(first: pd.Series, server: str, columns: list[str]) -> pd.DataFrame:
    prefix = "A" if server == "A" else "B"
    values = {
        "server_serve_rate": first[f"{prefix}_serve_rate"],
        "receiver_return_rate": first[f"{prefix}_return_rate"],
        "server_surface_serve_rate": first[f"{prefix}_surface_serve_rate"],
        "receiver_surface_return_rate": first[f"{prefix}_surface_return_rate"],
        "server_serve_rate_365d": first[f"{prefix}_serve_rate_365d"],
        "receiver_return_rate_365d": first[f"{prefix}_return_rate_365d"],
        "surface_clay": int(first["surface"] == "Clay"),
        "surface_grass": int(first["surface"] == "Grass"),
        "prematch_server": (
            first["prematch_probability_A"]
            if server == "A"
            else 1 - first["prematch_probability_A"]
        ),
        "is_break_point": 0,
        "is_game_point": 0,
        "is_set_point": 0,
        "is_match_point": 0,
        "is_tiebreak": 0,
        "deuce": 0,
        "server_leading_in_game": 0,
        "sets_diff_server": 0,
        "games_diff_server": 0,
    }
    return pd.DataFrame([{column: values[column] for column in columns}])


def _evaluate_match(group: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    group = group.copy()
    first = group.iloc[0]
    pipeline = artifact["pipeline"]
    columns = artifact["feature_columns"]
    fmt = match_format(str(first["rule_label"]), int(first["best_of"]))
    raw_a = float(pipeline.predict_proba(_neutral_row(first, "A", columns))[0, 1])
    raw_b = float(pipeline.predict_proba(_neutral_row(first, "B", columns))[0, 1])
    prior = float(first["prematch_probability_A"])
    first_server = str(first["server"])
    p_a_serve, p_b_serve, delta = align_serve_probabilities_to_prematch(
        raw_a,
        raw_b,
        prior,
        first_server=first_server,
        match_format=fmt,
    )
    raw_point_server = pipeline.predict_proba(group[columns])[:, 1]
    signed_delta = np.where(group["server"].eq("A"), delta, -delta)
    adjusted_server = _logit_shift(raw_point_server, signed_delta)
    group["p_next_point_A"] = np.where(
        group["server"].eq("A"), adjusted_server, 1 - adjusted_server
    )
    games: list[float] = []
    sets: list[float] = []
    matches: list[float] = []
    for point in group.itertuples(index=False):
        probabilities = live_probabilities(
            sets_a=int(point.sets_A),
            sets_b=int(point.sets_B),
            games_a=int(point.games_A),
            games_b=int(point.games_B),
            points_a=int(point.points_A),
            points_b=int(point.points_B),
            server=str(point.server),
            is_tiebreak=bool(point.is_tiebreak),
            p_a_serve=p_a_serve,
            p_b_serve=p_b_serve,
            match_format=fmt,
        )
        games.append(probabilities.p_current_game_a)
        sets.append(probabilities.p_current_set_a)
        matches.append(probabilities.p_match_a)
    group["p_current_game_A"] = games
    group["p_current_set_A"] = sets
    group["p_match_A"] = matches
    group["p_A_serve"] = p_a_serve
    group["p_B_serve"] = p_b_serve
    group["consistency_logit_shift"] = delta
    group["point_index"] = np.arange(len(group))
    clear_probability_caches()
    return group


def _phase_rows(frame: pd.DataFrame) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    for _, group in frame.groupby("match_id", sort=False):
        candidates: list[tuple[str, pd.DataFrame]] = [
            ("match_start", group.iloc[[0]]),
            ("after_10_points", group.iloc[[min(10, len(group) - 1)]]),
            ("match_midpoint", group.iloc[[len(group) // 2]]),
        ]
        after_set = group.loc[(group["sets_A"] + group["sets_B"]) >= 1]
        if not after_set.empty:
            first = after_set.iloc[[0]]
            phase = (
                "after_A_wins_set"
                if int(first.iloc[0]["sets_A"]) > int(first.iloc[0]["sets_B"])
                else "after_A_loses_set"
            )
            candidates.append((phase, first))
        for phase, rows in candidates:
            rows = rows.copy()
            rows["phase"] = phase
            selected.append(rows)
    return pd.concat(selected, ignore_index=True)


def _write_plot(replays: pd.DataFrame) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    matches = list(replays["match_id"].drop_duplicates())
    if not matches:
        return False
    figure, axes = plt.subplots(len(matches), 1, figsize=(11, 3.2 * len(matches)), squeeze=False)
    for axis, match_id in zip(axes[:, 0], matches, strict=True):
        match = replays.loc[replays["match_id"] == match_id]
        year = pd.Timestamp(match.iloc[0]["date"]).year
        label = f"{year} — {match.iloc[0].player_A_name} — {match.iloc[0].player_B_name}"
        axis.plot(match["point_index"], match["p_match_A"], color="#1769aa", linewidth=1.5)
        axis.axhline(
            float(match.iloc[0]["prematch_probability_A"]),
            color="#666666",
            linestyle="--",
            linewidth=1,
            label="PRE-MATCH V1",
        )
        axis.set(title=label, ylabel="P(A gagne)", ylim=(0, 1))
        axis.grid(alpha=0.2)
        axis.legend(loc="best")
    axes[-1, 0].set_xlabel("Point du match (état avant point)")
    figure.tight_layout()
    figure.savefig(REPORTS_DIR / "live_replay_examples.png", dpi=160)
    plt.close(figure)
    return True


def finalize_existing_evaluation() -> dict[str, Any]:
    """Finish reports after an interrupted plotting step without recomputing DP states."""
    result = pd.read_parquet(OUTPUT_PATH)
    metrics = pd.read_csv(METRICS_PATH)
    replays = pd.read_csv(REPLAY_PATH)
    plotted = _write_plot(replays)
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "population": "men, test 2023-2026, PRE-MATCH V1 available",
        "points": len(result),
        "matches": int(result["match_id"].nunique()),
        "source_matches": result.groupby("source")["match_id"].nunique().to_dict(),
        "metrics": metrics.to_dict("records"),
        "replay_matches": list(replays["match_id"].drop_duplicates()),
        "plot_written": plotted,
    }
    (REPORTS_DIR / "live_probability_validation.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return summary


def run_live_evaluation() -> dict[str, Any]:
    artifact = joblib.load(MODEL_DIR / "model.joblib")
    columns = sorted(
        {
            "match_id",
            "source",
            "date",
            "surface",
            "gender",
            "player_A_name",
            "player_B_name",
            "best_of",
            "rule_label",
            "server",
            "sets_A",
            "sets_B",
            "games_A",
            "games_B",
            "points_A",
            "points_B",
            "is_tiebreak",
            "A_wins_next_point",
            "current_game_winner_A",
            "current_set_winner_A",
            "final_match_winner_A",
            "prematch_probability_A",
            *artifact["feature_columns"],
            *[
                f"{side}_{metric}"
                for side in ("A", "B")
                for metric in (
                    "serve_rate",
                    "return_rate",
                    "surface_serve_rate",
                    "surface_return_rate",
                    "serve_rate_365d",
                    "return_rate_365d",
                )
            ],
        }
    )
    frame = pd.read_parquet(FEATURE_PATH, columns=columns)
    frame["date"] = pd.to_datetime(frame["date"])
    test = frame.loc[
        (frame["gender"] == "men")
        & (frame["date"] > pd.Timestamp("2022-12-31"))
        & frame["prematch_probability_A"].notna()
    ].copy()
    evaluated = [
        _evaluate_match(group, artifact)
        for _, group in test.groupby("match_id", sort=False, observed=True)
    ]
    result = pd.concat(evaluated, ignore_index=True)
    output_columns = [
        "match_id",
        "source",
        "date",
        "surface",
        "player_A_name",
        "player_B_name",
        "point_index",
        "server",
        "sets_A",
        "sets_B",
        "games_A",
        "games_B",
        "points_A",
        "points_B",
        "is_tiebreak",
        "A_wins_next_point",
        "current_game_winner_A",
        "current_set_winner_A",
        "final_match_winner_A",
        "prematch_probability_A",
        "p_next_point_A",
        "p_current_game_A",
        "p_current_set_A",
        "p_match_A",
        "p_A_serve",
        "p_B_serve",
        "consistency_logit_shift",
    ]
    result[output_columns].to_parquet(OUTPUT_PATH, index=False)

    rows = []
    for target, probability in (
        ("next_point", "p_next_point_A"),
        ("current_game", "p_current_game_A"),
        ("current_set", "p_current_set_A"),
        ("match", "p_match_A"),
    ):
        truth = {
            "next_point": "A_wins_next_point",
            "current_game": "current_game_winner_A",
            "current_set": "current_set_winner_A",
            "match": "final_match_winner_A",
        }[target]
        rows.append(_metric_row(result[truth], result[probability], target_name=target, scope="all"))
    rows.append(
        _metric_row(
            result["final_match_winner_A"],
            result["prematch_probability_A"],
            target_name="match",
            scope="all_states",
            model="PREMATCH_STATIC",
        )
    )
    phases = _phase_rows(result)
    for phase, group in phases.groupby("phase", sort=True):
        rows.append(
            _metric_row(
                group["final_match_winner_A"],
                group["p_match_A"],
                target_name="match",
                scope=phase,
            )
        )
        rows.append(
            _metric_row(
                group["final_match_winner_A"],
                group["prematch_probability_A"],
                target_name="match",
                scope=phase,
                model="PREMATCH_STATIC",
            )
        )
    for segment_type, column in (("source", "source"), ("surface", "surface")):
        for value, group in result.groupby(column, sort=True):
            rows.append(
                _metric_row(
                    group["final_match_winner_A"],
                    group["p_match_A"],
                    target_name="match",
                    scope=f"{segment_type}:{value}",
                )
            )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(METRICS_PATH, index=False)

    calibration = result.assign(
        bucket=pd.cut(result["p_match_A"], np.linspace(0, 1, 11), include_lowest=True)
    )
    calibration = (
        calibration.groupby("bucket", observed=False)
        .agg(
            states=("final_match_winner_A", "size"),
            predicted=("p_match_A", "mean"),
            observed=("final_match_winner_A", "mean"),
        )
        .reset_index()
    )
    calibration["bucket"] = calibration["bucket"].astype(str)
    calibration.to_csv(REPORTS_DIR / "live_probability_calibration.csv", index=False)

    preferred = [
        "slam:2023-wimbledon-1701",
        "slam:2024-wimbledon-1701",
        "slam:2024-usopen-1701",
    ]
    available = [match_id for match_id in preferred if match_id in set(result["match_id"])]
    if not available:
        available = list(result["match_id"].drop_duplicates().head(3))
    replays = result.loc[result["match_id"].isin(available), output_columns].copy()
    replays.to_csv(REPLAY_PATH, index=False)
    plotted = _write_plot(replays)

    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "population": "men, test 2023-2026, PRE-MATCH V1 available",
        "points": len(result),
        "matches": int(result["match_id"].nunique()),
        "source_matches": result.groupby("source")["match_id"].nunique().to_dict(),
        "metrics": rows,
        "replay_matches": available,
        "plot_written": plotted,
    }
    (REPORTS_DIR / "live_probability_validation.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return summary
