from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from tennis_analytics.live.evaluation import match_format
from tennis_analytics.live.markov import (
    align_serve_probabilities_to_prematch,
    clear_probability_caches,
    infer_tiebreak_first_server,
    live_probabilities,
    regular_game_probability_a,
    tiebreak_probability_a,
)
from tennis_analytics.paths import PROCESSED_DIR, PROJECT_ROOT, REPORTS_DIR
from tennis_analytics.point_by_point.in_match import V2_PATH

MODEL_PATH = PROJECT_ROOT / "models" / "point_v2" / "model.joblib"
V1_LIVE_PATH = PROCESSED_DIR / "live_probabilities_test.parquet"
OUTPUT_PATH = PROCESSED_DIR / "live_v2_probabilities_test.parquet"


def _sigmoid_shift(probability: np.ndarray | float, delta: np.ndarray | float) -> np.ndarray:
    values = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(np.log(values / (1 - values)) + delta)))


def _metric(
    target: pd.Series | np.ndarray,
    probability: pd.Series | np.ndarray,
    *,
    model: str,
    target_name: str,
    scope: str,
) -> dict[str, Any]:
    y = np.asarray(target, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "model": model,
        "target": target_name,
        "scope": scope,
        "observations": len(y),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else math.nan,
    }


def _required_columns(features: list[str]) -> list[str]:
    columns = {
        "match_id",
        "date",
        "source",
        "gender",
        "surface",
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
        "v1_probability_server",
        "points_played_so_far",
        *features,
    }
    for side in ("A", "B"):
        for suffix in (
            "serve_rate",
            "return_rate",
            "surface_serve_rate",
            "surface_return_rate",
            "serve_rate_365d",
            "return_rate_365d",
            "points_won_match",
            "service_points_played_match",
            "service_points_won_match",
            "return_points_played_match",
            "return_points_won_match",
            "service_points_won_pct_so_far",
            "return_points_won_pct_so_far",
            "break_points_faced_so_far",
            "break_points_saved_so_far",
            "games_won_last_3",
            "break_recently",
            "was_broken_recently",
            "consecutive_games_won",
            "previous_set_won",
            "previous_set_margin",
            "won_previous_set_tiebreak",
            "break_clutch_history",
            "tiebreak_clutch_history",
            "deciding_clutch_history",
        ):
            columns.add(f"{side}_{suffix}")
        for window in (3, 5, 10):
            columns.add(f"{side}_points_won_last_{window}")
            columns.add(f"{side}_service_points_won_last_{window}")
            columns.add(f"{side}_return_points_won_last_{window}")
    return sorted(columns)


def _neutral_features(
    frame: pd.DataFrame, side: str, columns: list[str], prior_weight: int
) -> pd.DataFrame:
    other = "B" if side == "A" else "A"
    values: dict[str, Any] = {}
    zero = np.zeros(len(frame), dtype=float)
    neutral = {
        "is_break_point",
        "is_game_point",
        "is_set_point",
        "is_match_point",
        "is_tiebreak",
        "deuce",
        "server_leading_in_game",
        "sets_diff_server",
        "games_diff_server",
        "break_clutch_edge_active",
        "tiebreak_clutch_edge_active",
        "deciding_clutch_edge_active",
    }
    custom = {
        "server_serve_rate": f"{side}_serve_rate",
        "receiver_return_rate": f"{side}_return_rate",
        "server_surface_serve_rate": f"{side}_surface_serve_rate",
        "receiver_surface_return_rate": f"{side}_surface_return_rate",
        "server_serve_rate_365d": f"{side}_serve_rate_365d",
        "receiver_return_rate_365d": f"{side}_return_rate_365d",
    }
    for feature in columns:
        if feature in neutral:
            values[feature] = zero
        elif feature in custom:
            values[feature] = frame[custom[feature]].to_numpy()
        elif feature == "surface_clay":
            values[feature] = frame["surface"].eq("Clay").astype(int).to_numpy()
        elif feature == "surface_grass":
            values[feature] = frame["surface"].eq("Grass").astype(int).to_numpy()
        elif feature == "prematch_server":
            prior = frame["prematch_probability_A"].to_numpy()
            values[feature] = prior if side == "A" else 1 - prior
        elif feature == "points_played_so_far":
            values[feature] = frame[feature].to_numpy()
        elif feature.startswith("server_"):
            suffix = feature.removeprefix("server_")
            values[feature] = frame[f"{side}_{suffix}"].to_numpy()
        elif feature.startswith("receiver_"):
            suffix = feature.removeprefix("receiver_")
            values[feature] = frame[f"{other}_{suffix}"].to_numpy()
        elif feature.startswith("dynamic_serve_w"):
            attempts = frame[f"{side}_service_points_played_match"].to_numpy()
            wins = frame[f"{side}_service_points_won_match"].to_numpy()
            prior = frame[f"{side}_serve_rate"].to_numpy()
            values[feature] = (prior_weight * prior + wins) / (prior_weight + attempts)
        elif feature.startswith("dynamic_return_w"):
            attempts = frame[f"{other}_return_points_played_match"].to_numpy()
            wins = frame[f"{other}_return_points_won_match"].to_numpy()
            prior = frame[f"{side}_return_rate"].to_numpy()
            values[feature] = (prior_weight * prior + wins) / (prior_weight + attempts)
        elif feature.startswith("serve_today_delta_w"):
            attempts = frame[f"{side}_service_points_played_match"].to_numpy()
            wins = frame[f"{side}_service_points_won_match"].to_numpy()
            prior = frame[f"{side}_serve_rate"].to_numpy()
            values[feature] = (prior_weight * prior + wins) / (
                prior_weight + attempts
            ) - prior
        elif feature.startswith("return_today_delta_w"):
            attempts = frame[f"{other}_return_points_played_match"].to_numpy()
            wins = frame[f"{other}_return_points_won_match"].to_numpy()
            prior = frame[f"{side}_return_rate"].to_numpy()
            values[feature] = (prior_weight * prior + wins) / (
                prior_weight + attempts
            ) - prior
        else:
            raise KeyError(f"No neutral mapping for V2 feature: {feature}")
    return pd.DataFrame(values, index=frame.index)[columns]


def _format_target(frame: pd.DataFrame, index: int) -> int:
    fmt = match_format(str(frame.at[index, "rule_label"]), int(frame.at[index, "best_of"]))
    is_final = int(frame.at[index, "sets_A"] + frame.at[index, "sets_B"]) == fmt.best_of - 1
    return fmt.final_set_tiebreak_points if is_final else fmt.tiebreak_points


def run_live_evaluation_v2() -> dict[str, Any]:
    artifact = joblib.load(MODEL_PATH)
    feature_columns = artifact["feature_columns"]
    prior_weight = int(artifact["selected_prior_weight"])
    frame = pd.read_parquet(
        V2_PATH,
        columns=_required_columns(feature_columns),
        filters=[("gender", "==", "men"), ("date", ">", pd.Timestamp("2022-12-31"))],
    )
    frame = frame.loc[frame["prematch_probability_A"].notna()].reset_index(drop=True)
    frame["point_index"] = frame.groupby("match_id", sort=False).cumcount()
    pipeline = artifact["pipeline"]
    raw_current = pipeline.predict_proba(frame[feature_columns])[:, 1]
    raw_a = pipeline.predict_proba(
        _neutral_features(frame, "A", feature_columns, prior_weight)
    )[:, 1]
    raw_b = pipeline.predict_proba(
        _neutral_features(frame, "B", feature_columns, prior_weight)
    )[:, 1]

    match_deltas: dict[str, float] = {}
    for match_id, group in frame.groupby("match_id", sort=False, observed=True):
        index = int(group.index[0])
        fmt = match_format(str(group.iloc[0]["rule_label"]), int(group.iloc[0]["best_of"]))
        _, _, delta = align_serve_probabilities_to_prematch(
            float(raw_a[index]),
            float(raw_b[index]),
            float(group.iloc[0]["prematch_probability_A"]),
            first_server=str(group.iloc[0]["server"]),
            match_format=fmt,
        )
        match_deltas[str(match_id)] = delta
        clear_probability_caches()
    delta = frame["match_id"].map(match_deltas).to_numpy(dtype=float)
    p_a_serve = _sigmoid_shift(raw_a, delta)
    p_b_serve = _sigmoid_shift(raw_b, -delta)
    signed = np.where(frame["server"].eq("A"), delta, -delta)
    adjusted_server = _sigmoid_shift(raw_current, signed)
    frame["p_next_point_A_v2"] = np.where(
        frame["server"].eq("A"), adjusted_server, 1 - adjusted_server
    )
    frame["p_A_serve_v2"] = p_a_serve
    frame["p_B_serve_v2"] = p_b_serve
    frame["v2_consistency_shift"] = delta

    game_probabilities: list[float] = []
    for index, point in frame.iterrows():
        if bool(point["is_tiebreak"]):
            first = infer_tiebreak_first_server(
                str(point["server"]), int(point["points_A"] + point["points_B"])
            )
            game_probability = tiebreak_probability_a(
                int(point["points_A"]),
                int(point["points_B"]),
                first_server=first,
                p_a_serve=float(p_a_serve[index]),
                p_b_serve=float(p_b_serve[index]),
                target=_format_target(frame, index),
            )
        else:
            game_probability = regular_game_probability_a(
                int(point["points_A"]),
                int(point["points_B"]),
                server=str(point["server"]),
                p_a_serve=float(p_a_serve[index]),
                p_b_serve=float(p_b_serve[index]),
            )
        game_probabilities.append(game_probability)
    frame["p_current_game_A_v2"] = game_probabilities

    structural = frame["points_A"].eq(0) & frame["points_B"].eq(0) & frame["games_A"].eq(0) & frame["games_B"].eq(0)
    sample_mask = frame["point_index"].mod(10).eq(0) | structural
    frame["live_sampled"] = sample_mask
    frame["p_current_set_A_v2"] = np.nan
    frame["p_match_A_v2"] = np.nan
    for index in frame.index[sample_mask]:
        point = frame.loc[index]
        fmt = match_format(str(point["rule_label"]), int(point["best_of"]))
        probabilities = live_probabilities(
            sets_a=int(point["sets_A"]),
            sets_b=int(point["sets_B"]),
            games_a=int(point["games_A"]),
            games_b=int(point["games_B"]),
            points_a=int(point["points_A"]),
            points_b=int(point["points_B"]),
            server=str(point["server"]),
            is_tiebreak=bool(point["is_tiebreak"]),
            p_a_serve=float(p_a_serve[index]),
            p_b_serve=float(p_b_serve[index]),
            match_format=fmt,
        )
        frame.at[index, "p_current_set_A_v2"] = probabilities.p_current_set_a
        frame.at[index, "p_match_A_v2"] = probabilities.p_match_a
        clear_probability_caches()

    baseline_columns = [
        "match_id",
        "point_index",
        "p_next_point_A",
        "p_current_game_A",
        "p_current_set_A",
        "p_match_A",
    ]
    baseline = pd.read_parquet(V1_LIVE_PATH, columns=baseline_columns)
    frame = frame.merge(
        baseline,
        on=["match_id", "point_index"],
        how="left",
        validate="one_to_one",
    )

    metrics_rows = []
    all_targets = [
        ("next_point", "A_wins_next_point", "p_next_point_A", "p_next_point_A_v2", np.ones(len(frame), dtype=bool)),
        ("current_game", "current_game_winner_A", "p_current_game_A", "p_current_game_A_v2", np.ones(len(frame), dtype=bool)),
        ("current_set", "current_set_winner_A", "p_current_set_A", "p_current_set_A_v2", sample_mask.to_numpy()),
        ("match", "final_match_winner_A", "p_match_A", "p_match_A_v2", sample_mask.to_numpy()),
    ]
    for target_name, truth, v1_column, v2_column, mask in all_targets:
        for model, column in (("LIVE_V1", v1_column), ("LIVE_V2", v2_column)):
            metrics_rows.append(
                _metric(
                    frame.loc[mask, truth],
                    frame.loc[mask, column],
                    model=model,
                    target_name=target_name,
                    scope="all_evaluated_states",
                )
            )
    progress = pd.cut(
        frame["point_index"],
        [-1, 19, 49, 99, np.inf],
        labels=["points_1_20", "points_21_50", "points_51_100", "points_100_plus"],
    )
    for target_name, truth, v1_column, v2_column, target_mask in all_targets:
        for segment in progress.dropna().unique():
            mask = progress.eq(segment).to_numpy() & target_mask
            for model, column in (("LIVE_V1", v1_column), ("LIVE_V2", v2_column)):
                metrics_rows.append(
                    _metric(
                        frame.loc[mask, truth],
                        frame.loc[mask, column],
                        model=model,
                        target_name=target_name,
                        scope=str(segment),
                    )
                )
    for source in sorted(frame["source"].unique()):
        for target_name, truth, v1_column, v2_column, target_mask in all_targets:
            mask = frame["source"].eq(source).to_numpy() & target_mask
            for model, column in (("LIVE_V1", v1_column), ("LIVE_V2", v2_column)):
                metrics_rows.append(
                    _metric(
                        frame.loc[mask, truth],
                        frame.loc[mask, column],
                        model=model,
                        target_name=target_name,
                        scope=f"source:{source}",
                    )
                )
    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_frame.to_csv(REPORTS_DIR / "live_v2_validation.csv", index=False)

    calibration_rows = []
    for target_name, truth, v1_column, v2_column, mask in all_targets:
        for model, column in (("LIVE_V1", v1_column), ("LIVE_V2", v2_column)):
            subset = frame.loc[mask, [truth, column]].copy()
            subset["bucket"] = pd.cut(
                subset[column], np.linspace(0, 1, 11), include_lowest=True
            )
            table = (
                subset.groupby("bucket", observed=False)
                .agg(states=(truth, "size"), predicted=(column, "mean"), observed=(truth, "mean"))
                .reset_index()
            )
            table.insert(0, "target", target_name)
            table.insert(0, "model", model)
            calibration_rows.append(table)
    calibration = pd.concat(calibration_rows, ignore_index=True)
    calibration["bucket"] = calibration["bucket"].astype(str)
    calibration.to_csv(REPORTS_DIR / "live_v2_calibration.csv", index=False)

    favored = (
        frame["prematch_probability_A"].gt(0.5)
        & frame["sets_A"].eq(0)
        & frame["sets_B"].eq(0)
        & sample_mask
        & frame["p_match_A_v2"].lt(0.5)
    )
    reversals = (
        frame.loc[favored]
        .sort_values(["match_id", "point_index"])
        .groupby("match_id", as_index=False)
        .first()
    )
    if not reversals.empty:
        reversals["reversal_correct_B_won"] = 1 - reversals["final_match_winner_A"]
    reversal_columns = [
        "match_id",
        "date",
        "source",
        "surface",
        "player_A_name",
        "player_B_name",
        "point_index",
        "prematch_probability_A",
        "p_match_A",
        "p_match_A_v2",
        "final_match_winner_A",
        "reversal_correct_B_won",
    ]
    reversals.reindex(columns=reversal_columns).to_csv(
        REPORTS_DIR / "live_v2_first_set_reversals.csv", index=False
    )

    frame["absolute_dynamic_delta"] = np.maximum(
        np.abs(frame[f"serve_today_delta_w{prior_weight}"]),
        np.abs(frame[f"return_today_delta_w{prior_weight}"]),
    )
    anomalies = (
        frame.sort_values("absolute_dynamic_delta", ascending=False)
        .groupby("match_id", as_index=False)
        .first()
        .sort_values("absolute_dynamic_delta", ascending=False)
        .head(100)
    )
    anomaly_columns = [
        "match_id",
        "date",
        "source",
        "player_A_name",
        "player_B_name",
        "point_index",
        f"serve_today_delta_w{prior_weight}",
        f"return_today_delta_w{prior_weight}",
        "absolute_dynamic_delta",
        "p_next_point_A",
        "p_next_point_A_v2",
        "final_match_winner_A",
    ]
    anomalies[anomaly_columns].to_csv(REPORTS_DIR / "live_v2_anomalies.csv", index=False)

    output_columns = [
        "match_id",
        "date",
        "source",
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
        "p_next_point_A_v2",
        "p_current_game_A",
        "p_current_game_A_v2",
        "p_current_set_A",
        "p_current_set_A_v2",
        "p_match_A",
        "p_match_A_v2",
        "p_A_serve_v2",
        "p_B_serve_v2",
        "v2_consistency_shift",
        "live_sampled",
    ]
    frame[output_columns].to_parquet(OUTPUT_PATH, index=False)
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "states": len(frame),
        "matches": int(frame["match_id"].nunique()),
        "set_match_sampled_states": int(sample_mask.sum()),
        "sample_policy": "every tenth point plus every set start",
        "selected_prior_weight": prior_weight,
        "first_set_reversals": len(reversals),
        "reversal_success_rate": (
            float(reversals["reversal_correct_B_won"].mean()) if len(reversals) else None
        ),
        "metrics": metrics_rows,
    }
    (REPORTS_DIR / "live_v2_validation.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return summary
