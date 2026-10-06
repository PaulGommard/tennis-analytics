from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd

from tennis_analytics.modeling.point_model import (
    FEATURE_BLOCKS,
    TRAIN_END,
    VALIDATION_END,
    calibration_table,
    make_pipeline,
    metrics,
)
from tennis_analytics.paths import PROJECT_ROOT, REPORTS_DIR
from tennis_analytics.point_by_point.in_match import PRIOR_WEIGHTS, V2_PATH

MODEL_DIR = PROJECT_ROOT / "models" / "point_v2"
BASE_FEATURES = FEATURE_BLOCKS["P5_RECENT_SCORE"]

CUMULATIVE_FEATURES = [
    "points_played_so_far",
    "server_service_points_played_match",
    "server_service_points_won_match",
    "receiver_return_points_played_match",
    "receiver_return_points_won_match",
    "server_service_points_won_pct_so_far",
    "receiver_return_points_won_pct_so_far",
    "server_break_points_faced_so_far",
    "server_break_points_saved_so_far",
]
RECENT_POINT_FEATURES = [
    *[f"server_points_won_last_{window}" for window in (3, 5, 10)],
    *[f"receiver_points_won_last_{window}" for window in (3, 5, 10)],
    *[f"server_service_points_won_last_{window}" for window in (3, 5, 10)],
    *[f"receiver_return_points_won_last_{window}" for window in (3, 5, 10)],
]
GAME_FEATURES = [
    "server_games_won_last_3",
    "receiver_games_won_last_3",
    "server_break_recently",
    "receiver_break_recently",
    "server_was_broken_recently",
    "receiver_was_broken_recently",
    "server_consecutive_games_won",
    "receiver_consecutive_games_won",
]
SET_FEATURES = [
    "server_previous_set_won",
    "receiver_previous_set_won",
    "server_previous_set_margin",
    "receiver_previous_set_margin",
    "server_won_previous_set_tiebreak",
    "receiver_won_previous_set_tiebreak",
]
CLUTCH_FEATURES = [
    "break_clutch_edge_active",
    "tiebreak_clutch_edge_active",
    "deciding_clutch_edge_active",
]


def dynamic_features(weight: int) -> list[str]:
    return [
        f"dynamic_serve_w{weight}",
        f"dynamic_return_w{weight}",
        f"serve_today_delta_w{weight}",
        f"return_today_delta_w{weight}",
    ]


def _effect(
    frame: pd.DataFrame,
    mask: pd.Series,
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    split: str,
    model: str,
) -> dict[str, Any]:
    target = frame.loc[mask, "server_won_point"].to_numpy(dtype=int)
    base = np.clip(baseline, 1e-6, 1 - 1e-6)
    tested = np.clip(candidate, 1e-6, 1 - 1e-6)
    loss_base = -(target * np.log(base) + (1 - target) * np.log(1 - base))
    loss_tested = -(target * np.log(tested) + (1 - target) * np.log(1 - tested))
    paired = pd.DataFrame(
        {
            "match_id": frame.loc[mask, "match_id"].to_numpy(),
            "improvement": loss_base - loss_tested,
        }
    )
    by_match = paired.groupby("match_id")["improvement"].mean()
    standard_error = float(by_match.std(ddof=1) / np.sqrt(len(by_match)))
    mean = float(by_match.mean())
    return {
        "model": model,
        "split": split,
        "matches": len(by_match),
        "points": len(paired),
        "point_weighted_log_loss_improvement": float(paired["improvement"].mean()),
        "mean_match_log_loss_improvement": mean,
        "ci95_low": mean - 1.96 * standard_error,
        "ci95_high": mean + 1.96 * standard_error,
        "share_matches_improved": float((by_match > 0).mean()),
    }


def _load_frame() -> pd.DataFrame:
    all_features = {
        *BASE_FEATURES,
        *CUMULATIVE_FEATURES,
        *RECENT_POINT_FEATURES,
        *GAME_FEATURES,
        *SET_FEATURES,
        *CLUTCH_FEATURES,
        *[feature for weight in PRIOR_WEIGHTS for feature in dynamic_features(weight)],
    }
    columns = sorted(
        {
            "match_id",
            "date",
            "source",
            "gender",
            "surface",
            "server_won_point",
            "v1_probability_server",
            *all_features,
        }
    )
    frame = pd.read_parquet(
        V2_PATH,
        columns=columns,
        filters=[("gender", "==", "men")],
    )
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def run_point_model_v2() -> dict[str, Any]:
    frame = _load_frame()
    train = frame["date"] <= TRAIN_END
    validation = frame["date"].between(
        TRAIN_END + pd.Timedelta(days=1), VALIDATION_END
    )
    test = frame["date"] > VALIDATION_END

    tuning_rows: list[dict[str, Any]] = []
    dynamic_pipelines: dict[int, Any] = {}
    for weight in PRIOR_WEIGHTS:
        columns = BASE_FEATURES + dynamic_features(weight)
        pipeline = make_pipeline()
        pipeline.fit(frame.loc[train, columns], frame.loc[train, "server_won_point"])
        dynamic_pipelines[weight] = pipeline
        for split_name, mask in (("validation", validation), ("test", test)):
            probability = pipeline.predict_proba(frame.loc[mask, columns])[:, 1]
            tuning_rows.append(
                {
                    "prior_weight": weight,
                    "split": split_name,
                    "points": int(mask.sum()),
                    **metrics(frame.loc[mask, "server_won_point"], probability),
                }
            )
    tuning = pd.DataFrame(tuning_rows)
    tuning.to_csv(REPORTS_DIR / "point_v2_prior_weight_tuning.csv", index=False)
    selected_weight = int(
        tuning.loc[tuning["split"] == "validation"].sort_values("log_loss").iloc[0][
            "prior_weight"
        ]
    )

    candidates = {
        "P1_CUMULATIVE": BASE_FEATURES + CUMULATIVE_FEATURES,
        "P2_DYNAMIC": BASE_FEATURES + dynamic_features(selected_weight),
        "P3_POINT_MOMENTUM": BASE_FEATURES + RECENT_POINT_FEATURES,
        "P4_GAME_SET_PRESSURE": BASE_FEATURES + GAME_FEATURES + SET_FEATURES + CLUTCH_FEATURES,
        "P5_ALL": list(
            dict.fromkeys(
                BASE_FEATURES
                + CUMULATIVE_FEATURES
                + dynamic_features(selected_weight)
                + RECENT_POINT_FEATURES
                + GAME_FEATURES
                + SET_FEATURES
                + CLUTCH_FEATURES
            )
        ),
        "EXP_GAMES": BASE_FEATURES + GAME_FEATURES,
        "EXP_PREVIOUS_SET": BASE_FEATURES + SET_FEATURES,
        "EXP_CLUTCH": BASE_FEATURES + CLUTCH_FEATURES,
    }
    pipelines: dict[str, Any] = {"P2_DYNAMIC": dynamic_pipelines[selected_weight]}
    rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, str], np.ndarray] = {}
    for split_name, mask in (("validation", validation), ("test", test)):
        probability = frame.loc[mask, "v1_probability_server"].to_numpy()
        predictions[("P0_POINT_V1", split_name)] = probability
        rows.append(
            {
                "model": "P0_POINT_V1",
                "split": split_name,
                "points": int(mask.sum()),
                **metrics(frame.loc[mask, "server_won_point"], probability),
            }
        )
    for name, columns in candidates.items():
        pipeline = pipelines.get(name)
        if pipeline is None:
            pipeline = make_pipeline()
            pipeline.fit(frame.loc[train, columns], frame.loc[train, "server_won_point"])
            pipelines[name] = pipeline
        for split_name, mask in (("validation", validation), ("test", test)):
            probability = pipeline.predict_proba(frame.loc[mask, columns])[:, 1]
            predictions[(name, split_name)] = probability
            rows.append(
                {
                    "model": name,
                    "split": split_name,
                    "points": int(mask.sum()),
                    **metrics(frame.loc[mask, "server_won_point"], probability),
                }
            )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(REPORTS_DIR / "point_v2_comparison.csv", index=False)
    selected_v2 = str(
        comparison.loc[
            (comparison["split"] == "validation")
            & comparison["model"].str.startswith("P")
            & comparison["model"].ne("P0_POINT_V1")
        ]
        .sort_values("log_loss")
        .iloc[0]["model"]
    )
    selected_columns = candidates[selected_v2]
    selected_pipeline = pipelines[selected_v2]

    effects = []
    for name in candidates:
        for split_name, mask in (("validation", validation), ("test", test)):
            effects.append(
                _effect(
                    frame,
                    mask,
                    predictions[("P0_POINT_V1", split_name)],
                    predictions[(name, split_name)],
                    split=split_name,
                    model=name,
                )
            )
    effects_frame = pd.DataFrame(effects)
    effects_frame.to_csv(REPORTS_DIR / "point_v2_clustered_effects.csv", index=False)

    segment_rows = []
    test_points = frame.loc[test, "points_played_so_far"]
    buckets = pd.cut(
        test_points,
        [-1, 19, 49, 99, np.inf],
        labels=["points_1_20", "points_21_50", "points_51_100", "points_100_plus"],
    )
    for segment, segment_index in buckets.groupby(buckets, observed=True).groups.items():
        positions = frame.index.get_indexer(segment_index)
        test_positions = np.flatnonzero(test.to_numpy())
        relative = np.flatnonzero(np.isin(test_positions, positions))
        target = frame.loc[segment_index, "server_won_point"]
        for name in ("P0_POINT_V1", selected_v2):
            probability = predictions[(name, "test")][relative]
            segment_rows.append(
                {
                    "segment_type": "match_progress",
                    "segment": str(segment),
                    "model": name,
                    "points": len(target),
                    **metrics(target, probability),
                }
            )
    for segment_type, column in (("source", "source"), ("surface", "surface")):
        test_frame = frame.loc[test]
        for value in sorted(test_frame[column].dropna().unique()):
            relative = test_frame[column].eq(value).to_numpy()
            target = test_frame.loc[relative, "server_won_point"]
            for name in ("P0_POINT_V1", selected_v2):
                probability = predictions[(name, "test")][relative]
                segment_rows.append(
                    {
                        "segment_type": segment_type,
                        "segment": value,
                        "model": name,
                        "points": len(target),
                        **metrics(target, probability),
                    }
                )
    segments = pd.DataFrame(segment_rows)
    segments.to_csv(REPORTS_DIR / "point_v2_segments.csv", index=False)

    test_target = frame.loc[test, "server_won_point"]
    calibration_frames = []
    for name in ("P0_POINT_V1", selected_v2):
        calibration = calibration_table(test_target, predictions[(name, "test")])
        calibration.insert(0, "model", name)
        calibration_frames.append(calibration)
    calibration = pd.concat(calibration_frames, ignore_index=True)
    calibration.to_csv(REPORTS_DIR / "point_v2_calibration.csv", index=False)

    coefficient_rows = []
    for name in ("P3_POINT_MOMENTUM", "EXP_GAMES", "EXP_PREVIOUS_SET", "EXP_CLUTCH", selected_v2):
        pipeline = pipelines[name]
        for feature, coefficient in zip(
            candidates[name], pipeline.named_steps["logistic"].coef_[0], strict=True
        ):
            coefficient_rows.append(
                {
                    "model": name,
                    "feature": feature,
                    "standardized_coefficient": coefficient,
                }
            )
    coefficients = pd.DataFrame(coefficient_rows)
    coefficients.to_csv(REPORTS_DIR / "point_v2_coefficients.csv", index=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_version": "point_v2_experimental",
        "selected_model": selected_v2,
        "feature_columns": selected_columns,
        "pipeline": selected_pipeline,
        "selected_prior_weight": selected_weight,
        "v1_baseline": "models/point_v1/model.joblib",
        "train_end": TRAIN_END.date().isoformat(),
        "validation_end": VALIDATION_END.date().isoformat(),
    }
    joblib.dump(artifact, MODEL_DIR / "model.joblib")
    validation_rows = comparison.loc[comparison["split"] == "validation"].set_index("model")
    test_rows = comparison.loc[comparison["split"] == "test"].set_index("model")
    recommended = bool(
        validation_rows.loc[selected_v2, "log_loss"]
        < validation_rows.loc["P0_POINT_V1", "log_loss"]
        and test_rows.loc[selected_v2, "log_loss"]
        < test_rows.loc["P0_POINT_V1", "log_loss"]
    )
    metadata = {
        key: value for key, value in artifact.items() if key != "pipeline"
    }
    metadata.update(
        {
            "created_at": datetime.now(UTC).isoformat(),
            "dataset_rows_men": len(frame),
            "recommended_to_replace_v1": recommended,
            "recommendation_scope": (
                "next-point candidate only; LIVE match promotion requires separate validation"
            ),
            "selection_metric": "validation log_loss",
        }
    )
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (MODEL_DIR / "feature_schema.json").write_text(
        json.dumps(
            {
                "target": "server_won_point",
                "state_timing": "strictly before current point",
                "ordered_features": selected_columns,
                "candidate_features": candidates,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "selected": selected_v2,
        "selected_prior_weight": selected_weight,
        "recommended_to_replace_v1": recommended,
        "validation": validation_rows.loc[selected_v2].to_dict(),
        "test": test_rows.loc[selected_v2].to_dict(),
        "v1_test": test_rows.loc["P0_POINT_V1"].to_dict(),
    }
