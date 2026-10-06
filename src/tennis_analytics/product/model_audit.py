from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def prediction_decomposition(
    artifact: dict[str, Any], feature_values: dict[str, float | None]
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Decompose the frozen logistic model and its Platt calibration exactly."""
    columns = list(artifact["feature_columns"])
    frame = pd.DataFrame([[feature_values[name] for name in columns]], columns=columns)
    pipeline = artifact["pipeline"]
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    logistic = pipeline.named_steps["logistic"]
    imputed = imputer.transform(frame)[0]
    standardized = scaler.transform(imputed.reshape(1, -1))[0]
    coefficients = logistic.coef_[0]
    contributions = standardized * coefficients
    raw_logit = float(logistic.intercept_[0] + contributions.sum())
    raw_probability = 1 / (1 + math.exp(-raw_logit))
    calibrator = artifact.get("calibrator")
    if calibrator is None:
        calibrated_logit = raw_logit
        final_probability = raw_probability
        calibration_intercept = 0.0
        calibration_slope = 1.0
    else:
        calibration_intercept = float(calibrator.intercept_[0])
        calibration_slope = float(calibrator.coef_[0, 0])
        calibrated_logit = calibration_intercept + calibration_slope * raw_logit
        final_probability = 1 / (1 + math.exp(-calibrated_logit))
    rows = pd.DataFrame(
        {
            "feature": columns,
            "raw_value": [feature_values[name] for name in columns],
            "imputed_value": imputed,
            "imputer_median": imputer.statistics_,
            "mean": scaler.mean_,
            "std": scaler.scale_,
            "standardized_value": standardized,
            "coefficient": coefficients,
            "contribution_to_logit": contributions,
            "direction": np.where(
                contributions > 0,
                "favors_player_A",
                np.where(contributions < 0, "favors_player_B", "neutral"),
            ),
            "was_missing": frame.iloc[0].isna().to_numpy(),
        }
    ).sort_values("contribution_to_logit", key=lambda values: values.abs(), ascending=False)
    summary = {
        "intercept": float(logistic.intercept_[0]),
        "sum_contributions": float(contributions.sum()),
        "raw_logit": raw_logit,
        "raw_probability": raw_probability,
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
        "calibrated_logit": calibrated_logit,
        "final_probability": final_probability,
        "sum_positive": float(contributions[contributions > 0].sum()),
        "sum_negative": float(contributions[contributions < 0].sum()),
    }
    return rows.reset_index(drop=True), summary


def reverse_feature_values(
    feature_values: dict[str, float | None],
) -> dict[str, float | None]:
    """Swap A/B orientation under the Phase 5 feature contract."""
    reversed_values = dict(feature_values)
    for name, value in feature_values.items():
        if value is None:
            continue
        antisymmetric = (
            name.endswith("_diff")
            or "_diff_x_" in name
            or name in {"elo_logit", "elo_logit_x_clay", "elo_logit_x_grass", "h2h_edge_A"}
        )
        if antisymmetric:
            reversed_values[name] = -float(value)
    return reversed_values


def diagnostic_ablation_probability(
    artifact: dict[str, Any],
    feature_values: dict[str, float | None],
    removed_features: set[str],
) -> float:
    """Set selected standardized coordinates to zero for diagnosis only."""
    columns = list(artifact["feature_columns"])
    frame = pd.DataFrame([[feature_values[name] for name in columns]], columns=columns)
    pipeline = artifact["pipeline"]
    standardized = pipeline.named_steps["scaler"].transform(
        pipeline.named_steps["imputer"].transform(frame)
    )[0]
    for index, name in enumerate(columns):
        if name in removed_features:
            standardized[index] = 0.0
    logistic = pipeline.named_steps["logistic"]
    raw_logit = float(logistic.intercept_[0] + standardized @ logistic.coef_[0])
    calibrator = artifact.get("calibrator")
    if calibrator is None:
        return 1 / (1 + math.exp(-raw_logit))
    calibrated_logit = float(calibrator.intercept_[0] + calibrator.coef_[0, 0] * raw_logit)
    return 1 / (1 + math.exp(-calibrated_logit))
