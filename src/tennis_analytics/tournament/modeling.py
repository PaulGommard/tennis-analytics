from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tennis_analytics.paths import PROCESSED_DIR, REPORTS_DIR
from tennis_analytics.tournament.replay import (
    MODEL_DIR,
    OUTPUT_PATH,
    PLAYER_PRIORS,
    TOURNAMENT_PRIORS,
)

WORKLOAD = [
    "matches_played_diff",
    "sets_played_diff",
    "games_played_diff",
    "sets_lost_diff",
    "games_lost_diff",
    "tiebreaks_diff",
    "five_set_matches_diff",
    "retirements_diff",
]
MINUTES = ["minutes_played_diff", "minutes_coverage_diff"]
RAW_PERFORMANCE = [
    "service_points_won_rate_tournament_diff",
    "return_points_won_rate_tournament_diff",
    "ace_rate_tournament_diff",
    "double_fault_rate_tournament_diff",
    "break_rate_tournament_diff",
]
ROUND = [
    f"round_{label}_x_logit_p_base"
    for label in ("R128", "R64", "R32", "R16", "QF", "SF", "F", "RR", "OTHER")
]
HISTORY = [
    "same_tournament_history_matches_diff",
    "same_tournament_wins_above_expected_c20_diff",
]
CONTEXT = [
    f"{name}_x_logit_p_base"
    for name in (
        "surface_clay",
        "surface_grass",
        "level_grand_slam",
        "level_masters",
        "best_of_5",
    )
]


def performance_features(prior: int) -> list[str]:
    return [
        f"{name}_c{prior}_diff"
        for name in (
            "wins_above_expected",
            "serve_over_expected",
            "return_over_expected",
            "ace_over_expected",
            "df_over_expected",
            "break_over_expected",
        )
    ]


def environment_features(prior: int) -> list[str]:
    return [
        f"tournament_serve_environment_c{prior}_x_style_serve_diff",
        f"tournament_return_environment_c{prior}_x_style_return_diff",
        f"tournament_ace_environment_c{prior}_x_style_ace_diff",
        f"tournament_df_environment_c{prior}_x_style_df_diff",
        f"tournament_break_environment_c{prior}_x_style_return_diff",
        f"tournament_speed_proxy_c{prior}_x_style_speed_diff",
        f"tournament_upset_environment_c{prior}_x_logit_p_base",
        f"tournament_workload_environment_c{prior}_x_games_played_diff",
    ]


def _sigmoid(value: np.ndarray) -> np.ndarray:
    positive = value >= 0
    result = np.empty_like(value, dtype=float)
    result[positive] = 1 / (1 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    result[~positive] = exponential / (1 + exponential)
    return result


@dataclass
class TournamentOffsetModel:
    feature_columns: list[str]
    regularization: float = 1.0
    medians_: np.ndarray | None = None
    scales_: np.ndarray | None = None
    coefficients_: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> TournamentOffsetModel:
        values = frame[self.feature_columns].to_numpy(dtype=float)
        # Every adapter feature is oriented A-minus-B. Missing tournament
        # information therefore means "no correction", represented by zero.
        medians = np.zeros(values.shape[1], dtype=float)
        values = np.where(np.isfinite(values), values, medians)
        scales = values.std(axis=0)
        scales[scales < 1e-12] = 1.0
        design = values / scales
        offset = frame["logit_p_base"].to_numpy(dtype=float)
        y = target.to_numpy(dtype=float)
        beta = np.zeros(design.shape[1], dtype=float)
        penalty = np.eye(design.shape[1]) * self.regularization
        for _ in range(100):
            probability = _sigmoid(offset + design @ beta)
            weights = np.clip(probability * (1 - probability), 1e-8, None)
            gradient = design.T @ (y - probability) - penalty @ beta
            hessian = (design.T * weights) @ design + penalty
            step = np.linalg.solve(hessian, gradient)
            beta += step
            if np.max(np.abs(step)) < 1e-8:
                break
        self.medians_ = medians
        self.scales_ = scales
        self.coefficients_ = beta
        return self

    def _design(self, frame: pd.DataFrame) -> np.ndarray:
        if any(value is None for value in (self.medians_, self.scales_)):
            raise RuntimeError("TournamentOffsetModel is not fitted")
        values = frame[self.feature_columns].to_numpy(dtype=float)
        values = np.where(np.isfinite(values), values, self.medians_)
        return values / self.scales_

    def correction(self, frame: pd.DataFrame) -> np.ndarray:
        if self.coefficients_ is None:
            raise RuntimeError("TournamentOffsetModel is not fitted")
        return self._design(frame) @ self.coefficients_

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        probability = _sigmoid(
            frame["logit_p_base"].to_numpy(dtype=float) + self.correction(frame)
        )
        return np.column_stack([1 - probability, probability])

    def contributions(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.coefficients_ is None:
            raise RuntimeError("TournamentOffsetModel is not fitted")
        design = self._design(frame)
        values = design * self.coefficients_
        result = pd.DataFrame(values, columns=self.feature_columns, index=frame.index)
        return result


@dataclass
class LogitScaleCalibrator:
    """One-parameter temperature calibration preserving A/B symmetry."""

    slope: float = 1.0

    def transform(self, probability: np.ndarray) -> np.ndarray:
        probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
        logits = np.log(probability / (1 - probability))
        return _sigmoid(self.slope * logits)


def _fit_logit_scale(probability: np.ndarray, target: pd.Series) -> LogitScaleCalibrator:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    x = np.log(probability / (1 - probability))
    y = np.asarray(target, dtype=float)
    slope = 1.0
    for _ in range(100):
        fitted = _sigmoid(slope * x)
        weight = np.clip(fitted * (1 - fitted), 1e-8, None)
        step = np.sum(x * (y - fitted)) / (np.sum(weight * x * x) + 1e-8)
        slope += step
        if abs(step) < 1e-10:
            break
    return LogitScaleCalibrator(float(slope))


def _metrics(target: pd.Series | np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(target, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "matches": len(y),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan"),
    }


def _walk_forward(
    frame: pd.DataFrame,
    features: list[str],
    regularization: float,
) -> pd.DataFrame:
    rows = []
    for year in (2019, 2020, 2021, 2022):
        train = frame["year"] < year
        validation = frame["year"] == year
        if not validation.any():
            continue
        model = TournamentOffsetModel(features, regularization).fit(
            frame.loc[train], frame.loc[train, "player_A_win"]
        )
        prediction = model.predict_proba(frame.loc[validation])[:, 1]
        part = frame.loc[
            validation,
            ["match_id", "tournament_edition_id", "year", "round", "p_base", "player_A_win"],
        ].copy()
        part["p_adjusted"] = prediction
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def _candidate_blocks(player_prior: int, tournament_prior: int) -> dict[str, list[str]]:
    performance = performance_features(player_prior)
    environment = environment_features(tournament_prior)
    return {
        "T1_WORKLOAD": WORKLOAD,
        "T1B_WORKLOAD_WITH_MINUTES": WORKLOAD + MINUTES,
        "T2_PERFORMANCE_VS_EXPECTED": performance,
        "T2B_RAW_TOURNAMENT_STATS": RAW_PERFORMANCE,
        "T2C_RAW_PLUS_EXPECTED": RAW_PERFORMANCE + performance,
        "T3_TOURNAMENT_ENVIRONMENT": environment,
        "T4_ROUND": ROUND,
        "T5_SAME_TOURNAMENT_HISTORY": HISTORY,
        "T6_ALL_NO_MINUTES": list(
            dict.fromkeys(
                WORKLOAD + RAW_PERFORMANCE + performance + environment + ROUND + HISTORY + CONTEXT
            )
        ),
        "T6B_ALL_WITH_MINUTES": list(
            dict.fromkeys(
                WORKLOAD
                + MINUTES
                + RAW_PERFORMANCE
                + performance
                + environment
                + ROUND
                + HISTORY
                + CONTEXT
            )
        ),
    }


def _cap(probability: np.ndarray, baseline: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return probability
    if mode.startswith("probability_"):
        maximum = float(mode.split("_")[1])
        return np.clip(probability, baseline - maximum, baseline + maximum)
    if mode.startswith("logit_"):
        maximum = float(mode.split("_")[1])
        base_logit = np.log(baseline / (1 - baseline))
        candidate_logit = np.log(probability / (1 - probability))
        return _sigmoid(base_logit + np.clip(candidate_logit - base_logit, -maximum, maximum))
    raise ValueError(f"Unknown cap: {mode}")


def _clustered_gain(frame: pd.DataFrame, candidate: np.ndarray) -> dict[str, float]:
    y = frame["player_A_win"].to_numpy(dtype=int)
    baseline = np.clip(frame["p_base"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    candidate = np.clip(candidate, 1e-6, 1 - 1e-6)
    base_loss = -(y * np.log(baseline) + (1 - y) * np.log(1 - baseline))
    candidate_loss = -(y * np.log(candidate) + (1 - y) * np.log(1 - candidate))
    grouped = pd.DataFrame(
        {"edition": frame["tournament_edition_id"], "gain": base_loss - candidate_loss}
    ).groupby("edition")["gain"].mean()
    se = grouped.std(ddof=1) / math.sqrt(len(grouped))
    mean = grouped.mean()
    return {
        "mean_edition_log_loss_gain": float(mean),
        "ci95_low": float(mean - 1.96 * se),
        "ci95_high": float(mean + 1.96 * se),
        "share_editions_improved": float((grouped > 0).mean()),
    }


def train_tournament_adapter() -> dict[str, Any]:
    frame = pd.read_parquet(OUTPUT_PATH)
    frame["date"] = pd.to_datetime(frame["date"])
    # Select shrinkage only with walk-forward validation predictions.
    shrink_rows = []
    for player_prior in PLAYER_PRIORS:
        for tournament_prior in TOURNAMENT_PRIORS:
            columns = _candidate_blocks(player_prior, tournament_prior)["T6_ALL_NO_MINUTES"]
            predictions = _walk_forward(frame, columns, regularization=1.0)
            shrink_rows.append(
                {
                    "player_prior": player_prior,
                    "tournament_prior": tournament_prior,
                    **_metrics(predictions["player_A_win"], predictions["p_adjusted"]),
                }
            )
    shrinkage = pd.DataFrame(shrink_rows).sort_values(["log_loss", "brier"])
    shrinkage.to_csv(REPORTS_DIR / "tournament_adapter_shrinkage_validation.csv", index=False)
    selected_player_prior = int(shrinkage.iloc[0]["player_prior"])
    selected_tournament_prior = int(shrinkage.iloc[0]["tournament_prior"])
    blocks = _candidate_blocks(selected_player_prior, selected_tournament_prior)

    regularization_rows = []
    for regularization in (0.1, 1.0, 10.0, 100.0):
        predictions = _walk_forward(
            frame, blocks["T6_ALL_NO_MINUTES"], regularization=regularization
        )
        regularization_rows.append(
            {"regularization": regularization, **_metrics(predictions["player_A_win"], predictions["p_adjusted"])}
        )
    regularization_frame = pd.DataFrame(regularization_rows).sort_values(["log_loss", "brier"])
    regularization_frame.to_csv(
        REPORTS_DIR / "tournament_adapter_regularization_validation.csv", index=False
    )
    selected_regularization = float(regularization_frame.iloc[0]["regularization"])

    ablation_rows = [
        {
            "model": "T0_PREMATCH_V1",
            "features": 0,
            **_metrics(
                frame.loc[frame["year"].between(2019, 2022), "player_A_win"],
                frame.loc[frame["year"].between(2019, 2022), "p_base"].to_numpy(),
            ),
        }
    ]
    ablation_predictions: dict[str, pd.DataFrame] = {}
    for name, columns in blocks.items():
        predictions = _walk_forward(frame, columns, selected_regularization)
        ablation_predictions[name] = predictions
        ablation_rows.append(
            {
                "model": name,
                "features": len(columns),
                **_metrics(predictions["player_A_win"], predictions["p_adjusted"]),
            }
        )
    ablation = pd.DataFrame(ablation_rows).sort_values(["log_loss", "brier"])
    ablation.to_csv(REPORTS_DIR / "tournament_adapter_ablation_validation.csv", index=False)
    selected_name = str(ablation.loc[ablation["model"] != "T0_PREMATCH_V1"].iloc[0]["model"])
    selected_features = blocks[selected_name]
    validation_predictions = ablation_predictions[selected_name]

    cap_rows = []
    capped_predictions: dict[str, np.ndarray] = {}
    baseline_validation = validation_predictions["p_base"].to_numpy(dtype=float)
    raw_validation = validation_predictions["p_adjusted"].to_numpy(dtype=float)
    for cap_mode in ("none", "probability_0.05", "probability_0.10", "logit_0.5"):
        values = _cap(raw_validation, baseline_validation, cap_mode)
        capped_predictions[cap_mode] = values
        cap_rows.append(
            {"cap": cap_mode, **_metrics(validation_predictions["player_A_win"], values)}
        )
    cap_frame = pd.DataFrame(cap_rows).sort_values(["log_loss", "brier"])
    cap_frame.to_csv(REPORTS_DIR / "tournament_adapter_cap_validation.csv", index=False)
    selected_cap = str(cap_frame.iloc[0]["cap"])
    validation_capped = capped_predictions[selected_cap]

    # Calibration is trained on 2019-2020 OOF and selected on untouched 2021-2022 OOF.
    calibration_train = validation_predictions["year"].between(2019, 2020).to_numpy()
    calibration_eval = validation_predictions["year"].between(2021, 2022).to_numpy()
    calibrator = _fit_logit_scale(
        validation_capped[calibration_train],
        validation_predictions.loc[calibration_train, "player_A_win"],
    )
    raw_eval = validation_capped[calibration_eval]
    calibrated_eval = calibrator.transform(raw_eval)
    calibration_frame = pd.DataFrame(
        [
            {
                "model": "raw",
                **_metrics(validation_predictions.loc[calibration_eval, "player_A_win"], raw_eval),
            },
            {
                "model": "temperature",
                **_metrics(
                    validation_predictions.loc[calibration_eval, "player_A_win"], calibrated_eval
                ),
            },
        ]
    )
    calibration_frame.to_csv(
        REPORTS_DIR / "tournament_adapter_calibration_validation.csv", index=False
    )
    raw_row = calibration_frame.loc[calibration_frame["model"] == "raw"].iloc[0]
    calibrated_row = calibration_frame.loc[
        calibration_frame["model"] == "temperature"
    ].iloc[0]
    use_calibration = bool(
        calibrated_row["log_loss"] < raw_row["log_loss"]
        and calibrated_row["brier"] <= raw_row["brier"]
    )

    # A temperature-only recalibration of PRE-MATCH V1 is a necessary control:
    # it reveals whether a test gain comes from context or calibration drift.
    base_calibrator = _fit_logit_scale(
        validation_predictions.loc[calibration_train, "p_base"].to_numpy(),
        validation_predictions.loc[calibration_train, "player_A_win"],
    )
    base_eval_raw = validation_predictions.loc[calibration_eval, "p_base"].to_numpy()
    base_eval_calibrated = base_calibrator.transform(base_eval_raw)
    base_use_calibration = bool(
        _metrics(
            validation_predictions.loc[calibration_eval, "player_A_win"],
            base_eval_calibrated,
        )["log_loss"]
        < _metrics(
            validation_predictions.loc[calibration_eval, "player_A_win"], base_eval_raw
        )["log_loss"]
    )

    # Controlled comparison with a full logistic that can relearn the baseline coefficient.
    full_columns = ["logit_p_base", *selected_features]
    full_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(max_iter=500, C=1.0)),
        ]
    )
    train_final = frame["year"] <= 2022
    test = frame["year"] >= 2023
    offset_model = TournamentOffsetModel(selected_features, selected_regularization).fit(
        frame.loc[train_final], frame.loc[train_final, "player_A_win"]
    )
    full_pipeline.fit(frame.loc[train_final, full_columns], frame.loc[train_final, "player_A_win"])
    test_frame = frame.loc[test].copy()
    offset_raw = offset_model.predict_proba(test_frame)[:, 1]
    offset_capped = _cap(offset_raw, test_frame["p_base"].to_numpy(dtype=float), selected_cap)
    offset_final = calibrator.transform(offset_capped) if use_calibration else offset_capped
    base_temperature = (
        base_calibrator.transform(test_frame["p_base"].to_numpy(dtype=float))
        if base_use_calibration
        else test_frame["p_base"].to_numpy(dtype=float)
    )
    full_probability = full_pipeline.predict_proba(test_frame[full_columns])[:, 1]
    benchmark = pd.DataFrame(
        [
            {"model": "PREMATCH_V1", **_metrics(test_frame["player_A_win"], test_frame["p_base"])},
            {
                "model": "PREMATCH_V1_TEMPERATURE_CONTROL",
                **_metrics(test_frame["player_A_win"], base_temperature),
            },
            {"model": "OFFSET_RAW", **_metrics(test_frame["player_A_win"], offset_raw)},
            {"model": "OFFSET_FINAL", **_metrics(test_frame["player_A_win"], offset_final)},
            {"model": "FULL_LOGISTIC", **_metrics(test_frame["player_A_win"], full_probability)},
        ]
    )
    benchmark.to_csv(REPORTS_DIR / "tournament_adapter_test_comparison.csv", index=False)

    predictions = test_frame[
        [
            "match_id",
            "tournament_edition_id",
            "tournament_name",
            "date",
            "surface",
            "level_group",
            "best_of",
            "round",
            "matches_observed_before_round",
            "player_A_id",
            "player_B_id",
            "player_A_win",
            "p_base",
        ]
    ].copy()
    predictions["p_adjusted_raw"] = offset_raw
    predictions["p_adjusted"] = offset_final
    predictions["delta_probability"] = predictions["p_adjusted"] - predictions["p_base"]
    predictions.to_parquet(
        PROCESSED_DIR / "tournament_adapter_v1_predictions_test.parquet", index=False
    )

    segment_rows = []
    for segment_type, series in (
        ("round", predictions["round"]),
        (
            "observed_matches",
            pd.cut(
                predictions["matches_observed_before_round"],
                [-1, 8, 20, 40, np.inf],
                labels=["0-8", "9-20", "21-40", "40+"],
            ),
        ),
        ("surface", predictions["surface"]),
        ("level", predictions["level_group"]),
        ("best_of", predictions["best_of"].astype(str)),
    ):
        for value in series.dropna().unique():
            mask = series == value
            for model_name, column in (("PREMATCH_V1", "p_base"), ("TOURNAMENT_ADAPTER", "p_adjusted")):
                segment_rows.append(
                    {
                        "segment_type": segment_type,
                        "segment": str(value),
                        "model": model_name,
                        **_metrics(predictions.loc[mask, "player_A_win"], predictions.loc[mask, column]),
                    }
                )
    segments = pd.DataFrame(segment_rows)
    segments.to_csv(REPORTS_DIR / "tournament_adapter_test_segments.csv", index=False)

    correction_audit = predictions.copy()
    correction_audit["abs_delta"] = correction_audit["delta_probability"].abs()
    correction_audit["correction_bucket"] = pd.cut(
        correction_audit["abs_delta"],
        [-1, 0.05, 0.10, 0.15, np.inf],
        labels=["0-5pp", "5-10pp", "10-15pp", "15pp+"],
    )
    correction_audit.sort_values("abs_delta", ascending=False).head(200).to_csv(
        REPORTS_DIR / "tournament_adapter_largest_corrections.csv", index=False
    )
    correction_summary = (
        correction_audit.groupby("correction_bucket", observed=True)
        .agg(
            matches=("match_id", "size"),
            mean_abs_delta=("abs_delta", "mean"),
            base_log_loss=("p_base", lambda p: log_loss(correction_audit.loc[p.index, "player_A_win"], p, labels=[0, 1])),
            adjusted_log_loss=("p_adjusted", lambda p: log_loss(correction_audit.loc[p.index, "player_A_win"], p, labels=[0, 1])),
        )
        .reset_index()
    )
    correction_summary.to_csv(REPORTS_DIR / "tournament_adapter_correction_buckets.csv", index=False)

    clustered_gain = _clustered_gain(test_frame, offset_final)
    base_test_metrics = _metrics(test_frame["player_A_win"], test_frame["p_base"])
    final_test_metrics = _metrics(test_frame["player_A_win"], offset_final)
    production_recommended = bool(
        final_test_metrics["log_loss"] < base_test_metrics["log_loss"]
        and final_test_metrics["brier"] <= base_test_metrics["brier"]
        and clustered_gain["ci95_low"] > 0
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_version": "tournament_adapter_v1",
        "status": (
            "production_candidate" if production_recommended else "validated_not_recommended"
        ),
        "model": offset_model,
        "calibrator": calibrator if use_calibration else None,
        "feature_columns": selected_features,
        "feature_groups": {
            "workload": [feature for feature in selected_features if feature in WORKLOAD + MINUTES],
            "player_form": [
                feature
                for feature in selected_features
                if feature in RAW_PERFORMANCE + performance_features(selected_player_prior)
            ],
            "tournament_conditions": [feature for feature in selected_features if feature.startswith("tournament_")],
            "round_context": [feature for feature in selected_features if feature in ROUND + CONTEXT],
            "same_tournament_history": [feature for feature in selected_features if feature in HISTORY],
        },
        "selected_player_prior": selected_player_prior,
        "selected_tournament_prior": selected_tournament_prior,
        "selected_regularization": selected_regularization,
        "selected_cap": selected_cap,
        "uses_temperature_calibration": use_calibration,
        "temperature_slope": calibrator.slope if use_calibration else 1.0,
        "base_temperature_control_used": base_use_calibration,
        "base_temperature_control_slope": (
            base_calibrator.slope if base_use_calibration else 1.0
        ),
        "trained_through": 2022,
        "p_base_contract": "exact frozen PRE-MATCH V1 probability",
        "production_recommended": production_recommended,
    }
    joblib.dump(artifact, MODEL_DIR / "model.joblib")
    metadata = {
        key: value
        for key, value in artifact.items()
        if key not in {"model", "calibrator"}
    }
    metadata["test"] = benchmark.set_index("model").to_dict("index")
    metadata["clustered_test_gain"] = clustered_gain
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    coefficients = pd.DataFrame(
        {
            "feature": selected_features,
            "standardized_logit_contribution": offset_model.coefficients_,
        }
    ).sort_values("standardized_logit_contribution", key=lambda values: values.abs(), ascending=False)
    coefficients.to_csv(REPORTS_DIR / "tournament_adapter_coefficients.csv", index=False)
    return metadata
