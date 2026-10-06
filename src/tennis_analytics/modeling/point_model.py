from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tennis_analytics.paths import PROJECT_ROOT, REPORTS_DIR
from tennis_analytics.point_by_point.normalize import FEATURE_PATH, normalize_name

MODEL_DIR = PROJECT_ROOT / "models" / "point_v1"
TRAIN_END = pd.Timestamp("2018-12-31")
VALIDATION_END = pd.Timestamp("2022-12-31")

FEATURE_BLOCKS = {
    "P1_SERVE_RETURN": ["server_serve_rate", "receiver_return_rate"],
    "P2_SURFACE": [
        "server_serve_rate",
        "receiver_return_rate",
        "server_surface_serve_rate",
        "receiver_surface_return_rate",
        "surface_clay",
        "surface_grass",
    ],
    "P3_PREMATCH": [
        "server_serve_rate",
        "receiver_return_rate",
        "server_surface_serve_rate",
        "receiver_surface_return_rate",
        "surface_clay",
        "surface_grass",
        "prematch_server",
    ],
    "P4_SCORE": [
        "server_serve_rate",
        "receiver_return_rate",
        "server_surface_serve_rate",
        "receiver_surface_return_rate",
        "surface_clay",
        "surface_grass",
        "prematch_server",
        "is_break_point",
        "is_game_point",
        "is_set_point",
        "is_match_point",
        "is_tiebreak",
        "deuce",
        "server_leading_in_game",
        "sets_diff_server",
        "games_diff_server",
    ],
    "P5_RECENT": [
        "server_serve_rate",
        "receiver_return_rate",
        "server_surface_serve_rate",
        "receiver_surface_return_rate",
        "surface_clay",
        "surface_grass",
        "prematch_server",
        "server_serve_rate_365d",
        "receiver_return_rate_365d",
    ],
    "P5_RECENT_SCORE": [
        "server_serve_rate",
        "receiver_return_rate",
        "server_surface_serve_rate",
        "receiver_surface_return_rate",
        "surface_clay",
        "surface_grass",
        "prematch_server",
        "server_serve_rate_365d",
        "receiver_return_rate_365d",
        "is_break_point",
        "is_game_point",
        "is_set_point",
        "is_match_point",
        "is_tiebreak",
        "deuce",
        "server_leading_in_game",
        "sets_diff_server",
        "games_diff_server",
    ],
}

LOAD_COLUMNS = sorted(
    {
        "match_id",
        "date",
        "source",
        "gender",
        "surface",
        "server_won_point",
        "player_A_id",
        "player_B_id",
        "player_A_name",
        "player_B_name",
        "server",
        *[column for columns in FEATURE_BLOCKS.values() for column in columns],
    }
)


def make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("logistic", LogisticRegression(C=1.0, max_iter=300)),
        ]
    )


def metrics(target: pd.Series | np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(target, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
    }


def split_masks(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    dates = pd.to_datetime(frame["date"])
    return (
        dates <= TRAIN_END,
        dates.between(TRAIN_END + pd.Timedelta(days=1), VALIDATION_END),
        dates > VALIDATION_END,
    )


def scope_mask(frame: pd.DataFrame, scope: str) -> pd.Series:
    men = frame["gender"] == "men"
    if scope == "combined_men":
        return men
    if scope == "slam_men":
        return men & (frame["source"] == "slam")
    if scope == "mcp_men":
        return men & (frame["source"] == "mcp")
    raise ValueError(scope)


def calibration_table(target: pd.Series, probability: np.ndarray) -> pd.DataFrame:
    edges = [0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.000001]
    labels = ["<50", "50-55", "55-60", "60-65", "65-70", "70-75", "75-80", "80+"]
    data = pd.DataFrame(
        {
            "bucket": pd.cut(probability, edges, labels=labels, right=False),
            "probability": probability,
            "target": target.to_numpy(),
        }
    )
    return (
        data.groupby("bucket", observed=False)
        .agg(
            points=("target", "size"),
            predicted=("probability", "mean"),
            observed=("target", "mean"),
        )
        .reset_index()
    )


def clustered_score_effect(
    frame: pd.DataFrame,
    mask: pd.Series,
    probability_without_score: np.ndarray,
    probability_with_score: np.ndarray,
    split: str,
) -> dict[str, Any]:
    """Paired loss improvement with the match as the independent unit."""
    target = frame.loc[mask, "server_won_point"].to_numpy(dtype=int)
    without = np.clip(probability_without_score, 1e-6, 1 - 1e-6)
    with_score = np.clip(probability_with_score, 1e-6, 1 - 1e-6)
    loss_without = -(target * np.log(without) + (1 - target) * np.log(1 - without))
    loss_with = -(target * np.log(with_score) + (1 - target) * np.log(1 - with_score))
    paired = pd.DataFrame(
        {
            "match_id": frame.loc[mask, "match_id"].to_numpy(),
            "improvement": loss_without - loss_with,
        }
    )
    by_match = paired.groupby("match_id")["improvement"].mean()
    standard_error = float(by_match.std(ddof=1) / np.sqrt(len(by_match)))
    mean = float(by_match.mean())
    return {
        "split": split,
        "comparison": "P5_RECENT_SCORE_minus_P5_RECENT",
        "matches": len(by_match),
        "points": len(paired),
        "point_weighted_log_loss_improvement": float(paired["improvement"].mean()),
        "mean_match_log_loss_improvement": mean,
        "ci95_low": mean - 1.96 * standard_error,
        "ci95_high": mean + 1.96 * standard_error,
        "share_matches_improved": float((by_match > 0).mean()),
    }
def build_profiles() -> pd.DataFrame:
    columns = [
        "date",
        "surface",
        "server",
        "server_won_point",
        "player_A_id",
        "player_B_id",
        "player_A_name",
        "player_B_name",
    ]
    points = pd.read_parquet(FEATURE_PATH, columns=columns)
    records: list[pd.DataFrame] = []
    for side, other in (("A", "B"), ("B", "A")):
        serving = points.loc[points["server"] == side].copy()
        serving["player_id"] = serving[f"player_{side}_id"]
        serving["player_name"] = serving[f"player_{side}_name"]
        serving["serve_wins"] = serving["server_won_point"]
        serving["serve_points"] = 1
        serving["return_wins"] = 0
        serving["return_points"] = 0
        returning = points.loc[points["server"] == other].copy()
        returning["player_id"] = returning[f"player_{side}_id"]
        returning["player_name"] = returning[f"player_{side}_name"]
        returning["serve_wins"] = 0
        returning["serve_points"] = 0
        returning["return_wins"] = 1 - returning["server_won_point"]
        returning["return_points"] = 1
        records.extend(
            [
                serving[
                    [
                        "date",
                        "surface",
                        "player_id",
                        "player_name",
                        "serve_wins",
                        "serve_points",
                        "return_wins",
                        "return_points",
                    ]
                ],
                returning[
                    [
                        "date",
                        "surface",
                        "player_id",
                        "player_name",
                        "serve_wins",
                        "serve_points",
                        "return_wins",
                        "return_points",
                    ]
                ],
            ]
        )
    data = pd.concat(records, ignore_index=True)
    global_serve = float(data["serve_wins"].sum() / data["serve_points"].sum())
    career = data.groupby("player_id", as_index=False).agg(
        player_name=("player_name", "last"),
        serve_wins=("serve_wins", "sum"),
        serve_points=("serve_points", "sum"),
        return_wins=("return_wins", "sum"),
        return_points=("return_points", "sum"),
        last_date=("date", "max"),
    )
    career["serve_rate"] = (career["serve_wins"] + 200 * global_serve) / (
        career["serve_points"] + 200
    )
    career["return_rate"] = (career["return_wins"] + 200 * (1 - global_serve)) / (
        career["return_points"] + 200
    )
    surfaces = data.groupby(["player_id", "surface"], as_index=False).agg(
        surface_serve_wins=("serve_wins", "sum"),
        surface_serve_points=("serve_points", "sum"),
        surface_return_wins=("return_wins", "sum"),
        surface_return_points=("return_points", "sum"),
    )
    surfaces = surfaces.merge(
        career[["player_id", "serve_rate", "return_rate"]], on="player_id", how="left"
    )
    surfaces["surface_serve_rate"] = (
        surfaces["surface_serve_wins"] + 200 * surfaces["serve_rate"]
    ) / (surfaces["surface_serve_points"] + 200)
    surfaces["surface_return_rate"] = (
        surfaces["surface_return_wins"] + 200 * surfaces["return_rate"]
    ) / (surfaces["surface_return_points"] + 200)
    surface_wide = surfaces.pivot(
        index="player_id", columns="surface", values=["surface_serve_rate", "surface_return_rate"]
    )
    surface_wide.columns = [
        f"{metric}_{surface.lower()}" for metric, surface in surface_wide.columns
    ]
    profile = career.merge(surface_wide.reset_index(), on="player_id", how="left")
    max_date = pd.to_datetime(data["date"]).max()
    recent = data.loc[pd.to_datetime(data["date"]) >= max_date - pd.Timedelta(days=365)]
    recent = recent.groupby("player_id", as_index=False).agg(
        recent_serve_wins=("serve_wins", "sum"),
        recent_serve_points=("serve_points", "sum"),
        recent_return_wins=("return_wins", "sum"),
        recent_return_points=("return_points", "sum"),
    )
    profile = profile.merge(recent, on="player_id", how="left")
    profile["serve_rate_365d"] = (profile["recent_serve_wins"].fillna(0) + 200 * global_serve) / (
        profile["recent_serve_points"].fillna(0) + 200
    )
    profile["return_rate_365d"] = (
        profile["recent_return_wins"].fillna(0) + 200 * (1 - global_serve)
    ) / (profile["recent_return_points"].fillna(0) + 200)
    profile["normalized_name"] = profile["player_name"].map(normalize_name)
    profile.attrs["global_server_rate"] = global_serve
    profile.attrs["through"] = max_date.date().isoformat()
    return profile


def run_point_model() -> dict[str, Any]:
    frame = pd.read_parquet(FEATURE_PATH, columns=LOAD_COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"])
    train_time, validation_time, test_time = split_masks(frame)
    rows: list[dict[str, Any]] = []
    fitted: dict[tuple[str, str], Pipeline] = {}
    for scope in ("slam_men", "mcp_men", "combined_men"):
        scope_rows = scope_mask(frame, scope)
        train = scope_rows & train_time
        validation = scope_rows & validation_time
        test = scope_rows & test_time
        constant = float(frame.loc[train, "server_won_point"].mean())
        for split_name, mask in (("validation", validation), ("test", test)):
            target = frame.loc[mask, "server_won_point"]
            rows.append(
                {
                    "scope": scope,
                    "model": "ALWAYS_SERVER",
                    "split": split_name,
                    "points": int(mask.sum()),
                    **metrics(target, np.full(mask.sum(), 1 - 1e-6)),
                }
            )
            rows.append(
                {
                    "scope": scope,
                    "model": "P0_CONSTANT",
                    "split": split_name,
                    "points": int(mask.sum()),
                    **metrics(target, np.full(mask.sum(), constant)),
                }
            )
        for name, columns in FEATURE_BLOCKS.items():
            pipeline = make_pipeline()
            pipeline.fit(frame.loc[train, columns], frame.loc[train, "server_won_point"])
            fitted[(scope, name)] = pipeline
            for split_name, mask in (("validation", validation), ("test", test)):
                probability = pipeline.predict_proba(frame.loc[mask, columns])[:, 1]
                rows.append(
                    {
                        "scope": scope,
                        "model": name,
                        "split": split_name,
                        "points": int(mask.sum()),
                        **metrics(frame.loc[mask, "server_won_point"], probability),
                    }
                )
    results = pd.DataFrame(rows)
    results.to_csv(REPORTS_DIR / "point_model_comparison.csv", index=False)
    combined_validation = results.loc[
        (results["scope"] == "combined_men")
        & (results["split"] == "validation")
        & results["model"].isin(FEATURE_BLOCKS)
    ].sort_values("log_loss")
    selected = str(combined_validation.iloc[0]["model"])
    selected_columns = FEATURE_BLOCKS[selected]
    selected_pipeline = fitted[("combined_men", selected)]
    combined_test = scope_mask(frame, "combined_men") & test_time
    test_probability = selected_pipeline.predict_proba(frame.loc[combined_test, selected_columns])[
        :, 1
    ]
    calibration = calibration_table(frame.loc[combined_test, "server_won_point"], test_probability)
    calibration.to_csv(REPORTS_DIR / "point_model_calibration.csv", index=False)
    segments = []
    for segment_type, column in (("source", "source"), ("surface", "surface")):
        for value in sorted(frame.loc[combined_test, column].dropna().unique()):
            mask = combined_test & (frame[column] == value)
            probability = selected_pipeline.predict_proba(frame.loc[mask, selected_columns])[:, 1]
            segments.append(
                {
                    "segment_type": segment_type,
                    "segment": value,
                    "points": int(mask.sum()),
                    **metrics(frame.loc[mask, "server_won_point"], probability),
                }
            )
    segments_frame = pd.DataFrame(segments)
    segments_frame.to_csv(REPORTS_DIR / "point_model_segments.csv", index=False)
    coefficients = pd.DataFrame(
        {
            "feature": selected_columns,
            "standardized_coefficient": selected_pipeline.named_steps["logistic"].coef_[0],
        }
    ).sort_values("standardized_coefficient", key=abs, ascending=False)
    coefficients.to_csv(REPORTS_DIR / "point_model_coefficients.csv", index=False)
    score_effect_rows = []
    for split_name, mask in (
        ("validation", scope_mask(frame, "combined_men") & validation_time),
        ("test", combined_test),
    ):
        probability_without = fitted[("combined_men", "P5_RECENT")].predict_proba(
            frame.loc[mask, FEATURE_BLOCKS["P5_RECENT"]]
        )[:, 1]
        probability_with = selected_pipeline.predict_proba(frame.loc[mask, selected_columns])[:, 1]
        score_effect_rows.append(
            clustered_score_effect(
                frame, mask, probability_without, probability_with, split_name
            )
        )
    score_effect = pd.DataFrame(score_effect_rows)
    score_effect.to_csv(REPORTS_DIR / "point_model_score_effect.csv", index=False)
    profiles = build_profiles()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_version": "point_v1",
        "selected_model": selected,
        "feature_columns": selected_columns,
        "pipeline": selected_pipeline,
        "profiles": profiles,
        "global_server_rate": profiles.attrs["global_server_rate"],
        "data_through": profiles.attrs["through"],
        "train_end": TRAIN_END.date().isoformat(),
        "validation_end": VALIDATION_END.date().isoformat(),
    }
    joblib.dump(artifact, MODEL_DIR / "model.joblib")
    metadata = {
        key: value for key, value in artifact.items() if key not in {"pipeline", "profiles"}
    }
    metadata["created_at"] = datetime.now(UTC).isoformat()
    metadata["dataset_rows"] = len(frame)
    metadata["primary_population"] = "ATP men; Grand Slam and MCP source adapters"
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (MODEL_DIR / "feature_schema.json").write_text(
        json.dumps(
            {
                "target": "server_won_point",
                "state_timing": "before_point",
                "ordered_features": selected_columns,
                "candidates": FEATURE_BLOCKS,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (MODEL_DIR / "config_snapshot.json").write_text(
        json.dumps(
            {
                "train_end": TRAIN_END.date().isoformat(),
                "validation_end": VALIDATION_END.date().isoformat(),
                "test_start": (VALIDATION_END + pd.Timedelta(days=1)).date().isoformat(),
                "primary_metric": "log_loss",
                "selection_scope": "combined_men",
                "selected_model": selected,
                "candidate_features": FEATURE_BLOCKS,
                "pipeline": {
                    "imputer": "median",
                    "scaler": "StandardScaler",
                    "estimator": "LogisticRegression(C=1.0, max_iter=300)",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "selected": selected,
        "selected_features": selected_columns,
        "validation": combined_validation.iloc[0].to_dict(),
        "test": {
            "points": int(combined_test.sum()),
            **metrics(frame.loc[combined_test, "server_won_point"], test_probability),
        },
        "results": results,
        "segments": segments_frame,
        "coefficients": coefficients,
        "score_effect": score_effect,
    }
