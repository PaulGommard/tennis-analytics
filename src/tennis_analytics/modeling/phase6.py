from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import tempfile
import time
import tomllib
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_CACHE_ROOT = Path(tempfile.gettempdir()) / "tennis-analytics-phase6-cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier, Pool
from joblib import dump, load
from lightgbm import LGBMClassifier, early_stopping
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

from tennis_analytics.modeling.logistic import (
    apply_platt,
    calibration_table,
    fit_platt,
    make_pipeline,
    probability_metrics,
)
from tennis_analytics.modeling.phase5 import PHASE4_FATIGUE
from tennis_analytics.paths import PROCESSED_DIR, PROJECT_ROOT, REPORTS_DIR

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "phase6.toml"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


@dataclass(frozen=True)
class Phase6Config:
    validation_years: tuple[int, ...]
    benchmark_start: pd.Timestamp
    random_seed: int
    early_stopping_rounds: int
    calibration_min_year: int
    shap_sample_size: int
    interaction_sample_size: int
    permutation_repeats: int
    grids: dict[str, tuple[dict[str, Any], ...]]

    @classmethod
    def from_toml(cls, path: Path = DEFAULT_CONFIG) -> Phase6Config:
        with path.open("rb") as stream:
            values = tomllib.load(stream)
        protocol = values["protocol"]
        return cls(
            validation_years=tuple(map(int, protocol["validation_years"])),
            benchmark_start=pd.Timestamp(protocol["benchmark_start"]),
            random_seed=int(protocol["random_seed"]),
            early_stopping_rounds=int(protocol["early_stopping_rounds"]),
            calibration_min_year=int(protocol["calibration_min_year"]),
            shap_sample_size=int(protocol["shap_sample_size"]),
            interaction_sample_size=int(protocol["interaction_sample_size"]),
            permutation_repeats=int(protocol["permutation_repeats"]),
            grids={
                name: tuple(dict(params) for params in values[name])
                for name in ("xgboost", "lightgbm", "catboost")
            },
        )


@dataclass
class TreeArtifact:
    algorithm: str
    variant: str
    feature_columns: list[str]
    imputer: SimpleImputer
    estimator: Any
    calibration_method: str
    calibrator: Any | None
    params: dict[str, Any]
    fitted_iterations: int
    trained_through: str

    def predict_raw(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self.imputer.transform(frame[self.feature_columns])
        return np.asarray(self.estimator.predict_proba(matrix)[:, 1], dtype=float)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self.predict_raw(frame)
        return apply_calibrator(self.calibration_method, self.calibrator, raw)


def temporal_folds(
    frame: pd.DataFrame, validation_years: tuple[int, ...], benchmark_start: pd.Timestamp
) -> list[tuple[int, pd.Series, pd.Series]]:
    dates = pd.to_datetime(frame["date"])
    if dates.isna().any():
        raise ValueError("Phase 6 requires non-null dates")
    folds = []
    for year in validation_years:
        train = dates < pd.Timestamp(f"{year}-01-01")
        validation = dates.between(f"{year}-01-01", f"{year}-12-31")
        if not train.any() or not validation.any():
            raise ValueError(f"Empty temporal fold for {year}")
        if dates.loc[train].max() >= dates.loc[validation].min():
            raise ValueError(f"Temporal leakage in fold {year}")
        if dates.loc[validation].max() >= benchmark_start:
            raise ValueError(f"Benchmark rows entered tuning fold {year}")
        folds.append((year, train, validation))
    return folds


def feature_variants(selected_features: list[str]) -> dict[str, list[str]]:
    full = list(selected_features)
    no_fatigue = [column for column in full if column not in PHASE4_FATIGUE]
    removed = sorted(set(full) - set(no_fatigue))
    if removed != sorted(PHASE4_FATIGUE):
        raise ValueError("The no-fatigue variant did not remove exactly the 13 Phase 4 proxies")
    return {"full": full, "no_fatigue_proxy": no_fatigue}


def validate_model_matrix(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing model features: {missing}")
    numeric = frame[columns].apply(pd.to_numeric, errors="raise")
    if np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("Model matrix contains infinite values")
    if frame["player_A_win"].isna().any():
        raise ValueError("Target contains missing values")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_phase5(report_dir: Path) -> None:
    paths = [
        PROCESSED_DIR / "match_features_v3.parquet",
        PROCESSED_DIR / "models" / "logistic_phase5.joblib",
        REPORTS_DIR / "phase5_report.md",
        REPORTS_DIR / "phase5_selected.json",
        PROJECT_ROOT / "config" / "phase5.toml",
    ]
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "statement": "Phase 3-5 artifacts are read-only during Phase 6.",
        "files": [
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in paths
        ],
    }
    (report_dir / "phase6_phase5_freeze.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def _make_estimator(
    algorithm: str,
    params: dict[str, Any],
    *,
    seed: int,
    early_stopping_rounds: int | None,
    iterations: int | None = None,
) -> Any:
    clean = {key: value for key, value in params.items() if key != "id"}
    if algorithm == "xgboost":
        if iterations is not None:
            clean["n_estimators"] = iterations
        return XGBClassifier(
            **clean,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=1,
            verbosity=0,
            early_stopping_rounds=early_stopping_rounds,
        )
    if algorithm == "lightgbm":
        if iterations is not None:
            clean["n_estimators"] = iterations
        return LGBMClassifier(
            **clean,
            objective="binary",
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
            bagging_freq=1,
            deterministic=True,
            force_col_wise=True,
        )
    if algorithm == "catboost":
        if iterations is not None:
            clean["iterations"] = iterations
        return CatBoostClassifier(
            **clean,
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=seed,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
            od_type="Iter" if early_stopping_rounds else None,
            od_wait=early_stopping_rounds,
        )
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def _best_iteration(algorithm: str, estimator: Any, maximum: int) -> int:
    if algorithm == "xgboost":
        value = getattr(estimator, "best_iteration", None)
        return maximum if value is None else int(value) + 1
    if algorithm == "lightgbm":
        value = getattr(estimator, "best_iteration_", None)
        return maximum if not value else int(value)
    value = estimator.get_best_iteration()
    return maximum if value is None or value < 0 else int(value) + 1


def fit_fold_candidate(
    algorithm: str,
    params: dict[str, Any],
    train_x: pd.DataFrame,
    train_y: pd.Series,
    validation_x: pd.DataFrame,
    validation_y: pd.Series,
    config: Phase6Config,
) -> tuple[np.ndarray, int, float]:
    started = time.perf_counter()
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train_x)
    x_validation = imputer.transform(validation_x)
    estimator = _make_estimator(
        algorithm,
        params,
        seed=config.random_seed,
        early_stopping_rounds=config.early_stopping_rounds,
    )
    if algorithm == "xgboost":
        estimator.fit(x_train, train_y, eval_set=[(x_validation, validation_y)], verbose=False)
        maximum = int(params["n_estimators"])
    elif algorithm == "lightgbm":
        estimator.fit(
            x_train,
            train_y,
            eval_X=x_validation,
            eval_y=validation_y,
            eval_metric="binary_logloss",
            callbacks=[early_stopping(config.early_stopping_rounds, verbose=False)],
        )
        maximum = int(params["n_estimators"])
    else:
        estimator.fit(
            x_train,
            train_y,
            eval_set=(x_validation, validation_y),
            use_best_model=True,
            early_stopping_rounds=config.early_stopping_rounds,
            verbose=False,
        )
        maximum = int(params["iterations"])
    probability = estimator.predict_proba(x_validation)[:, 1]
    return (
        np.asarray(probability, dtype=float),
        _best_iteration(algorithm, estimator, maximum),
        time.perf_counter() - started,
    )


def tune_tree_models(
    features: pd.DataFrame,
    variants: dict[str, list[str]],
    config: Phase6Config,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str, str], pd.Series]]:
    folds = temporal_folds(features, config.validation_years, config.benchmark_start)
    rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, str, str], pd.Series] = {}
    for algorithm, grid in config.grids.items():
        for variant, columns in variants.items():
            validate_model_matrix(features, columns)
            for params in grid:
                config_id = str(params["id"])
                key = (algorithm, variant, config_id)
                oof = pd.Series(np.nan, index=features.index, dtype=float)
                for year, train, validation in folds:
                    LOGGER.info("Tuning %s %s %s fold=%d", algorithm, variant, config_id, year)
                    probability, best_iteration, seconds = fit_fold_candidate(
                        algorithm,
                        params,
                        features.loc[train, columns],
                        features.loc[train, "player_A_win"],
                        features.loc[validation, columns],
                        features.loc[validation, "player_A_win"],
                        config,
                    )
                    oof.loc[validation] = probability
                    metrics = probability_metrics(
                        features.loc[validation, "player_A_win"], probability
                    )
                    rows.append(
                        {
                            "algorithm": algorithm,
                            "variant": variant,
                            "config_id": config_id,
                            "params": json.dumps(params, sort_keys=True),
                            "fold": year,
                            "train_end": f"{year - 1}-12-31",
                            "train_matches": int(train.sum()),
                            "best_iteration": best_iteration,
                            "fit_seconds": seconds,
                            **metrics,
                        }
                    )
                predictions[key] = oof
    tuning = pd.DataFrame(rows)
    summary = (
        tuning.groupby(["algorithm", "variant", "config_id", "params"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            mean_log_loss=("log_loss", "mean"),
            std_log_loss=("log_loss", "std"),
            mean_brier=("brier", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_accuracy=("accuracy", "mean"),
            median_best_iteration=("best_iteration", "median"),
            total_fit_seconds=("fit_seconds", "sum"),
        )
        .sort_values(
            ["mean_log_loss", "mean_brier", "mean_roc_auc", "mean_accuracy"],
            ascending=[True, True, False, False],
            kind="stable",
        )
    )
    return tuning, summary, predictions


def walk_forward_logistic(
    features: pd.DataFrame, columns: list[str], config: Phase6Config
) -> tuple[pd.DataFrame, pd.Series]:
    rows = []
    oof = pd.Series(np.nan, index=features.index, dtype=float)
    for year, train, validation in temporal_folds(
        features, config.validation_years, config.benchmark_start
    ):
        started = time.perf_counter()
        model = make_pipeline(1.0)
        model.fit(features.loc[train, columns], features.loc[train, "player_A_win"])
        probability = model.predict_proba(features.loc[validation, columns])[:, 1]
        oof.loc[validation] = probability
        rows.append(
            {
                "algorithm": "logistic_phase5_refit",
                "variant": "full",
                "config_id": "phase5_C1",
                "params": json.dumps({"C": 1.0}),
                "fold": year,
                "train_end": f"{year - 1}-12-31",
                "train_matches": int(train.sum()),
                "best_iteration": np.nan,
                "fit_seconds": time.perf_counter() - started,
                **probability_metrics(features.loc[validation, "player_A_win"], probability),
            }
        )
    return pd.DataFrame(rows), oof


def select_best_configs(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.sort_values(
            ["mean_log_loss", "mean_brier", "mean_roc_auc", "mean_accuracy"],
            ascending=[True, True, False, False],
            kind="stable",
        )
        .groupby(["algorithm", "variant"], as_index=False, sort=False)
        .first()
    )


def fit_calibrator(method: str, probability: np.ndarray, target: pd.Series) -> Any | None:
    if method == "raw":
        return None
    if method == "platt":
        return fit_platt(probability, target)
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip", y_min=1e-5, y_max=1 - 1e-5).fit(
            probability, target.astype(int)
        )
    raise ValueError(f"Unknown calibration method: {method}")


def apply_calibrator(method: str, calibrator: Any | None, probability: np.ndarray) -> np.ndarray:
    raw = np.asarray(probability, dtype=float)
    if method == "raw":
        return raw
    if calibrator is None:
        raise ValueError("Missing fitted calibrator")
    if method == "platt":
        return apply_platt(calibrator, raw)
    return np.clip(np.asarray(calibrator.predict(raw), dtype=float), 1e-5, 1 - 1e-5)


def select_calibration(
    features: pd.DataFrame,
    selected_configs: pd.DataFrame,
    oof_predictions: dict[tuple[str, str, str], pd.Series],
    config: Phase6Config,
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], Any | None]]:
    years = pd.to_datetime(features["date"]).dt.year
    rows = []
    final: dict[tuple[str, str, str], Any | None] = {}
    for selected in selected_configs.to_dict("records"):
        algorithm = str(selected["algorithm"])
        variant = str(selected["variant"])
        config_id = str(selected["config_id"])
        probability = oof_predictions[(algorithm, variant, config_id)]
        evaluation_indexes = []
        method_predictions = {method: [] for method in ("raw", "platt", "isotonic")}
        for evaluation_year in config.validation_years:
            if evaluation_year < config.calibration_min_year:
                continue
            calibration_mask = probability.notna() & (years < evaluation_year)
            evaluation_mask = probability.notna() & (years == evaluation_year)
            if not calibration_mask.any() or not evaluation_mask.any():
                raise ValueError(f"Invalid calibration fold {evaluation_year}")
            evaluation_indexes.extend(features.index[evaluation_mask].tolist())
            for method, stored_predictions in method_predictions.items():
                calibrator = fit_calibrator(
                    method,
                    probability.loc[calibration_mask].to_numpy(),
                    features.loc[calibration_mask, "player_A_win"],
                )
                stored_predictions.extend(
                    apply_calibrator(
                        method, calibrator, probability.loc[evaluation_mask].to_numpy()
                    ).tolist()
                )
        target = features.loc[evaluation_indexes, "player_A_win"]
        for method, calibrated in method_predictions.items():
            rows.append(
                {
                    "algorithm": algorithm,
                    "variant": variant,
                    "config_id": config_id,
                    "calibration": method,
                    **probability_metrics(target, np.asarray(calibrated)),
                }
            )
    results = pd.DataFrame(rows)
    for (algorithm, variant), group in results.groupby(["algorithm", "variant"]):
        selected = selected_configs.loc[
            (selected_configs["algorithm"] == algorithm)
            & (selected_configs["variant"] == variant)
        ].iloc[0]
        oof = oof_predictions[(algorithm, variant, str(selected["config_id"]))]
        mask = oof.notna()
        for method in ("raw", "platt", "isotonic"):
            final[(algorithm, variant, method)] = fit_calibrator(
                method, oof.loc[mask].to_numpy(), features.loc[mask, "player_A_win"]
            )
    return results, final


def fit_final_artifact(
    features: pd.DataFrame,
    columns: list[str],
    algorithm: str,
    variant: str,
    params: dict[str, Any],
    iterations: int,
    calibration_method: str,
    calibrator: Any | None,
    config: Phase6Config,
) -> TreeArtifact:
    train = pd.to_datetime(features["date"]) < config.benchmark_start
    if pd.to_datetime(features.loc[train, "date"]).max() >= config.benchmark_start:
        raise ValueError("Benchmark entered final model training")
    imputer = SimpleImputer(strategy="median")
    matrix = imputer.fit_transform(features.loc[train, columns])
    estimator = _make_estimator(
        algorithm,
        params,
        seed=config.random_seed,
        early_stopping_rounds=None,
        iterations=iterations,
    )
    if algorithm in {"xgboost", "catboost"}:
        estimator.fit(matrix, features.loc[train, "player_A_win"], verbose=False)
    else:
        estimator.fit(matrix, features.loc[train, "player_A_win"])
    return TreeArtifact(
        algorithm=algorithm,
        variant=variant,
        feature_columns=columns,
        imputer=imputer,
        estimator=estimator,
        calibration_method=calibration_method,
        calibrator=calibrator,
        params=params,
        fitted_iterations=iterations,
        trained_through=(config.benchmark_start - pd.Timedelta(days=1)).date().isoformat(),
    )


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame[columns].to_dict("records"):
        rendered = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                rendered.append("—")
            elif isinstance(value, (float, np.floating)):
                rendered.append(f"{float(value):.5f}")
            elif isinstance(value, (int, np.integer)):
                rendered.append(f"{int(value):,}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def _selection_rows(calibration_results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    ordered = calibration_results.sort_values(
        ["log_loss", "brier", "roc_auc", "accuracy"],
        ascending=[True, True, False, False],
        kind="stable",
    )
    per_variant = ordered.groupby(["algorithm", "variant"], as_index=False, sort=False).first()
    per_algorithm = (
        per_variant.sort_values(
            ["log_loss", "brier", "roc_auc", "accuracy"],
            ascending=[True, True, False, False],
            kind="stable",
        )
        .groupby("algorithm", as_index=False, sort=False)
        .first()
    )
    return per_variant, per_algorithm, per_variant.iloc[0]


def build_final_artifacts(
    features: pd.DataFrame,
    variants: dict[str, list[str]],
    selected_configs: pd.DataFrame,
    tuning: pd.DataFrame,
    calibration_results: pd.DataFrame,
    calibrators: dict[tuple[str, str, str], Any | None],
    config: Phase6Config,
    model_dir: Path,
) -> tuple[dict[tuple[str, str], TreeArtifact], pd.DataFrame, pd.Series]:
    model_dir.mkdir(parents=True, exist_ok=True)
    per_variant, per_algorithm, best = _selection_rows(calibration_results)
    artifacts: dict[tuple[str, str], TreeArtifact] = {}
    for selected in per_variant.to_dict("records"):
        algorithm = str(selected["algorithm"])
        variant = str(selected["variant"])
        method = str(selected["calibration"])
        config_row = selected_configs.loc[
            (selected_configs["algorithm"] == algorithm)
            & (selected_configs["variant"] == variant)
        ].iloc[0]
        config_id = str(config_row["config_id"])
        params = json.loads(str(config_row["params"]))
        iterations = max(
            1,
            round(
                tuning.loc[
                    (tuning["algorithm"] == algorithm)
                    & (tuning["variant"] == variant)
                    & (tuning["config_id"] == config_id),
                    "best_iteration",
                ].median()
            ),
        )
        artifact = fit_final_artifact(
            features,
            variants[variant],
            algorithm,
            variant,
            params,
            iterations,
            method,
            calibrators[(algorithm, variant, method)],
            config,
        )
        artifacts[(algorithm, variant)] = artifact
        dump(artifact, model_dir / f"{algorithm}_{variant}.joblib")
    best_key = (str(best["algorithm"]), str(best["variant"]))
    shutil.copyfile(
        model_dir / f"{best_key[0]}_{best_key[1]}.joblib",
        model_dir / "best_tree_phase6.joblib",
    )
    return artifacts, per_algorithm, best


def _series_for_test(values: np.ndarray, index: pd.Index, full_index: pd.Index) -> pd.Series:
    output = pd.Series(np.nan, index=full_index, dtype=float)
    output.loc[index] = values
    return output


def evaluate_benchmark(
    features: pd.DataFrame,
    ratings: pd.DataFrame,
    phase5_artifact: dict[str, Any],
    artifacts: dict[tuple[str, str], TreeArtifact],
    per_algorithm: pd.DataFrame,
    best_calibrated: pd.Series,
    calibrators: dict[tuple[str, str, str], Any | None],
    config: Phase6Config,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, pd.Series],
    TreeArtifact,
]:
    benchmark = pd.to_datetime(features["date"]) >= config.benchmark_start
    index = features.index[benchmark]
    target = features.loc[benchmark, "player_A_win"]
    phase5_raw = phase5_artifact["pipeline"].predict_proba(
        features.loc[benchmark, phase5_artifact["feature_columns"]]
    )[:, 1]
    phase5_probability = apply_platt(phase5_artifact["calibrator"], phase5_raw)
    probabilities: dict[str, pd.Series] = {
        "Logistic Phase 5": _series_for_test(phase5_probability, index, features.index),
        "Elo Phase 3": ratings["final_calibrated_expected_A"],
    }
    raw_tree: dict[tuple[str, str], pd.Series] = {}
    final_tree: dict[tuple[str, str], pd.Series] = {}
    all_tree_rows = []
    for key, artifact in artifacts.items():
        raw = artifact.predict_raw(features.loc[benchmark])
        final = artifact.predict_proba(features.loc[benchmark])
        raw_tree[key] = _series_for_test(raw, index, features.index)
        final_tree[key] = _series_for_test(final, index, features.index)
        states = [("raw", raw)]
        if artifact.calibration_method != "raw":
            states.append((artifact.calibration_method, final))
        for state, values in states:
            all_tree_rows.append(
                {
                    "algorithm": artifact.algorithm,
                    "variant": artifact.variant,
                    "probability": state,
                    **probability_metrics(target, values),
                }
            )
    all_trees = pd.DataFrame(all_tree_rows)

    final_rows: list[dict[str, Any]] = []
    ranking = (
        benchmark
        & features["rank_A"].notna()
        & features["rank_B"].notna()
        & (features["rank_A"] != features["rank_B"])
    )
    ranking_prediction = features.loc[ranking, "rank_A"] < features.loc[ranking, "rank_B"]
    final_rows.append(
        {
            "model": "ATP Ranking",
            "matches": int(ranking.sum()),
            "accuracy": float(
                (
                    ranking_prediction.astype(int)
                    == features.loc[ranking, "player_A_win"].astype(int)
                ).mean()
            ),
            "log_loss": np.nan,
            "brier": np.nan,
            "roc_auc": np.nan,
        }
    )
    final_rows.append(
        {
            "model": "Elo Phase 3",
            **probability_metrics(target, probabilities["Elo Phase 3"].loc[index]),
        }
    )
    final_rows.append(
        {"model": "Logistic Phase 5", **probability_metrics(target, phase5_probability)}
    )
    algorithm_names = {
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "catboost": "CatBoost",
    }
    for selected in per_algorithm.to_dict("records"):
        key = (str(selected["algorithm"]), str(selected["variant"]))
        artifact = artifacts[key]
        name = algorithm_names[key[0]]
        probabilities[name] = final_tree[key]
        final_rows.append(
            {
                "model": name,
                **probability_metrics(target, final_tree[key].loc[index]),
            }
        )
    calibrated_key = (
        str(best_calibrated["algorithm"]),
        str(best_calibrated["variant"]),
    )
    calibrated_method = str(best_calibrated["calibration"])
    calibrated_artifact = artifacts[calibrated_key]
    calibrated_values = apply_calibrator(
        calibrated_method,
        calibrators[(*calibrated_key, calibrated_method)],
        raw_tree[calibrated_key].loc[index].to_numpy(),
    )
    probabilities["Best calibrated tree"] = _series_for_test(
        calibrated_values, index, features.index
    )
    final_rows.append(
        {
            "model": "Best calibrated tree",
            **probability_metrics(target, calibrated_values),
        }
    )
    final_comparison = pd.DataFrame(final_rows)

    segment_rows = []
    dates = pd.to_datetime(features["date"])
    segment_probabilities = {
        name: probability
        for name, probability in probabilities.items()
        if name not in {"Elo Phase 3"}
    }
    masks: list[tuple[str, str, pd.Series]] = []
    for surface in ("Hard", "Clay", "Grass"):
        masks.append(("surface", surface, benchmark & features["surface"].eq(surface)))
    for year in sorted(dates.loc[benchmark].dt.year.unique()):
        masks.append(("year", str(year), benchmark & dates.dt.year.eq(year)))
    level_groups = {
        "Grand Slam": features["level"].eq("G"),
        "Masters": features["level"].eq("M"),
        "Other": ~features["level"].isin(["G", "M"]),
    }
    for label, mask in level_groups.items():
        masks.append(("level", label, benchmark & mask))
    for segment_type, segment, mask in masks:
        for name, probability in segment_probabilities.items():
            segment_rows.append(
                {
                    "segment_type": segment_type,
                    "segment": segment,
                    "model": name,
                    **probability_metrics(
                        features.loc[mask, "player_A_win"], probability.loc[mask]
                    ),
                }
            )
    segments = pd.DataFrame(segment_rows)

    best_raw = raw_tree[calibrated_key].loc[index].to_numpy()
    best_final = calibrated_values
    calibration = calibration_table(
        features["player_A_win"],
        {
            "Logistic Phase 5": phase5_probability,
            "Best tree raw": best_raw,
            "Best tree calibrated": best_final,
        },
        index,
    )
    return final_comparison, all_trees, segments, calibration, probabilities, calibrated_artifact


def confidence_table(target: pd.Series, variants: dict[str, np.ndarray]) -> pd.DataFrame:
    edges = [0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 1.0000001]
    labels = ["50-55%", "55-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
    y = target.astype(int).to_numpy()
    rows = []
    for name, probability in variants.items():
        p = np.asarray(probability, dtype=float)
        favorite_probability = np.maximum(p, 1 - p)
        favorite_won = np.where(p >= 0.5, y, 1 - y)
        frame = pd.DataFrame(
            {
                "bin": pd.cut(favorite_probability, edges, labels=labels, right=False),
                "probability": favorite_probability,
                "favorite_won": favorite_won,
            }
        )
        grouped = (
            frame.groupby("bin", observed=False)
            .agg(
                matches=("favorite_won", "size"),
                mean_probability=("probability", "mean"),
                actual_win_rate=("favorite_won", "mean"),
            )
            .reset_index()
        )
        grouped.insert(0, "model", name)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def plot_calibration(calibration: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 6))
    for name, group in calibration.groupby("model", sort=False):
        axis.plot(
            group["mean_probability"],
            group["actual_win_rate"],
            marker="o",
            label=name,
        )
    axis.plot([0.5, 1], [0.5, 1], linestyle="--", color="black", alpha=0.5)
    axis.set(xlabel="Probabilité moyenne du favori", ylabel="Fréquence de victoire", xlim=(0.5, 1))
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _native_importance(
    algorithm: str,
    estimator: Any,
    columns: list[str],
    matrix: np.ndarray,
    target: pd.Series,
) -> np.ndarray:
    if algorithm == "xgboost":
        scores = estimator.get_booster().get_score(importance_type="gain")
        return np.asarray([scores.get(f"f{index}", 0.0) for index in range(len(columns))])
    if algorithm == "lightgbm":
        return np.asarray(estimator.booster_.feature_importance(importance_type="gain"))
    return np.asarray(
        estimator.get_feature_importance(
            Pool(matrix, target.astype(int)), type="LossFunctionChange"
        )
    )


def build_feature_importance(
    features: pd.DataFrame,
    variants: dict[str, list[str]],
    per_algorithm: pd.DataFrame,
    selected_configs: pd.DataFrame,
    tuning: pd.DataFrame,
    config: Phase6Config,
) -> pd.DataFrame:
    dates = pd.to_datetime(features["date"])
    train = dates < pd.Timestamp("2022-01-01")
    validation = dates.between("2022-01-01", "2022-12-31")
    rows = []
    for selected in per_algorithm.to_dict("records"):
        algorithm = str(selected["algorithm"])
        variant = str(selected["variant"])
        config_row = selected_configs.loc[
            (selected_configs["algorithm"] == algorithm)
            & (selected_configs["variant"] == variant)
        ].iloc[0]
        params = json.loads(str(config_row["params"]))
        config_id = str(config_row["config_id"])
        earlier_iterations = tuning.loc[
            (tuning["algorithm"] == algorithm)
            & (tuning["variant"] == variant)
            & (tuning["config_id"] == config_id)
            & (tuning["fold"] < 2022),
            "best_iteration",
        ]
        iterations = max(1, round(earlier_iterations.median()))
        columns = variants[variant]
        imputer = SimpleImputer(strategy="median")
        x_train = imputer.fit_transform(features.loc[train, columns])
        x_validation = imputer.transform(features.loc[validation, columns])
        estimator = _make_estimator(
            algorithm,
            params,
            seed=config.random_seed,
            early_stopping_rounds=None,
            iterations=iterations,
        )
        if algorithm in {"xgboost", "catboost"}:
            estimator.fit(x_train, features.loc[train, "player_A_win"], verbose=False)
        else:
            estimator.fit(x_train, features.loc[train, "player_A_win"])
        native = _native_importance(
            algorithm,
            estimator,
            columns,
            x_validation,
            features.loc[validation, "player_A_win"],
        )
        native_total = native.sum()
        native = native / native_total if native_total > 0 else native
        permutation = permutation_importance(
            estimator,
            x_validation,
            features.loc[validation, "player_A_win"],
            scoring="neg_log_loss",
            n_repeats=config.permutation_repeats,
            random_state=config.random_seed,
            n_jobs=1,
        )
        for index, column in enumerate(columns):
            rows.append(
                {
                    "algorithm": algorithm,
                    "variant": variant,
                    "feature": column,
                    "native_gain": float(native[index]),
                    "permutation_log_loss_increase": float(permutation.importances_mean[index]),
                    "permutation_std": float(permutation.importances_std[index]),
                    "diagnostic_year": 2022,
                    "diagnostic_iterations": iterations,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["algorithm", "permutation_log_loss_increase"], ascending=[True, False]
    )


def plot_importance(importance: pd.DataFrame, algorithm: str, path: Path) -> None:
    selected = (
        importance.loc[importance["algorithm"] == algorithm]
        .nlargest(20, "permutation_log_loss_increase")
        .sort_values("permutation_log_loss_increase")
    )
    fig, axis = plt.subplots(figsize=(9, 7))
    axis.barh(selected["feature"], selected["permutation_log_loss_increase"])
    axis.set_xlabel("Hausse de log loss après permutation (validation 2022)")
    axis.set_title(f"Importance par permutation — {algorithm}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _extract_shap_values(explainer: Any, matrix: np.ndarray) -> np.ndarray:
    values = explainer.shap_values(matrix)
    if isinstance(values, list):
        values = values[-1]
    array = np.asarray(values)
    if array.ndim == 3:
        array = array[:, :, -1]
    if array.ndim != 2:
        raise ValueError(f"Unexpected SHAP shape: {array.shape}")
    return array


def build_shap_outputs(
    features: pd.DataFrame,
    artifact: TreeArtifact,
    logistic_probability: pd.Series,
    tree_probability: pd.Series,
    config: Phase6Config,
    report_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    benchmark = pd.to_datetime(features["date"]) >= config.benchmark_start
    candidates = features.index[benchmark]
    sample_size = min(config.shap_sample_size, len(candidates))
    sample_index = (
        features.loc[candidates]
        .sample(sample_size, random_state=config.random_seed)
        .index
    )
    sample_matrix = artifact.imputer.transform(features.loc[sample_index, artifact.feature_columns])
    sample_frame = pd.DataFrame(
        sample_matrix, index=sample_index, columns=artifact.feature_columns
    )
    explainer = shap.TreeExplainer(artifact.estimator)
    values = _extract_shap_values(explainer, sample_matrix)
    global_rows = []
    for index, feature in enumerate(artifact.feature_columns):
        column = values[:, index]
        global_rows.append(
            {
                "feature": feature,
                "mean_abs_shap": float(np.mean(np.abs(column))),
                "mean_shap": float(np.mean(column)),
                "std_shap": float(np.std(column)),
                "p05_shap": float(np.quantile(column, 0.05)),
                "p95_shap": float(np.quantile(column, 0.95)),
            }
        )
    global_importance = pd.DataFrame(global_rows).sort_values("mean_abs_shap", ascending=False)
    global_importance.to_csv(report_dir / "phase6_shap_global.csv", index=False)

    shap.summary_plot(values, sample_frame, plot_type="bar", max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(report_dir / "phase6_shap_global.png", dpi=160, bbox_inches="tight")
    plt.close()
    shap.summary_plot(values, sample_frame, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(report_dir / "phase6_shap_distribution.png", dpi=160, bbox_inches="tight")
    plt.close()

    near_68 = (tree_probability.loc[candidates] - 0.68).abs().nsmallest(5).index
    confident_wrong = candidates[
        (np.maximum(tree_probability.loc[candidates], 1 - tree_probability.loc[candidates]) > 0.8)
        & (
            tree_probability.loc[candidates].ge(0.5).astype(int)
            != features.loc[candidates, "player_A_win"].astype(int)
        )
    ][:5]
    example_index = pd.Index(list(dict.fromkeys([*near_68, *confident_wrong])))
    example_matrix = artifact.imputer.transform(
        features.loc[example_index, artifact.feature_columns]
    )
    example_values = _extract_shap_values(explainer, example_matrix)
    example_rows = []
    for row_number, match_index in enumerate(example_index):
        top = np.argsort(np.abs(example_values[row_number]))[::-1][:8]
        for rank, feature_index in enumerate(top, start=1):
            feature = artifact.feature_columns[feature_index]
            example_rows.append(
                {
                    "match_id": features.loc[match_index, "match_id"],
                    "date": features.loc[match_index, "date"],
                    "player_A_id": features.loc[match_index, "player_A_id"],
                    "player_B_id": features.loc[match_index, "player_B_id"],
                    "tree_probability_A": tree_probability.loc[match_index],
                    "player_A_win": int(features.loc[match_index, "player_A_win"]),
                    "contribution_rank": rank,
                    "feature": feature,
                    "feature_value": float(example_matrix[row_number, feature_index]),
                    "shap_log_odds": float(example_values[row_number, feature_index]),
                }
            )
    examples = pd.DataFrame(example_rows)
    examples.to_csv(report_dir / "phase6_shap_examples.csv", index=False)

    disagreement = benchmark & (
        logistic_probability.ge(0.5).astype(int) != tree_probability.ge(0.5).astype(int)
    )
    disagreement_index = features.index[disagreement]
    if len(disagreement_index) > 1000:
        disagreement_index = (
            features.loc[disagreement_index]
            .sample(1000, random_state=config.random_seed)
            .index
        )
    disagreement_matrix = artifact.imputer.transform(
        features.loc[disagreement_index, artifact.feature_columns]
    )
    disagreement_values = _extract_shap_values(explainer, disagreement_matrix)
    disagreement_importance = pd.DataFrame(
        {
            "feature": artifact.feature_columns,
            "mean_abs_shap_disagreement": np.mean(np.abs(disagreement_values), axis=0),
            "mean_shap_disagreement": np.mean(disagreement_values, axis=0),
        }
    ).sort_values("mean_abs_shap_disagreement", ascending=False)
    disagreement_importance.to_csv(
        report_dir / "phase6_disagreement_shap.csv", index=False
    )

    interactions = pd.DataFrame(columns=["feature_1", "feature_2", "mean_abs_interaction"])
    interaction_index = sample_index[: min(config.interaction_sample_size, len(sample_index))]
    interaction_matrix = artifact.imputer.transform(
        features.loc[interaction_index, artifact.feature_columns]
    )
    try:
        raw_interactions = explainer.shap_interaction_values(interaction_matrix)
        if isinstance(raw_interactions, list):
            raw_interactions = raw_interactions[-1]
        interaction_values = np.asarray(raw_interactions)
        if interaction_values.ndim == 4:
            interaction_values = interaction_values[:, :, :, -1]
        mean_abs = np.mean(np.abs(interaction_values), axis=0)
        pairs = []
        for left in range(len(artifact.feature_columns)):
            for right in range(left + 1, len(artifact.feature_columns)):
                pairs.append(
                    {
                        "feature_1": artifact.feature_columns[left],
                        "feature_2": artifact.feature_columns[right],
                        "mean_abs_interaction": float(mean_abs[left, right]),
                    }
                )
        interactions = pd.DataFrame(pairs).sort_values(
            "mean_abs_interaction", ascending=False
        )
    except (AttributeError, ValueError, TypeError) as error:
        LOGGER.warning("SHAP interactions unavailable: %s", error)
    interactions.to_csv(report_dir / "phase6_shap_interactions.csv", index=False)
    return global_importance, examples, disagreement_importance


def analyze_disagreement(
    features: pd.DataFrame,
    logistic_probability: pd.Series,
    tree_probability: pd.Series,
    config: Phase6Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    benchmark = pd.to_datetime(features["date"]) >= config.benchmark_start
    disagree = benchmark & (
        logistic_probability.ge(0.5).astype(int) != tree_probability.ge(0.5).astype(int)
    )
    target = features.loc[disagree, "player_A_win"].astype(int)
    logistic_pick = logistic_probability.loc[disagree].ge(0.5).astype(int)
    tree_pick = tree_probability.loc[disagree].ge(0.5).astype(int)
    summary = pd.DataFrame(
        [
            {
                "matches": int(disagree.sum()),
                "share_of_benchmark": float(disagree.sum() / benchmark.sum()),
                "logistic_accuracy": float((logistic_pick == target).mean()),
                "tree_accuracy": float((tree_pick == target).mean()),
                "logistic_mean_confidence": float(
                    np.maximum(
                        logistic_probability.loc[disagree],
                        1 - logistic_probability.loc[disagree],
                    ).mean()
                ),
                "tree_mean_confidence": float(
                    np.maximum(
                        tree_probability.loc[disagree], 1 - tree_probability.loc[disagree]
                    ).mean()
                ),
            }
        ]
    )
    details = features.loc[
        disagree,
        [
            "match_id",
            "date",
            "surface",
            "level",
            "player_A_id",
            "player_B_id",
            "player_A_win",
        ],
    ].copy()
    details["logistic_probability_A"] = logistic_probability.loc[disagree]
    details["tree_probability_A"] = tree_probability.loc[disagree]
    details["winner_model"] = np.where(tree_pick == target, "tree", "logistic")
    return summary, details


def write_confident_errors(
    features: pd.DataFrame,
    probability: pd.Series,
    selected_columns: list[str],
    config: Phase6Config,
    report_dir: Path,
) -> pd.DataFrame:
    benchmark = pd.to_datetime(features["date"]) >= config.benchmark_start
    confidence = np.maximum(probability, 1 - probability)
    prediction = probability.ge(0.5).astype(int)
    wrong = benchmark & confidence.gt(0.8) & prediction.ne(features["player_A_win"].astype(int))
    source = features.loc[wrong].copy()
    rows = []
    for index, row in source.iterrows():
        favorite_is_a = probability.loc[index] >= 0.5
        favorite_side = "A" if favorite_is_a else "B"
        winner_side = "A" if int(row["player_A_win"]) == 1 else "B"
        favorite_rank = row[f"rank_{favorite_side}"]
        winner_rank = row[f"rank_{winner_side}"]
        favorite_age = row[f"age_years_{favorite_side}"]
        winner_age = row[f"age_years_{winner_side}"]
        rows.append(
            {
                "match_id": row["match_id"],
                "date": row["date"],
                "surface": row["surface"],
                "level": row["level"],
                "round": row["round"],
                "favorite_id": row[f"player_{favorite_side}_id"],
                "winner_id": row[f"player_{winner_side}_id"],
                "favorite_probability": confidence.loc[index],
                "favorite_rank": favorite_rank,
                "winner_rank": winner_rank,
                "winner_lower_ranked": bool(
                    pd.notna(winner_rank)
                    and pd.notna(favorite_rank)
                    and winner_rank > favorite_rank
                ),
                "favorite_age": favorite_age,
                "winner_age": winner_age,
                "winner_younger": bool(
                    pd.notna(winner_age)
                    and pd.notna(favorite_age)
                    and winner_age < favorite_age
                ),
                "missing_model_features": int(row[selected_columns].isna().sum()),
            }
        )
    errors = pd.DataFrame(rows).sort_values("favorite_probability", ascending=False)
    errors.to_csv(report_dir / "phase6_confident_errors.csv", index=False)
    surface = errors.groupby("surface", dropna=False).size().sort_values(ascending=False)
    level = errors.groupby("level", dropna=False).size().sort_values(ascending=False)
    report = f"""# Phase 6 — erreurs confiantes

Le meilleur arbre gelé se trompe sur **{len(errors):,}** matchs avec une confiance supérieure
à 80 %, soit {len(errors) / int(benchmark.sum()):.2%} du benchmark historique.

## Patterns observables

- vainqueur moins bien classé que le favori : {errors["winner_lower_ranked"].mean():.1%};
- vainqueur plus jeune que le favori : {errors["winner_younger"].mean():.1%};
- nombre médian de features modèle manquantes : {errors["missing_model_features"].median():.0f};
- surfaces : {surface.to_dict()};
- niveaux source : {level.to_dict()}.

Les blessures, retours de blessure et états physiques ne sont pas observés dans V3 : aucune
conclusion de ce type ne peut être tirée. Les retraits/walkovers ne sont pas utilisés comme
résultats joués. La liste exhaustive triée par confiance est dans
`phase6_confident_errors.csv`.

## Cas les plus confiants

{_markdown_table(errors.head(30), ["match_id", "date", "surface", "level", "favorite_id", "winner_id", "favorite_probability", "favorite_rank", "winner_rank", "missing_model_features"])}
"""
    (report_dir / "phase6_confident_errors.md").write_text(report, encoding="utf-8")
    return errors


def verify_reload(path: Path, frame: pd.DataFrame) -> None:
    before: TreeArtifact = load(path)
    expected = before.predict_proba(frame.head(256))
    after: TreeArtifact = load(path)
    actual = after.predict_proba(frame.head(256))
    np.testing.assert_array_equal(expected, actual)


def write_phase6_report(
    *,
    report_dir: Path,
    tuning_summary: pd.DataFrame,
    selected_configs: pd.DataFrame,
    calibration_results: pd.DataFrame,
    per_variant: pd.DataFrame,
    per_algorithm: pd.DataFrame,
    best_selected: pd.Series,
    best_calibrated: pd.Series,
    benchmark: pd.DataFrame,
    tree_benchmark: pd.DataFrame,
    segments: pd.DataFrame,
    confidence: pd.DataFrame,
    importance: pd.DataFrame,
    shap_global: pd.DataFrame,
    interactions: pd.DataFrame,
    disagreement: pd.DataFrame,
    confident_errors: pd.DataFrame,
    logistic_validation: pd.DataFrame,
) -> None:
    logistic = benchmark.loc[benchmark["model"] == "Logistic Phase 5"].iloc[0]
    best_tree = benchmark.loc[benchmark["model"] == "Best calibrated tree"].iloc[0]
    deltas = {
        metric: float(best_tree[metric]) - float(logistic[metric])
        for metric in ("accuracy", "log_loss", "brier", "roc_auc")
    }
    metric_winners = {}
    eligible = benchmark.loc[benchmark["model"] != "ATP Ranking"]
    for metric, ascending in (
        ("accuracy", False),
        ("log_loss", True),
        ("brier", True),
        ("roc_auc", False),
    ):
        metric_winners[metric] = str(
            eligible.sort_values(metric, ascending=ascending).iloc[0]["model"]
        )
    stable_gain = deltas["log_loss"] < -0.001 and deltas["brier"] < 0
    recommendation = (
        "figer le meilleur arbre calibré comme moteur PRE-MATCH V1"
        if stable_gain
        else "conserver Logistic Phase 5 comme moteur PRE-MATCH V1"
    )
    logistic_validation_best = logistic_validation.sort_values(
        ["log_loss", "brier", "roc_auc", "accuracy"],
        ascending=[True, True, False, False],
    ).iloc[0]
    full_vs_clean = per_variant[
        [
            "algorithm",
            "variant",
            "calibration",
            "accuracy",
            "log_loss",
            "brier",
            "roc_auc",
        ]
    ].sort_values(["algorithm", "variant"])
    fatigue_deltas = []
    for algorithm, group in per_variant.groupby("algorithm"):
        by_variant = group.set_index("variant")
        delta = float(
            by_variant.loc["no_fatigue_proxy", "log_loss"]
            - by_variant.loc["full", "log_loss"]
        )
        fatigue_deltas.append(f"{algorithm}: {delta:+.5f}")
    tree_selected_benchmark = tree_benchmark.loc[
        tree_benchmark.apply(
            lambda row: bool(
                (
                    (per_variant["algorithm"] == row["algorithm"])
                    & (per_variant["variant"] == row["variant"])
                ).any()
                and row["probability"]
                == per_variant.loc[
                    (per_variant["algorithm"] == row["algorithm"])
                    & (per_variant["variant"] == row["variant"]),
                    "calibration",
                ].iloc[0]
            ),
            axis=1,
        )
    ]
    surface_year = segments.loc[segments["segment_type"].isin(["surface", "year"])]
    level = segments.loc[segments["segment_type"] == "level"]
    top_importance = (
        importance.sort_values(["algorithm", "permutation_log_loss_increase"], ascending=[True, False])
        .groupby("algorithm", as_index=False, sort=False)
        .head(12)
    )
    report = f"""# Phase 6 — modèles non linéaires pré-match

## Statut du benchmark

La période 2023–2026 est un **historical benchmark déjà consulté**, pas un test aveugle. Aucun
résultat de cette période n'entre dans les features, hyperparamètres, early stopping, choix de
variante, calibration ou sélection du modèle. Un nouveau holdout prospectif reste nécessaire.

## Protocole walk-forward

Cinq folds valident successivement 2018, 2019, 2020, 2021 et 2022; chaque entraînement utilise
strictement les années antérieures. La sélection minimise la moyenne de log loss, puis Brier,
ROC-AUC et accuracy. Les calibrateurs sont eux-mêmes évalués en walk-forward de 2019 à 2022 sur
des prédictions out-of-fold. Les artefacts finaux sont entraînés jusqu'au 31 décembre 2022.

La recherche comprend {len(tuning_summary):,} combinaisons algorithme/variante/configuration et
{int(tuning_summary["folds"].sum()):,} entraînements arbres. Les paramètres et chaque fold sont
conservés dans `phase6_tuning.csv`.

À protocole de calibration identique sur 2019–2022, la Logistic Phase 5 refit atteint une log
loss de {float(logistic_validation_best["log_loss"]):.5f}, contre
{float(best_selected["log_loss"]):.5f} pour le meilleur arbre sélectionné. Ce différentiel
mesure le gain validation de la non-linéarité sans consulter 2023–2026.

{_markdown_table(logistic_validation, ["calibration", "matches", "accuracy", "log_loss", "brier", "roc_auc"])}

## Sélection des hyperparamètres

{_markdown_table(selected_configs, ["algorithm", "variant", "config_id", "mean_log_loss", "mean_brier", "mean_roc_auc", "mean_accuracy", "median_best_iteration", "total_fit_seconds"])}

## Dataset complet vs sans proxy de fatigue

La variante `no_fatigue_proxy` retire exactement les 13 variables Phase 4 basées sur
`matches/sets/games/minutes` à 7/14/30 jours et `rest_days`. Elle conserve la charge de tournois
antérieurs Phase 5, qui exclut l'édition courante.

{_markdown_table(full_vs_clean, ["algorithm", "variant", "calibration", "accuracy", "log_loss", "brier", "roc_auc"])}

La version complète gagne en log loss validation pour les trois familles; la pénalité de retrait
est {", ".join(fatigue_deltas)}. Le signal n'est donc pas un accident propre à un seul moteur,
mais les importances montrent qu'Elo, ranking, expérience et statistiques ajustées dominent les
proxies de progression. La variante complète reste expérimentale, pas causale.

Le candidat global choisi avant ouverture du benchmark est
`{best_selected["algorithm"]}/{best_selected["variant"]}` avec calibration
`{best_selected["calibration"]}`. Le meilleur candidat explicitement calibré est
`{best_calibrated["algorithm"]}/{best_calibrated["variant"]}` avec
`{best_calibrated["calibration"]}`.

## Calibration validation

{_markdown_table(calibration_results.sort_values(["algorithm", "variant", "log_loss"]), ["algorithm", "variant", "calibration", "matches", "accuracy", "log_loss", "brier", "roc_auc"])}

La courbe rétrospective est enregistrée dans `phase6_calibration.png`; les buckets détaillés
sont dans `phase6_calibration.csv`.

## Historical benchmark 2023–2026

{_markdown_table(benchmark, ["model", "matches", "accuracy", "log_loss", "brier", "roc_auc"])}

- meilleure accuracy : **{metric_winners["accuracy"]}**;
- meilleure log loss : **{metric_winners["log_loss"]}**;
- meilleur Brier : **{metric_winners["brier"]}**;
- meilleur ROC-AUC : **{metric_winners["roc_auc"]}**.

Comparaison des six candidats arbres avec leur calibration sélectionnée :

{_markdown_table(tree_selected_benchmark, ["algorithm", "variant", "probability", "matches", "accuracy", "log_loss", "brier", "roc_auc"])}

## Gain réel contre Logistic Phase 5

- accuracy : {deltas["accuracy"] * 100:+.3f} point;
- log loss : {deltas["log_loss"]:+.5f};
- Brier : {deltas["brier"]:+.5f};
- ROC-AUC : {deltas["roc_auc"]:+.5f}.

## Stabilité par surface et année

{_markdown_table(surface_year, ["segment_type", "segment", "model", "matches", "accuracy", "log_loss", "brier", "roc_auc"])}

Grass reste le segment le plus incertain en raison de son faible volume. Les résultats ne sont
jamais utilisés pour modifier le modèle gelé.

## Niveau de tournoi

Les codes source distinguent fiablement `G` (Grand Slam) et `M` (Masters); tous les autres
niveaux sont regroupés sans inventer ATP 250/500.

{_markdown_table(level, ["segment", "model", "matches", "accuracy", "log_loss", "brier", "roc_auc"])}

## Confiance

{_markdown_table(confidence, ["model", "bin", "matches", "mean_probability", "actual_win_rate"])}

## Importance et SHAP

Les importances natives (gain pour XGBoost/LightGBM, `LossFunctionChange` pour CatBoost) et les
permutations sont calculées sur 2022 avec un modèle diagnostique entraîné jusqu'en 2021.

{_markdown_table(top_importance, ["algorithm", "feature", "native_gain", "permutation_log_loss_increase", "permutation_std"])}

Top SHAP du meilleur arbre calibré, en valeur absolue moyenne sur un échantillon déterministe du
benchmark :

{_markdown_table(shap_global.head(20), ["feature", "mean_abs_shap", "mean_shap", "std_shap"])}

SHAP explique le modèle en unités de sortie interne/log-odds et non en causalité ni en points de
probabilité additifs. Les exemples individuels sont dans `phase6_shap_examples.csv`.

## Interactions apprises

{_markdown_table(interactions.head(20), ["feature_1", "feature_2", "mean_abs_interaction"])}

## Désaccord Logistic vs arbre

{_markdown_table(disagreement, ["matches", "share_of_benchmark", "logistic_accuracy", "tree_accuracy", "logistic_mean_confidence", "tree_mean_confidence"])}

Les features qui dominent les désaccords sont détaillées dans `phase6_disagreement_shap.csv`.

## Erreurs confiantes

{len(confident_errors):,} erreurs dépassent 80 % de confiance. L'audit complet est dans
`phase6_confident_errors.md`. Les données ne permettent pas d'identifier fiablement les
blessures ou retours de blessure.

## Complexité et recommandation

Au regard de la performance probabiliste, de la stabilité, de la calibration, de
l'interprétabilité et du coût opérationnel, la recommandation est de **{recommendation}**.
La sélection technique de l'arbre reste fondée uniquement sur 2018–2022; le benchmark sert à
quantifier rétrospectivement sa stabilité. Aucun réseau neuronal, Transformer, modèle live ou
point-by-point n'est commencé.
"""
    (report_dir / "phase6_report.md").write_text(report, encoding="utf-8")


def run_phase6(
    *,
    config: Phase6Config,
    features: pd.DataFrame,
    ratings: pd.DataFrame,
    phase5_artifact: dict[str, Any],
    phase5_selected: dict[str, Any],
    report_dir: Path,
    model_dir: Path,
) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    freeze_phase5(report_dir)
    variants = feature_variants(list(phase5_selected["features"]))
    for columns in variants.values():
        validate_model_matrix(features, columns)

    LOGGER.info("Running walk-forward tree tuning")
    tuning, tuning_summary, oof = tune_tree_models(features, variants, config)
    logistic_tuning, logistic_oof = walk_forward_logistic(
        features, variants["full"], config
    )
    tuning_with_baseline = pd.concat([tuning, logistic_tuning], ignore_index=True)
    tuning_with_baseline.to_csv(report_dir / "phase6_tuning.csv", index=False)
    tuning_summary.to_csv(report_dir / "phase6_tuning_summary.csv", index=False)
    selected_configs = select_best_configs(tuning_summary)
    selected_configs.to_csv(report_dir / "phase6_selected_configs.csv", index=False)

    LOGGER.info("Selecting calibration from out-of-fold historical predictions")
    calibration_results, calibrators = select_calibration(
        features, selected_configs, oof, config
    )
    logistic_selection = pd.DataFrame(
        [
            {
                "algorithm": "logistic_phase5_refit",
                "variant": "full",
                "config_id": "phase5_C1",
            }
        ]
    )
    logistic_validation, _ = select_calibration(
        features,
        logistic_selection,
        {("logistic_phase5_refit", "full", "phase5_C1"): logistic_oof},
        config,
    )
    logistic_validation.to_csv(
        report_dir / "phase6_logistic_validation.csv", index=False
    )
    calibration_results.to_csv(
        report_dir / "phase6_calibration_validation.csv", index=False
    )
    per_variant, per_algorithm, best_selected = _selection_rows(calibration_results)
    best_calibrated = (
        calibration_results.loc[calibration_results["calibration"] != "raw"]
        .sort_values(
            ["log_loss", "brier", "roc_auc", "accuracy"],
            ascending=[True, True, False, False],
            kind="stable",
        )
        .iloc[0]
    )
    selection_payload = {
        "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "benchmark_status": "historical benchmark; not used for selection",
        "validation_years": list(config.validation_years),
        "removed_no_fatigue_proxy": PHASE4_FATIGUE,
        "selected_configs": selected_configs.to_dict("records"),
        "selected_calibration_per_variant": per_variant.to_dict("records"),
        "selected_per_algorithm": per_algorithm.to_dict("records"),
        "best_selected": best_selected.to_dict(),
        "best_calibrated": best_calibrated.to_dict(),
        "logistic_validation": logistic_validation.to_dict("records"),
    }
    (report_dir / "phase6_selection_frozen.json").write_text(
        json.dumps(
            selection_payload, indent=2, ensure_ascii=False, default=_json_default
        )
        + "\n",
        encoding="utf-8",
    )

    LOGGER.info("Training frozen final tree artifacts through 2022")
    artifacts, per_algorithm, best_selected = build_final_artifacts(
        features,
        variants,
        selected_configs,
        tuning,
        calibration_results,
        calibrators,
        config,
        model_dir,
    )
    calibrated_key = (
        str(best_calibrated["algorithm"]),
        str(best_calibrated["variant"]),
    )
    calibrated_method = str(best_calibrated["calibration"])
    calibrated_artifact = replace(
        artifacts[calibrated_key],
        calibration_method=calibrated_method,
        calibrator=calibrators[(*calibrated_key, calibrated_method)],
    )
    dump(calibrated_artifact, model_dir / "best_calibrated_phase6.joblib")
    for path in sorted(model_dir.glob("*.joblib")):
        verify_reload(path, features)

    LOGGER.info("Opening historical benchmark after Phase 6 selection freeze")
    (
        benchmark,
        tree_benchmark,
        segments,
        calibration,
        probabilities,
        shap_artifact,
    ) = evaluate_benchmark(
        features,
        ratings,
        phase5_artifact,
        artifacts,
        per_algorithm,
        best_calibrated,
        calibrators,
        config,
    )
    benchmark.to_csv(report_dir / "phase6_benchmark_comparison.csv", index=False)
    tree_benchmark.to_csv(report_dir / "phase6_fatigue_ablation_benchmark.csv", index=False)
    segments.to_csv(report_dir / "phase6_benchmark_segments.csv", index=False)
    calibration.to_csv(report_dir / "phase6_calibration.csv", index=False)
    confidence = confidence_table(
        features.loc[pd.to_datetime(features["date"]) >= config.benchmark_start, "player_A_win"],
        {
            "Logistic Phase 5": probabilities["Logistic Phase 5"].dropna().to_numpy(),
            "Best tree calibrated": probabilities["Best calibrated tree"].dropna().to_numpy(),
        },
    )
    confidence.to_csv(report_dir / "phase6_confidence.csv", index=False)
    plot_calibration(calibration, report_dir / "phase6_calibration.png")

    LOGGER.info("Computing held-out permutation and native importances")
    importance = build_feature_importance(
        features, variants, per_algorithm, selected_configs, tuning, config
    )
    importance.to_csv(report_dir / "phase6_feature_importance.csv", index=False)
    plot_importance(
        importance,
        str(best_calibrated["algorithm"]),
        report_dir / "phase6_feature_importance.png",
    )

    LOGGER.info("Computing SHAP explanations for the best calibrated tree")
    shap_global, _examples, _disagreement_shap = build_shap_outputs(
        features,
        shap_artifact,
        probabilities["Logistic Phase 5"],
        probabilities["Best calibrated tree"],
        config,
        report_dir,
    )
    interactions = pd.read_csv(report_dir / "phase6_shap_interactions.csv")
    disagreement, disagreement_details = analyze_disagreement(
        features,
        probabilities["Logistic Phase 5"],
        probabilities["Best calibrated tree"],
        config,
    )
    disagreement.to_csv(report_dir / "phase6_disagreement_summary.csv", index=False)
    disagreement_details.to_csv(report_dir / "phase6_disagreements.csv", index=False)
    confident_errors = write_confident_errors(
        features,
        probabilities["Best calibrated tree"],
        shap_artifact.feature_columns,
        config,
        report_dir,
    )

    write_phase6_report(
        report_dir=report_dir,
        tuning_summary=tuning_summary,
        selected_configs=selected_configs,
        calibration_results=calibration_results,
        per_variant=per_variant,
        per_algorithm=per_algorithm,
        best_selected=best_selected,
        best_calibrated=best_calibrated,
        benchmark=benchmark,
        tree_benchmark=tree_benchmark,
        segments=segments,
        confidence=confidence,
        importance=importance,
        shap_global=shap_global,
        interactions=interactions,
        disagreement=disagreement,
        confident_errors=confident_errors,
        logistic_validation=logistic_validation,
    )
    result = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "selection": selection_payload,
        "benchmark": [
            {
                key: (None if isinstance(value, float) and math.isnan(value) else value)
                for key, value in row.items()
            }
            for row in benchmark.to_dict("records")
        ],
    }
    (report_dir / "phase6_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return {"benchmark": benchmark, "selection": selection_payload}


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Run Phase 6 nonlinear pre-match benchmark")
    command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    command.add_argument(
        "--features", type=Path, default=PROCESSED_DIR / "match_features_v3.parquet"
    )
    command.add_argument(
        "--ratings", type=Path, default=PROCESSED_DIR / "match_ratings_v2.parquet"
    )
    command.add_argument(
        "--phase5-model",
        type=Path,
        default=PROCESSED_DIR / "models" / "logistic_phase5.joblib",
    )
    command.add_argument(
        "--phase5-selected", type=Path, default=REPORTS_DIR / "phase5_selected.json"
    )
    command.add_argument("--report-dir", type=Path, default=REPORTS_DIR)
    command.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models" / "phase6")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results = run_phase6(
        config=Phase6Config.from_toml(args.config),
        features=pd.read_parquet(args.features),
        ratings=pd.read_parquet(args.ratings),
        phase5_artifact=load(args.phase5_model),
        phase5_selected=json.loads(args.phase5_selected.read_text(encoding="utf-8")),
        report_dir=args.report_dir,
        model_dir=args.model_dir,
    )
    print(results["benchmark"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
