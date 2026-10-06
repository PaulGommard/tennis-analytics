from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from joblib import dump, load
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from tennis_analytics.modeling.phase5 import PHASE4_FATIGUE
from tennis_analytics.modeling.phase6 import (
    Phase6Config,
    TreeArtifact,
    feature_variants,
    fit_fold_candidate,
    select_calibration,
    temporal_folds,
    validate_model_matrix,
    verify_reload,
)


def _config() -> Phase6Config:
    return Phase6Config(
        validation_years=(2018, 2019, 2020, 2021, 2022),
        benchmark_start=pd.Timestamp("2023-01-01"),
        random_seed=42,
        early_stopping_rounds=5,
        calibration_min_year=2019,
        shap_sample_size=20,
        interaction_sample_size=10,
        permutation_repeats=1,
        grids={"xgboost": (), "lightgbm": (), "catboost": ()},
    )


def _year_frame() -> pd.DataFrame:
    dates = pd.to_datetime(
        [f"{year}-{month:02d}-01" for year in range(2016, 2024) for month in (1, 7)]
    )
    return pd.DataFrame(
        {
            "date": dates,
            "feature": np.linspace(-1, 1, len(dates)),
            "player_A_win": np.arange(len(dates)) % 2,
        }
    )


def test_walk_forward_is_strictly_temporal_and_excludes_benchmark():
    frame = _year_frame()
    folds = temporal_folds(frame, _config().validation_years, _config().benchmark_start)
    assert [year for year, _, _ in folds] == [2018, 2019, 2020, 2021, 2022]
    for year, train, validation in folds:
        assert frame.loc[train, "date"].max() < frame.loc[validation, "date"].min()
        assert frame.loc[validation, "date"].dt.year.eq(year).all()
        assert frame.loc[validation, "date"].max() < pd.Timestamp("2023-01-01")


def test_benchmark_cannot_be_a_tuning_fold():
    with pytest.raises(ValueError, match="Benchmark rows entered"):
        temporal_folds(_year_frame(), (2023,), pd.Timestamp("2023-01-01"))


def test_no_fatigue_variant_removes_exactly_the_phase4_proxies():
    selected = ["elo_logit", *PHASE4_FATIGUE, "prior_tournament_matches_14d_diff"]
    variants = feature_variants(selected)
    assert variants["full"] == selected
    assert set(variants["full"]) - set(variants["no_fatigue_proxy"]) == set(PHASE4_FATIGUE)
    assert "prior_tournament_matches_14d_diff" in variants["no_fatigue_proxy"]


def test_model_matrix_rejects_infinity_but_allows_missing_values():
    frame = pd.DataFrame({"x": [1.0, np.nan], "player_A_win": [0, 1]})
    validate_model_matrix(frame, ["x"])
    frame.loc[0, "x"] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        validate_model_matrix(frame, ["x"])


def test_xgboost_fold_is_deterministic_and_imputes_missing_values():
    rng = np.random.default_rng(7)
    x = pd.DataFrame({"x1": rng.normal(size=240), "x2": rng.normal(size=240)})
    x.loc[::11, "x1"] = np.nan
    y = pd.Series((x["x2"] + x["x1"].fillna(0) > 0).astype(int))
    params = {
        "id": "tiny",
        "max_depth": 2,
        "learning_rate": 0.1,
        "n_estimators": 30,
        "min_child_weight": 2.0,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.0,
        "reg_lambda": 2.0,
    }
    first = fit_fold_candidate(
        "xgboost", params, x.iloc[:180], y.iloc[:180], x.iloc[180:], y.iloc[180:], _config()
    )[0]
    second = fit_fold_candidate(
        "xgboost", params, x.iloc[:180], y.iloc[:180], x.iloc[180:], y.iloc[180:], _config()
    )[0]
    np.testing.assert_array_equal(first, second)


def test_calibration_is_evaluated_only_on_later_oof_years():
    frame = _year_frame().loc[lambda value: value.date.dt.year <= 2022].reset_index(drop=True)
    selected = pd.DataFrame(
        [
            {
                "algorithm": "xgboost",
                "variant": "full",
                "config_id": "tiny",
                "params": json.dumps({"id": "tiny"}),
            }
        ]
    )
    probability = pd.Series(np.linspace(0.2, 0.8, len(frame)), index=frame.index)
    results, calibrators = select_calibration(
        frame,
        selected,
        {("xgboost", "full", "tiny"): probability},
        _config(),
    )
    expected = int(frame.date.dt.year.between(2019, 2022).sum())
    assert results.matches.eq(expected).all()
    assert set(results.calibration) == {"raw", "platt", "isotonic"}
    assert ("xgboost", "full", "platt") in calibrators


def test_serialized_artifact_has_identical_predictions_after_reload(tmp_path):
    frame = pd.DataFrame({"x": [-1.0, np.nan, 0.5, 2.0], "player_A_win": [0, 0, 1, 1]})
    imputer = SimpleImputer(strategy="median").fit(frame[["x"]])
    estimator = LogisticRegression().fit(imputer.transform(frame[["x"]]), frame.player_A_win)
    artifact = TreeArtifact(
        algorithm="test",
        variant="full",
        feature_columns=["x"],
        imputer=imputer,
        estimator=estimator,
        calibration_method="raw",
        calibrator=None,
        params={},
        fitted_iterations=1,
        trained_through="2022-12-31",
    )
    path = tmp_path / "artifact.joblib"
    dump(artifact, path)
    expected = artifact.predict_proba(frame)
    actual = load(path).predict_proba(frame)
    np.testing.assert_array_equal(expected, actual)
    verify_reload(path, frame)
