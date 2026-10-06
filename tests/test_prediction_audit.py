from __future__ import annotations

import math

import joblib
import numpy as np
import pandas as pd
import pytest

from tennis_analytics.paths import PROCESSED_DIR, PROJECT_ROOT
from tennis_analytics.product.model_audit import (
    prediction_decomposition,
    reverse_feature_values,
)


@pytest.fixture(scope="module")
def artifact():
    return joblib.load(PROJECT_ROOT / "models" / "prematch_v1" / "model.joblib")


@pytest.fixture(scope="module")
def historical_values(artifact):
    frame = pd.read_parquet(
        PROCESSED_DIR / "match_features_v3.parquet",
        filters=[("match_id", "==", "atp:2026-520:2026-05-25:198")],
        columns=artifact["feature_columns"],
    )
    return {
        name: None if pd.isna(frame.iloc[0][name]) else float(frame.iloc[0][name])
        for name in artifact["feature_columns"]
    }


def test_contribution_sum_is_exact_raw_logit(artifact, historical_values):
    rows, summary = prediction_decomposition(artifact, historical_values)

    assert summary["intercept"] + rows["contribution_to_logit"].sum() == pytest.approx(
        summary["raw_logit"], abs=1e-14
    )
    assert artifact["pipeline"].predict_proba(
        pd.DataFrame([historical_values])[artifact["feature_columns"]]
    )[0, 1] == pytest.approx(summary["raw_probability"], abs=1e-14)


def test_nulls_use_training_median_before_standardization(artifact, historical_values):
    values = dict(historical_values)
    values["h2h_edge_A"] = None
    rows, _ = prediction_decomposition(artifact, values)
    h2h = rows.set_index("feature").loc["h2h_edge_A"]

    assert bool(h2h["was_missing"])
    assert h2h["imputed_value"] == pytest.approx(h2h["imputer_median"])
    assert math.isfinite(h2h["standardized_value"])


def test_contribution_direction_and_ab_reverse_contract(artifact, historical_values):
    rows, summary = prediction_decomposition(artifact, historical_values)
    reversed_values = reverse_feature_values(historical_values)
    _, reversed_summary = prediction_decomposition(artifact, reversed_values)

    assert (rows.loc[rows["contribution_to_logit"] > 0, "direction"] == "favors_player_A").all()
    assert (rows.loc[rows["contribution_to_logit"] < 0, "direction"] == "favors_player_B").all()
    for name, value in historical_values.items():
        if value is None:
            continue
        if name.endswith("_diff") or "_diff_x_" in name or name in {
            "elo_logit", "elo_logit_x_clay", "elo_logit_x_grass", "h2h_edge_A",
        }:
            assert reversed_values[name] == pytest.approx(-value)
    # Do not impose an exact probability complement on the frozen model: its
    # learned scaler means and model/calibration intercepts are non-zero.  The
    # permanent contract is the exact orientation of the input features; the
    # audit reports the resulting probability asymmetry explicitly.
    assert 0.0 < summary["final_probability"] < 1.0
    assert 0.0 < reversed_summary["final_probability"] < 1.0
    assert summary["final_probability"] + reversed_summary["final_probability"] != pytest.approx(
        1.0, abs=1e-9
    )
    assert reverse_feature_values(reversed_values) == pytest.approx(historical_values)


def test_frozen_runtime_artifact_has_strict_research_parity(artifact, historical_values):
    research = joblib.load(PROCESSED_DIR / "models" / "logistic_phase5.joblib")
    frozen_rows, frozen = prediction_decomposition(artifact, historical_values)
    research_rows, original = prediction_decomposition(research, historical_values)

    assert artifact["feature_columns"] == research["feature_columns"]
    assert frozen["final_probability"] == pytest.approx(original["final_probability"], abs=1e-15)
    np.testing.assert_allclose(
        frozen_rows["contribution_to_logit"],
        research_rows["contribution_to_logit"],
        rtol=0,
        atol=1e-15,
    )
