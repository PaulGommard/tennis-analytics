from __future__ import annotations

import numpy as np
import pandas as pd

from tennis_analytics.live.phase8c import (
    ProbabilityCalibrator,
    _blend,
    _checkpoint_indices,
)


def test_progressive_blend_starts_at_v1_and_moves_toward_v2() -> None:
    v1 = np.array([0.60, 0.60, 0.60])
    v2 = np.array([0.80, 0.80, 0.80])
    points = np.array([0.0, 50.0, 200.0])
    result = _blend(v1, v2, points, 200, "linear")
    assert result[0] == v1[0]
    assert v1[0] < result[1] < result[2] < v2[0]
    assert result[2] == 0.70


def test_tempered_logit_stays_between_v1_and_v2() -> None:
    result = _blend(
        np.array([0.2, 0.8]),
        np.array([0.7, 0.3]),
        np.array([100.0, 100.0]),
        100,
        "tempered_logit",
    )
    assert 0.2 < result[0] < 0.7
    assert 0.3 < result[1] < 0.8


def test_none_calibrator_is_identity() -> None:
    probability = np.array([0.1, 0.4, 0.9])
    target = np.array([0, 1, 1])
    calibrator = ProbabilityCalibrator("none").fit(
        probability, target, np.ones(len(target))
    )
    np.testing.assert_allclose(calibrator.predict(probability), probability)


def test_depth_checkpoints_never_use_future_row() -> None:
    group = pd.DataFrame(
        {
            "point_index": range(121),
            "sets_A": [0] * 60 + [1] * 61,
            "sets_B": [0] * 100 + [1] * 21,
        }
    )
    checkpoints = {label: index for index, label in _checkpoint_indices(group)}
    assert checkpoints["before_match"] == 0
    assert checkpoints["after_20_points"] == 20
    assert checkpoints["after_50_points"] == 50
    assert checkpoints["after_100_points"] == 100
    assert checkpoints["after_1_sets"] == 60
    assert checkpoints["after_2_sets"] == 100
