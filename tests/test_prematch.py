import json

import pandas as pd
import pytest

from tennis_analytics.prematch.predictor import (
    Prediction,
    PredictionInputError,
    PreMatchPredictor,
)
from tennis_analytics.prematch.registry import append_prediction, evaluate_ledger, record_result


def prediction() -> Prediction:
    return Prediction(
        model_version="prematch_v1",
        prediction_id="test-id",
        match_id="prospective:test",
        generated_at="2026-10-06T00:00:00+00:00",
        match_date="2026-10-07",
        surface="Hard",
        player_a_id="a",
        player_a_name="A",
        player_b_id="b",
        player_b_name="B",
        probability_a=0.6,
        probability_b=0.4,
        raw_probability_a=0.61,
        favorite="A",
        feature_snapshot_hash="abc",
        warnings=(),
        top_contributions=(),
        features={"elo_logit": 0.2},
    )


def test_prospective_ledger_is_append_only(tmp_path) -> None:
    path = tmp_path / "ledger.parquet"
    assert append_prediction(prediction(), path)
    assert not append_prediction(prediction(), path)
    ledger = pd.read_parquet(path)
    assert len(ledger) == 1
    assert json.loads(ledger.iloc[0].features_json) == {"elo_logit": 0.2}


def test_result_is_immutable_and_evaluable(tmp_path) -> None:
    path = tmp_path / "ledger.parquet"
    append_prediction(prediction(), path)
    record_result("test-id", 1, path)
    record_result("test-id", 1, path)
    with pytest.raises(ValueError, match="immutable"):
        record_result("test-id", 0, path)
    metrics = evaluate_ledger(path)
    assert metrics["evaluated"] == 1
    assert metrics["accuracy"] == 1.0


def test_input_error_has_explicit_contract() -> None:
    assert issubclass(PredictionInputError, ValueError)


def test_player_resolution_is_explicit_for_unknown_and_ambiguous_names() -> None:
    predictor = PreMatchPredictor()
    unknown = predictor.resolve_player("A Totally New Player")
    assert unknown.status == "cold_start"
    assert unknown.player_id.startswith("new:")
    with pytest.raises(PredictionInputError, match="Ambiguous player"):
        predictor.resolve_player("Martin Landaluce")


@pytest.mark.parametrize(
    ("match_id", "expected"),
    [
        ("atp:2000-339:2000-01-03:1", 0.5099154291694334),
        ("atp:2012-773:2012-04-23:8", 0.3552746713544534),
        ("atp:2026-520:2026-05-25:226", 0.7611394131384859),
    ],
)
def test_historical_inference_replays_frozen_probability(match_id, expected) -> None:
    replay = PreMatchPredictor().replay(match_id)
    assert replay.probability_a == pytest.approx(expected, abs=1e-15)
    assert len(replay.feature_snapshot_hash) == 64
