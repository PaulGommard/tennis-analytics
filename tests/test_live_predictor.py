from __future__ import annotations

import pytest

from tennis_analytics.live.predictor import LiveInputError, LivePredictor, parse_point_score


def test_parse_regular_and_tiebreak_point_scores() -> None:
    assert parse_point_score("AD", "40", False) == (4, 3)
    assert parse_point_score("8", "7", True) == (8, 7)
    with pytest.raises(LiveInputError):
        parse_point_score("AD", "30", False)


def test_live_predictor_matches_supplied_prematch_at_start() -> None:
    predictor = LivePredictor()
    prediction = predictor.predict(
        player_a="Jannik Sinner",
        player_b="Carlos Alcaraz",
        surface="Clay",
        sets_a=0,
        sets_b=0,
        games_a=0,
        games_b=0,
        points_a="0",
        points_b="0",
        server="A",
        first_server="A",
        prematch_probability_a=0.58,
    )
    assert prediction.p_match_a == pytest.approx(0.58, abs=1e-6)
    assert 0 < prediction.p_next_point_a < 1
