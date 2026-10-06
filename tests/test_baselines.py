import pandas as pd
import pytest

from tennis_analytics.evaluation.baselines import metrics_for


def test_ranking_convention_and_missing_values():
    frame = pd.DataFrame(
        {
            "player_A_win": [1, 0, 1],
            "rank_A": [1, 30, None],
            "rank_B": [20, 10, 5],
        }
    )
    result = metrics_for(frame, "ATP Ranking")
    assert result["accuracy_n"] == 2
    assert result["accuracy"] == 1.0


def test_elo_probability_metrics_use_pre_match_probability():
    frame = pd.DataFrame(
        {
            "player_A_win": [1, 0],
            "elo_diff": [100.0, -100.0],
            "elo_expected_A": [0.75, 0.25],
        }
    )
    result = metrics_for(frame, "Global Elo")
    assert result["accuracy"] == 1.0
    assert result["log_loss"] == pytest.approx(
        -pd.Series([0.75, 0.75]).map(__import__("math").log).mean()
    )
    assert result["brier"] == pytest.approx(0.0625)
