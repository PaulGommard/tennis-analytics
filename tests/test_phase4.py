import numpy as np
import pandas as pd
import pytest

from tennis_analytics.features.score import parse_score_workload
from tennis_analytics.features.stat_history import (
    build_stat_history_base,
    materialize_features,
    metric_contributions,
)
from tennis_analytics.modeling.logistic import (
    Phase4Config,
    make_pipeline,
    select_best,
    temporal_masks,
)


def stat_row(match_id: str, player_id: str, *, service_points: int = 10) -> dict[str, object]:
    return {
        "match_id": match_id,
        "player_id": player_id,
        "aces": 2,
        "double_faults": 1,
        "service_points": service_points,
        "first_serve_in": 6,
        "first_serve_won": 4,
        "second_serve_won": 2,
        "service_games": 2,
        "break_points_saved": 1,
        "break_points_faced": 2,
    }


def phase4_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratings = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date": pd.Timestamp("2020-01-01"),
                "surface": "Hard",
                "level": "A",
                "round": "R32",
                "player_A_id": "A",
                "player_B_id": "B",
                "player_A_win": 1,
                "chronology_batch_id": 0,
                "elo_diff": 0.0,
                "final_raw_expected_A": 0.5,
                "rank_A": 1,
                "rank_B": 2,
                "rank_diff": 1,
                "career_matches_A": 0,
                "career_matches_B": 0,
                "surface_matches_A": 0,
                "surface_matches_B": 0,
                "days_since_last_match_A": np.nan,
                "days_since_last_match_B": np.nan,
            },
            {
                "match_id": "m2",
                "date": pd.Timestamp("2020-01-01"),
                "surface": "Hard",
                "level": "A",
                "round": "R32",
                "player_A_id": "A",
                "player_B_id": "C",
                "player_A_win": 0,
                "chronology_batch_id": 0,
                "elo_diff": 0.0,
                "final_raw_expected_A": 0.5,
                "rank_A": 1,
                "rank_B": 3,
                "rank_diff": 2,
                "career_matches_A": 0,
                "career_matches_B": 0,
                "surface_matches_A": 0,
                "surface_matches_B": 0,
                "days_since_last_match_A": np.nan,
                "days_since_last_match_B": np.nan,
            },
            {
                "match_id": "m3",
                "date": pd.Timestamp("2020-01-01"),
                "surface": "Hard",
                "level": "A",
                "round": "R16",
                "player_A_id": "A",
                "player_B_id": "D",
                "player_A_win": 1,
                "chronology_batch_id": 1,
                "elo_diff": 10.0,
                "final_raw_expected_A": 0.52,
                "rank_A": 1,
                "rank_B": 4,
                "rank_diff": 3,
                "career_matches_A": 2,
                "career_matches_B": 0,
                "surface_matches_A": 2,
                "surface_matches_B": 0,
                "days_since_last_match_A": 0,
                "days_since_last_match_B": np.nan,
            },
        ]
    )
    timeline = ratings[["match_id"]].copy()
    timeline["age_years_A"] = 25.0
    timeline["age_years_B"] = 27.0
    timeline["rank_points_A"] = 1000
    timeline["rank_points_B"] = 800
    matches = pd.DataFrame(
        {
            "match_id": ["m1", "m2", "m3"],
            "minutes": [60, 90, 120],
            "score": ["6-4 6-4", "7-6(5) 6-3", "6-0 6-0"],
        }
    )
    stats = pd.DataFrame(
        [
            stat_row(match_id, player)
            for match_id, players in (("m1", ("A", "B")), ("m2", ("A", "C")), ("m3", ("A", "D")))
            for player in players
        ]
    )
    return ratings, timeline, matches, stats


def config() -> Phase4Config:
    return Phase4Config(
        point_prior_strengths=(50.0,),
        surface_prior_strengths=(100.0,),
        logistic_c_values=(1.0,),
        train_end=pd.Timestamp("2018-12-31"),
        validation_start=pd.Timestamp("2019-01-01"),
        validation_end=pd.Timestamp("2022-12-31"),
        test_start=pd.Timestamp("2023-01-01"),
    )


def test_score_workload_handles_tiebreak_retirement_and_walkover():
    assert parse_score_workload("7-6(5) 6-4").sets == 2
    assert parse_score_workload("7-6(5) 6-4").games == 23
    retired = parse_score_workload("6-4 2-1 RET")
    assert (retired.sets, retired.games, retired.status) == (2, 13, "retired")
    assert parse_score_workload("W/O").status == "not_played"
    match_tiebreak = parse_score_workload("6-4 3-6 [10-8]")
    assert (match_tiebreak.sets, match_tiebreak.games, match_tiebreak.match_tiebreaks) == (3, 19, 1)


def test_metric_formulas_and_invalid_ratio_filtering():
    own = stat_row("m", "A")
    opponent = stat_row("m", "B")
    metrics = metric_contributions(own, opponent)
    assert metrics["service_points_won_rate"] == (6.0, 10.0)
    assert metrics["return_points_won_rate"] == (4.0, 10.0)
    assert metrics["hold_rate"] == (1.0, 2.0)
    own["second_serve_won"] = 5
    assert "second_serve_won_rate" not in metric_contributions(own, opponent)


def test_current_round_stats_are_frozen_then_weighted_by_opportunities():
    ratings, timeline, matches, stats = phase4_inputs()
    stats.loc[(stats.match_id == "m2") & (stats.player_id == "A"), "service_points"] = 20
    stats.loc[(stats.match_id == "m2") & (stats.player_id == "A"), "first_serve_in"] = 10
    stats.loc[(stats.match_id == "m2") & (stats.player_id == "A"), "first_serve_won"] = 5
    base = build_stat_history_base(ratings, timeline, matches, stats).set_index("match_id")
    assert base.loc["m1", "ace_rate_career_den_A"] == 0
    assert base.loc["m2", "ace_rate_career_den_A"] == 0
    assert base.loc["m3", "ace_rate_career_num_A"] == 4
    assert base.loc["m3", "ace_rate_career_den_A"] == 30
    assert base.loc["m3", "matches_7d_A"] == 2
    assert base.loc["m3", "minutes_7d_A"] == 150


def test_current_match_stats_and_duration_cannot_change_own_features():
    ratings, timeline, matches, stats = phase4_inputs()
    original = build_stat_history_base(ratings, timeline, matches, stats)
    changed_stats = stats.copy()
    changed_stats.loc[changed_stats.match_id == "m3", "aces"] = 9
    changed_matches = matches.copy()
    changed_matches.loc[changed_matches.match_id == "m3", "minutes"] = 300
    changed = build_stat_history_base(ratings, timeline, changed_matches, changed_stats)
    pd.testing.assert_series_equal(
        original.set_index("match_id").loc["m3"],
        changed.set_index("match_id").loc["m3"],
    )


def test_surface_and_circuit_priors_are_strictly_historical():
    ratings, timeline, matches, stats = phase4_inputs()
    base = build_stat_history_base(ratings, timeline, matches, stats).set_index("match_id")
    assert pd.isna(base.loc["m1", "circuit_ace_rate_prior"])
    assert pd.isna(base.loc["m2", "circuit_ace_rate_prior"])
    assert base.loc["m3", "circuit_ace_rate_prior"] == pytest.approx(0.2)
    assert base.loc["m3", "service_points_won_rate_surface_den_A"] == 20

    features = materialize_features(
        base.reset_index(), point_prior_strength=50, surface_prior_strength=100
    ).set_index("match_id")
    assert features.loc["m3", "service_points_won_rate_surface_opportunities_A"] == 20


def test_preprocessing_and_temporal_selection_do_not_fit_on_test():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2018-01-01", "2018-02-01", "2020-01-01", "2023-01-01"]),
            "x": [0.0, 2.0, 3.0, 1000.0],
            "player_A_win": [0, 1, 1, 0],
        }
    )
    train, validation, test = temporal_masks(frame, config())
    pipeline = make_pipeline(1.0)
    pipeline.fit(frame.loc[train, ["x"]], frame.loc[train, "player_A_win"])
    assert pipeline.named_steps["imputer"].statistics_[0] == 1.0
    assert pipeline.named_steps["scaler"].mean_[0] == 1.0
    assert train.sum() == 2 and validation.sum() == 1 and test.sum() == 1

    trials = pd.DataFrame(
        [
            {
                "name": "validation",
                "validation_log_loss": 0.60,
                "validation_brier": 0.2,
                "validation_accuracy": 0.6,
                "test_log_loss": 0.9,
            },
            {
                "name": "test",
                "validation_log_loss": 0.61,
                "validation_brier": 0.19,
                "validation_accuracy": 0.7,
                "test_log_loss": 0.5,
            },
        ]
    )
    assert select_best(trials)["name"] == "validation"
