import pandas as pd
import pytest

from tennis_analytics.features.opponent_adjusted import (
    build_opponent_history_base,
    materialize_phase5_features,
)


def _stat(match_id: str, player: str, service_won: int = 6) -> dict[str, object]:
    first_won = min(4, service_won)
    return {
        "match_id": match_id,
        "player_id": player,
        "aces": 1,
        "double_faults": 1,
        "service_points": 10,
        "first_serve_in": 6,
        "first_serve_won": first_won,
        "second_serve_won": service_won - first_won,
        "service_games": 2,
        "break_points_saved": 1,
        "break_points_faced": 2,
    }


def phase5_inputs():
    rows = [
        ("m1", "2020-01-01", "Hard", "A", "B", 1, 0, 0.60, 1500, 1600, "e1"),
        ("m2", "2020-01-01", "Hard", "A", "C", 0, 0, 0.70, 1500, 1700, "e1"),
        ("m3", "2020-01-01", "Hard", "A", "B", 1, 1, 0.55, 1510, 1590, "e1"),
        ("m4", "2020-03-01", "Clay", "A", "B", 0, 2, 0.52, 1520, 1580, "e2"),
    ]
    features = []
    ratings = []
    timeline = []
    matches = []
    stats = []
    for (
        match_id,
        date,
        surface,
        player_a,
        player_b,
        win,
        batch,
        probability,
        elo_a,
        elo_b,
        edition,
    ) in rows:
        features.append(
            {
                "match_id": match_id,
                "date": pd.Timestamp(date),
                "surface": surface,
                "level": "A",
                "round": "R32" if batch == 0 else "R16",
                "player_A_id": player_a,
                "player_B_id": player_b,
                "player_A_win": win,
                "rank_A": 10,
                "rank_B": 20,
                "service_points_won_rate_career_A": 0.60,
                "service_points_won_rate_career_B": 0.60,
                "return_points_won_rate_career_A": 0.40,
                "return_points_won_rate_career_B": 0.40,
                "service_points_won_rate_90d_A": 0.60,
                "service_points_won_rate_90d_B": 0.60,
                "return_points_won_rate_90d_A": 0.40,
                "return_points_won_rate_90d_B": 0.40,
                "elo_logit": 0.0,
                "service_points_won_rate_90d_diff": 0.0,
                "return_points_won_rate_90d_diff": 0.0,
            }
        )
        ratings.append(
            {
                "match_id": match_id,
                "chronology_batch_id": batch,
                "elo_A": elo_a,
                "elo_B": elo_b,
                "final_raw_expected_A": probability,
            }
        )
        timeline.append({"match_id": match_id, "tournament_edition_id": edition})
        matches.append({"match_id": match_id, "minutes": 60, "score": "6-4 6-4"})
        a_service_won = 8 if match_id == "m2" else 6
        stats.extend([_stat(match_id, player_a, a_service_won), _stat(match_id, player_b, 6)])
    return tuple(map(pd.DataFrame, (features, ratings, timeline, matches, stats)))


def test_opponent_strength_form_and_residuals_are_pre_batch():
    features, ratings, timeline, matches, stats = phase5_inputs()
    base = build_opponent_history_base(features, ratings, timeline, matches, stats).set_index(
        "match_id"
    )
    assert pd.isna(base.loc["m1", "opponent_elo_mean_90d_A"])
    assert pd.isna(base.loc["m2", "opponent_elo_mean_90d_A"])
    assert base.loc["m3", "opponent_elo_mean_90d_A"] == 1650
    # m1: 1 - .60; m2: 0 - .70, using pre-match Elo probabilities.
    assert base.loc["m3", "wins_above_expected_90d_A"] == pytest.approx(-0.30)
    # m1 residual is 0/10; m2 is +2/10 against a 60% expected service rate.
    assert base.loc["m3", "serve_residual_career_num_A"] == 2
    assert base.loc["m3", "serve_residual_career_den_A"] == 20


def test_h2h_global_and_surface_exclude_current_match():
    features, ratings, timeline, matches, stats = phase5_inputs()
    base = build_opponent_history_base(features, ratings, timeline, matches, stats).set_index(
        "match_id"
    )
    assert base.loc["m1", "h2h_matches_before"] == 0
    assert base.loc["m3", "h2h_matches_before"] == 1
    assert base.loc["m3", "h2h_wins_A_before"] == 1
    assert base.loc["m4", "h2h_matches_before"] == 2
    assert base.loc["m4", "h2h_surface_matches_before"] == 0


def test_future_result_and_stats_cannot_change_past_phase5_features():
    features, ratings, timeline, matches, stats = phase5_inputs()
    original = build_opponent_history_base(features, ratings, timeline, matches, stats)
    changed_features = features.copy()
    changed_features.loc[changed_features.match_id == "m4", "player_A_win"] = 1
    changed_stats = stats.copy()
    changed_stats.loc[changed_stats.match_id == "m4", "first_serve_won"] = 6
    changed = build_opponent_history_base(
        changed_features, ratings, timeline, matches, changed_stats
    )
    # Even the target row is a pre-match snapshot: changing its result or stats
    # cannot change any feature exposed for that target.
    pd.testing.assert_frame_equal(original, changed)


def test_surface_residual_and_h2h_shrinkage_are_historical():
    features, ratings, timeline, matches, stats = phase5_inputs()
    base = build_opponent_history_base(features, ratings, timeline, matches, stats)
    materialized = materialize_phase5_features(
        features,
        base,
        adjusted_point_prior=20,
        adjusted_surface_prior=100,
        h2h_prior=10,
    ).set_index("match_id")
    assert materialized.loc["m3", "serve_above_expected_career_A"] == 2 / 40
    assert materialized.loc["m3", "serve_above_expected_surface_A"] == 2 / 120
    assert materialized.loc["m3", "h2h_edge_A"] == (1 + 5) / 11 - 0.5
    assert materialized.loc["m4", "h2h_surface_edge_A"] == 0


def test_prior_tournament_workload_excludes_current_edition():
    features, ratings, timeline, matches, stats = phase5_inputs()
    base = build_opponent_history_base(features, ratings, timeline, matches, stats).set_index(
        "match_id"
    )
    assert base.loc["m3", "prior_tournament_matches_30d_A"] == 0
    # e1 is 60 days before e2, hence outside 30d rather than being mistaken for current load.
    assert base.loc["m4", "prior_tournament_matches_30d_A"] == 0
