import pandas as pd
import pytest

from tennis_analytics.features.temporal import build_temporal_features
from tennis_analytics.ratings.elo import EloConfig, expected_score


def match(
    match_id: str,
    date: str,
    player_a: str,
    player_b: str,
    a_wins: int,
    *,
    surface: str = "Hard",
) -> dict[str, object]:
    return {
        "match_id": match_id,
        "date": pd.Timestamp(date),
        "date_precision": "tournament_start",
        "tournament": "Test Open",
        "tournament_id": f"t-{date}",
        "tournament_edition_id": f"te-{date}",
        "level": "A",
        "surface": surface,
        "round": "R32",
        "player_A_id": player_a,
        "player_B_id": player_b,
        "player_A_name": player_a,
        "player_B_name": player_b,
        "player_A_win": a_wins,
        "rank_A": 10,
        "rank_B": 20,
        "temporal_batch_size": 1,
        "is_elo_eligible": True,
    }


def three_match_timeline() -> pd.DataFrame:
    return pd.DataFrame(
        [
            match("m1", "2020-01-01", "A", "B", 1),
            match("m2", "2020-01-02", "A", "C", 1),
            match("m3", "2020-01-03", "B", "C", 1),
        ]
    )


def test_classic_elo_and_pre_match_ordering():
    features, _ = build_temporal_features(three_match_timeline(), EloConfig())
    first = features.set_index("match_id").loc["m1"]
    second = features.set_index("match_id").loc["m2"]
    assert first["elo_A"] == 1500
    assert first["elo_B"] == 1500
    assert first["elo_expected_A"] == 0.5
    assert second["elo_A"] == 1516
    assert second["elo_B"] == 1500
    assert second["elo_expected_A"] == pytest.approx(expected_score(1516, 1500))
    assert second["career_matches_A"] == 1
    assert second["matches_30d_A"] == 1
    assert second["wins_30d_A"] == 1
    assert second["days_since_last_match_A"] == 1


def test_same_date_is_a_frozen_batch():
    timeline = pd.DataFrame(
        [
            match("m1", "2020-01-01", "A", "B", 1),
            match("m2", "2020-01-01", "A", "C", 1),
            match("m3", "2020-01-08", "A", "D", 1),
        ]
    )
    features, _ = build_temporal_features(timeline, EloConfig())
    rows = features.set_index("match_id")
    assert rows.loc["m1", "elo_A"] == 1500
    assert rows.loc["m2", "elo_A"] == 1500
    assert rows.loc["m1", "career_matches_A"] == 0
    assert rows.loc["m2", "career_matches_A"] == 0
    assert rows.loc["m3", "elo_A"] == 1532
    assert rows.loc["m3", "career_matches_A"] == 2


def test_current_target_cannot_change_current_features():
    original = pd.DataFrame([match("m1", "2020-01-01", "A", "B", 1)])
    flipped = original.copy()
    flipped.loc[0, "player_A_win"] = 0
    original_features, _ = build_temporal_features(original, EloConfig())
    flipped_features, _ = build_temporal_features(flipped, EloConfig())
    columns = [column for column in original_features if column != "player_A_win"]
    pd.testing.assert_frame_equal(original_features[columns], flipped_features[columns])


def test_future_result_cannot_change_past_features():
    original = three_match_timeline()
    changed_future = original.copy()
    changed_future.loc[changed_future["match_id"] == "m3", "player_A_win"] = 0
    original_features, _ = build_temporal_features(original, EloConfig())
    changed_features, _ = build_temporal_features(changed_future, EloConfig())
    columns = [column for column in original_features if column != "player_A_win"]
    pd.testing.assert_frame_equal(
        original_features.loc[:1, columns], changed_features.loc[:1, columns]
    )


def test_surface_history_updates_only_matching_surface():
    timeline = pd.DataFrame(
        [
            match("m1", "2020-01-01", "A", "B", 1, surface="Clay"),
            match("m2", "2020-01-08", "A", "C", 1, surface="Hard"),
            match("m3", "2020-01-15", "A", "D", 1, surface="Clay"),
        ]
    )
    features, _ = build_temporal_features(timeline, EloConfig())
    rows = features.set_index("match_id")
    assert rows.loc["m2", "surface_elo_A"] == 1500
    assert rows.loc["m2", "career_matches_surface_A"] == 0
    assert rows.loc["m3", "surface_elo_A"] == 1516
    assert rows.loc["m3", "career_matches_surface_A"] == 1


def test_input_is_processed_chronologically():
    timeline = three_match_timeline().iloc[::-1].reset_index(drop=True)
    features, _ = build_temporal_features(timeline, EloConfig())
    assert features["match_id"].tolist() == ["m1", "m2", "m3"]
