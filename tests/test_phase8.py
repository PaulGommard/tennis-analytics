from __future__ import annotations

import numpy as np
import pandas as pd

from tennis_analytics.point_by_point.in_match import (
    _build_clutch_snapshots,
    add_sequential_features,
)


def sample_points(first_winner: str = "A") -> pd.DataFrame:
    winners = [first_winner, "A", "B", "A", "A", "B"]
    servers = ["A", "A", "A", "A", "B", "B"]
    return pd.DataFrame(
        {
            "match_id": ["m1"] * 6,
            "server": servers,
            "point_winner": winners,
            "server_won_point": [int(winner == server) for winner, server in zip(winners, servers)],
            "ace": [False] * 6,
            "double_fault": [False] * 6,
            "is_break_point": [0, 0, 1, 0, 0, 0],
            "game_ended_after": [0, 0, 0, 1, 0, 1],
            "set_ended_after": [0, 0, 0, 0, 1, 0],
            "games_A": [0, 0, 0, 0, 1, 0],
            "games_B": [0, 0, 0, 0, 0, 0],
            "is_tiebreak": [False] * 6,
            "server_serve_rate": [0.64] * 6,
            "receiver_return_rate": [0.36] * 6,
        }
    )


def test_current_point_never_enters_own_features() -> None:
    won = add_sequential_features(sample_points("A"))
    lost = add_sequential_features(sample_points("B"))
    columns = [column for column in won if column not in sample_points().columns]
    pd.testing.assert_series_equal(won.loc[0, columns], lost.loc[0, columns])
    assert won.loc[1, "A_points_won_match"] == 1
    assert lost.loc[1, "A_points_won_match"] == 0


def test_recent_windows_use_only_previous_points() -> None:
    result = add_sequential_features(sample_points())
    assert np.isnan(result.loc[0, "A_points_won_last_10"])
    assert result.loc[3, "A_points_won_last_10"] == 2 / 3
    assert result.loc[3, "server_service_points_won_last_3"] == 2 / 3


def test_match_updates_and_previous_set_are_delayed() -> None:
    result = add_sequential_features(sample_points())
    assert result.loc[2, "server_break_points_faced_so_far"] == 0
    assert result.loc[3, "server_break_points_faced_so_far"] == 1
    assert np.isnan(result.loc[4, "A_previous_set_won"])
    assert result.loc[5, "B_previous_set_won"] == 0
    assert result.loc[5, "B_previous_set_margin"] < 0


def test_shrinkage_starts_at_prior_and_updates_after_point() -> None:
    result = add_sequential_features(sample_points())
    assert result.loc[0, "dynamic_serve_w20"] == 0.64
    expected = (20 * 0.64 + 1) / 21
    assert result.loc[1, "dynamic_serve_w20"] == expected


def test_clutch_history_does_not_cross_same_date_batch() -> None:
    records = []
    for match_id, date, contribution in (
        ("m1", "2024-01-01", 1.0),
        ("m2", "2024-01-01", 1.0),
        ("m3", "2024-01-02", 0.0),
    ):
        row: dict[str, object] = {
            "match_id": match_id,
            "date": pd.Timestamp(date),
            "player_A_id": "player",
            "player_B_id": f"opponent:{match_id}",
        }
        for side in ("A", "B"):
            for category in ("break", "tiebreak", "deciding"):
                row[f"{side}_{category}_sum"] = contribution if side == "A" else 0.0
                row[f"{side}_{category}_count"] = 1
        records.append(row)
    snapshots = _build_clutch_snapshots(records).set_index("match_id")
    assert snapshots.loc["m1", "A_break_clutch_samples"] == 0
    assert snapshots.loc["m2", "A_break_clutch_samples"] == 0
    assert snapshots.loc["m3", "A_break_clutch_samples"] == 2
