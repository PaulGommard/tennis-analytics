from __future__ import annotations

import pandas as pd

from tennis_analytics.point_by_point.normalize import replay_mcp_match, replay_slam_match


class Resolver:
    def resolve(self, name: str, source: str) -> tuple[str, str]:
        return f"{source}:{name}", "test"


class Prematch:
    def for_mcp(self, *args: object) -> float:
        return 0.6

    def for_slam(self, *args: object) -> float:
        return 0.6


def test_mcp_adapter_reads_server_oriented_score_before_point() -> None:
    points = pd.DataFrame(
        [
            {"Pt": 1, "Svr": 1, "PtWinner": 1, "Set1": 0, "Set2": 0, "Gm1": 0, "Gm2": 0, "Pts": "0-0"},
            {"Pt": 2, "Svr": 1, "PtWinner": 2, "Set1": 0, "Set2": 0, "Gm1": 0, "Gm2": 0, "Pts": "15-0"},
        ]
    )
    metadata = pd.Series(
        {
            "match_id": "m1",
            "Player 1": "Alpha",
            "Player 2": "Beta",
            "Date": 20240101,
            "Surface": "Hard",
            "Best of": 3,
            "Tournament": "Test",
        }
    )
    rows, audit = replay_mcp_match(points, metadata, Resolver(), Prematch())
    assert audit["score_valid"]
    assert [(row["point_score_A"], row["point_score_B"]) for row in rows] == [
        ("0", "0"),
        ("15", "0"),
    ]
    assert rows[1]["A_wins_next_point"] == 0


def test_slam_adapter_converts_after_point_source_to_before_point_state() -> None:
    points = pd.DataFrame(
        [
            {
                "PointNumber": 1,
                "PointServer": 1,
                "PointWinner": 1,
                "SetNo": 1,
                "P1Ace": 1,
                "P2Ace": 0,
                "P1DoubleFault": 0,
                "P2DoubleFault": 0,
                "P1GamesWon": 0,
                "P2GamesWon": 0,
                "P1Score": "15",
                "P2Score": "0",
            }
        ]
    )
    metadata = pd.Series(
        {
            "match_id": "s1",
            "year": 2024,
            "slam": "usopen",
            "player1": "Alpha",
            "player2": "Beta",
            "match_num": 1001,
            "status": "",
        }
    )
    rows, audit = replay_slam_match(points, metadata, Resolver(), Prematch())
    assert audit["score_valid"]
    assert rows[0]["point_score_A"] == "0"
    assert rows[0]["point_score_B"] == "0"
    assert rows[0]["ace"] is True
