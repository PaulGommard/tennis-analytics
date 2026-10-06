from datetime import date

import pytest

from tennis_analytics.preprocessing.parsing import (
    canonical_player_id,
    normalize_name,
    normalize_surface,
    parse_tourney_date,
    valid_score,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("20240115", date(2024, 1, 15)), (20240229, date(2024, 2, 29)), ("bad", None)],
)
def test_parse_tourney_date(raw, expected):
    assert parse_tourney_date(raw) == expected


def test_surface_and_name_normalization():
    assert normalize_surface(" hard ") == "Hard"
    assert normalize_surface("CLAY") == "Clay"
    assert normalize_name("  Alexander   Zverev ") == "Alexander Zverev"


def test_player_ids_prefer_source_and_synthetic_are_stable():
    assert canonical_player_id(123.0, "A. Player")[0] == "atp:123"
    first = canonical_player_id(None, "Émile  Test", None, "FRA")
    second = canonical_player_id(None, "Emile Test", None, "FRA")
    assert first == second
    assert first[1] == "synthetic"


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        ("6-4 7-6(5)", True),
        ("6-2 2-1 RET", True),
        ("W/O", True),
        ("6-4 Played and abandoned", True),
        ("6-5 Def.", True),
        (None, True),
        ("six-four", False),
    ],
)
def test_score_syntax(score, expected):
    assert valid_score(score) is expected
