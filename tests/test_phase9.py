from __future__ import annotations

from tennis_analytics.shots.build import _checkpoint_rows, _score_flags
from tennis_analytics.shots.encoding import direction_relation, parse_point


def test_parses_documented_winner_with_directions() -> None:
    result = parse_point("4ffbbf*")
    assert result.valid
    assert result.derived_winner == "receiver"
    assert len(result.events) == 6
    assert result.events[0].serve_direction == "wide"
    assert result.events[-1].outcome == "winner"


def test_parses_second_serve_return_depth_and_error() -> None:
    result = parse_point("5d", "4s39b3b1w@")
    assert result.valid
    assert result.active_serve_number == 2
    assert result.events[0].outcome == "fault"
    assert result.events[2].depth == "deep"
    assert result.events[-1].outcome == "unforced_error"
    assert result.derived_winner == "server"


def test_parses_position_and_net_cord_modifiers() -> None:
    result = parse_point("4+s28v1f-3*")
    assert result.valid
    assert result.events[0].is_approach
    assert result.events[-1].at_net
    cord = parse_point("5f;1*")
    assert cord.valid and cord.events[-1].net_cord


def test_parses_double_fault_and_unreturned_serve() -> None:
    double = parse_point("4n", "5d")
    assert double.valid
    assert double.terminal_outcome == "double_fault"
    assert double.derived_winner == "receiver"
    unreturned = parse_point("6#")
    assert unreturned.valid
    assert unreturned.derived_winner == "server"


def test_special_or_unknown_syntax_is_never_silently_interpreted() -> None:
    special = parse_point("S")
    assert not special.valid
    assert special.special_code == "unobserved_server_won"
    invalid = parse_point("6f3?")
    assert not invalid.valid
    assert invalid.reason == "unknown_shot_attribute:?"


def test_direction_relation_respects_handedness() -> None:
    right = parse_point("6f1*").events[-1]
    left = parse_point("6f3*").events[-1]
    assert direction_relation(right, "R") == "crosscourt"
    assert direction_relation(left, "L") == "crosscourt"


def test_prefix_events_contain_no_future_or_terminal_information() -> None:
    result = parse_point("6f19b3f1w@")
    observed = [event for event in result.events if (event.active_shot_index or 0) <= 3]
    assert observed
    assert all(not event.is_terminal for event in observed)
    assert all(event.outcome == "in_play" for event in observed)


def test_modern_depth_and_historical_modifier_order_are_supported() -> None:
    result = parse_point("6f19b38f1;*")
    assert result.valid
    assert result.events[2].depth == "medium"
    assert result.events[-1].net_cord
    shank = parse_point("4b!1w#")
    assert shank.valid
    assert shank.events[-1].is_shank
    assert shank.events[-1].error_type == "shank+wide"


def test_ad_score_is_not_misclassified_as_tiebreak() -> None:
    assert _score_flags("AD-40", 1) == (False, False)
    assert _score_flags("40-AD", 1) == (False, True)
    assert _score_flags("5-4", 1) == (True, False)


def test_checkpoint_rows_exclude_terminal_prefixes_and_forbidden_fields() -> None:
    parsed = parse_point("6f19b3f1w@")
    active = [event for event in parsed.events if event.active_shot_index is not None]
    base = {
        "match_id": "m",
        "point_id": "m:1",
        "date": None,
        "year": 2024,
        "surface": "Hard",
        "point_number": 1,
        "server_id": "a",
        "receiver_id": "b",
        "server_won_point": 0,
        "is_tiebreak": False,
        "is_break_point": False,
    }
    rows = list(_checkpoint_rows(base, active, {"server": "R", "receiver": "R"}))
    assert {row["checkpoint"] for row in rows} == {0, 1, 2, 3}
    forbidden = {"final_rally_length", "terminal_outcome", "point_winner_number"}
    assert all(not forbidden.intersection(row) for row in rows)
