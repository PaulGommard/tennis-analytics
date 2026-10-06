import pytest

from tennis_analytics.scoring.engine import TennisScore


def game(score: TennisScore, winner: str) -> None:
    for _ in range(4):
        score.point_won_by(winner)


def set_won(score: TennisScore, winner: str) -> None:
    for _ in range(6):
        game(score, winner)


def test_love_game_and_server_switch() -> None:
    score = TennisScore(first_server="A")
    states = score.play("AAAA")
    assert [state.display_a for state in states[:3]] == ["15", "30", "40"]
    assert states[-1].games_a == 1
    assert states[-1].server == "B"


def test_deuce_advantage_and_revert() -> None:
    score = TennisScore()
    score.play("AAABBB")
    assert (score.snapshot().display_a, score.snapshot().display_b) == ("40", "40")
    score.point_won_by("A")
    assert score.snapshot().display_a == "AD"
    score.point_won_by("B")
    assert (score.snapshot().display_a, score.snapshot().display_b) == ("40", "40")
    score.play("BB")
    assert score.snapshot().games_b == 1


def test_break_is_just_a_game_won_by_receiver() -> None:
    score = TennisScore(first_server="A")
    game(score, "B")
    assert score.snapshot().games_b == 1
    assert score.snapshot().server == "B"


def test_straight_sets_best_of_three() -> None:
    score = TennisScore(best_of=3)
    set_won(score, "A")
    assert score.snapshot().sets_a == 1
    set_won(score, "A")
    assert score.snapshot().match_over
    assert score.snapshot().winner == "A"


def test_best_of_five_requires_three_sets() -> None:
    score = TennisScore(best_of=5)
    set_won(score, "B")
    set_won(score, "B")
    assert not score.snapshot().match_over
    set_won(score, "B")
    assert score.snapshot().winner == "B"


def test_tiebreak_server_rotation_and_set() -> None:
    score = TennisScore(first_server="A")
    for index in range(12):
        game(score, "A" if index % 2 == 0 else "B")
    assert score.snapshot().is_tiebreak
    assert score.snapshot().server == "A"
    score.point_won_by("A")
    assert score.snapshot().server == "B"
    score.point_won_by("A")
    assert score.snapshot().server == "B"
    score.point_won_by("A")
    assert score.snapshot().server == "A"
    score.play("AAAA")
    state = score.snapshot()
    assert state.sets_a == 1
    assert state.games_a == state.games_b == 0
    assert state.server == "B"


def test_tiebreak_requires_two_point_margin() -> None:
    score = TennisScore(tiebreak_at=1)
    game(score, "A")
    game(score, "B")
    score.play("ABABABABABAB")
    assert score.snapshot().is_tiebreak
    score.play("AA")
    assert score.snapshot().sets_a == 1


def test_final_set_can_use_ten_point_tiebreak() -> None:
    score = TennisScore(best_of=3, final_set_tiebreak_points=10)
    set_won(score, "A")
    set_won(score, "B")
    for index in range(12):
        game(score, "A" if index % 2 == 0 else "B")
    assert score.snapshot().is_tiebreak
    score.play("AAAAAAAAA")
    assert not score.snapshot().match_over
    score.point_won_by("A")
    assert score.snapshot().winner == "A"


def test_wimbledon_style_final_tiebreak_at_twelve_all() -> None:
    score = TennisScore(best_of=3, final_set_tiebreak_at=12)
    set_won(score, "A")
    set_won(score, "B")
    for index in range(24):
        game(score, "A" if index % 2 == 0 else "B")
    assert score.snapshot().is_tiebreak


def test_advantage_final_set_without_tiebreak() -> None:
    score = TennisScore(best_of=3, final_set_tiebreak=False)
    set_won(score, "A")
    set_won(score, "B")
    for index in range(12):
        game(score, "A" if index % 2 == 0 else "B")
    assert not score.snapshot().is_tiebreak
    game(score, "A")
    game(score, "A")
    assert score.snapshot().winner == "A"


def test_reject_invalid_configuration_and_post_match_points() -> None:
    with pytest.raises(ValueError):
        TennisScore(best_of=1)
    score = TennisScore()
    set_won(score, "A")
    set_won(score, "A")
    with pytest.raises(RuntimeError):
        score.point_won_by("B")
