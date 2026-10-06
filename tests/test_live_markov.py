import pytest

from tennis_analytics.live.markov import (
    MatchFormat,
    align_serve_probabilities_to_prematch,
    live_probabilities,
    regular_game_probability_a,
    tiebreak_probability_a,
    tiebreak_server,
)


def test_game_probability_is_exact_at_deuce() -> None:
    p = 0.65
    expected = p**2 / (p**2 + (1 - p) ** 2)
    assert regular_game_probability_a(
        3, 3, server="A", p_a_serve=p, p_b_serve=0.6
    ) == pytest.approx(expected)


def test_game_probability_moves_monotonically_after_a_point() -> None:
    before = regular_game_probability_a(2, 2, server="A", p_a_serve=0.64, p_b_serve=0.61)
    after_win = regular_game_probability_a(3, 2, server="A", p_a_serve=0.64, p_b_serve=0.61)
    after_loss = regular_game_probability_a(2, 3, server="A", p_a_serve=0.64, p_b_serve=0.61)
    assert after_loss < before < after_win


def test_tiebreak_service_rotation() -> None:
    assert [tiebreak_server("A", n) for n in range(7)] == ["A", "B", "B", "A", "A", "B", "B"]


def test_symmetric_tiebreak_is_fifty_fifty() -> None:
    probability = tiebreak_probability_a(0, 0, first_server="A", p_a_serve=0.5, p_b_serve=0.5)
    assert probability == pytest.approx(0.5)


@pytest.mark.parametrize("best_of", [3, 5])
def test_symmetric_match_is_fifty_fifty(best_of) -> None:
    result = live_probabilities(
        sets_a=0,
        sets_b=0,
        games_a=0,
        games_b=0,
        points_a=0,
        points_b=0,
        server="A",
        is_tiebreak=False,
        p_a_serve=0.62,
        p_b_serve=0.62,
        match_format=MatchFormat(best_of=best_of),
    )
    assert result.p_match_a == pytest.approx(0.5, abs=1e-12)


def test_match_point_is_near_one_for_strong_server() -> None:
    result = live_probabilities(
        sets_a=1,
        sets_b=0,
        games_a=5,
        games_b=0,
        points_a=3,
        points_b=0,
        server="A",
        is_tiebreak=False,
        p_a_serve=0.9,
        p_b_serve=0.6,
    )
    assert result.p_match_a > 0.999


def test_advantage_final_set_is_supported() -> None:
    result = live_probabilities(
        sets_a=1,
        sets_b=1,
        games_a=8,
        games_b=8,
        points_a=0,
        points_b=0,
        server="A",
        is_tiebreak=False,
        p_a_serve=0.65,
        p_b_serve=0.62,
        match_format=MatchFormat(final_set_tiebreak=False),
    )
    assert 0 < result.p_match_a < 1


def test_alignment_matches_prematch_probability() -> None:
    fmt = MatchFormat(best_of=3)
    pa, pb, delta = align_serve_probabilities_to_prematch(
        0.64, 0.62, 0.58, first_server="A", match_format=fmt
    )
    result = live_probabilities(
        sets_a=0,
        sets_b=0,
        games_a=0,
        games_b=0,
        points_a=0,
        points_b=0,
        server="A",
        is_tiebreak=False,
        p_a_serve=pa,
        p_b_serve=pb,
        match_format=fmt,
    )
    assert abs(delta) < 1
    assert result.p_match_a == pytest.approx(0.58, abs=1e-10)
