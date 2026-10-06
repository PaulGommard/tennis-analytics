from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
from typing import Literal

import numpy as np

Player = Literal["A", "B"]


@dataclass(frozen=True)
class MatchFormat:
    best_of: int = 3
    tiebreak_at: int = 6
    tiebreak_points: int = 7
    final_set_tiebreak: bool = True
    final_set_tiebreak_at: int = 6
    final_set_tiebreak_points: int = 7

    def __post_init__(self) -> None:
        if self.best_of not in (3, 5):
            raise ValueError("best_of must be 3 or 5")
        if min(self.tiebreak_at, self.final_set_tiebreak_at) < 1:
            raise ValueError("tie-break game thresholds must be positive")
        if min(self.tiebreak_points, self.final_set_tiebreak_points) < 2:
            raise ValueError("tie-break point targets must be at least two")

    @property
    def sets_to_win(self) -> int:
        return self.best_of // 2 + 1


@dataclass(frozen=True)
class LiveProbabilities:
    p_next_point_a: float
    p_current_game_a: float
    p_current_set_a: float
    p_match_a: float


def _other(player: Player) -> Player:
    return "B" if player == "A" else "A"


def _point_probability_a(server: Player, p_a_serve: float, p_b_serve: float) -> float:
    return p_a_serve if server == "A" else 1.0 - p_b_serve


def regular_game_probability_a(
    points_a: int,
    points_b: int,
    *,
    server: Player,
    p_a_serve: float,
    p_b_serve: float,
) -> float:
    """Exact probability that A wins a standard game from its current point score."""
    p = _point_probability_a(server, p_a_serve, p_b_serve)
    if not 0 < p < 1:
        raise ValueError("point probabilities must be strictly between zero and one")

    @cache
    def solve(a: int, b: int) -> float:
        if a >= 4 and a - b >= 2:
            return 1.0
        if b >= 4 and b - a >= 2:
            return 0.0
        if a >= 3 and b >= 3:
            deuce = p * p / (p * p + (1 - p) * (1 - p))
            if a == b:
                return deuce
            if a == b + 1:
                return p + (1 - p) * deuce
            if b == a + 1:
                return p * deuce
        return p * solve(a + 1, b) + (1 - p) * solve(a, b + 1)

    return float(solve(int(points_a), int(points_b)))


def tiebreak_server(first_server: Player, total_points: int) -> Player:
    if total_points == 0:
        return first_server
    block = (total_points - 1) // 2
    return _other(first_server) if block % 2 == 0 else first_server


def infer_tiebreak_first_server(current_server: Player, total_points: int) -> Player:
    return current_server if tiebreak_server("A", total_points) == "A" else _other(current_server)


def tiebreak_probability_a(
    points_a: int,
    points_b: int,
    *,
    first_server: Player,
    p_a_serve: float,
    p_b_serve: float,
    target: int = 7,
) -> float:
    """Exact absorbing-Markov solution for a win-by-two tie-break."""
    if target < 2:
        raise ValueError("target must be at least two")
    if points_a >= target and points_a - points_b >= 2:
        return 1.0
    if points_b >= target and points_b - points_a >= 2:
        return 0.0

    states = [(difference, phase) for difference in (-1, 0, 1) for phase in range(4)]
    positions = {state: index for index, state in enumerate(states)}
    matrix = np.eye(len(states))
    rhs = np.zeros(len(states))
    for state, row in positions.items():
        difference, phase = state
        server = tiebreak_server(first_server, phase)
        p = _point_probability_a(server, p_a_serve, p_b_serve)
        for probability, next_difference, terminal in (
            (p, difference + 1, 1.0),
            (1 - p, difference - 1, 0.0),
        ):
            if abs(next_difference) >= 2:
                rhs[row] += probability * terminal
            else:
                matrix[row, positions[(next_difference, (phase + 1) % 4)]] -= probability
    extended = np.linalg.solve(matrix, rhs)

    @cache
    def solve(a: int, b: int) -> float:
        if a >= target and a - b >= 2:
            return 1.0
        if b >= target and b - a >= 2:
            return 0.0
        if a >= target - 1 and b >= target - 1:
            return float(extended[positions[(a - b, (a + b) % 4)]])
        server = tiebreak_server(first_server, a + b)
        p = _point_probability_a(server, p_a_serve, p_b_serve)
        return p * solve(a + 1, b) + (1 - p) * solve(a, b + 1)

    return float(solve(int(points_a), int(points_b)))


def _merge_outcomes(
    target: dict[tuple[Player, Player], float],
    source: dict[tuple[Player, Player], float],
    weight: float,
) -> None:
    for key, probability in source.items():
        target[key] = target.get(key, 0.0) + weight * probability


@cache
def _boundary_outcomes(
    games_a: int,
    games_b: int,
    server: Player,
    p_a_serve: float,
    p_b_serve: float,
    match_format: MatchFormat,
    is_final_set: bool,
) -> tuple[tuple[Player, Player, float], ...]:
    threshold = match_format.final_set_tiebreak_at if is_final_set else match_format.tiebreak_at
    use_tiebreak = not (is_final_set and not match_format.final_set_tiebreak)
    target = (
        match_format.final_set_tiebreak_points if is_final_set else match_format.tiebreak_points
    )
    if games_a >= 6 and games_a - games_b >= 2:
        return (("A", server, 1.0),)
    if games_b >= 6 and games_b - games_a >= 2:
        return (("B", server, 1.0),)
    if use_tiebreak and games_a == games_b == threshold:
        probability = tiebreak_probability_a(
            0,
            0,
            first_server=server,
            p_a_serve=p_a_serve,
            p_b_serve=p_b_serve,
            target=target,
        )
        following = _other(server)
        return (("A", following, probability), ("B", following, 1 - probability))
    if not use_tiebreak and games_a >= 6 and games_b >= 6:
        shift = min(games_a, games_b) - 6
        if shift:
            return _boundary_outcomes(
                games_a - shift,
                games_b - shift,
                server,
                p_a_serve,
                p_b_serve,
                match_format,
                is_final_set,
            )
        if games_a == games_b:
            q1 = regular_game_probability_a(
                0, 0, server=server, p_a_serve=p_a_serve, p_b_serve=p_b_serve
            )
            q2 = regular_game_probability_a(
                0, 0, server=_other(server), p_a_serve=p_a_serve, p_b_serve=p_b_serve
            )
            repeat = q1 * (1 - q2) + (1 - q1) * q2
            probability = q1 * q2 / (1 - repeat)
            return (("A", server, probability), ("B", server, 1 - probability))
    q = regular_game_probability_a(0, 0, server=server, p_a_serve=p_a_serve, p_b_serve=p_b_serve)
    after_server = _other(server)
    merged: dict[tuple[Player, Player], float] = {}
    for weight, next_a, next_b in ((q, games_a + 1, games_b), (1 - q, games_a, games_b + 1)):
        outcomes = _boundary_outcomes(
            next_a,
            next_b,
            after_server,
            p_a_serve,
            p_b_serve,
            match_format,
            is_final_set,
        )
        _merge_outcomes(
            merged,
            {(winner, srv): probability for winner, srv, probability in outcomes},
            weight,
        )
    return tuple((winner, srv, probability) for (winner, srv), probability in merged.items())


def set_outcomes(
    *,
    games_a: int,
    games_b: int,
    points_a: int,
    points_b: int,
    server: Player,
    is_tiebreak: bool,
    p_a_serve: float,
    p_b_serve: float,
    match_format: MatchFormat,
    is_final_set: bool,
) -> dict[tuple[Player, Player], float]:
    """Distribution of (set winner, first server next set)."""
    target = (
        match_format.final_set_tiebreak_points if is_final_set else match_format.tiebreak_points
    )
    if is_tiebreak:
        first_server = infer_tiebreak_first_server(server, points_a + points_b)
        probability = tiebreak_probability_a(
            points_a,
            points_b,
            first_server=first_server,
            p_a_serve=p_a_serve,
            p_b_serve=p_b_serve,
            target=target,
        )
        following = _other(first_server)
        return {("A", following): probability, ("B", following): 1 - probability}

    game_probability = regular_game_probability_a(
        points_a,
        points_b,
        server=server,
        p_a_serve=p_a_serve,
        p_b_serve=p_b_serve,
    )
    after_server = _other(server)
    result: dict[tuple[Player, Player], float] = {}
    _merge_outcomes(
        result,
        {
            (winner, srv): prob
            for winner, srv, prob in _boundary_outcomes(
                games_a + 1,
                games_b,
                after_server,
                p_a_serve,
                p_b_serve,
                match_format,
                is_final_set,
            )
        },
        game_probability,
    )
    _merge_outcomes(
        result,
        {
            (winner, srv): prob
            for winner, srv, prob in _boundary_outcomes(
                games_a,
                games_b + 1,
                after_server,
                p_a_serve,
                p_b_serve,
                match_format,
                is_final_set,
            )
        },
        1 - game_probability,
    )
    return result


@cache
def _fresh_match_probability(
    sets_a: int,
    sets_b: int,
    first_server: Player,
    p_a_serve: float,
    p_b_serve: float,
    match_format: MatchFormat,
) -> float:
    if sets_a >= match_format.sets_to_win:
        return 1.0
    if sets_b >= match_format.sets_to_win:
        return 0.0
    outcomes = _boundary_outcomes(
        0,
        0,
        first_server,
        p_a_serve,
        p_b_serve,
        match_format,
        sets_a + sets_b == match_format.best_of - 1,
    )
    return sum(
        probability
        * _fresh_match_probability(
            sets_a + int(winner == "A"),
            sets_b + int(winner == "B"),
            next_server,
            p_a_serve,
            p_b_serve,
            match_format,
        )
        for winner, next_server, probability in outcomes
    )


def live_probabilities(
    *,
    sets_a: int,
    sets_b: int,
    games_a: int,
    games_b: int,
    points_a: int,
    points_b: int,
    server: Player,
    is_tiebreak: bool,
    p_a_serve: float,
    p_b_serve: float,
    match_format: MatchFormat | None = None,
) -> LiveProbabilities:
    if match_format is None:
        match_format = MatchFormat()
    if sets_a >= match_format.sets_to_win or sets_b >= match_format.sets_to_win:
        terminal = float(sets_a >= match_format.sets_to_win)
        return LiveProbabilities(terminal, terminal, terminal, terminal)
    next_point = _point_probability_a(server, p_a_serve, p_b_serve)
    if is_tiebreak:
        first = infer_tiebreak_first_server(server, points_a + points_b)
        target = (
            match_format.final_set_tiebreak_points
            if sets_a + sets_b == match_format.best_of - 1
            else match_format.tiebreak_points
        )
        current_game = tiebreak_probability_a(
            points_a,
            points_b,
            first_server=first,
            p_a_serve=p_a_serve,
            p_b_serve=p_b_serve,
            target=target,
        )
    else:
        current_game = regular_game_probability_a(
            points_a,
            points_b,
            server=server,
            p_a_serve=p_a_serve,
            p_b_serve=p_b_serve,
        )
    current_outcomes = set_outcomes(
        games_a=games_a,
        games_b=games_b,
        points_a=points_a,
        points_b=points_b,
        server=server,
        is_tiebreak=is_tiebreak,
        p_a_serve=p_a_serve,
        p_b_serve=p_b_serve,
        match_format=match_format,
        is_final_set=sets_a + sets_b == match_format.best_of - 1,
    )
    current_set = sum(
        probability for (winner, _), probability in current_outcomes.items() if winner == "A"
    )

    match = 0.0
    for (winner, next_server), probability in current_outcomes.items():
        match += probability * _fresh_match_probability(
            sets_a + int(winner == "A"),
            sets_b + int(winner == "B"),
            next_server,
            p_a_serve,
            p_b_serve,
            match_format,
        )
    return LiveProbabilities(next_point, current_game, current_set, match)


def align_serve_probabilities_to_prematch(
    p_a_serve: float,
    p_b_serve: float,
    prematch_a: float,
    *,
    first_server: Player,
    match_format: MatchFormat,
) -> tuple[float, float, float]:
    """Apply the smallest symmetric log-odds shift matching the pre-match prior."""
    target = min(max(float(prematch_a), 1e-4), 1 - 1e-4)

    def sigmoid(value: float) -> float:
        return 1.0 / (1.0 + math.exp(-value))

    def logit(value: float) -> float:
        clipped = min(max(value, 1e-6), 1 - 1e-6)
        return math.log(clipped / (1 - clipped))

    def shifted(delta: float) -> tuple[float, float]:
        return sigmoid(logit(p_a_serve) + delta), sigmoid(logit(p_b_serve) - delta)

    low, high = -4.0, 4.0
    # 40 bisections keep the start-of-match invariant stable to test precision
    # while remaining practical for bulk historical replay.
    for _ in range(40):
        middle = (low + high) / 2
        pa, pb = shifted(middle)
        probability = live_probabilities(
            sets_a=0,
            sets_b=0,
            games_a=0,
            games_b=0,
            points_a=0,
            points_b=0,
            server=first_server,
            is_tiebreak=False,
            p_a_serve=pa,
            p_b_serve=pb,
            match_format=match_format,
        ).p_match_a
        if probability < target:
            low = middle
        else:
            high = middle
    delta = (low + high) / 2
    pa, pb = shifted(delta)
    return pa, pb, delta


def clear_probability_caches() -> None:
    """Release probability-specific DP states between independent bulk matches."""
    _boundary_outcomes.cache_clear()
    _fresh_match_probability.cache_clear()
