from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Player = Literal["A", "B"]


@dataclass(frozen=True)
class ScoreSnapshot:
    sets_a: int
    sets_b: int
    games_a: int
    games_b: int
    points_a: int
    points_b: int
    display_a: str
    display_b: str
    server: Player
    is_tiebreak: bool
    match_over: bool
    winner: Player | None


class TennisScore:
    """Deterministic standard tennis scorer for best-of-three or best-of-five."""

    def __init__(
        self,
        *,
        best_of: int = 3,
        first_server: Player = "A",
        tiebreak_at: int = 6,
        tiebreak_points: int = 7,
        final_set_tiebreak: bool = True,
        final_set_tiebreak_at: int | None = None,
        final_set_tiebreak_points: int | None = None,
    ) -> None:
        if best_of not in (3, 5):
            raise ValueError("best_of must be 3 or 5")
        if first_server not in ("A", "B"):
            raise ValueError("first_server must be A or B")
        if tiebreak_at < 1 or tiebreak_points < 2:
            raise ValueError("tiebreak_at and tiebreak_points must be positive")
        self.best_of = best_of
        self.sets_to_win = best_of // 2 + 1
        self.tiebreak_at = tiebreak_at
        self.tiebreak_points = tiebreak_points
        self.final_set_tiebreak = final_set_tiebreak
        self.final_set_tiebreak_at = (
            tiebreak_at if final_set_tiebreak_at is None else final_set_tiebreak_at
        )
        self.final_set_tiebreak_points = (
            tiebreak_points if final_set_tiebreak_points is None else final_set_tiebreak_points
        )
        if self.final_set_tiebreak_at < 1 or self.final_set_tiebreak_points < 2:
            raise ValueError("final-set tiebreak limits must be positive")
        self.sets = {"A": 0, "B": 0}
        self.games = {"A": 0, "B": 0}
        self.points = {"A": 0, "B": 0}
        self.server: Player = first_server
        self.tiebreak_first_server: Player | None = None
        self.match_over = False
        self.winner: Player | None = None

    @staticmethod
    def _other(player: Player) -> Player:
        return "B" if player == "A" else "A"

    @property
    def is_final_set(self) -> bool:
        return self.sets["A"] + self.sets["B"] == self.best_of - 1

    @property
    def is_tiebreak(self) -> bool:
        return self.tiebreak_first_server is not None

    def _display(self) -> tuple[str, str]:
        if self.is_tiebreak:
            return str(self.points["A"]), str(self.points["B"])
        a, b = self.points["A"], self.points["B"]
        if a >= 3 and b >= 3:
            if a == b:
                return "40", "40"
            return ("AD", "40") if a > b else ("40", "AD")
        labels = ("0", "15", "30", "40")
        return labels[min(a, 3)], labels[min(b, 3)]

    def snapshot(self) -> ScoreSnapshot:
        display_a, display_b = self._display()
        return ScoreSnapshot(
            sets_a=self.sets["A"],
            sets_b=self.sets["B"],
            games_a=self.games["A"],
            games_b=self.games["B"],
            points_a=self.points["A"],
            points_b=self.points["B"],
            display_a=display_a,
            display_b=display_b,
            server=self.server,
            is_tiebreak=self.is_tiebreak,
            match_over=self.match_over,
            winner=self.winner,
        )

    def point_won_by(self, winner: Player) -> ScoreSnapshot:
        if winner not in ("A", "B"):
            raise ValueError("point winner must be A or B")
        if self.match_over:
            raise RuntimeError("cannot add a point after match completion")
        self.points[winner] += 1
        if self.is_tiebreak:
            self._advance_tiebreak(winner)
        else:
            self._advance_regular_game(winner)
        return self.snapshot()

    def _advance_regular_game(self, winner: Player) -> None:
        loser = self._other(winner)
        if self.points[winner] >= 4 and self.points[winner] - self.points[loser] >= 2:
            self.games[winner] += 1
            self.points = {"A": 0, "B": 0}
            self.server = self._other(self.server)
            self._after_game(winner)

    def _after_game(self, winner: Player) -> None:
        loser = self._other(winner)
        games_winner, games_loser = self.games[winner], self.games[loser]
        if games_winner >= 6 and games_winner - games_loser >= 2:
            self._complete_set(winner)
            return
        threshold = self.final_set_tiebreak_at if self.is_final_set else self.tiebreak_at
        at_tiebreak = self.games["A"] == self.games["B"] == threshold
        use_tiebreak = not (self.is_final_set and not self.final_set_tiebreak)
        if at_tiebreak and use_tiebreak:
            self.tiebreak_first_server = self.server

    def _advance_tiebreak(self, winner: Player) -> None:
        loser = self._other(winner)
        target = self.final_set_tiebreak_points if self.is_final_set else self.tiebreak_points
        if self.points[winner] >= target and self.points[winner] - self.points[loser] >= 2:
            first_server = self.tiebreak_first_server
            self.games[winner] += 1
            self._complete_set(winner)
            if not self.match_over and first_server is not None:
                self.server = self._other(first_server)
            return
        total_points = self.points["A"] + self.points["B"]
        first_server = self.tiebreak_first_server
        if first_server is None:
            raise RuntimeError("missing tiebreak first server")
        block = (total_points + 1) // 2
        self.server = first_server if block % 2 == 0 else self._other(first_server)

    def _complete_set(self, winner: Player) -> None:
        self.sets[winner] += 1
        self.games = {"A": 0, "B": 0}
        self.points = {"A": 0, "B": 0}
        self.tiebreak_first_server = None
        if self.sets[winner] == self.sets_to_win:
            self.match_over = True
            self.winner = winner

    def play(self, winners: str) -> list[ScoreSnapshot]:
        """Apply an iterable-like string of A/B point winners and return every state."""
        return [self.point_won_by(winner) for winner in winners]
