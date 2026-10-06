from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class TennisDataProvider(ABC):
    """Provider boundary for schedules, brackets, scores, and point streams."""

    @abstractmethod
    def get_tournaments(self, *, limit: int = 24) -> list[dict[str, Any]]: ...

    @abstractmethod
    def get_tournament(self, tournament_id: str) -> dict[str, Any]: ...

    @abstractmethod
    def get_matches(self, tournament_id: str) -> list[dict[str, Any]]: ...

    @abstractmethod
    def get_bracket(self, tournament_id: str) -> list[dict[str, Any]]: ...

    @abstractmethod
    def get_match(self, match_id: str) -> dict[str, Any]: ...

    @abstractmethod
    def get_live_score(self, match_id: str, point_index: int) -> dict[str, Any]: ...

    @abstractmethod
    def get_point_by_point(self, match_id: str) -> list[dict[str, Any]]: ...
