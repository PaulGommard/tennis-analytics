from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class OnDemandTennisDataProvider(ABC):
    """Small provider contract for prospective, user-triggered predictions."""

    provider_name: str

    @abstractmethod
    def get_upcoming_matches(self, *, days: int = 3) -> list[dict[str, Any]]: ...

    @abstractmethod
    def get_player_recent_matches(
        self, external_player_id: str, *, year: int
    ) -> dict[str, Any]: ...

    @abstractmethod
    def get_rankings(self) -> dict[str, Any]: ...

    @property
    @abstractmethod
    def request_count(self) -> int: ...
