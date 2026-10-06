from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

from tennis_analytics.paths import PROJECT_ROOT


@dataclass(frozen=True)
class EloConfig:
    initial_rating: float = 1500.0
    k_factor: float = 32.0
    scale: float = 400.0
    surfaces: tuple[str, ...] = ("Hard", "Clay", "Grass", "Carpet")

    @classmethod
    def from_toml(cls, path: Path = PROJECT_ROOT / "config" / "elo.toml") -> EloConfig:
        with path.open("rb") as stream:
            values = tomllib.load(stream)["elo"]
        return cls(
            initial_rating=float(values["initial_rating"]),
            k_factor=float(values["k_factor"]),
            scale=float(values["scale"]),
            surfaces=tuple(values["surfaces"]),
        )


def expected_score(rating_a: float, rating_b: float, scale: float = 400.0) -> float:
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / scale))


def rating_delta(
    rating_a: float,
    rating_b: float,
    result_a: float,
    *,
    k_factor: float = 32.0,
    scale: float = 400.0,
) -> float:
    return k_factor * (result_a - expected_score(rating_a, rating_b, scale))
