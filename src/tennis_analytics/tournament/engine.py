from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd

from tennis_analytics.tournament.modeling import _cap
from tennis_analytics.tournament.replay import MODEL_DIR


@dataclass(frozen=True)
class TournamentMatchPrediction:
    p_base: float
    p_adjusted: float
    delta_probability: float
    adjustments: dict[str, dict[str, float]]
    model_version: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_artifact(model_path: str | Path | None) -> dict[str, Any]:
    path = Path(model_path) if model_path is not None else MODEL_DIR / "model.joblib"
    return joblib.load(path)


def predict_tournament_match(
    p_base: float,
    context: Mapping[str, float] | None = None,
    *,
    model_path: str | Path | None = None,
) -> TournamentMatchPrediction:
    """Apply the adapter to a PRE-MATCH V1 probability and frozen context.

    Missing context means no available evidence and contributes zero. Returned
    contributions are model attributions, not causal effects.
    """
    if not 0 < p_base < 1:
        raise ValueError("p_base must be strictly between zero and one")
    artifact = _load_artifact(model_path)
    values: dict[str, float] = {
        feature: float((context or {}).get(feature, 0.0))
        for feature in artifact["feature_columns"]
    }
    values["p_base"] = float(p_base)
    values["logit_p_base"] = math.log(p_base / (1 - p_base))
    frame = pd.DataFrame([values])
    model = artifact["model"]
    raw_probability = float(model.predict_proba(frame)[0, 1])
    capped_probability = float(
        _cap(
            np.array([raw_probability]),
            np.array([p_base]),
            artifact["selected_cap"],
        )[0]
    )
    calibrator = artifact.get("calibrator")
    adjusted = (
        float(calibrator.transform(np.array([capped_probability]))[0])
        if calibrator is not None
        else capped_probability
    )
    if abs(adjusted - p_base) < 1e-15:
        adjusted = float(p_base)

    feature_contributions = model.contributions(frame).iloc[0]
    grouped_logit: dict[str, float] = {}
    assigned: set[str] = set()
    for group, features in artifact["feature_groups"].items():
        available = [feature for feature in features if feature in feature_contributions]
        grouped_logit[group] = float(feature_contributions[available].sum())
        assigned.update(available)
    remainder = [feature for feature in feature_contributions.index if feature not in assigned]
    if remainder:
        grouped_logit["other"] = float(feature_contributions[remainder].sum())

    total_logit = sum(grouped_logit.values())
    delta = adjusted - p_base
    adjustments = {
        group: {
            "logit_contribution": value,
            "probability_points_allocated": (
                100 * delta * value / total_logit if abs(total_logit) > 1e-12 else 0.0
            ),
        }
        for group, value in grouped_logit.items()
    }
    return TournamentMatchPrediction(
        p_base=float(p_base),
        p_adjusted=adjusted,
        delta_probability=delta,
        adjustments=adjustments,
        model_version=str(artifact["model_version"]),
        status=str(artifact["status"]),
    )


def _round_labels(draw_size: int) -> list[str]:
    standard = {128: "R128", 64: "R64", 32: "R32", 16: "R16", 8: "QF", 4: "SF", 2: "F"}
    labels = []
    remaining = draw_size
    while remaining >= 2:
        labels.append(standard.get(remaining, f"R{remaining}"))
        remaining //= 2
    return labels


def simulate_tournament(
    first_round_pairs: Sequence[tuple[str, str]],
    base_probability: Callable[[str, str], float],
    *,
    n_simulations: int = 100_000,
    seed: int = 42,
    mode: Literal["pre_tournament", "live"] = "pre_tournament",
    context_provider: Callable[[str, str, str], Mapping[str, float]] | None = None,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Simulate a fixed knockout bracket from its next unplayed round.

    Pre-tournament mode deliberately uses PRE-MATCH V1 alone. Live mode uses a
    frozen real-world context snapshot supplied by ``context_provider``; it does
    not invent future match statistics inside simulations.
    """
    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive")
    players = [player for pair in first_round_pairs for player in pair]
    if len(players) < 2 or len(players) & (len(players) - 1):
        raise ValueError("The bracket must contain a power-of-two number of players")
    if len(set(players)) != len(players):
        raise ValueError("Each player must occur exactly once in the opening bracket")
    if mode == "live" and context_provider is None:
        raise ValueError("Live mode requires a frozen context_provider")

    labels = _round_labels(len(players))
    reached: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    champions: defaultdict[str, int] = defaultdict(int)
    probability_cache: dict[tuple[str, str, str], float] = {}
    rng = np.random.default_rng(seed)

    def match_probability(player_a: str, player_b: str, round_name: str) -> float:
        key = (player_a, player_b, round_name)
        if key in probability_cache:
            return probability_cache[key]
        base = float(base_probability(player_a, player_b))
        if not 0 < base < 1:
            raise ValueError("base_probability must return values strictly between zero and one")
        if mode == "live":
            context = context_provider(player_a, player_b, round_name)  # type: ignore[misc]
            value = predict_tournament_match(base, context, model_path=model_path).p_adjusted
        else:
            value = base
        probability_cache[key] = value
        return value

    for _ in range(n_simulations):
        pairs = list(first_round_pairs)
        for round_name in labels:
            participants = [player for pair in pairs for player in pair]
            for player in participants:
                reached[player][round_name] += 1
            winners = []
            for player_a, player_b in pairs:
                probability = match_probability(player_a, player_b, round_name)
                winners.append(player_a if rng.random() < probability else player_b)
            if len(winners) == 1:
                champions[winners[0]] += 1
                break
            pairs = list(zip(winners[::2], winners[1::2], strict=True))

    results = {
        player: {
            **{
                f"P({round_name})": reached[player][round_name] / n_simulations
                for round_name in labels
            },
            "P(Champion)": champions[player] / n_simulations,
        }
        for player in players
    }
    stage_probability_sums = {
        round_name: sum(result[f"P({round_name})"] for result in results.values())
        for round_name in labels
    }
    stage_probability_sums["Champion"] = sum(
        result["P(Champion)"] for result in results.values()
    )
    return {
        "mode": mode,
        "n_simulations": n_simulations,
        "seed": seed,
        "players": results,
        "stage_probability_sums": stage_probability_sums,
        "future_context_policy": (
            "PRE-MATCH V1 only"
            if mode == "pre_tournament"
            else "frozen observed context; no simulated future statistics"
        ),
    }
