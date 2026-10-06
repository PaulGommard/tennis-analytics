from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from tennis_analytics.live.markov import (
    LiveProbabilities,
    MatchFormat,
    align_serve_probabilities_to_prematch,
    live_probabilities,
)
from tennis_analytics.paths import PROJECT_ROOT
from tennis_analytics.point_by_point.normalize import Rules, _point_flags, normalize_name
from tennis_analytics.prematch.predictor import PreMatchPredictor
from tennis_analytics.scoring.engine import ScoreSnapshot

MODEL_PATH = PROJECT_ROOT / "models" / "point_v1" / "model.joblib"


class LiveInputError(ValueError):
    pass


@dataclass(frozen=True)
class LivePrediction:
    model_version: str
    player_a: str
    player_b: str
    surface: str
    prematch_probability_a: float
    p_server_a: float
    p_server_b: float
    consistency_logit_shift: float
    p_next_point_a: float
    p_current_game_a: float
    p_current_set_a: float
    p_match_a: float
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_point_score(value_a: str | int, value_b: str | int, is_tiebreak: bool) -> tuple[int, int]:
    if is_tiebreak:
        try:
            points = int(value_a), int(value_b)
        except (TypeError, ValueError) as exc:
            raise LiveInputError("Tie-break points must be non-negative integers.") from exc
        if min(points) < 0:
            raise LiveInputError("Tie-break points must be non-negative integers.")
        return points
    token_a, token_b = str(value_a).upper(), str(value_b).upper()
    base = {"0": 0, "15": 1, "30": 2, "40": 3}
    if token_a == "AD" and token_b == "40":
        return 4, 3
    if token_a == "40" and token_b == "AD":
        return 3, 4
    if token_a not in base or token_b not in base:
        raise LiveInputError("Regular point score must use 0, 15, 30, 40, or one valid AD.")
    points = base[token_a], base[token_b]
    if max(points) == 3 and min(points) < 3 and abs(points[0] - points[1]) > 1:
        # 40-0 and 40-15 are valid; this branch intentionally documents no rejection.
        pass
    return points


def _logit_shift(probability: float, delta: float) -> float:
    value = min(max(probability, 1e-6), 1 - 1e-6)
    return 1.0 / (1.0 + math.exp(-(math.log(value / (1 - value)) + delta)))


class LivePredictor:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"POINT MODEL V1 not found at {model_path}. Run scripts/train_point_model_v1.py."
            )
        self.artifact = joblib.load(model_path)
        self.profiles = self.artifact["profiles"].copy()

    def _resolve(self, value: str) -> tuple[pd.Series | None, list[str]]:
        exact_id = self.profiles.loc[self.profiles["player_id"] == value]
        if len(exact_id) == 1:
            return exact_id.iloc[0], []
        normalized = normalize_name(value)
        matches = self.profiles.loc[self.profiles["normalized_name"] == normalized]
        if len(matches) > 1:
            options = ", ".join(
                f"{row.player_name} ({row.player_id})" for row in matches.itertuples()
            )
            raise LiveInputError(f"Ambiguous player '{value}': {options}. Use player_id.")
        if len(matches) == 1:
            return matches.iloc[0], []
        return None, [f"unknown point-profile player '{value}': global priors used"]

    def _rate(self, profile: pd.Series | None, kind: str, surface: str) -> float:
        global_serve = float(self.artifact["global_server_rate"])
        if profile is None:
            return global_serve if "serve" in kind else 1 - global_serve
        if kind == "surface_serve":
            return (
                float(profile.get(f"surface_serve_rate_{surface.lower()}", np.nan))
                if pd.notna(profile.get(f"surface_serve_rate_{surface.lower()}", np.nan))
                else float(profile["serve_rate"])
            )
        if kind == "surface_return":
            return (
                float(profile.get(f"surface_return_rate_{surface.lower()}", np.nan))
                if pd.notna(profile.get(f"surface_return_rate_{surface.lower()}", np.nan))
                else float(profile["return_rate"])
            )
        return float(profile[kind])

    def _feature_row(
        self,
        *,
        server: str,
        profile_a: pd.Series | None,
        profile_b: pd.Series | None,
        surface: str,
        prematch_a: float,
        snapshot: ScoreSnapshot,
        match_format: MatchFormat,
    ) -> pd.DataFrame:
        server_profile, receiver_profile = (
            (profile_a, profile_b) if server == "A" else (profile_b, profile_a)
        )
        rules = Rules(
            match_format.best_of,
            match_format.final_set_tiebreak,
            match_format.final_set_tiebreak_at,
            match_format.final_set_tiebreak_points,
        )
        flags = _point_flags(snapshot, rules)
        values = {
            "server_serve_rate": self._rate(server_profile, "serve_rate", surface),
            "receiver_return_rate": self._rate(receiver_profile, "return_rate", surface),
            "server_surface_serve_rate": self._rate(server_profile, "surface_serve", surface),
            "receiver_surface_return_rate": self._rate(receiver_profile, "surface_return", surface),
            "surface_clay": int(surface == "Clay"),
            "surface_grass": int(surface == "Grass"),
            "prematch_server": prematch_a if server == "A" else 1 - prematch_a,
            "server_serve_rate_365d": self._rate(server_profile, "serve_rate_365d", surface),
            "receiver_return_rate_365d": self._rate(receiver_profile, "return_rate_365d", surface),
            "server_leading_in_game": int(
                (server == "A" and snapshot.points_a > snapshot.points_b)
                or (server == "B" and snapshot.points_b > snapshot.points_a)
            ),
            "sets_diff_server": (
                snapshot.sets_a - snapshot.sets_b
                if server == "A"
                else snapshot.sets_b - snapshot.sets_a
            ),
            "games_diff_server": (
                snapshot.games_a - snapshot.games_b
                if server == "A"
                else snapshot.games_b - snapshot.games_a
            ),
            "is_tiebreak": int(snapshot.is_tiebreak),
            **flags,
        }
        columns = self.artifact["feature_columns"]
        return pd.DataFrame([{column: values.get(column, 0.0) for column in columns}])

    def _server_probability(
        self,
        *,
        server: str,
        profile_a: pd.Series | None,
        profile_b: pd.Series | None,
        surface: str,
        prematch_a: float,
        snapshot: ScoreSnapshot,
        match_format: MatchFormat,
    ) -> float:
        frame = self._feature_row(
            server=server,
            profile_a=profile_a,
            profile_b=profile_b,
            surface=surface,
            prematch_a=prematch_a,
            snapshot=snapshot,
            match_format=match_format,
        )
        return float(self.artifact["pipeline"].predict_proba(frame)[:, 1][0])

    def predict(
        self,
        *,
        player_a: str,
        player_b: str,
        surface: str,
        sets_a: int,
        sets_b: int,
        games_a: int,
        games_b: int,
        points_a: str | int,
        points_b: str | int,
        server: str,
        first_server: str | None = None,
        is_tiebreak: bool = False,
        best_of: int = 3,
        prematch_probability_a: float | None = None,
        match_date: str | date | None = None,
        final_set_tiebreak: bool = True,
        final_set_tiebreak_at: int = 6,
        final_set_tiebreak_points: int = 7,
    ) -> LivePrediction:
        canonical_surface = surface.title()
        if canonical_surface not in {"Hard", "Clay", "Grass", "Carpet"}:
            raise LiveInputError("surface must be Hard, Clay, Grass, or Carpet")
        canonical_server = server.upper()
        if canonical_server not in {"A", "B"}:
            raise LiveInputError("server must be A or B")
        canonical_first_server = first_server.upper() if first_server is not None else None
        if canonical_first_server not in {None, "A", "B"}:
            raise LiveInputError("first_server must be A or B")
        raw_a, raw_b = parse_point_score(points_a, points_b, is_tiebreak)
        match_format = MatchFormat(
            best_of=best_of,
            final_set_tiebreak=final_set_tiebreak,
            final_set_tiebreak_at=final_set_tiebreak_at,
            final_set_tiebreak_points=final_set_tiebreak_points,
        )
        if min(sets_a, sets_b, games_a, games_b) < 0:
            raise LiveInputError("sets and games must be non-negative")
        profile_a, warnings_a = self._resolve(player_a)
        profile_b, warnings_b = self._resolve(player_b)
        warnings = warnings_a + warnings_b
        if canonical_first_server is None:
            canonical_first_server = canonical_server
            warnings.append(
                "first_server not supplied: current server used for PRE-MATCH alignment"
            )
        if prematch_probability_a is None:
            effective_date = match_date or datetime.now(UTC).date().isoformat()
            prematch_prediction = PreMatchPredictor().predict(
                player_a=player_a,
                player_b=player_b,
                surface=canonical_surface,
                match_date=effective_date,
            )
            prematch_probability_a = prematch_prediction.probability_a
            warnings.extend(prematch_prediction.warnings)
        if not 0 < prematch_probability_a < 1:
            raise LiveInputError("prematch_probability_a must be strictly between 0 and 1")
        display_a = str(points_a).upper()
        display_b = str(points_b).upper()
        snapshot = ScoreSnapshot(
            sets_a=sets_a,
            sets_b=sets_b,
            games_a=games_a,
            games_b=games_b,
            points_a=raw_a,
            points_b=raw_b,
            display_a=display_a,
            display_b=display_b,
            server=canonical_server,
            is_tiebreak=is_tiebreak,
            match_over=False,
            winner=None,
        )
        neutral_a = ScoreSnapshot(0, 0, 0, 0, 0, 0, "0", "0", "A", False, False, None)
        neutral_b = ScoreSnapshot(0, 0, 0, 0, 0, 0, "0", "0", "B", False, False, None)
        raw_p_a_serve = self._server_probability(
            server="A",
            profile_a=profile_a,
            profile_b=profile_b,
            surface=canonical_surface,
            prematch_a=prematch_probability_a,
            snapshot=neutral_a,
            match_format=match_format,
        )
        raw_p_b_serve = self._server_probability(
            server="B",
            profile_a=profile_a,
            profile_b=profile_b,
            surface=canonical_surface,
            prematch_a=prematch_probability_a,
            snapshot=neutral_b,
            match_format=match_format,
        )
        p_a_serve, p_b_serve, delta = align_serve_probabilities_to_prematch(
            raw_p_a_serve,
            raw_p_b_serve,
            prematch_probability_a,
            first_server=canonical_first_server,
            match_format=match_format,
        )
        raw_current_server = self._server_probability(
            server=canonical_server,
            profile_a=profile_a,
            profile_b=profile_b,
            surface=canonical_surface,
            prematch_a=prematch_probability_a,
            snapshot=snapshot,
            match_format=match_format,
        )
        adjusted_server = _logit_shift(
            raw_current_server, delta if canonical_server == "A" else -delta
        )
        probabilities: LiveProbabilities = live_probabilities(
            sets_a=sets_a,
            sets_b=sets_b,
            games_a=games_a,
            games_b=games_b,
            points_a=raw_a,
            points_b=raw_b,
            server=canonical_server,
            is_tiebreak=is_tiebreak,
            p_a_serve=p_a_serve,
            p_b_serve=p_b_serve,
            match_format=match_format,
        )
        p_next_a = adjusted_server if canonical_server == "A" else 1 - adjusted_server
        return LivePrediction(
            model_version="live_v1/point_v1",
            player_a=player_a,
            player_b=player_b,
            surface=canonical_surface,
            prematch_probability_a=float(prematch_probability_a),
            p_server_a=p_a_serve,
            p_server_b=p_b_serve,
            consistency_logit_shift=delta,
            p_next_point_a=p_next_a,
            p_current_game_a=probabilities.p_current_game_a,
            p_current_set_a=probabilities.p_current_set_a,
            p_match_a=probabilities.p_match_a,
            warnings=tuple(warnings),
        )


def predict_live_state(**kwargs: Any) -> LivePrediction:
    return LivePredictor().predict(**kwargs)
