from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd

from tennis_analytics.live.evaluation import match_format
from tennis_analytics.live.evaluation_v2 import _neutral_features, _sigmoid_shift
from tennis_analytics.live.markov import (
    align_serve_probabilities_to_prematch,
    live_probabilities,
)
from tennis_analytics.live.phase8c import _blend
from tennis_analytics.paths import PROJECT_ROOT
from tennis_analytics.product.providers.historical import HistoricalTennisDataProvider
from tennis_analytics.product.storage import ProductStore


class HistoricalLiveService:
    """Official live-stack inference over frozen historical point snapshots."""

    def __init__(
        self, provider: HistoricalTennisDataProvider, store: ProductStore
    ) -> None:
        self.provider = provider
        self.store = store
        self.v1 = joblib.load(PROJECT_ROOT / "models" / "point_v1" / "model.joblib")
        self.v2 = joblib.load(PROJECT_ROOT / "models" / "point_v2" / "model.joblib")
        self.v3 = joblib.load(PROJECT_ROOT / "models" / "live_v3" / "model.joblib")
        self._prediction_cache: dict[tuple[str, int], dict[str, Any]] = {}

    def _model_state(
        self,
        row_frame: pd.DataFrame,
        artifact: dict[str, Any],
        first_server: str,
    ) -> tuple[float, float, float]:
        columns = artifact["feature_columns"]
        prior_weight = int(artifact.get("selected_prior_weight", 50))
        pipeline = artifact["pipeline"]
        raw_current = float(pipeline.predict_proba(row_frame[columns])[0, 1])
        raw_a = float(
            pipeline.predict_proba(
                _neutral_features(row_frame, "A", columns, prior_weight)
            )[0, 1]
        )
        raw_b = float(
            pipeline.predict_proba(
                _neutral_features(row_frame, "B", columns, prior_weight)
            )[0, 1]
        )
        row = row_frame.iloc[0]
        fmt = match_format(str(row["rule_label"]), int(row["best_of"]))
        p_a, p_b, delta = align_serve_probabilities_to_prematch(
            raw_a,
            raw_b,
            float(row["prematch_probability_A"]),
            first_server=first_server,
            match_format=fmt,
        )
        signed_delta = delta if row["server"] == "A" else -delta
        adjusted_server = float(_sigmoid_shift(raw_current, signed_delta))
        p_next_a = adjusted_server if row["server"] == "A" else 1 - adjusted_server
        return p_a, p_b, p_next_a

    def prediction(self, match_id: str, point_index: int) -> dict[str, Any]:
        cache_key = (match_id, int(point_index))
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]
        frame = self.provider.point_frame(match_id)
        index = min(max(int(point_index), 0), len(frame) - 1)
        row_frame = frame.iloc[[index]].copy()
        row = row_frame.iloc[0]
        first_server = str(frame.iloc[0]["server"])
        v1_a, v1_b, _ = self._model_state(row_frame, self.v1, first_server)
        v2_a, v2_b, p_next_v2 = self._model_state(row_frame, self.v2, first_server)
        fmt = match_format(str(row["rule_label"]), int(row["best_of"]))

        v2_probabilities = live_probabilities(
            sets_a=int(row["sets_A"]),
            sets_b=int(row["sets_B"]),
            games_a=int(row["games_A"]),
            games_b=int(row["games_B"]),
            points_a=int(row["points_A"]),
            points_b=int(row["points_B"]),
            server=str(row["server"]),
            is_tiebreak=bool(row["is_tiebreak"]),
            p_a_serve=v2_a,
            p_b_serve=v2_b,
            match_format=fmt,
        )
        points_played = np.array([float(row["points_played_so_far"])])
        blend_constant = float(self.v3["blend_constant"])
        blend_method = str(self.v3["blend_method"])
        blended_a = float(
            _blend(
                np.array([v1_a]), np.array([v2_a]), points_played, blend_constant, blend_method
            )[0]
        )
        blended_b = float(
            _blend(
                np.array([v1_b]), np.array([v2_b]), points_played, blend_constant, blend_method
            )[0]
        )
        v3_probabilities = live_probabilities(
            sets_a=int(row["sets_A"]),
            sets_b=int(row["sets_B"]),
            games_a=int(row["games_A"]),
            games_b=int(row["games_B"]),
            points_a=int(row["points_A"]),
            points_b=int(row["points_B"]),
            server=str(row["server"]),
            is_tiebreak=bool(row["is_tiebreak"]),
            p_a_serve=blended_a,
            p_b_serve=blended_b,
            match_format=fmt,
        )
        p_match = v3_probabilities.p_match_a
        calibrator = self.v3.get("calibrators", {}).get("match")
        if calibrator is not None:
            p_match = float(calibrator.predict(np.array([p_match]))[0])
        match = self.provider.get_match(match_id)
        orientation = self.provider.point_orientation(match_id)
        if orientation == -1:
            p_next_v2 = 1 - p_next_v2
            game_a = 1 - v2_probabilities.p_current_game_a
            set_a = 1 - v2_probabilities.p_current_set_a
            p_match = 1 - p_match
            next_winner = "B" if row["point_winner"] == "A" else "A"
        else:
            game_a = v2_probabilities.p_current_game_a
            set_a = v2_probabilities.p_current_set_a
            next_winner = str(row["point_winner"])
        payload = {
            "match_id": match_id,
            "point_index": index,
            "total_points": len(frame),
            "player_a": match["player_a"],
            "player_b": match["player_b"],
            "score": self.provider.get_live_score(match_id, index),
            "probabilities": {
                "next_point_a": p_next_v2,
                "game_a": game_a,
                "set_a": set_a,
                "match_a": p_match,
            },
            "models": {
                "next_point": "POINT_V2",
                "game": "LIVE_V2",
                "set": "LIVE_V2",
                "match": "LIVE_V3",
            },
            "replay": {
                "source": "historical_point_by_point",
                "point_match_id": match["point_match_id"],
                "next_point_winner": next_winner,
            },
        }
        self.store.store_live_state(match_id, index, "LIVE_V3", payload)
        if len(self._prediction_cache) >= 8_192:
            self._prediction_cache.pop(next(iter(self._prediction_cache)))
        self._prediction_cache[cache_key] = payload
        return payload

    def history(self, match_id: str, through_point: int) -> list[dict[str, Any]]:
        frame = self.provider.point_frame(match_id)
        maximum = min(max(int(through_point), 0), len(frame) - 1)
        step = max(1, (maximum + 1) // 120)
        indices = list(range(0, maximum + 1, step))
        if maximum not in indices:
            indices.append(maximum)
        return [
            {
                "point_index": index,
                "p_match_a": self.prediction(match_id, index)["probabilities"]["match_a"],
            }
            for index in indices
        ]
