from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from tennis_analytics.modeling.logistic import apply_platt
from tennis_analytics.paths import PROCESSED_DIR, PROJECT_ROOT
from tennis_analytics.product.providers.historical import HistoricalTennisDataProvider
from tennis_analytics.product.storage import ProductStore

MODEL_VERSION = "PREMATCH_V1"


class FastPreMatchService:
    """Fast immutable inference over frozen historical feature snapshots.

    Actual historical matches are bit-identical to research. Counterfactual bracket
    pairs use cached player snapshots and the same frozen pipeline; unavailable
    pair-specific features are left missing for the model's frozen imputer.
    """

    def __init__(
        self, provider: HistoricalTennisDataProvider, store: ProductStore
    ) -> None:
        self.provider = provider
        self.store = store
        self.artifact = joblib.load(PROJECT_ROOT / "models" / "prematch_v1" / "model.joblib")
        self.columns = list(self.artifact["feature_columns"])
        feature_path = PROCESSED_DIR / "match_features_v3.parquet"
        schema_columns = set(pq.read_schema(feature_path).names)
        self._side_bases = [
            column.removesuffix("_diff")
            for column in self.columns
            if column.endswith("_diff")
            and f"{column.removesuffix('_diff')}_A" in schema_columns
            and f"{column.removesuffix('_diff')}_B" in schema_columns
        ]
        side_columns = [
            side_column
            for base in self._side_bases
            for side_column in (f"{base}_A", f"{base}_B")
        ]
        required = [
            "match_id",
            "date",
            "surface",
            "player_A_id",
            "player_B_id",
            *self.columns,
            *side_columns,
        ]
        self.features = pd.read_parquet(
            feature_path, columns=list(dict.fromkeys(required))
        )
        self.features["date"] = pd.to_datetime(self.features["date"])
        self.features_by_id = self.features.set_index("match_id", drop=False)
        probability_path = PROCESSED_DIR / "tournament_context_v1.parquet"
        if probability_path.exists():
            probabilities = pd.read_parquet(
                probability_path, columns=["match_id", "p_base"]
            )
            self.probability_by_id = probabilities.set_index("match_id")["p_base"]
        else:
            # Tournament-context data is an optional research output. Historical
            # PRE-MATCH V1 inference can be reconstructed directly from the frozen
            # model and the stored 86-feature row.
            self.probability_by_id = pd.Series(dtype=float)
        self.ratings = pd.read_parquet(
            PROCESSED_DIR / "match_ratings_v2.parquet",
            columns=[
                "date",
                "surface",
                "player_A_id",
                "player_B_id",
                "elo_A",
                "elo_B",
                "surface_elo_A",
                "surface_elo_B",
                "surface_matches_A",
                "surface_matches_B",
            ],
        )
        self.ratings["date"] = pd.to_datetime(self.ratings["date"])
        self._feature_positions: defaultdict[str, list[int]] = defaultdict(list)
        self._rating_positions: defaultdict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(self.features.itertuples(index=False)):
            player_a, player_b = str(row.player_A_id), str(row.player_B_id)
            self._feature_positions[player_a].append(index)
            self._feature_positions[player_b].append(index)
        for index, row in enumerate(self.ratings.itertuples(index=False)):
            self._rating_positions[str(row.player_A_id)].append(index)
            self._rating_positions[str(row.player_B_id)].append(index)
        self._state_cache: dict[tuple[str, str, str], dict[str, float]] = {}
        self._pair_probability_cache: dict[tuple[str, str, str, str], float] = {}

    @staticmethod
    def _feature_hash(values: dict[str, float | None]) -> str:
        canonical = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self.artifact["pipeline"].predict_proba(frame[self.columns])[:, 1]
        calibrator = self.artifact.get("calibrator")
        return apply_platt(calibrator, raw) if calibrator is not None else raw

    def _top_contributions(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        pipeline = self.artifact["pipeline"]
        transformed = pipeline.named_steps["scaler"].transform(
            pipeline.named_steps["imputer"].transform(frame[self.columns])
        )[0]
        coefficients = pipeline.named_steps["logistic"].coef_[0]
        rows = [
            {
                "feature": feature,
                "log_odds_contribution": float(value * coefficient),
                "value": None if pd.isna(frame.iloc[0][feature]) else float(frame.iloc[0][feature]),
            }
            for feature, value, coefficient in zip(
                self.columns, transformed, coefficients, strict=True
            )
        ]
        return sorted(rows, key=lambda row: abs(row["log_odds_contribution"]), reverse=True)[:6]

    def prediction(self, match_id: str) -> dict[str, Any]:
        cached = self.store.get_prediction(match_id, MODEL_VERSION)
        if cached is not None:
            cached["cache"] = "hit"
            return cached
        if match_id not in self.features_by_id.index:
            raise KeyError(match_id)
        row = self.features_by_id.loc[[match_id]].copy()
        match = self.provider.get_match(match_id)
        probability = (
            float(self.probability_by_id.loc[match_id])
            if match_id in self.probability_by_id.index
            else float(self._predict_frame(row)[0])
        )
        values = {
            column: None if pd.isna(row.iloc[0][column]) else float(row.iloc[0][column])
            for column in self.columns
        }
        payload = {
            "match_id": match_id,
            "player_a": match["player_a"],
            "player_b": match["player_b"],
            "surface": match["surface"],
            "probability": {"player_a": probability, "player_b": 1 - probability},
            "model_version": MODEL_VERSION,
            "prediction_created_at": datetime.now(UTC).isoformat(),
            "feature_snapshot_hash": self._feature_hash(values),
            "top_contributions": self._top_contributions(row),
            "interpretation": "Model probability, not a certainty.",
            "cache": "miss",
        }
        return self.store.store_prediction(match_id, payload)

    def _player_states(
        self, player_ids: tuple[str, ...], date: str, surface: str
    ) -> dict[str, dict[str, float]]:
        cutoff = pd.Timestamp(date)
        states: dict[str, dict[str, float]] = {}
        for player_id in player_ids:
            cache_key = (player_id, date, surface)
            if cache_key in self._state_cache:
                states[player_id] = self._state_cache[cache_key]
                continue
            appearances = self.features.iloc[self._feature_positions[player_id]]
            appearances = appearances.loc[appearances["date"] < cutoff].sort_values("date")
            if appearances.empty:
                states[player_id] = {"last_date": float("nan")}
                self._state_cache[cache_key] = states[player_id]
                continue
            latest = appearances.iloc[-1]
            side = "A" if latest["player_A_id"] == player_id else "B"
            state = {
                base: float(latest.get(f"{base}_{side}", np.nan))
                for base in self._side_bases
            }
            # Surface-specific snapshots must come from the requested surface,
            # not from the player's most recent match on another surface.
            surface_appearances = appearances.loc[appearances["surface"].eq(surface)]
            if not surface_appearances.empty:
                surface_latest = surface_appearances.iloc[-1]
                surface_side = (
                    "A" if surface_latest["player_A_id"] == player_id else "B"
                )
                for base in (*[b for b in self._side_bases if b.endswith("_surface")], "surface_matches"):
                    state[base] = float(
                        surface_latest.get(f"{base}_{surface_side}", np.nan)
                    )
            else:
                for base in (*[b for b in self._side_bases if b.endswith("_surface")], "surface_matches"):
                    state[base] = np.nan
            state["last_date_days"] = float((cutoff - latest["date"]).days)
            rating_history = self.ratings.iloc[self._rating_positions[player_id]]
            rating_history = rating_history.loc[rating_history["date"].lt(cutoff)].sort_values(
                "date"
            )
            if not rating_history.empty:
                global_row = rating_history.iloc[-1]
                global_side = "A" if global_row["player_A_id"] == player_id else "B"
                global_elo = float(global_row[f"elo_{global_side}"])
                surface_history = rating_history.loc[rating_history["surface"].eq(surface)]
                if not surface_history.empty:
                    surface_row = surface_history.iloc[-1]
                    surface_side = (
                        "A" if surface_row["player_A_id"] == player_id else "B"
                    )
                    surface_elo = float(surface_row[f"surface_elo_{surface_side}"])
                    matches = float(surface_row[f"surface_matches_{surface_side}"])
                    weight = matches / (matches + 75.0)
                    state["elo_blended"] = weight * surface_elo + (1 - weight) * global_elo
                else:
                    state["elo_blended"] = global_elo
            states[player_id] = state
            self._state_cache[cache_key] = state
        return states

    def pair_probability(
        self,
        player_a_id: str,
        player_b_id: str,
        date: str,
        surface: str,
    ) -> float:
        cache_key = (player_a_id, player_b_id, date, surface)
        if cache_key in self._pair_probability_cache:
            return self._pair_probability_cache[cache_key]
        frame = self.pair_feature_frame(player_a_id, player_b_id, date, surface)
        result = float(self._predict_frame(frame)[0])
        if len(self._pair_probability_cache) >= 16_384:
            self._pair_probability_cache.pop(next(iter(self._pair_probability_cache)))
        self._pair_probability_cache[cache_key] = result
        return result

    def pair_feature_frame(
        self,
        player_a_id: str,
        player_b_id: str,
        date: str,
        surface: str,
    ) -> pd.DataFrame:
        """Build the frozen 86-column vector without running inference."""
        states = self._player_states((player_a_id, player_b_id), date, surface)
        state_a, state_b = states[player_a_id], states[player_b_id]
        values = {column: np.nan for column in self.columns}
        for base in self._side_bases:
            value_a, value_b = state_a.get(base, np.nan), state_b.get(base, np.nan)
            if np.isfinite(value_a) and np.isfinite(value_b):
                values[f"{base}_diff"] = (
                    value_b - value_a if base == "rank" else value_a - value_b
                )

        # Recreate Phase 3's surface-blended Elo logit from cached player state.
        elo_a = state_a.get("elo_blended", np.nan)
        elo_b = state_b.get("elo_blended", np.nan)
        if np.isfinite(elo_a) and np.isfinite(elo_b):
            elo_probability = 1 / (1 + 10 ** ((elo_b - elo_a) / 300.0))
            values["elo_logit"] = math.log(elo_probability / (1 - elo_probability))
        elif np.isfinite(state_a.get("rank", np.nan)) and np.isfinite(
            state_b.get("rank", np.nan)
        ):
            rank_edge = math.log((state_b["rank"] + 10) / (state_a["rank"] + 10))
            values["elo_logit"] = rank_edge

        values["rest_days_diff"] = state_a.get("last_date_days", np.nan) - state_b.get(
            "last_date_days", np.nan
        )
        clay = float(surface == "Clay")
        grass = float(surface == "Grass")
        values["surface_clay"] = clay
        values["surface_grass"] = grass
        if np.isfinite(values["elo_logit"]):
            values["elo_logit_x_clay"] = values["elo_logit"] * clay
            values["elo_logit_x_grass"] = values["elo_logit"] * grass
        for base in (
            "service_points_won_rate_90d",
            "return_points_won_rate_90d",
            "serve_above_expected_90d",
            "return_above_expected_90d",
        ):
            feature = f"{base}_diff"
            if np.isfinite(values.get(feature, np.nan)):
                values[f"{feature}_x_clay"] = values[feature] * clay
                values[f"{feature}_x_grass"] = values[feature] * grass
        return pd.DataFrame([values], columns=self.columns)
