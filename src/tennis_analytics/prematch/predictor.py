from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from tennis_analytics.features.opponent_adjusted import (
    build_opponent_history_base,
    materialize_phase5_features,
)
from tennis_analytics.features.stat_history import build_stat_history_base, materialize_features
from tennis_analytics.modeling.logistic import apply_platt
from tennis_analytics.paths import PROCESSED_DIR, PROJECT_ROOT
from tennis_analytics.ratings.elo import EloConfig
from tennis_analytics.ratings.engine import (
    InactivityConfig,
    blended_surface_probability,
    run_elo_engine,
)

MODEL_DIR = PROJECT_ROOT / "models" / "prematch_v1"
SUPPORTED_SURFACES = ("Hard", "Clay", "Grass", "Carpet")


class PredictionInputError(ValueError):
    """An actionable error in prediction inputs."""


@dataclass(frozen=True)
class PlayerResolution:
    player_id: str
    name: str
    status: str
    suggestions: tuple[str, ...] = ()


@dataclass(frozen=True)
class Prediction:
    model_version: str
    prediction_id: str
    match_id: str
    generated_at: str
    match_date: str
    surface: str
    player_a_id: str
    player_a_name: str
    player_b_id: str
    player_b_name: str
    probability_a: float
    probability_b: float
    raw_probability_a: float
    favorite: str
    feature_snapshot_hash: str
    warnings: tuple[str, ...]
    top_contributions: tuple[dict[str, Any], ...]
    features: dict[str, float | None]

    def to_dict(self, *, include_features: bool = False) -> dict[str, Any]:
        result = asdict(self)
        if not include_features:
            result.pop("features")
        return result


def _normalize_name(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", folded.lower()).strip()


def _json_value(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


class PreMatchPredictor:
    """Generate a frozen V1 prediction using only observations before the query batch."""

    def __init__(
        self,
        *,
        model_dir: Path = MODEL_DIR,
        processed_dir: Path = PROCESSED_DIR,
    ) -> None:
        model_path = model_dir / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Frozen model not found at {model_path}. Run scripts/freeze_prematch_v1.py."
            )
        self.model_dir = model_dir
        self.processed_dir = processed_dir
        self.artifact = joblib.load(model_path)
        metadata_path = model_dir / "metadata.json"
        self.metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        self.players = pd.read_parquet(processed_dir / "atp" / "players.parquet")
        self.players["normalized_name"] = self.players["full_name"].map(_normalize_name)

    def resolve_player(self, name_or_id: str) -> PlayerResolution:
        value = name_or_id.strip()
        if not value:
            raise PredictionInputError("Player name/id cannot be empty.")
        by_id = self.players.loc[self.players["player_id"] == value]
        if len(by_id) == 1:
            row = by_id.iloc[0]
            return PlayerResolution(str(row.player_id), str(row.full_name), "known")
        normalized = _normalize_name(value)
        exact = self.players.loc[self.players["normalized_name"] == normalized]
        if len(exact) > 1:
            options = ", ".join(f"{row.full_name} ({row.player_id})" for row in exact.itertuples())
            raise PredictionInputError(f"Ambiguous player '{value}': {options}. Use player_id.")
        if len(exact) == 1:
            row = exact.iloc[0]
            return PlayerResolution(str(row.player_id), str(row.full_name), "known")

        tokens = set(normalized.split())
        scored: list[tuple[int, str]] = []
        for row in self.players.itertuples():
            overlap = len(tokens.intersection(str(row.normalized_name).split()))
            if overlap:
                scored.append((overlap, str(row.full_name)))
        suggestions = tuple(name for _, name in sorted(scored, reverse=True)[:3])
        slug = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-") or "unknown"
        digest = hashlib.sha256(normalized.encode()).hexdigest()[:10]
        return PlayerResolution(f"new:{slug}:{digest}", value, "cold_start", suggestions)

    @staticmethod
    def _validate_inputs(surface: str, match_date: str | date) -> tuple[str, pd.Timestamp]:
        canonical = surface.strip().title()
        if canonical not in SUPPORTED_SURFACES:
            raise PredictionInputError(
                f"Unsupported surface '{surface}'. Expected one of: {', '.join(SUPPORTED_SURFACES)}."
            )
        try:
            query_date = pd.Timestamp(match_date).normalize()
        except Exception as exc:
            raise PredictionInputError(f"Invalid ISO date '{match_date}'.") from exc
        if pd.isna(query_date):
            raise PredictionInputError(f"Invalid ISO date '{match_date}'.")
        return canonical, query_date

    def _latest_context(
        self, player_id: str, query_date: pd.Timestamp, timeline: pd.DataFrame
    ) -> tuple[float, float, float, list[str]]:
        warnings: list[str] = []
        if player_id.startswith("new:"):
            return np.nan, np.nan, np.nan, ["cold_start: no historical player record"]
        rows: list[dict[str, Any]] = []
        for side in ("A", "B"):
            subset = timeline.loc[
                (timeline[f"player_{side}_id"] == player_id)
                & (pd.to_datetime(timeline["date"]) < query_date),
                ["date", f"rank_{side}", f"rank_points_{side}", f"age_years_{side}"],
            ].copy()
            subset.columns = ["date", "rank", "rank_points", "age"]
            rows.extend(subset.to_dict("records"))
        if not rows:
            return np.nan, np.nan, np.nan, ["missing_history: no prior match context"]
        latest = max(rows, key=lambda item: pd.Timestamp(item["date"]))
        age = latest["age"]
        birth = self.players.loc[self.players["player_id"] == player_id, "birth_date"]
        if len(birth) and pd.notna(birth.iloc[0]):
            born = pd.Timestamp(birth.iloc[0])
            age = (query_date - born).days / 365.25
        lag = (query_date - pd.Timestamp(latest["date"])).days
        if lag > 90:
            warnings.append(f"stale_rank_context: last observation is {lag} days old")
        if pd.isna(latest["rank"]):
            warnings.append("missing_rank: model median imputation will apply")
        if pd.isna(latest["rank_points"]):
            warnings.append("missing_rank_points: model median imputation will apply")
        return float(latest["rank"]), float(latest["rank_points"]), float(age), warnings

    def _synthetic_rows(
        self,
        player_a: PlayerResolution,
        player_b: PlayerResolution,
        surface: str,
        query_date: pd.Timestamp,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
        ratings = pd.read_parquet(self.processed_dir / "match_ratings_v2.parquet")
        timeline = pd.read_parquet(self.processed_dir / "analytics" / "match_timeline.parquet")
        matches = pd.read_parquet(self.processed_dir / "atp" / "matches.parquet")
        stats = pd.read_parquet(self.processed_dir / "atp" / "match_stats.parquet")
        history = ratings.loc[pd.to_datetime(ratings["date"]) < query_date].copy()
        max_batch = int(history["chronology_batch_id"].max()) if len(history) else -1
        query_id = (
            "prospective:"
            + hashlib.sha256(
                f"{query_date.date()}|{surface}|{player_a.player_id}|{player_b.player_id}".encode()
            ).hexdigest()[:20]
        )
        query_input = pd.DataFrame(
            [
                {
                    "match_id": query_id,
                    "date": query_date,
                    "surface": surface,
                    "level": "PROSPECTIVE",
                    "round": "PRED",
                    "player_A_id": player_a.player_id,
                    "player_B_id": player_b.player_id,
                    "player_A_win": 0,
                    "chronology_batch_id": max_batch + 1,
                    "chronology_order": max_batch + 1,
                    "chronology_stage": "PROSPECTIVE",
                    "chronology_strategy": "strict_date_cutoff",
                }
            ]
        )
        history["chronology_order"] = history["chronology_batch_id"]
        elo_input = pd.concat([history, query_input], ignore_index=True, sort=False)
        rebuilt = run_elo_engine(
            elo_input,
            EloConfig(k_factor=20.0, scale=300.0),
            InactivityConfig(threshold_days=90, strength=0.10),
        )
        rebuilt["surface_blended_expected_A"] = blended_surface_probability(
            rebuilt, c_value=75.0, scale=300.0
        )
        rebuilt["final_raw_expected_A"] = rebuilt["surface_blended_expected_A"]
        rebuilt["rank_A"] = np.nan
        rebuilt["rank_B"] = np.nan
        rebuilt["rank_diff"] = np.nan

        warnings: list[str] = []
        source_max_date = pd.to_datetime(ratings["date"]).max()
        if query_date <= source_max_date:
            warnings.append(
                "historical_date: calendar-date cutoff used; use replay(match_id) for exact "
                "round-batch reconstruction"
            )
        contexts: dict[str, tuple[float, float, float]] = {}
        for side, player in (("A", player_a), ("B", player_b)):
            rank, points, age, local_warnings = self._latest_context(
                player.player_id, query_date, timeline
            )
            contexts[side] = (rank, points, age)
            warnings.extend(f"player_{side}: {warning}" for warning in local_warnings)
            if player.status == "cold_start":
                hint = (
                    f"; suggestions: {', '.join(player.suggestions)}" if player.suggestions else ""
                )
                warnings.append(f"player_{side}: unknown player treated as cold start{hint}")
        query_index = rebuilt.index[rebuilt["match_id"] == query_id][0]
        rebuilt.loc[query_index, "rank_A"] = contexts["A"][0]
        rebuilt.loc[query_index, "rank_B"] = contexts["B"][0]
        rebuilt.loc[query_index, "rank_diff"] = contexts["B"][0] - contexts["A"][0]

        timeline_columns = [
            "match_id",
            "tournament_edition_id",
            "rank_points_A",
            "rank_points_B",
            "age_years_A",
            "age_years_B",
        ]
        timeline_history = timeline.loc[
            timeline["match_id"].isin(history["match_id"]), timeline_columns
        ].copy()
        timeline_query = {
            "match_id": query_id,
            "tournament_edition_id": query_id,
            "rank_points_A": contexts["A"][1],
            "rank_points_B": contexts["B"][1],
            "age_years_A": contexts["A"][2],
            "age_years_B": contexts["B"][2],
        }
        timeline_all = timeline_history.reset_index(drop=True)
        timeline_all.loc[len(timeline_all)] = timeline_query
        matches_history = matches.loc[
            matches["match_id"].isin(history["match_id"]), ["match_id", "minutes", "score"]
        ].copy()
        matches_all = matches_history.reset_index(drop=True)
        matches_all.loc[len(matches_all), "match_id"] = query_id
        stats_history = stats.loc[stats["match_id"].isin(history["match_id"])].copy()
        return rebuilt, timeline_all, matches_all, stats_history, warnings

    def _feature_row(
        self,
        player_a: PlayerResolution,
        player_b: PlayerResolution,
        surface: str,
        query_date: pd.Timestamp,
    ) -> tuple[pd.Series, list[str]]:
        ratings, timeline, matches, stats, warnings = self._synthetic_rows(
            player_a, player_b, surface, query_date
        )
        phase4_base = build_stat_history_base(ratings, timeline, matches, stats)
        features_v2 = materialize_features(
            phase4_base, point_prior_strength=50.0, surface_prior_strength=1000.0
        )
        phase5_base = build_opponent_history_base(features_v2, ratings, timeline, matches, stats)
        features_v3 = materialize_phase5_features(
            features_v2,
            phase5_base,
            adjusted_point_prior=500.0,
            adjusted_surface_prior=100.0,
            h2h_prior=5.0,
        )
        row = features_v3.iloc[-1]
        for side in ("A", "B"):
            career_matches = row.get(f"career_matches_{side}", np.nan)
            surface_matches = row.get(f"surface_matches_{side}", np.nan)
            service_opportunities = row.get(
                f"service_points_won_rate_career_opportunities_{side}", np.nan
            )
            if pd.isna(career_matches) or float(career_matches) < 5:
                warnings.append(
                    f"player_{side}: very_limited_history ({_json_value(career_matches)} matches)"
                )
            if pd.isna(surface_matches) or float(surface_matches) == 0:
                warnings.append(f"player_{side}: no_prior_matches_on_{surface.lower()}")
            if pd.isna(service_opportunities) or float(service_opportunities) == 0:
                warnings.append(f"player_{side}: no_prior_service_statistics")
        missing = sum(
            pd.isna(row.get(column, np.nan)) for column in self.artifact["feature_columns"]
        )
        if missing:
            warnings.append(
                f"missing_model_features: {missing}/{len(self.artifact['feature_columns'])}; "
                "frozen median imputation applied"
            )
        return row, warnings

    def replay(self, match_id: str) -> Prediction:
        """Replay one stored research row whose features were frozen at T-1."""
        features = pd.read_parquet(self.processed_dir / "match_features_v3.parquet")
        rows = features.loc[features["match_id"] == match_id]
        if len(rows) != 1:
            raise PredictionInputError(
                f"Expected one research row for '{match_id}', found {len(rows)}."
            )
        row = rows.iloc[0]
        names = self.players.set_index("player_id")["full_name"]
        name_a = str(names.get(row.player_A_id, row.player_A_id))
        name_b = str(names.get(row.player_B_id, row.player_B_id))
        return self._predict_row(
            row,
            PlayerResolution(str(row.player_A_id), name_a, "historical"),
            PlayerResolution(str(row.player_B_id), name_b, "historical"),
            (),
        )

    def predict(
        self, *, player_a: str, player_b: str, surface: str, match_date: str | date
    ) -> Prediction:
        canonical_surface, query_date = self._validate_inputs(surface, match_date)
        resolved_a = self.resolve_player(player_a)
        resolved_b = self.resolve_player(player_b)
        if resolved_a.player_id == resolved_b.player_id:
            raise PredictionInputError("Player A and player B must be different.")
        row, warnings = self._feature_row(resolved_a, resolved_b, canonical_surface, query_date)
        return self._predict_row(row, resolved_a, resolved_b, tuple(warnings))

    def _predict_row(
        self,
        row: pd.Series,
        player_a: PlayerResolution,
        player_b: PlayerResolution,
        warnings: tuple[str, ...],
    ) -> Prediction:
        columns = list(self.artifact["feature_columns"])
        frame = pd.DataFrame([{column: row.get(column, np.nan) for column in columns}])
        raw = float(self.artifact["pipeline"].predict_proba(frame)[:, 1][0])
        calibrator = self.artifact.get("calibrator")
        probability = float(apply_platt(calibrator, np.array([raw]))[0]) if calibrator else raw
        features = {column: _json_value(frame.iloc[0][column]) for column in columns}
        canonical = json.dumps(features, sort_keys=True, separators=(",", ":"), allow_nan=False)
        feature_hash = hashlib.sha256(canonical.encode()).hexdigest()

        pipeline = self.artifact["pipeline"]
        transformed = pipeline.named_steps["scaler"].transform(
            pipeline.named_steps["imputer"].transform(frame)
        )[0]
        coefficients = pipeline.named_steps["logistic"].coef_[0]
        contributions = sorted(
            (
                {
                    "feature": column,
                    "log_odds_contribution": float(value * coefficient),
                    "value": features[column],
                }
                for column, value, coefficient in zip(
                    columns, transformed, coefficients, strict=True
                )
            ),
            key=lambda item: abs(item["log_odds_contribution"]),
            reverse=True,
        )[:10]
        now = pd.Timestamp.now(tz="UTC").isoformat()
        prediction_key = f"{row['date']}|{row['surface']}|{player_a.player_id}|{player_b.player_id}|{feature_hash}"
        prediction_id = hashlib.sha256(prediction_key.encode()).hexdigest()[:24]
        return Prediction(
            model_version=str(self.metadata.get("model_version", "prematch_v1")),
            prediction_id=prediction_id,
            match_id=str(row["match_id"]),
            generated_at=now,
            match_date=pd.Timestamp(row["date"]).date().isoformat(),
            surface=str(row["surface"]),
            player_a_id=player_a.player_id,
            player_a_name=player_a.name,
            player_b_id=player_b.player_id,
            player_b_name=player_b.name,
            probability_a=probability,
            probability_b=1.0 - probability,
            raw_probability_a=raw,
            favorite=player_a.name if probability >= 0.5 else player_b.name,
            feature_snapshot_hash=feature_hash,
            warnings=warnings,
            top_contributions=tuple(contributions),
            features=features,
        )
