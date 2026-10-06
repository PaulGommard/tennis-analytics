from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from tennis_analytics.paths import PROCESSED_DIR
from tennis_analytics.product.prematch import MODEL_VERSION, FastPreMatchService
from tennis_analytics.product.providers.espn import EspnTennisProvider, ProviderError
from tennis_analytics.product.providers.historical import HistoricalTennisDataProvider, _normalize
from tennis_analytics.product.storage import ProductStore


class IdentityResolutionError(ValueError):
    pass


class OnDemandPredictionService:
    """Orchestrate a bounded, explicit refresh followed by frozen-model inference."""

    def __init__(
        self,
        provider: EspnTennisProvider,
        historical: HistoricalTennisDataProvider,
        prematch: FastPreMatchService,
        store: ProductStore,
    ) -> None:
        self.provider = provider
        self.historical = historical
        self.prematch = prematch
        self.store = store
        players_a = historical.timeline[["player_A_id", "player_A_name"]].rename(
            columns={"player_A_id": "id", "player_A_name": "name"}
        )
        players_b = historical.timeline[["player_B_id", "player_B_name"]].rename(
            columns={"player_B_id": "id", "player_B_name": "name"}
        )
        self.players = pd.concat([players_a, players_b]).drop_duplicates()
        self.players["normalized"] = self.players["name"].map(_normalize)
        player_records = pd.read_parquet(
            PROCESSED_DIR / "atp" / "players.parquet",
            columns=["player_id", "birth_date"],
        )
        self.birth_dates = player_records.set_index("player_id")["birth_date"].to_dict()

    @classmethod
    def create(cls, store: ProductStore | None = None) -> OnDemandPredictionService:
        actual_store = store or ProductStore()
        historical = HistoricalTennisDataProvider()
        return cls(
            EspnTennisProvider(), historical,
            FastPreMatchService(historical, actual_store), actual_store,
        )

    def upcoming(self, *, days: int = 3) -> list[dict[str, Any]]:
        matches = self.provider.get_upcoming_matches(days=days)
        self.store.store_upcoming(matches)
        return matches

    def match(self, external_match_id: str) -> dict[str, Any]:
        stored = self.store.get_upcoming(external_match_id)
        if stored:
            return stored
        matches = self.upcoming(days=7)
        for match in matches:
            if match["external_match_id"] == external_match_id:
                return match
        raise KeyError(external_match_id)

    def match_detail(self, external_match_id: str) -> dict[str, Any]:
        match = self.match(external_match_id)
        prediction = self.cached_prediction(external_match_id)
        ranks = (prediction or {}).get("freshness", {})
        return {
            "match_id": external_match_id,
            "status": "upcoming",
            "scheduled_at": match["starts_at"],
            "tournament": {
                "name": match["tournament"],
                "surface": match["surface"],
                "round": match["round"],
                "venue": match.get("venue"),
            },
            "player_a": {
                **match["player_a"],
                "ranking": ranks.get("player_a", {}).get("rank"),
            },
            "player_b": {
                **match["player_b"],
                "ranking": ranks.get("player_b", {}).get("rank"),
            },
            "provider": match["provider"],
            "prediction_available": prediction is not None,
            "future_capabilities": {"live": False, "finished": False},
        }

    def cached_prediction(self, external_match_id: str) -> dict[str, Any] | None:
        return self.store.get_prediction(external_match_id, MODEL_VERSION)

    def prediction_history(self, external_match_id: str) -> list[dict[str, Any]]:
        return self.store.get_prediction_history(external_match_id, MODEL_VERSION)

    def _resolve(self, external_id: str, name: str) -> dict[str, str]:
        mapped = self.store.get_external_mapping(self.provider.provider_name, external_id)
        if mapped:
            return {
                "player_id": str(mapped["internal_player_id"]),
                "name": name,
                "resolution": str(mapped["resolution"]),
            }
        normalized = _normalize(name)
        controlled_aliases = {
            "zhang zhizhen": "zhizhen zhang",
            "wu yibing": "yibing wu",
            "zhou yi": "yi zhou",
        }
        target = controlled_aliases.get(normalized, normalized)
        exact = self.players.loc[self.players["normalized"].eq(target)]
        ids = exact["id"].astype(str).unique()
        if len(ids) != 1:
            reason = "ambiguous" if len(ids) > 1 else "unresolved"
            raise IdentityResolutionError(
                f"Player identity {reason}: {name!r} (ESPN {external_id}); "
                "no fuzzy match was applied."
            )
        resolution = "controlled_alias" if target != normalized else "exact_name"
        result = {"player_id": str(ids[0]), "name": name, "resolution": resolution}
        self.store.store_external_mapping(
            self.provider.provider_name, external_id, result["player_id"], name, result["resolution"]
        )
        return result

    @staticmethod
    def _score_totals(score: str) -> tuple[int, int]:
        sets = games = 0
        for left, right in re.findall(r"(?<!\d)(\d{1,2})-(\d{1,2})(?!\d)", str(score or "")):
            a, b = int(left), int(right)
            if a <= 7 and b <= 7:
                sets += 1
                games += a + b
        return sets, games

    @staticmethod
    def _recent_stats(matches: list[dict[str, Any]], query: pd.Timestamp, days: int) -> dict[str, float]:
        start = query - pd.Timedelta(int(days), unit="D")
        selected = [
            item for item in matches
            if start <= pd.Timestamp(item["date"]) < query
        ]
        totals = [OnDemandPredictionService._score_totals(item["score"]) for item in selected]
        return {
            "matches": float(len(selected)),
            "sets": float(sum(item[0] for item in totals)),
            "games": float(sum(item[1] for item in totals)),
            "win_rate": float(sum(item["result"] == "W" for item in selected) / len(selected))
            if selected else np.nan,
        }

    def _combined_results(
        self,
        player_id: str,
        remote_matches: list[dict[str, Any]],
        query: pd.Timestamp,
    ) -> list[dict[str, Any]]:
        timeline = self.historical.timeline
        local = timeline.loc[
            (
                timeline["player_A_id"].eq(player_id)
                | timeline["player_B_id"].eq(player_id)
            )
            & timeline["date"].lt(query)
        ].sort_values(["date", "match_id"])
        results: list[dict[str, Any]] = []
        for row in local.itertuples(index=False):
            side = "A" if str(row.player_A_id) == player_id else "B"
            opponent_side = "B" if side == "A" else "A"
            results.append(
                {
                    "date": pd.Timestamp(row.date),
                    "tournament": str(row.tournament),
                    "surface": str(row.surface),
                    "opponent_internal_id": str(getattr(row, f"player_{opponent_side}_id")),
                    "opponent_external_id": None,
                    "opponent_name": str(getattr(row, f"player_{opponent_side}_name")),
                    "result": "W" if str(row.winner_id) == player_id else "L",
                    "score": row.score,
                    "minutes": float(row.minutes) if pd.notna(row.minutes) else None,
                    "source": "internal_historical",
                    "date_precision": "match_day",
                }
            )
        local_cutoff = local["date"].max() if not local.empty else pd.Timestamp.min
        for item in remote_matches:
            event_date = pd.Timestamp(item["date"])
            if not (local_cutoff < event_date < query):
                continue
            results.append(
                {
                    "date": event_date,
                    "tournament": item["tournament"],
                    "surface": item["surface"],
                    "opponent_internal_id": None,
                    "opponent_external_id": item["opponent"]["external_id"],
                    "opponent_name": item["opponent"]["name"],
                    "result": item["result"],
                    "score": item["score"],
                    "minutes": None,
                    "source": self.provider.provider_name,
                    "date_precision": "tournament_start",
                }
            )
        return sorted(results, key=lambda item: item["date"], reverse=True)

    def _state(
        self, player: dict[str, Any], match: dict[str, Any], rankings: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, str]]:
        resolved = self._resolve(player["external_id"], player["name"])
        query = pd.Timestamp(match["starts_at"]).tz_localize(None)
        remote = self.provider.get_player_recent_matches(
            player["external_id"], year=query.year
        )
        ranked = (rankings or {}).get("players", {}).get(player["external_id"])
        if ranked:
            remote["rank"] = ranked["rank"]
            remote["rank_points"] = ranked["rank_points"]
            remote["ranking_as_of"] = rankings.get("as_of")
        prior = [item for item in remote["matches"] if pd.Timestamp(item["date"]) < query]
        state_as_of = max((item["date"] for item in prior), default=remote["source_fetched_at"])
        state = {
            **remote,
            "internal_player_id": resolved["player_id"],
            "identity_resolution": resolved["resolution"],
            "state_as_of": state_as_of,
            "refreshed_at": datetime.now(UTC).isoformat(),
            "quality": "MEDIUM" if remote["rank"] is not None else "LOW",
        }
        self.store.store_runtime_state(state)
        return state, resolved

    def _apply_refresh(
        self,
        frame: pd.DataFrame,
        states: tuple[dict[str, Any], dict[str, Any]],
        internal_ids: tuple[str, str],
        match: dict[str, Any],
    ) -> tuple[pd.DataFrame, list[str], dict[str, dict[str, Any]]]:
        values = frame.iloc[0].to_dict()
        query = pd.Timestamp(match["starts_at"]).tz_localize(None)
        historical_states = self.prematch._player_states(
            internal_ids, query.isoformat(), match["surface"]
        )
        side_values: list[dict[str, float]] = []
        combined_results: list[list[dict[str, Any]]] = []
        warnings = [
            "service/return statistics remain at their last compatible historical snapshot",
            "recent durations are unavailable and are passed as missing values to the frozen imputer",
            "ESPN result dates are tournament-level dates; workload is approximate",
            "Elo is not recursively refreshed for every recent opponent",
        ]
        for state, player_id in zip(states, internal_ids, strict=True):
            side = dict(historical_states[player_id])
            recent = self._combined_results(player_id, state["matches"], query)
            combined_results.append(recent)
            if state.get("rank") is not None:
                side["rank"] = float(state["rank"])
            if state.get("rank_points") is not None:
                side["rank_points"] = float(state["rank_points"])
            born = self.birth_dates.get(player_id)
            if born is not None and pd.notna(born):
                side["age_years"] = float((query - pd.Timestamp(born)).days / 365.25)
            prior_dates = [pd.Timestamp(item["date"]) for item in recent]
            if prior_dates:
                side["last_date_days"] = float((query - max(prior_dates)).days)
            for days in (7, 14, 30):
                stats = self._recent_stats(recent, query, days)
                side[f"matches_{days}d"] = stats["matches"]
                side[f"sets_{days}d"] = stats["sets"]
                side[f"games_{days}d"] = stats["games"]
                selected = [
                    item for item in recent
                    if query - pd.Timedelta(int(days), unit="D")
                    <= pd.Timestamp(item["date"]) < query
                ]
                side[f"minutes_{days}d"] = (
                    0.0 if not selected else
                    sum(float(item["minutes"]) for item in selected)
                    if all(item["minutes"] is not None for item in selected) else np.nan
                )
            for days in (90, 365):
                stats = self._recent_stats(recent, query, days)
                if np.isfinite(stats["win_rate"]):
                    side[f"win_rate_{days}d"] = stats["win_rate"]
            if recent:
                side["win_rate_career"] = float(
                    sum(item["result"] == "W" for item in recent) / len(recent)
                )
            side["career_matches"] = float(len(recent))
            known_surface = [item for item in recent if item["surface"] == match["surface"]]
            side["surface_matches"] = float(len(known_surface))
            side_values.append(side)

        a, b = side_values
        for base in self.prematch._side_bases:
            va, vb = a.get(base, np.nan), b.get(base, np.nan)
            if np.isfinite(va) and np.isfinite(vb):
                values[f"{base}_diff"] = vb - va if base == "rank" else va - vb
            else:
                values[f"{base}_diff"] = np.nan
        values["rest_days_diff"] = a.get("last_date_days", np.nan) - b.get("last_date_days", np.nan)
        for days in (14, 30):
            prior_stats: list[dict[str, float]] = []
            for recent in combined_results:
                selected = [
                    item for item in recent
                    if query - pd.Timedelta(int(days), unit="D")
                    <= pd.Timestamp(item["date"]) < query
                    and _normalize(item["tournament"]) != _normalize(match["tournament"])
                ]
                totals = [self._score_totals(item["score"]) for item in selected]
                prior_stats.append(
                    {
                        "matches": float(len(selected)),
                        "sets": float(sum(item[0] for item in totals)),
                        "games": float(sum(item[1] for item in totals)),
                    }
                )
            for measure in ("matches", "sets", "games"):
                values[f"prior_tournament_{measure}_{days}d_diff"] = (
                    prior_stats[0][measure] - prior_stats[1][measure]
                )
            values[f"prior_tournament_minutes_{days}d_diff"] = np.nan

        local_h2h = self.historical.timeline.loc[
            (
                self.historical.timeline["player_A_id"].isin(internal_ids)
                & self.historical.timeline["player_B_id"].isin(internal_ids)
            )
            & self.historical.timeline["date"].lt(query)
        ]
        external_h2h = [
            item for item in combined_results[0]
            if item["source"] != "internal_historical"
            and item["opponent_external_id"] == states[1]["external_player_id"]
        ]
        h2h_matches = len(local_h2h) + len(external_h2h)
        h2h_wins_a = int(local_h2h["winner_id"].eq(internal_ids[0]).sum()) + sum(
            item["result"] == "W" for item in external_h2h
        )
        values["h2h_matches_before"] = float(h2h_matches)
        values["h2h_edge_A"] = (h2h_wins_a + 2.5) / (h2h_matches + 5.0) - 0.5
        clay, grass = float(match["surface"] == "Clay"), float(match["surface"] == "Grass")
        values["surface_clay"], values["surface_grass"] = clay, grass
        for base in (
            "service_points_won_rate_90d", "return_points_won_rate_90d",
            "serve_above_expected_90d", "return_above_expected_90d",
        ):
            feature = f"{base}_diff"
            if np.isfinite(values.get(feature, np.nan)):
                values[f"{feature}_x_clay"] = values[feature] * clay
                values[f"{feature}_x_grass"] = values[feature] * grass
            else:
                values[f"{feature}_x_clay"] = np.nan
                values[f"{feature}_x_grass"] = np.nan
        def clean(value: Any) -> float | None:
            try:
                return float(value) if np.isfinite(value) else None
            except (TypeError, ValueError):
                return None

        player_metrics: dict[str, dict[str, Any]] = {}
        for key, side, state in zip(("player_a", "player_b"), side_values, states, strict=True):
            prior = [
                item for item in state["matches"]
                if pd.Timestamp(item["date"]) < query
            ][:10]
            player_metrics[key] = {
                "name": state["name"],
                "ranking": clean(side.get("rank")),
                "ranking_points": clean(side.get("rank_points")),
                "elo_composite": clean(side.get("elo_blended")),
                "age_years": clean(side.get("age_years")),
                "hand": state.get("hand"),
                "recent_record": {
                    "wins": sum(item["result"] == "W" for item in prior),
                    "losses": sum(item["result"] == "L" for item in prior),
                    "sample": len(prior),
                },
                "win_rate_90d": clean(side.get("win_rate_90d")),
                "win_rate_365d": clean(side.get("win_rate_365d")),
                "service_points_won_rate_90d": clean(side.get("service_points_won_rate_90d")),
                "first_serve_won_rate_90d": clean(side.get("first_serve_won_rate_90d")),
                "second_serve_won_rate_90d": clean(side.get("second_serve_won_rate_90d")),
                "ace_rate_90d": clean(side.get("ace_rate_90d")),
                "return_points_won_rate_90d": clean(side.get("return_points_won_rate_90d")),
                "rest_days": clean(side.get("last_date_days")),
                "matches_30d": clean(side.get("matches_30d")),
                "sets_30d": clean(side.get("sets_30d")),
                "games_30d": clean(side.get("games_30d")),
                "career_matches": clean(side.get("career_matches")),
                "surface_matches": clean(side.get("surface_matches")),
                "state_as_of": state["state_as_of"],
                "identity_resolution": state["identity_resolution"],
            }
        return (
            pd.DataFrame([values], columns=self.prematch.columns),
            warnings,
            player_metrics,
        )

    def refresh_prediction(self, external_match_id: str) -> dict[str, Any]:
        self.provider.reset_budget()
        match = self.match(external_match_id)
        if match["surface"] == "unknown":
            raise ValueError(
                f"Surface unresolved for {match['tournament']!r}; prediction refused instead of guessing."
            )
        get_rankings = getattr(self.provider, "get_rankings", None)
        rankings = get_rankings() if get_rankings else None
        state_a, resolved_a = self._state(match["player_a"], match, rankings)
        state_b, resolved_b = self._state(match["player_b"], match, rankings)
        ids = (resolved_a["player_id"], resolved_b["player_id"])
        model_date = pd.Timestamp(match["starts_at"]).tz_localize(None).isoformat()
        baseline_frame = self.prematch.pair_feature_frame(
            *ids, model_date, match["surface"]
        )
        baseline = float(self.prematch._predict_frame(baseline_frame)[0])
        frame, warnings, player_metrics = self._apply_refresh(
            baseline_frame, (state_a, state_b), ids, match
        )
        probability = float(self.prematch._predict_frame(frame)[0])
        feature_values = {
            column: None if pd.isna(frame.iloc[0][column]) else float(frame.iloc[0][column])
            for column in self.prematch.columns
        }
        feature_hash = hashlib.sha256(
            json.dumps(feature_values, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        qualities = [state_a["quality"], state_b["quality"]]
        quality = "MEDIUM" if all(item == "MEDIUM" for item in qualities) else "LOW"
        payload = {
            "match_id": external_match_id,
            "external_match_id": external_match_id,
            "match": match,
            "player_a": {**match["player_a"], "internal_id": ids[0]},
            "player_b": {**match["player_b"], "internal_id": ids[1]},
            "probability": {"player_a": probability, "player_b": 1 - probability},
            "baseline_probability_a": baseline,
            "refresh_impact": probability - baseline,
            "model_version": MODEL_VERSION,
            "prediction_created_at": datetime.now(UTC).isoformat(),
            "feature_snapshot_hash": feature_hash,
            "feature_count": len(self.prematch.columns),
            "features": feature_values,
            "player_metrics": player_metrics,
            "top_contributions": self.prematch._top_contributions(frame),
            "data_quality": quality,
            "freshness": {
                "player_a": {"state_as_of": state_a["state_as_of"], "rank": state_a["rank"]},
                "player_b": {"state_as_of": state_b["state_as_of"], "rank": state_b["rank"]},
                "service_stats_as_of": "2026-05-25 or earlier",
            },
            "warnings": warnings,
            "provider": self.provider.provider_name,
            "data_sources": {
                "fixture": match["provider"],
                "recent_matches": self.provider.provider_name,
                "historical_features": "internal historical dataset through 2026-05-25",
            },
            "external_requests_used": self.provider.request_count,
            "interpretation": "Model probability, not a certainty.",
        }
        stored = self.store.store_prediction(external_match_id, payload)
        return {
            **stored,
            "refresh_checked_at": datetime.now(UTC).isoformat(),
            "cache": "unchanged_features"
            if stored["prediction_created_at"] != payload["prediction_created_at"]
            else "new_snapshot",
        }


__all__ = [
    "IdentityResolutionError", "OnDemandPredictionService", "ProviderError",
]
