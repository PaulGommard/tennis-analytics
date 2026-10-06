from __future__ import annotations

import json
from typing import Any

import pandas as pd

from tennis_analytics.paths import PROJECT_ROOT, REPORTS_DIR
from tennis_analytics.product.live import HistoricalLiveService
from tennis_analytics.product.prematch import FastPreMatchService
from tennis_analytics.product.providers.historical import HistoricalTennisDataProvider
from tennis_analytics.product.storage import ProductStore
from tennis_analytics.product.tournament import TournamentSimulationService

DEFAULT_TOURNAMENT = "atp:2024-540:2024-07-01"


class TennisProductService:
    def __init__(self, *, store: ProductStore | None = None) -> None:
        self.store = store or ProductStore()
        self.provider = HistoricalTennisDataProvider()
        self.prematch = FastPreMatchService(self.provider, self.store)
        self.live = HistoricalLiveService(self.provider, self.store)
        self.tournament = TournamentSimulationService(
            self.provider, self.prematch, self.store
        )
        self.store.register_models(self.model_registry())
        if self.store.replay_state() is None:
            default = DEFAULT_TOURNAMENT
            try:
                self.provider.get_tournament(default)
            except KeyError:
                default = self.provider.get_tournaments(limit=1)[0]["id"]
            self.store.set_replay_state(default)

    @staticmethod
    def model_registry() -> list[dict[str, Any]]:
        registry = [
            ("PREMATCH_V1", "pre_match", True, "models/prematch_v1/metadata.json"),
            ("POINT_V2", "next_point", True, "models/point_v2/metadata.json"),
            ("LIVE_V2", "game_and_set", True, "reports/phase8_report.md"),
            ("LIVE_V3", "match_live", True, "models/live_v3/metadata.json"),
            ("TOURNAMENT_SIM_V1", "tournament_simulation", True, None),
            (
                "TOURNAMENT_ADAPTER_V1",
                "research_only_disabled",
                False,
                "models/tournament_adapter_v1/metadata.json",
            ),
        ]
        rows = []
        for version, role, official, metadata_path in registry:
            metadata: dict[str, Any] = {}
            if metadata_path and metadata_path.endswith(".json"):
                path = PROJECT_ROOT / metadata_path
                if path.exists():
                    metadata = json.loads(path.read_text())
            rows.append(
                {
                    "model_version": version,
                    "role": role,
                    "official": official,
                    "metadata": metadata,
                }
            )
        return rows

    def replay_state(self) -> dict[str, Any]:
        state = self.store.replay_state() or {}
        if not state:
            return state
        tournament = self.provider.get_tournament(state["tournament_id"])
        state["tournament"] = tournament
        state["next_round"] = next(
            (
                round_name
                for round_name in tournament["rounds"]
                if self._round_order(round_name) > state["completed_round_order"]
            ),
            None,
        )
        return state

    @staticmethod
    def _round_order(round_name: str) -> int:
        from tennis_analytics.tournament.replay import ROUND_ORDER

        return ROUND_ORDER.get(round_name, -1)

    def select_replay(self, tournament_id: str) -> dict[str, Any]:
        self.provider.get_tournament(tournament_id)
        self.store.set_replay_state(tournament_id)
        return self.replay_state()

    def advance_round(self) -> dict[str, Any]:
        state = self.store.replay_state() or {}
        tournament = self.provider.get_tournament(state["tournament_id"])
        next_order = self.tournament.next_completed_order(
            tournament, int(state["completed_round_order"])
        )
        self.store.set_replay_state(
            state["tournament_id"], completed_round_order=next_order
        )
        self.tournament.probabilities(state["tournament_id"], next_order)
        return self.replay_state()

    def performance(self) -> dict[str, Any]:
        prematch = pd.read_csv(REPORTS_DIR / "phase5_test_comparison.csv")
        live = pd.read_csv(REPORTS_DIR / "phase8c_test_comparison.csv")
        tournament = pd.read_csv(REPORTS_DIR / "tournament_adapter_test_comparison.csv")
        return {
            "prospective_notice": (
                "Research benchmarks are shown separately. New immutable production "
                "predictions accumulate in the local SQLite ledger."
            ),
            "prematch": prematch.to_dict("records"),
            "live": live.to_dict("records"),
            "tournament_research": tournament.to_dict("records"),
        }
