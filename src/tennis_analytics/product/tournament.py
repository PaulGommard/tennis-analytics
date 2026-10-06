from __future__ import annotations

from typing import Any

from tennis_analytics.product.prematch import FastPreMatchService
from tennis_analytics.product.providers.historical import HistoricalTennisDataProvider
from tennis_analytics.product.storage import ProductStore
from tennis_analytics.tournament.engine import simulate_tournament
from tennis_analytics.tournament.replay import ROUND_ORDER

MAIN_ROUNDS = ("R128", "R64", "R32", "R16", "QF", "SF", "F")


class TournamentSimulationService:
    def __init__(
        self,
        provider: HistoricalTennisDataProvider,
        prematch: FastPreMatchService,
        store: ProductStore,
    ) -> None:
        self.provider = provider
        self.prematch = prematch
        self.store = store

    def _opening_pairs(
        self, tournament_id: str, completed_round_order: int
    ) -> tuple[str, list[tuple[str, str]], dict[str, str]]:
        matches = self.provider.get_matches(tournament_id)
        future = [
            match
            for match in matches
            if match["round"] in MAIN_ROUNDS and match["round_order"] > completed_round_order
        ]
        if not future:
            return "complete", [], {}
        next_order = min(match["round_order"] for match in future)
        round_matches = sorted(
            (match for match in future if match["round_order"] == next_order),
            key=lambda match: match["source_match_num"],
        )
        pairs = [
            (match["player_a"]["id"], match["player_b"]["id"])
            for match in round_matches
        ]
        names = {
            player["id"]: player["name"]
            for match in round_matches
            for player in (match["player_a"], match["player_b"])
        }
        return str(round_matches[0]["round"]), pairs, names

    def probabilities(
        self,
        tournament_id: str,
        completed_round_order: int,
        *,
        n_simulations: int = 2_000,
    ) -> dict[str, Any]:
        snapshot_key = f"state_store_v1:after_order_{completed_round_order}:n{n_simulations}"
        cached = self.store.get_tournament_prediction(
            tournament_id, snapshot_key, "TOURNAMENT_SIM_V1"
        )
        if cached is not None:
            cached["cache"] = "hit"
            return cached
        tournament = self.provider.get_tournament(tournament_id)
        next_round, pairs, names = self._opening_pairs(tournament_id, completed_round_order)
        if not pairs:
            matches = self.provider.get_matches(tournament_id)
            final = max(matches, key=lambda match: match["round_order"])
            champion = final["player_a"] if final["winner"] == "A" else final["player_b"]
            payload = {
                "tournament_id": tournament_id,
                "snapshot_key": snapshot_key,
                "next_round": None,
                "players": [
                    {
                        "player_id": champion["id"],
                        "player_name": champion["name"],
                        "P(Champion)": 1.0,
                    }
                ],
                "model_version": "TOURNAMENT_SIM_V1",
                "official_probability_source": "PREMATCH_V1",
                "cache": "miss",
            }
        else:
            draw_size = len(pairs) * 2
            if draw_size & (draw_size - 1):
                raise ValueError(
                    f"Replay round {next_round} has a non power-of-two draw ({draw_size})"
                )

            def probability(player_a: str, player_b: str) -> float:
                return self.prematch.pair_probability(
                    player_a,
                    player_b,
                    tournament["start_date"],
                    tournament["surface"],
                )

            simulation = simulate_tournament(
                pairs,
                probability,
                n_simulations=n_simulations,
                seed=20261006 + completed_round_order,
                mode="pre_tournament",
            )
            player_rows = [
                {"player_id": player_id, "player_name": names.get(player_id, player_id), **values}
                for player_id, values in simulation["players"].items()
            ]
            player_rows.sort(key=lambda row: row["P(Champion)"], reverse=True)
            payload = {
                "tournament_id": tournament_id,
                "snapshot_key": snapshot_key,
                "next_round": next_round,
                "completed_round_order": completed_round_order,
                "players": player_rows,
                "stage_probability_sums": simulation["stage_probability_sums"],
                "n_simulations": n_simulations,
                "model_version": "TOURNAMENT_SIM_V1",
                "official_probability_source": "PREMATCH_V1",
                "tournament_adapter_used": False,
                "counterfactual_policy": (
                    "Frozen PREMATCH_V1 pipeline with player-state snapshots strictly before "
                    "the tournament; only the realized bracket is updated during replay."
                ),
                "cache": "miss",
            }
        self.store.store_tournament_prediction(
            tournament_id, snapshot_key, "TOURNAMENT_SIM_V1", payload
        )
        return payload

    @staticmethod
    def next_completed_order(tournament: dict[str, Any], current: int) -> int:
        orders = sorted(
            ROUND_ORDER[round_name]
            for round_name in tournament["rounds"]
            if round_name in MAIN_ROUNDS and ROUND_ORDER[round_name] > current
        )
        return orders[0] if orders else current
