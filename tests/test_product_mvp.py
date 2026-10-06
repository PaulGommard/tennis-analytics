from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from tennis_analytics.paths import PROJECT_ROOT
from tennis_analytics.product.on_demand import OnDemandPredictionService
from tennis_analytics.product.service import TennisProductService
from tennis_analytics.product.storage import ProductStore

WIMBLEDON_2024 = "atp:2024-540:2024-07-01"


@pytest.fixture(scope="module")
def product(tmp_path_factory: pytest.TempPathFactory) -> TennisProductService:
    database = tmp_path_factory.mktemp("product") / "test.db"
    return TennisProductService(store=ProductStore(database))


def _replay_match(product: TennisProductService) -> dict:
    return next(
        match
        for match in product.provider.get_matches(WIMBLEDON_2024)
        if match["live_replay_available"] and match["prediction_available"]
    )


def test_historical_provider_exposes_complete_bracket_and_replay(product):
    tournament = product.provider.get_tournament(WIMBLEDON_2024)
    bracket = product.provider.get_bracket(WIMBLEDON_2024)

    assert tournament["matches"] == 127
    assert tournament["players"] == 128
    assert [(round_["round"], len(round_["matches"])) for round_ in bracket] == [
        ("R128", 64),
        ("R64", 32),
        ("R32", 16),
        ("R16", 8),
        ("QF", 4),
        ("SF", 2),
        ("F", 1),
    ]
    match = _replay_match(product)
    assert product.provider.point_orientation(match["id"]) in {-1, 1}
    assert len(product.provider.point_frame(match["id"])) > 0


def test_historical_prematch_prediction_is_exact_and_immutable(product):
    match = _replay_match(product)
    expected = float(product.prematch.probability_by_id.loc[match["id"]])

    first = product.prematch.prediction(match["id"])
    second = product.prematch.prediction(match["id"])

    assert first["probability"]["player_a"] == expected
    assert first["model_version"] == "PREMATCH_V1"
    assert len(first["feature_snapshot_hash"]) == 64
    assert second["probability"] == first["probability"]
    assert second["prediction_created_at"] == first["prediction_created_at"]
    assert second["cache"] == "hit"


def test_counterfactual_prematch_uses_real_player_snapshots(product):
    match = _replay_match(product)
    probability = product.prematch.pair_probability(
        match["player_a"]["id"],
        match["player_b"]["id"],
        "2024-07-01",
        "Grass",
    )

    assert len(product.prematch._side_bases) >= 40
    assert 0.0 < probability < 1.0


def test_live_stack_uses_official_horizons_and_keeps_orientation(product):
    match = _replay_match(product)
    prematch = product.prematch.prediction(match["id"])
    live = product.live.prediction(match["id"], 0)

    assert live["models"] == {
        "next_point": "POINT_V2",
        "game": "LIVE_V2",
        "set": "LIVE_V2",
        "match": "LIVE_V3",
    }
    assert all(0.0 <= value <= 1.0 for value in live["probabilities"].values())
    assert live["probabilities"]["match_a"] == pytest.approx(
        prematch["probability"]["player_a"], abs=1e-9
    )
    assert live["score"]["point_index"] == 0


def test_tournament_simulation_is_coherent_and_adapter_stays_disabled(product):
    result = product.tournament.probabilities(
        WIMBLEDON_2024, -1, n_simulations=200
    )

    assert result["official_probability_source"] == "PREMATCH_V1"
    assert result["tournament_adapter_used"] is False
    assert result["stage_probability_sums"] == {
        "R128": 128.0,
        "R64": 64.0,
        "R32": 32.0,
        "R16": 16.0,
        "QF": 8.0,
        "SF": 4.0,
        "F": 2.0,
        "Champion": 1.0,
    }
    cached = product.tournament.probabilities(
        WIMBLEDON_2024, -1, n_simulations=200
    )
    assert cached["cache"] == "hit"
    assert cached["players"] == result["players"]


def test_sqlite_prediction_ledger_rejects_immutable_collision(tmp_path: Path):
    store = ProductStore(tmp_path / "ledger.db")
    payload = {
        "model_version": "PREMATCH_V1",
        "prediction_created_at": "2026-10-06T00:00:00+00:00",
        "feature_snapshot_hash": "snapshot",
        "probability": {"player_a": 0.6, "player_b": 0.4},
    }
    store.store_prediction("match-1", payload)
    changed = {
        **payload,
        "probability": {"player_a": 0.7, "player_b": 0.3},
    }

    with pytest.raises(ValueError, match="Immutable prediction collision"):
        store.store_prediction("match-1", changed)


def test_api_and_static_frontend_are_wired(product, monkeypatch):
    monkeypatch.syspath_prepend(str(PROJECT_ROOT))
    backend_app = importlib.import_module("backend.app")
    monkeypatch.setattr(backend_app, "service", lambda: product)
    client = TestClient(backend_app.app)
    match = _replay_match(product)

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["tournament_adapter_enabled"] is False
    assert client.get(f"/api/tournaments/{WIMBLEDON_2024}").status_code == 200
    assert client.get(f"/api/matches/{match['id']}/prediction").status_code == 200
    live = client.get(f"/api/matches/{match['id']}/live?point=0")
    assert live.status_code == 200
    assert live.json()["models"]["match"] == "LIVE_V3"
    homepage = client.get("/")
    assert homepage.status_code == 200
    assert "TENNIS PREDICTION PLATFORM V1" in homepage.text


def test_performance_report_remains_separate_from_product_ledger(product):
    performance = product.performance()

    assert set(performance) == {
        "prospective_notice",
        "prematch",
        "live",
        "tournament_research",
    }
    assert isinstance(pd.DataFrame(performance["prematch"]), pd.DataFrame)
    assert any(not model["official"] for model in product.model_registry())


class _FakeOnDemandProvider:
    provider_name = "fake_on_demand"

    def __init__(self) -> None:
        self.request_count = 0

    def reset_budget(self) -> None:
        self.request_count = 0

    def get_upcoming_matches(self, *, days: int = 3) -> list[dict]:
        return []

    def get_player_recent_matches(self, external_player_id: str, *, year: int) -> dict:
        self.request_count += 1
        player_a = external_player_id == "2622"
        return {
            "external_player_id": external_player_id,
            "name": "Matteo Berrettini" if player_a else "Aleksandar Kovacevic",
            "rank": 30 if player_a else 70,
            "rank_points": None,
            "birth_date_text": None,
            "matches": [{
                "date": "2026-09-27", "tournament": "Japan Open", "surface": "Hard",
                "round": "1st", "opponent": {"external_id": "other", "name": "Other"},
                "result": "W" if player_a else "L", "score": "6-4, 6-4",
            }],
            "provider": self.provider_name,
            "source_fetched_at": "2026-10-06T00:00:00+00:00",
            "source_cache": "miss",
        }


def test_on_demand_refresh_keeps_frozen_86_feature_contract(product):
    provider = _FakeOnDemandProvider()
    runtime = OnDemandPredictionService(
        provider, product.provider, product.prematch, product.store
    )
    match = {
        "external_match_id": "espn:test-upcoming",
        "starts_at": "2026-10-07T04:00:00+00:00",
        "tournament": "Rolex Shanghai Masters",
        "round": "Round 1",
        "surface": "Hard",
        "venue": "Shanghai",
        "player_a": {"external_id": "2622", "name": "Matteo Berrettini"},
        "player_b": {"external_id": "3849", "name": "Aleksandar Kovacevic"},
        "provider": provider.provider_name,
        "source_fetched_at": "2026-10-06T00:00:00+00:00",
        "source_cache": "miss",
    }
    product.store.store_upcoming([match])

    prediction = runtime.refresh_prediction(match["external_match_id"])

    assert prediction["model_version"] == "PREMATCH_V1"
    assert prediction["feature_count"] == 86
    assert prediction["data_quality"] == "MEDIUM"
    assert prediction["external_requests_used"] == 2
    assert 0 < prediction["probability"]["player_a"] < 1
    assert prediction["player_metrics"]["player_a"]["ranking"] == 30
    assert prediction["player_metrics"]["player_b"]["ranking"] == 70
    assert prediction["features"]["rank_diff"] == 40
    assert prediction["features"]["matches_7d_diff"] == 0
    assert prediction["features"]["minutes_7d_diff"] == 0
    assert prediction["features"]["minutes_14d_diff"] is None


def _detail_runtime(product: TennisProductService) -> tuple[OnDemandPredictionService, dict]:
    provider = _FakeOnDemandProvider()
    runtime = OnDemandPredictionService(
        provider, product.provider, product.prematch, product.store
    )
    match = {
        "external_match_id": "espn:detail-test",
        "starts_at": "2026-10-07T14:00:00+00:00",
        "tournament": "Rolex Shanghai Masters",
        "round": "Quarterfinal",
        "surface": "Hard",
        "venue": "Shanghai",
        "player_a": {"external_id": "2622", "name": "Matteo Berrettini"},
        "player_b": {"external_id": "3849", "name": "Aleksandar Kovacevic"},
        "provider": provider.provider_name,
        "source_fetched_at": "2026-10-07T00:00:00+00:00",
        "source_cache": "miss",
    }
    product.store.store_upcoming([match])
    return runtime, match


def test_on_demand_detail_api_cache_features_history_and_ab_mapping(product, monkeypatch):
    runtime, match = _detail_runtime(product)
    prediction = runtime.refresh_prediction(match["external_match_id"])
    monkeypatch.syspath_prepend(str(PROJECT_ROOT))
    backend_app = importlib.import_module("backend.app")
    monkeypatch.setattr(backend_app, "service", lambda: product)
    monkeypatch.setattr(backend_app, "on_demand_service", lambda: runtime)
    client = TestClient(backend_app.app)
    encoded = "espn%3Adetail-test"

    detail = client.get(f"/api/matches/{encoded}")
    cached = client.get(f"/api/matches/{encoded}/prediction")
    features = client.get(f"/api/matches/{encoded}/features")
    history = client.get(f"/api/matches/{encoded}/prediction-history")
    refreshed = client.post(f"/api/matches/{encoded}/refresh-prediction")

    assert detail.status_code == cached.status_code == features.status_code == 200
    assert refreshed.status_code == history.status_code == 200
    assert detail.json()["player_a"]["name"] == "Matteo Berrettini"
    assert cached.json()["probability"] == prediction["probability"]
    assert cached.json()["player_metrics"]["player_a"]["ranking"] == 30
    assert cached.json()["player_metrics"]["player_b"]["ranking"] == 70
    assert features.json()["feature_count"] == 86
    assert features.json()["features"]["rank_diff"] == 40
    assert len(history.json()["history"]) == 1
    assert client.get(f"/matches/{encoded}").status_code == 200


def test_on_demand_detail_unknown_match_is_explicit(product, monkeypatch):
    runtime, _ = _detail_runtime(product)
    monkeypatch.syspath_prepend(str(PROJECT_ROOT))
    backend_app = importlib.import_module("backend.app")
    monkeypatch.setattr(backend_app, "on_demand_service", lambda: runtime)
    client = TestClient(backend_app.app)

    response = client.get("/api/matches/espn%3Aunknown")

    assert response.status_code == 404
    assert "Unknown resource" in response.json()["detail"]


def test_external_player_mapping_is_persisted_without_probability_inversion(product):
    runtime, match = _detail_runtime(product)
    prediction = runtime.refresh_prediction(match["external_match_id"])
    mapping_a = product.store.get_external_mapping("fake_on_demand", "2622")
    mapping_b = product.store.get_external_mapping("fake_on_demand", "3849")

    assert mapping_a and mapping_a["internal_player_id"] == prediction["player_a"]["internal_id"]
    assert mapping_b and mapping_b["internal_player_id"] == prediction["player_b"]["internal_id"]
    assert prediction["probability"]["player_b"] == pytest.approx(
        1 - prediction["probability"]["player_a"]
    )


def test_runtime_feature_vector_reverses_player_orientation(product):
    runtime, match = _detail_runtime(product)
    original = runtime.refresh_prediction(match["external_match_id"])
    reversed_match = {
        **match,
        "external_match_id": "espn:detail-test-reversed",
        "player_a": match["player_b"],
        "player_b": match["player_a"],
    }
    product.store.store_upcoming([reversed_match])
    reversed_prediction = runtime.refresh_prediction(reversed_match["external_match_id"])

    for name, value in original["features"].items():
        reversed_value = reversed_prediction["features"][name]
        if value is None:
            assert reversed_value is None
        elif name.endswith("_diff") or "_diff_x_" in name or name in {
            "elo_logit", "elo_logit_x_clay", "elo_logit_x_grass", "h2h_edge_A",
        }:
            assert reversed_value == pytest.approx(-value)
        else:
            assert reversed_value == pytest.approx(value)
    assert (
        original["probability"]["player_a"]
        + reversed_prediction["probability"]["player_a"]
    ) == pytest.approx(1.0, abs=0.02)
