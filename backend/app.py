from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tennis_analytics.paths import PROJECT_ROOT
from tennis_analytics.product.on_demand import (
    IdentityResolutionError,
    OnDemandPredictionService,
    ProviderError,
)
from tennis_analytics.product.providers.historical import _json_value
from tennis_analytics.product.service import TennisProductService

app = FastAPI(
    title="Tennis Prediction Platform V1",
    version="1.0.0",
    description="Local historical-replay MVP exposing frozen official models.",
)


@lru_cache(maxsize=1)
def service() -> TennisProductService:
    return TennisProductService()


@lru_cache(maxsize=1)
def on_demand_service() -> OnDemandPredictionService:
    return OnDemandPredictionService.create(service().store)


def _not_found(error: KeyError) -> HTTPException:
    return HTTPException(status_code=404, detail=f"Unknown resource: {error.args[0]}")


class ReplaySelection(BaseModel):
    tournament_id: str


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "provider": "historical_replay",
        "models": [
            model["model_version"]
            for model in service().model_registry()
            if model["official"]
        ],
        "tournament_adapter_enabled": False,
    }


@app.get("/api/home")
def home() -> dict[str, Any]:
    product = service()
    state = product.replay_state()
    tournament_id = state["tournament_id"]
    matches = product.provider.get_matches(tournament_id)
    completed = int(state["completed_round_order"])
    live_candidates = [
        match
        for match in matches
        if match["live_replay_available"] and match["round_order"] > completed
    ]
    return {
        "replay": state,
        "tournaments": product.provider.get_tournaments(limit=12),
        "featured_matches": live_candidates[:6],
        "model_policy": {
            "prematch": "PREMATCH_V1",
            "live": ["POINT_V2", "LIVE_V2", "LIVE_V3"],
            "tournament": "TOURNAMENT_SIM_V1 using PREMATCH_V1",
            "tournament_adapter": "disabled",
        },
    }


@app.get("/api/upcoming")
def upcoming(days: int = Query(3, ge=1, le=7)) -> dict[str, Any]:
    try:
        matches = on_demand_service().upcoming(days=days)
    except ProviderError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return {
        "matches": matches,
        "days": days,
        "provider": "espn_public",
        "notice": "Listing only: no player refresh or model inference has run.",
    }


@app.get("/api/upcoming/{external_match_id:path}")
def upcoming_match(external_match_id: str) -> dict[str, Any]:
    try:
        return on_demand_service().match(external_match_id)
    except KeyError as error:
        raise _not_found(error) from error
    except ProviderError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@app.post("/api/upcoming/{external_match_id:path}/refresh-prediction")
def refresh_upcoming_prediction(external_match_id: str) -> dict[str, Any]:
    try:
        return on_demand_service().refresh_prediction(external_match_id)
    except KeyError as error:
        raise _not_found(error) from error
    except (IdentityResolutionError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except ProviderError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@app.get("/api/tournaments")
def tournaments(limit: int = Query(24, ge=1, le=100)) -> list[dict[str, Any]]:
    return service().provider.get_tournaments(limit=limit)


@app.get("/api/tournaments/{tournament_id}")
def tournament(tournament_id: str) -> dict[str, Any]:
    try:
        return service().provider.get_tournament(tournament_id)
    except KeyError as error:
        raise _not_found(error) from error


@app.get("/api/tournaments/{tournament_id}/bracket")
def bracket(tournament_id: str) -> dict[str, Any]:
    product = service()
    try:
        rounds = product.provider.get_bracket(tournament_id)
    except KeyError as error:
        raise _not_found(error) from error
    state = product.store.replay_state() or {}
    completed = (
        int(state["completed_round_order"])
        if state.get("tournament_id") == tournament_id
        else -1
    )
    for item in rounds:
        if item["round_order"] <= completed:
            item["status"] = "completed"
        elif item["round_order"] == min(
            (
                candidate["round_order"]
                for candidate in rounds
                if candidate["round_order"] > completed
            ),
            default=999,
        ):
            item["status"] = "current"
        else:
            item["status"] = "upcoming"
    return {"tournament_id": tournament_id, "completed_round_order": completed, "rounds": rounds}


@app.get("/api/tournaments/{tournament_id}/predictions")
def tournament_predictions(
    tournament_id: str,
    simulations: int = Query(2_000, ge=200, le=20_000),
) -> dict[str, Any]:
    product = service()
    state = product.store.replay_state() or {}
    completed = (
        int(state["completed_round_order"])
        if state.get("tournament_id") == tournament_id
        else -1
    )
    try:
        return product.tournament.probabilities(
            tournament_id, completed, n_simulations=simulations
        )
    except KeyError as error:
        raise _not_found(error) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/matches/{match_id}")
def match(match_id: str) -> dict[str, Any]:
    if match_id.startswith("espn:"):
        try:
            return on_demand_service().match_detail(match_id)
        except KeyError as error:
            raise _not_found(error) from error
        except ProviderError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
    try:
        return service().provider.get_match(match_id)
    except KeyError as error:
        raise _not_found(error) from error


@app.get("/api/matches/{match_id}/prediction")
def match_prediction(match_id: str) -> dict[str, Any]:
    if match_id.startswith("espn:"):
        prediction = on_demand_service().cached_prediction(match_id)
        if prediction is None:
            raise HTTPException(status_code=404, detail="Prediction not calculated")
        return prediction
    try:
        return service().prematch.prediction(match_id)
    except KeyError as error:
        raise _not_found(error) from error


@app.post("/api/matches/{match_id}/refresh-prediction")
def refresh_match_prediction(match_id: str) -> dict[str, Any]:
    if not match_id.startswith("espn:"):
        raise HTTPException(
            status_code=405,
            detail="Historical predictions are immutable and cannot be refreshed.",
        )
    try:
        return on_demand_service().refresh_prediction(match_id)
    except KeyError as error:
        raise _not_found(error) from error
    except (IdentityResolutionError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except ProviderError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@app.get("/api/matches/{match_id}/features")
def match_features(match_id: str) -> dict[str, Any]:
    prediction = on_demand_service().cached_prediction(match_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not calculated")
    return {
        "match_id": match_id,
        "model_version": prediction["model_version"],
        "feature_snapshot_hash": prediction["feature_snapshot_hash"],
        "feature_count": prediction.get("feature_count", len(prediction.get("features", {}))),
        "features": prediction.get("features", {}),
        "player_metrics": prediction.get("player_metrics", {}),
    }


@app.get("/api/matches/{match_id}/prediction-history")
def match_prediction_history(match_id: str) -> dict[str, Any]:
    history = on_demand_service().prediction_history(match_id)
    return {
        "match_id": match_id,
        "history": [
            {
                "prediction_created_at": item["prediction_created_at"],
                "probability": item["probability"],
                "data_quality": item.get("data_quality"),
                "feature_snapshot_hash": item["feature_snapshot_hash"],
            }
            for item in history
        ],
    }


@app.get("/api/matches/{match_id}/live")
def live_match(match_id: str, point: int = Query(0, ge=0)) -> dict[str, Any]:
    try:
        return service().live.prediction(match_id, point)
    except KeyError as error:
        raise _not_found(error) from error


@app.get("/api/matches/{match_id}/live/history")
def live_history(match_id: str, through_point: int = Query(0, ge=0)) -> dict[str, Any]:
    try:
        return {
            "match_id": match_id,
            "history": service().live.history(match_id, through_point),
        }
    except KeyError as error:
        raise _not_found(error) from error


@app.get("/api/players/{player_id}")
def player(player_id: str) -> dict[str, Any]:
    timeline = service().provider.timeline
    matches = timeline.loc[
        timeline["player_A_id"].eq(player_id) | timeline["player_B_id"].eq(player_id)
    ].sort_values("date", ascending=False)
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Unknown player: {player_id}")
    latest = matches.iloc[0]
    side = "A" if latest["player_A_id"] == player_id else "B"
    return {
        "id": player_id,
        "name": str(latest[f"player_{side}_name"]),
        "latest_rank": _json_value(latest[f"rank_{side}"]),
        "matches_in_dataset": len(matches),
        "recent_matches": [
            service().provider.get_match(str(match_id))
            for match_id in matches.head(10)["match_id"]
        ],
    }


@app.get("/api/models")
def models() -> list[dict[str, Any]]:
    return service().model_registry()


@app.get("/api/admin/performance")
def performance() -> dict[str, Any]:
    return service().performance()


@app.get("/api/replay")
def replay() -> dict[str, Any]:
    return service().replay_state()


@app.post("/api/replay/select")
def replay_select(selection: ReplaySelection) -> dict[str, Any]:
    try:
        return service().select_replay(selection.tournament_id)
    except KeyError as error:
        raise _not_found(error) from error


@app.post("/api/replay/reset")
def replay_reset() -> dict[str, Any]:
    state = service().store.replay_state() or {}
    return service().select_replay(state["tournament_id"])


@app.post("/api/replay/advance-round")
def replay_advance_round() -> dict[str, Any]:
    return service().advance_round()


FRONTEND_DIR = Path(PROJECT_ROOT) / "frontend"


@app.get("/upcoming", include_in_schema=False)
def upcoming_page() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/matches/{match_id:path}", include_in_schema=False)
def match_page(match_id: str) -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
