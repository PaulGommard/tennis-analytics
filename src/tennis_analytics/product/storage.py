from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tennis_analytics.paths import DATA_DIR

DEFAULT_DATABASE = DATA_DIR / "product" / "tennis_product.db"


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS players (
    player_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    country_code TEXT,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tournaments (
    tournament_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    start_date TEXT NOT NULL,
    surface TEXT,
    level TEXT,
    provider TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tournament_entries (
    tournament_id TEXT NOT NULL,
    player_id TEXT NOT NULL,
    seed INTEGER,
    status TEXT NOT NULL DEFAULT 'active',
    PRIMARY KEY (tournament_id, player_id)
);
CREATE TABLE IF NOT EXISTS matches (
    match_id TEXT PRIMARY KEY,
    tournament_id TEXT NOT NULL,
    round TEXT NOT NULL,
    player_a_id TEXT NOT NULL,
    player_b_id TEXT NOT NULL,
    status TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS match_scores (
    match_id TEXT NOT NULL,
    point_index INTEGER NOT NULL,
    score_json TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    PRIMARY KEY (match_id, point_index)
);
CREATE TABLE IF NOT EXISTS model_versions (
    model_version TEXT PRIMARY KEY,
    role TEXT NOT NULL,
    official INTEGER NOT NULL,
    metadata_json TEXT NOT NULL,
    registered_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS match_predictions (
    prediction_id TEXT PRIMARY KEY,
    match_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    prediction_created_at TEXT NOT NULL,
    probability_a REAL NOT NULL,
    probability_b REAL NOT NULL,
    feature_snapshot_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    result_a INTEGER,
    UNIQUE (match_id, model_version, feature_snapshot_hash)
);
CREATE TABLE IF NOT EXISTS live_states (
    match_id TEXT NOT NULL,
    point_index INTEGER NOT NULL,
    model_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (match_id, point_index, model_version)
);
CREATE TABLE IF NOT EXISTS tournament_predictions (
    tournament_id TEXT NOT NULL,
    snapshot_key TEXT NOT NULL,
    model_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (tournament_id, snapshot_key, model_version)
);
CREATE TABLE IF NOT EXISTS replay_sessions (
    session_id TEXT PRIMARY KEY,
    tournament_id TEXT NOT NULL,
    completed_round_order INTEGER NOT NULL DEFAULT -1,
    selected_match_id TEXT,
    point_index INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS external_player_mappings (
    provider TEXT NOT NULL,
    external_player_id TEXT NOT NULL,
    internal_player_id TEXT NOT NULL,
    external_name TEXT NOT NULL,
    resolution TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (provider, external_player_id)
);
CREATE TABLE IF NOT EXISTS player_runtime_states (
    provider TEXT NOT NULL,
    external_player_id TEXT NOT NULL,
    internal_player_id TEXT NOT NULL,
    state_as_of TEXT NOT NULL,
    refreshed_at TEXT NOT NULL,
    quality TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (provider, external_player_id)
);
CREATE TABLE IF NOT EXISTS upcoming_matches (
    external_match_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    starts_at TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ProductStore:
    def __init__(self, path: Path = DEFAULT_DATABASE) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def register_models(self, models: list[dict[str, Any]]) -> None:
        with self.connect() as connection:
            for model in models:
                connection.execute(
                    """
                    INSERT INTO model_versions
                        (model_version, role, official, metadata_json, registered_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(model_version) DO UPDATE SET
                        role=excluded.role,
                        official=excluded.official,
                        metadata_json=excluded.metadata_json
                    """,
                    (
                        model["model_version"],
                        model["role"],
                        int(model["official"]),
                        json.dumps(model.get("metadata", {}), sort_keys=True),
                        _now(),
                    ),
                )

    def store_prediction(self, match_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        prediction_id = hashlib.sha256(
            f"{match_id}|{payload['model_version']}|{payload['feature_snapshot_hash']}".encode()
        ).hexdigest()[:24]
        values = (
            prediction_id,
            match_id,
            payload["model_version"],
            payload["prediction_created_at"],
            payload["probability"]["player_a"],
            payload["probability"]["player_b"],
            payload["feature_snapshot_hash"],
            canonical,
        )
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT payload_json FROM match_predictions WHERE prediction_id = ?",
                (prediction_id,),
            ).fetchone()
            if existing is not None:
                stored = json.loads(existing["payload_json"])
                immutable = ("model_version", "feature_snapshot_hash", "probability")
                if any(stored[key] != payload[key] for key in immutable):
                    raise ValueError("Immutable prediction collision")
                # Permit additive response-schema enrichment while preserving the
                # immutable prediction identity, timestamp and probability.
                enriched = {**stored, **payload, "prediction_created_at": stored["prediction_created_at"]}
                connection.execute(
                    "UPDATE match_predictions SET payload_json = ? WHERE prediction_id = ?",
                    (json.dumps(enriched, sort_keys=True, separators=(",", ":")), prediction_id),
                )
                return enriched
            connection.execute(
                """
                INSERT INTO match_predictions
                    (prediction_id, match_id, model_version, prediction_created_at,
                     probability_a, probability_b, feature_snapshot_hash, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
        return payload

    def get_prediction(self, match_id: str, model_version: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json FROM match_predictions
                WHERE match_id = ? AND model_version = ?
                ORDER BY prediction_created_at DESC LIMIT 1
                """,
                (match_id, model_version),
            ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def get_prediction_history(
        self, match_id: str, model_version: str, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM match_predictions
                WHERE match_id = ? AND model_version = ?
                ORDER BY prediction_created_at DESC LIMIT ?
                """,
                (match_id, model_version, limit),
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def store_live_state(
        self, match_id: str, point_index: int, model_version: str, payload: dict[str, Any]
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO live_states
                    (match_id, point_index, model_version, created_at, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (match_id, point_index, model_version, _now(), json.dumps(payload)),
            )

    def store_tournament_prediction(
        self,
        tournament_id: str,
        snapshot_key: str,
        model_version: str,
        payload: dict[str, Any],
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO tournament_predictions
                    (tournament_id, snapshot_key, model_version, created_at, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (tournament_id, snapshot_key, model_version, _now(), json.dumps(payload)),
            )

    def get_tournament_prediction(
        self, tournament_id: str, snapshot_key: str, model_version: str
    ) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json FROM tournament_predictions
                WHERE tournament_id = ? AND snapshot_key = ? AND model_version = ?
                """,
                (tournament_id, snapshot_key, model_version),
            ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def replay_state(self, session_id: str = "local") -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM replay_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return dict(row) if row else None

    def set_replay_state(
        self,
        tournament_id: str,
        *,
        completed_round_order: int = -1,
        selected_match_id: str | None = None,
        point_index: int = 0,
        session_id: str = "local",
    ) -> dict[str, Any]:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO replay_sessions
                    (session_id, tournament_id, completed_round_order,
                     selected_match_id, point_index, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    tournament_id=excluded.tournament_id,
                    completed_round_order=excluded.completed_round_order,
                    selected_match_id=excluded.selected_match_id,
                    point_index=excluded.point_index,
                    updated_at=excluded.updated_at
                """,
                (
                    session_id,
                    tournament_id,
                    completed_round_order,
                    selected_match_id,
                    point_index,
                    _now(),
                ),
            )
        return self.replay_state(session_id) or {}

    def store_upcoming(self, matches: list[dict[str, Any]]) -> None:
        with self.connect() as connection:
            for match in matches:
                connection.execute(
                    """
                    INSERT INTO upcoming_matches
                        (external_match_id, provider, starts_at, fetched_at, payload_json)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(external_match_id) DO UPDATE SET
                        starts_at=excluded.starts_at,
                        fetched_at=excluded.fetched_at,
                        payload_json=excluded.payload_json
                    """,
                    (
                        match["external_match_id"], match["provider"], match["starts_at"],
                        _now(), json.dumps(match, sort_keys=True),
                    ),
                )

    def get_upcoming(self, external_match_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM upcoming_matches WHERE external_match_id = ?",
                (external_match_id,),
            ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def store_external_mapping(
        self, provider: str, external_id: str, internal_id: str, name: str, resolution: str
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO external_player_mappings VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(provider, external_player_id) DO UPDATE SET
                    internal_player_id=excluded.internal_player_id,
                    external_name=excluded.external_name,
                    resolution=excluded.resolution,
                    updated_at=excluded.updated_at
                """,
                (provider, external_id, internal_id, name, resolution, _now()),
            )

    def get_external_mapping(self, provider: str, external_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM external_player_mappings WHERE provider=? AND external_player_id=?",
                (provider, external_id),
            ).fetchone()
        return dict(row) if row else None

    def store_runtime_state(self, state: dict[str, Any]) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO player_runtime_states VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(provider, external_player_id) DO UPDATE SET
                    internal_player_id=excluded.internal_player_id,
                    state_as_of=excluded.state_as_of,
                    refreshed_at=excluded.refreshed_at,
                    quality=excluded.quality,
                    payload_json=excluded.payload_json
                """,
                (
                    state["provider"], state["external_player_id"], state["internal_player_id"],
                    state["state_as_of"], state["refreshed_at"], state["quality"],
                    json.dumps(state, sort_keys=True),
                ),
            )
