from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tennis_analytics.paths import EXTERNAL_DIR, PROCESSED_DIR, PROJECT_ROOT, REPORTS_DIR
from tennis_analytics.preprocessing.parsing import normalize_name
from tennis_analytics.shots.encoding import ShotEvent, direction_relation, parse_point

MCP_DIR = EXTERNAL_DIR / "match_charting_project"
MATCHES_PATH = MCP_DIR / "charting-m-matches.csv"
POINT_GLOB = "charting-m-points-*.csv"
EVENTS_PATH = PROCESSED_DIR / "shot_events_v1.parquet"
POINTS_PATH = PROCESSED_DIR / "shot_points_v1.parquet"
CHECKPOINTS_PATH = PROCESSED_DIR / "shot_level" / "shot_checkpoints_v1.parquet"
AUDIT_JSON_PATH = PROCESSED_DIR / "shot_level" / "build_audit.json"

EVENT_SCHEMA = pa.schema(
    [
        ("match_id", pa.string()),
        ("point_id", pa.string()),
        ("date", pa.date32()),
        ("year", pa.int16()),
        ("gender", pa.string()),
        ("surface", pa.string()),
        ("tournament", pa.string()),
        ("point_number", pa.int32()),
        ("shot_index", pa.int16()),
        ("active_shot_index", pa.int16()),
        ("hitter_role", pa.string()),
        ("hitter_id", pa.string()),
        ("hitter_name", pa.string()),
        ("is_serve", pa.bool_()),
        ("serve_number", pa.int8()),
        ("serve_direction", pa.string()),
        ("shot_code", pa.string()),
        ("shot_type", pa.string()),
        ("wing", pa.string()),
        ("direction", pa.string()),
        ("direction_relation", pa.string()),
        ("depth", pa.string()),
        ("position", pa.string()),
        ("is_approach", pa.bool_()),
        ("at_net", pa.bool_()),
        ("net_cord", pa.bool_()),
        ("stop_volley", pa.bool_()),
        ("is_shank", pa.bool_()),
        ("outcome", pa.string()),
        ("error_type", pa.string()),
        ("is_terminal", pa.bool_()),
        ("won_point", pa.bool_()),
        ("raw_token", pa.string()),
        ("has_explicit_unknown", pa.bool_()),
        ("tactical_role", pa.string()),
    ]
)

POINT_SCHEMA = pa.schema(
    [
        ("match_id", pa.string()),
        ("point_id", pa.string()),
        ("date", pa.date32()),
        ("year", pa.int16()),
        ("gender", pa.string()),
        ("surface", pa.string()),
        ("tournament", pa.string()),
        ("round", pa.string()),
        ("player1_id", pa.string()),
        ("player2_id", pa.string()),
        ("player1_name", pa.string()),
        ("player2_name", pa.string()),
        ("player1_hand", pa.string()),
        ("player2_hand", pa.string()),
        ("point_number", pa.int32()),
        ("sets_player1", pa.int8()),
        ("sets_player2", pa.int8()),
        ("games_player1", pa.int8()),
        ("games_player2", pa.int8()),
        ("score_raw", pa.string()),
        ("server_number", pa.int8()),
        ("server_id", pa.string()),
        ("receiver_id", pa.string()),
        ("point_winner_number", pa.int8()),
        ("server_won_point", pa.int8()),
        ("active_serve_number", pa.int8()),
        ("terminal_outcome", pa.string()),
        ("final_shot_count", pa.int16()),
        ("final_active_shot_count", pa.int16()),
        ("final_rally_length", pa.int16()),
        ("is_tiebreak", pa.bool_()),
        ("is_break_point", pa.bool_()),
        ("raw_first", pa.string()),
        ("raw_second", pa.string()),
        ("sequence_text", pa.string()),
        ("sequence_tokens", pa.string()),
        ("sequence_json", pa.string()),
    ]
)

CHECKPOINT_SCHEMA = pa.schema(
    [
        ("match_id", pa.string()),
        ("point_id", pa.string()),
        ("date", pa.date32()),
        ("year", pa.int16()),
        ("surface", pa.string()),
        ("point_number", pa.int32()),
        ("server_id", pa.string()),
        ("receiver_id", pa.string()),
        ("server_won_point", pa.int8()),
        ("checkpoint", pa.int8()),
        ("observed_shots", pa.int8()),
        ("serve_number", pa.int8()),
        ("serve_direction", pa.string()),
        ("return_type", pa.string()),
        ("return_direction", pa.string()),
        ("return_depth", pa.string()),
        ("last_shot_type", pa.string()),
        ("last_shot_direction", pa.string()),
        ("last_shot_relation", pa.string()),
        ("next_hitter_role", pa.string()),
        ("is_tiebreak", pa.bool_()),
        ("is_break_point", pa.bool_()),
    ]
)


def _clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _integer(value: Any) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _player_id(name: str) -> str:
    normalized = normalize_name(name) or "unknown"
    digest = hashlib.sha256(normalized.encode()).hexdigest()[:16]
    return f"mcp:name:{digest}"


def _surface(value: Any) -> str:
    text = _clean(value).title()
    return text if text in {"Hard", "Clay", "Grass", "Carpet"} else "Unknown"


def _score_flags(score: str, server_number: int) -> tuple[bool, bool]:
    parts = [
        {"A": "AD", "ADV": "AD"}.get(token, token)
        for token in score.upper().split("-")
    ]
    if len(parts) != 2:
        return False, False
    regular = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}
    if parts[0] not in regular or parts[1] not in regular:
        return True, False
    server_points = regular[parts[server_number - 1]]
    receiver_points = regular[parts[1 if server_number == 1 else 0]]
    return False, receiver_points >= 3 and receiver_points > server_points


def _coarse_token(event: ShotEvent, hand: str | None) -> str:
    if event.is_serve:
        return f"serve_{event.serve_number}:{event.serve_direction}"
    relation = direction_relation(event, hand) or "unknown"
    return f"{event.shot_code}:{relation}"


def _tactical_role(event: ShotEvent) -> str:
    if event.active_shot_index == 1:
        return "serve"
    if event.active_shot_index == 2:
        return "return"
    if event.active_shot_index == 3:
        return "serve_plus_1"
    if event.active_shot_index == 4:
        return "return_plus_1"
    if event.active_shot_index is None:
        return "first_serve_fault"
    return "rally"


def _table(rows: list[dict[str, Any]], schema: pa.Schema) -> pa.Table:
    columns = {field.name: [row.get(field.name) for row in rows] for field in schema}
    return pa.Table.from_pydict(columns, schema=schema)


class _Writer:
    def __init__(self, path: Path, schema: pa.Schema) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = pq.ParquetWriter(path, schema, compression="zstd")
        self.schema = schema
        self.rows = 0

    def write(self, rows: list[dict[str, Any]]) -> None:
        if rows:
            self.writer.write_table(_table(rows, self.schema))
            self.rows += len(rows)

    def close(self) -> None:
        self.writer.close()


def _freeze_inputs() -> dict[str, Any]:
    selected = [
        PROJECT_ROOT / "models" / "prematch_v1" / "model.joblib",
        PROJECT_ROOT / "models" / "point_v2" / "model.joblib",
        PROJECT_ROOT / "models" / "live_v3" / "model.joblib",
        PROJECT_ROOT / "models" / "live_v3" / "metadata.json",
        REPORTS_DIR / "phase8c_report.md",
        REPORTS_DIR / "phase8c_test_comparison.csv",
    ]
    records = []
    for path in selected:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": digest,
                "bytes": path.stat().st_size,
            }
        )
    payload = {
        "purpose": "Phase 9 read-only freeze of official Phase 8C architecture",
        "files": records,
    }
    target = PROJECT_ROOT / "models" / "shot_v1" / "phase8c_freeze.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _checkpoint_rows(
    base: dict[str, Any],
    active_events: list[ShotEvent],
    hand_by_role: dict[str, str],
) -> Iterable[dict[str, Any]]:
    for checkpoint in (0, 1, 2, 3, 5):
        if checkpoint > len(active_events):
            continue
        observed = active_events[:checkpoint]
        if observed and observed[-1].is_terminal:
            continue
        serve = active_events[0] if active_events else None
        returned = active_events[1] if len(active_events) > 1 else None
        last = observed[-1] if observed else None
        next_role = "server" if checkpoint % 2 == 0 else "receiver"
        yield {
            **base,
            "checkpoint": checkpoint,
            "observed_shots": checkpoint,
            "serve_number": serve.serve_number if checkpoint >= 1 and serve else None,
            "serve_direction": serve.serve_direction if checkpoint >= 1 and serve else None,
            "return_type": returned.shot_type if checkpoint >= 2 and returned else None,
            "return_direction": returned.direction if checkpoint >= 2 and returned else None,
            "return_depth": returned.depth if checkpoint >= 2 and returned else None,
            "last_shot_type": last.shot_type if last else None,
            "last_shot_direction": last.direction if last else None,
            "last_shot_relation": (
                direction_relation(last, hand_by_role.get(last.hitter)) if last else None
            ),
            "next_hitter_role": next_role,
        }


def build_shot_datasets(chunk_size: int = 10_000) -> dict[str, Any]:
    """Build canonical shot/point/checkpoint tables and a machine-readable audit."""
    _freeze_inputs()
    metadata = pd.read_csv(MATCHES_PATH, dtype=str)
    duplicate_meta = int(metadata["match_id"].duplicated().sum())
    metadata["_quality"] = (
        metadata["Surface"].isin(["Hard", "Clay", "Grass", "Carpet"]).astype(int) * 10
        + metadata["Player 1"].notna().astype(int)
        + metadata["Player 2"].notna().astype(int)
    )
    metadata = (
        metadata.sort_values("_quality", ascending=False)
        .drop_duplicates("match_id", keep="first")
        .drop(columns="_quality")
    )
    match_lookup = metadata.set_index("match_id").to_dict("index")

    event_writer = _Writer(EVENTS_PATH, EVENT_SCHEMA)
    point_writer = _Writer(POINTS_PATH, POINT_SCHEMA)
    checkpoint_writer = _Writer(CHECKPOINTS_PATH, CHECKPOINT_SCHEMA)
    audit: dict[str, Any] = {
        "metadata_rows": len(metadata),
        "metadata_duplicate_ids": duplicate_meta,
        "source_files": {},
        "raw_points": 0,
        "valid_points": 0,
        "excluded_points": 0,
        "valid_events": 0,
        "reason_counts": Counter(),
        "reason_examples": defaultdict(list),
        "terminal_counts": Counter(),
        "surface_points": Counter(),
        "year_points": Counter(),
        "unknown_code_events": 0,
        "direction_known_events": 0,
        "direction_eligible_events": 0,
        "return_depth_known": 0,
        "return_depth_eligible": 0,
        "winner_mismatches": 0,
        "missing_metadata": 0,
    }
    event_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []

    try:
        for source_path in sorted(MCP_DIR.glob(POINT_GLOB)):
            file_points = 0
            for frame in pd.read_csv(source_path, dtype=str, chunksize=chunk_size):
                frame = frame.rename(columns={"Pt": "point_number", "1st": "first", "2nd": "second"})
                for row in frame.itertuples(index=False):
                    record = row._asdict()
                    audit["raw_points"] += 1
                    file_points += 1
                    match_id = _clean(record.get("match_id"))
                    point_number = _integer(record.get("point_number"))
                    parsed = parse_point(record.get("first"), record.get("second"))
                    meta = match_lookup.get(match_id)
                    reason = parsed.reason
                    if meta is None:
                        reason = "missing_metadata"
                        audit["missing_metadata"] += 1
                    server_number = _integer(record.get("Svr"))
                    winner_number = _integer(record.get("PtWinner"))
                    if reason is None and (server_number not in (1, 2) or winner_number not in (1, 2)):
                        reason = "invalid_server_or_winner"
                    derived_number = None
                    if reason is None:
                        derived_number = (
                            server_number
                            if parsed.derived_winner == "server"
                            else 3 - int(server_number)
                        )
                        if derived_number != winner_number:
                            reason = "winner_mismatch"
                            audit["winner_mismatches"] += 1
                    if reason is not None:
                        audit["excluded_points"] += 1
                        audit["reason_counts"][reason] += 1
                        examples = audit["reason_examples"][reason]
                        if len(examples) < 5:
                            examples.append(
                                {
                                    "match_id": match_id,
                                    "point": point_number,
                                    "first": _clean(record.get("first")),
                                    "second": _clean(record.get("second")),
                                }
                            )
                        continue

                    assert meta is not None and server_number is not None and winner_number is not None
                    player_names = {1: _clean(meta.get("Player 1")), 2: _clean(meta.get("Player 2"))}
                    player_ids = {number: _player_id(name) for number, name in player_names.items()}
                    player_hands = {1: _clean(meta.get("Pl 1 hand")) or "U", 2: _clean(meta.get("Pl 2 hand")) or "U"}
                    role_number = {"server": server_number, "receiver": 3 - server_number}
                    hand_by_role = {role: player_hands[number] for role, number in role_number.items()}
                    raw_date = _clean(meta.get("Date"))
                    date = pd.to_datetime(raw_date, format="%Y%m%d", errors="coerce")
                    if pd.isna(date):
                        audit["excluded_points"] += 1
                        audit["reason_counts"]["invalid_date"] += 1
                        continue
                    date_value = date.date()
                    year = int(date.year)
                    surface = _surface(meta.get("Surface"))
                    point_id = f"{match_id}:{point_number}"
                    server_won = int(winner_number == server_number)
                    score_raw = _clean(record.get("Pts"))
                    is_tiebreak, is_break_point = _score_flags(score_raw, server_number)
                    active_events = [event for event in parsed.events if event.active_shot_index is not None]
                    tokens = [
                        _coarse_token(event, hand_by_role[event.hitter]) for event in active_events
                    ]
                    sequence_json = []
                    for event in parsed.events:
                        number = role_number[event.hitter]
                        enriched = {
                            **asdict(event),
                            "hitter_number": number,
                            "hitter_id": player_ids[number],
                            "direction_relation": direction_relation(event, player_hands[number]),
                            "tactical_role": _tactical_role(event),
                        }
                        sequence_json.append(enriched)
                        if event.has_explicit_unknown:
                            audit["unknown_code_events"] += 1
                        if not event.is_serve:
                            audit["direction_eligible_events"] += 1
                            if event.direction not in (None, "unknown"):
                                audit["direction_known_events"] += 1
                            if event.active_shot_index == 2:
                                audit["return_depth_eligible"] += 1
                                if event.depth not in (None, "unknown"):
                                    audit["return_depth_known"] += 1
                        event_rows.append(
                            {
                                "match_id": match_id,
                                "point_id": point_id,
                                "date": date_value,
                                "year": year,
                                "gender": "men",
                                "surface": surface,
                                "tournament": _clean(meta.get("Tournament")),
                                "point_number": point_number,
                                "shot_index": event.shot_index,
                                "active_shot_index": event.active_shot_index,
                                "hitter_role": event.hitter,
                                "hitter_id": player_ids[number],
                                "hitter_name": player_names[number],
                                "is_serve": event.is_serve,
                                "serve_number": event.serve_number,
                                "serve_direction": event.serve_direction,
                                "shot_code": event.shot_code,
                                "shot_type": event.shot_type,
                                "wing": event.wing,
                                "direction": event.direction,
                                "direction_relation": direction_relation(event, player_hands[number]),
                                "depth": event.depth,
                                "position": event.position,
                                "is_approach": event.is_approach,
                                "at_net": event.at_net,
                                "net_cord": event.net_cord,
                                "stop_volley": event.stop_volley,
                                "is_shank": event.is_shank,
                                "outcome": event.outcome,
                                "error_type": event.error_type,
                                "is_terminal": event.is_terminal,
                                "won_point": number == winner_number,
                                "raw_token": event.raw_token,
                                "has_explicit_unknown": event.has_explicit_unknown,
                                "tactical_role": _tactical_role(event),
                            }
                        )
                    rally_length = sum(
                        event.outcome not in {"fault", "forced_error", "unforced_error"}
                        for event in active_events
                    )
                    point_rows.append(
                        {
                            "match_id": match_id,
                            "point_id": point_id,
                            "date": date_value,
                            "year": year,
                            "gender": "men",
                            "surface": surface,
                            "tournament": _clean(meta.get("Tournament")),
                            "round": _clean(meta.get("Round")),
                            "player1_id": player_ids[1],
                            "player2_id": player_ids[2],
                            "player1_name": player_names[1],
                            "player2_name": player_names[2],
                            "player1_hand": player_hands[1],
                            "player2_hand": player_hands[2],
                            "point_number": point_number,
                            "sets_player1": _integer(record.get("Set1")),
                            "sets_player2": _integer(record.get("Set2")),
                            "games_player1": _integer(record.get("Gm1")),
                            "games_player2": _integer(record.get("Gm2")),
                            "score_raw": score_raw,
                            "server_number": server_number,
                            "server_id": player_ids[server_number],
                            "receiver_id": player_ids[3 - server_number],
                            "point_winner_number": winner_number,
                            "server_won_point": server_won,
                            "active_serve_number": parsed.active_serve_number,
                            "terminal_outcome": parsed.terminal_outcome,
                            "final_shot_count": len(parsed.events),
                            "final_active_shot_count": len(active_events),
                            "final_rally_length": rally_length,
                            "is_tiebreak": is_tiebreak,
                            "is_break_point": is_break_point,
                            "raw_first": _clean(record.get("first")),
                            "raw_second": _clean(record.get("second")),
                            "sequence_text": " ".join(event.raw_token for event in parsed.events),
                            "sequence_tokens": " ".join(tokens),
                            "sequence_json": json.dumps(sequence_json, separators=(",", ":")),
                        }
                    )
                    checkpoint_base = {
                        "match_id": match_id,
                        "point_id": point_id,
                        "date": date_value,
                        "year": year,
                        "surface": surface,
                        "point_number": point_number,
                        "server_id": player_ids[server_number],
                        "receiver_id": player_ids[3 - server_number],
                        "server_won_point": server_won,
                        "is_tiebreak": is_tiebreak,
                        "is_break_point": is_break_point,
                    }
                    checkpoint_rows.extend(
                        _checkpoint_rows(checkpoint_base, active_events, hand_by_role)
                    )
                    audit["valid_points"] += 1
                    audit["valid_events"] += len(parsed.events)
                    audit["terminal_counts"][str(parsed.terminal_outcome)] += 1
                    audit["surface_points"][surface] += 1
                    audit["year_points"][str(year)] += 1

                event_writer.write(event_rows)
                point_writer.write(point_rows)
                checkpoint_writer.write(checkpoint_rows)
                event_rows.clear()
                point_rows.clear()
                checkpoint_rows.clear()
            audit["source_files"][source_path.name] = file_points
    finally:
        event_writer.close()
        point_writer.close()
        checkpoint_writer.close()

    audit["event_rows_written"] = event_writer.rows
    audit["point_rows_written"] = point_writer.rows
    audit["checkpoint_rows_written"] = checkpoint_writer.rows
    for key in ("reason_counts", "terminal_counts", "surface_points", "year_points"):
        audit[key] = dict(sorted(audit[key].items()))
    audit["reason_examples"] = dict(audit["reason_examples"])
    AUDIT_JSON_PATH.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    return audit
