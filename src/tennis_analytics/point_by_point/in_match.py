from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tennis_analytics.modeling.point_model import MODEL_DIR as V1_MODEL_DIR
from tennis_analytics.paths import PROCESSED_DIR, PROJECT_ROOT, REPORTS_DIR

V1_PARTS_DIR = PROCESSED_DIR / "point_features_v1_parts"
V2_PARTS_DIR = PROCESSED_DIR / "point_features_v2_parts"
V2_PATH = PROCESSED_DIR / "point_features_v2.parquet"
V2_MODEL_DIR = PROJECT_ROOT / "models" / "point_v2"
PRIOR_WEIGHTS = (10, 20, 50, 100)
POINT_WINDOWS = (3, 5, 10, 20)
SERVICE_WINDOWS = (3, 5, 10)
CLUTCH_PRIOR_WEIGHT = 200


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_phase7b_freeze() -> dict[str, Any]:
    paths = [
        PROCESSED_DIR / "point_features_v1.parquet",
        V1_MODEL_DIR / "model.joblib",
        V1_MODEL_DIR / "metadata.json",
        V1_MODEL_DIR / "feature_schema.json",
        V1_MODEL_DIR / "config_snapshot.json",
        REPORTS_DIR / "phase7b_report.md",
        REPORTS_DIR / "point_model_audit.md",
        REPORTS_DIR / "live_probability_validation.md",
    ]
    snapshot = {
        "created_at": datetime.now(UTC).isoformat(),
        "policy": "Phase 7B inputs are read-only baselines for Phase 8",
        "artifacts": [
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in paths
        ],
    }
    V2_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (V2_MODEL_DIR / "phase7b_freeze.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return snapshot


def verify_phase7b_freeze() -> list[str]:
    snapshot_path = V2_MODEL_DIR / "phase7b_freeze.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    mismatches = []
    for artifact in snapshot["artifacts"]:
        path = PROJECT_ROOT / artifact["path"]
        if (
            not path.exists()
            or path.stat().st_size != artifact["bytes"]
            or _sha256(path) != artifact["sha256"]
        ):
            mismatches.append(str(artifact["path"]))
    return mismatches


def _recent_rate(values: list[int], window: int) -> float:
    if not values:
        return np.nan
    recent = values[-window:]
    return float(sum(recent) / len(recent))


def _empty_arrays(length: int) -> dict[str, np.ndarray]:
    names = ["points_played_so_far"]
    for side in ("A", "B"):
        names.extend(
            [
                f"{side}_points_won_match",
                f"{side}_service_points_played_match",
                f"{side}_service_points_won_match",
                f"{side}_return_points_played_match",
                f"{side}_return_points_won_match",
                f"{side}_service_points_won_pct_so_far",
                f"{side}_return_points_won_pct_so_far",
                f"{side}_aces_so_far",
                f"{side}_double_faults_so_far",
                f"{side}_break_points_faced_so_far",
                f"{side}_break_points_saved_so_far",
                f"{side}_games_won_last_3",
                f"{side}_break_recently",
                f"{side}_was_broken_recently",
                f"{side}_consecutive_games_won",
                f"{side}_previous_set_won",
                f"{side}_previous_set_margin",
                f"{side}_won_previous_set_tiebreak",
            ]
        )
        names.extend(f"{side}_points_won_last_{window}" for window in POINT_WINDOWS)
        names.extend(
            f"{side}_service_points_won_last_{window}" for window in SERVICE_WINDOWS
        )
        names.extend(f"{side}_return_points_won_last_{window}" for window in SERVICE_WINDOWS)
    return {name: np.full(length, np.nan, dtype=np.float32) for name in names}


def add_sequential_features(points: pd.DataFrame) -> pd.DataFrame:
    """Compute all match-current features from prior points only."""
    points = points.reset_index(drop=True)
    arrays = _empty_arrays(len(points))
    for _, group in points.groupby("match_id", sort=False, observed=True):
        point_wins = {"A": 0, "B": 0}
        point_hist: dict[str, list[int]] = {"A": [], "B": []}
        serve_hist: dict[str, list[int]] = {"A": [], "B": []}
        return_hist: dict[str, list[int]] = {"A": [], "B": []}
        aces = {"A": 0, "B": 0}
        double_faults = {"A": 0, "B": 0}
        break_faced = {"A": 0, "B": 0}
        break_saved = {"A": 0, "B": 0}
        games: list[tuple[str, str]] = []
        previous_set: tuple[str, int, bool] | None = None
        event_stats_available = bool(group["ace"].notna().any())
        for row in group.itertuples(index=True):
            index = int(row.Index)
            arrays["points_played_so_far"][index] = len(point_hist["A"])
            for side in ("A", "B"):
                other = "B" if side == "A" else "A"
                arrays[f"{side}_points_won_match"][index] = point_wins[side]
                arrays[f"{side}_service_points_played_match"][index] = len(serve_hist[side])
                arrays[f"{side}_service_points_won_match"][index] = sum(serve_hist[side])
                arrays[f"{side}_return_points_played_match"][index] = len(return_hist[side])
                arrays[f"{side}_return_points_won_match"][index] = sum(return_hist[side])
                arrays[f"{side}_service_points_won_pct_so_far"][index] = _recent_rate(
                    serve_hist[side], len(serve_hist[side])
                )
                arrays[f"{side}_return_points_won_pct_so_far"][index] = _recent_rate(
                    return_hist[side], len(return_hist[side])
                )
                if event_stats_available:
                    arrays[f"{side}_aces_so_far"][index] = aces[side]
                    arrays[f"{side}_double_faults_so_far"][index] = double_faults[side]
                arrays[f"{side}_break_points_faced_so_far"][index] = break_faced[side]
                arrays[f"{side}_break_points_saved_so_far"][index] = break_saved[side]
                for window in POINT_WINDOWS:
                    arrays[f"{side}_points_won_last_{window}"][index] = _recent_rate(
                        point_hist[side], window
                    )
                for window in SERVICE_WINDOWS:
                    arrays[f"{side}_service_points_won_last_{window}"][index] = _recent_rate(
                        serve_hist[side], window
                    )
                    arrays[f"{side}_return_points_won_last_{window}"][index] = _recent_rate(
                        return_hist[side], window
                    )
                last_games = games[-3:]
                arrays[f"{side}_games_won_last_3"][index] = sum(
                    winner == side for winner, _ in last_games
                )
                arrays[f"{side}_break_recently"][index] = int(
                    any(winner == side and server == other for winner, server in last_games)
                )
                arrays[f"{side}_was_broken_recently"][index] = int(
                    any(winner == other and server == side for winner, server in last_games)
                )
                streak = 0
                for winner, _ in reversed(games):
                    if winner != side:
                        break
                    streak += 1
                arrays[f"{side}_consecutive_games_won"][index] = streak
                if previous_set is not None:
                    winner, margin, was_tiebreak = previous_set
                    arrays[f"{side}_previous_set_won"][index] = int(winner == side)
                    arrays[f"{side}_previous_set_margin"][index] = (
                        margin if winner == side else -margin
                    )
                    arrays[f"{side}_won_previous_set_tiebreak"][index] = int(
                        winner == side and was_tiebreak
                    )

            winner = str(row.point_winner)
            server = str(row.server)
            receiver = "B" if server == "A" else "A"
            point_wins[winner] += 1
            point_hist["A"].append(int(winner == "A"))
            point_hist["B"].append(int(winner == "B"))
            serve_hist[server].append(int(row.server_won_point))
            return_hist[receiver].append(1 - int(row.server_won_point))
            if event_stats_available:
                aces[server] += int(bool(row.ace)) if pd.notna(row.ace) else 0
                double_faults[server] += (
                    int(bool(row.double_fault)) if pd.notna(row.double_fault) else 0
                )
            if int(row.is_break_point):
                break_faced[server] += 1
                break_saved[server] += int(row.server_won_point)
            if int(row.game_ended_after):
                games.append((winner, server))
            if int(row.set_ended_after):
                final_a = int(row.games_A) + int(winner == "A")
                final_b = int(row.games_B) + int(winner == "B")
                previous_set = (winner, abs(final_a - final_b), bool(row.is_tiebreak))

    sequential = pd.DataFrame(arrays)
    server_a = points["server"].eq("A").to_numpy()

    def orient(server_suffix: str, receiver_suffix: str | None = None) -> np.ndarray:
        receiver_suffix = receiver_suffix or server_suffix
        return np.where(
            server_a,
            sequential[f"A_{server_suffix}"],
            sequential[f"B_{server_suffix}"],
        ), np.where(
            server_a,
            sequential[f"B_{receiver_suffix}"],
            sequential[f"A_{receiver_suffix}"],
        )

    pairs = {
        "points_won_match": "points_won_match",
        "service_points_played_match": "return_points_played_match",
        "service_points_won_match": "return_points_won_match",
        "service_points_won_pct_so_far": "return_points_won_pct_so_far",
        "aces_so_far": "aces_so_far",
        "double_faults_so_far": "double_faults_so_far",
        "break_points_faced_so_far": "break_points_faced_so_far",
        "break_points_saved_so_far": "break_points_saved_so_far",
        "games_won_last_3": "games_won_last_3",
        "break_recently": "break_recently",
        "was_broken_recently": "was_broken_recently",
        "consecutive_games_won": "consecutive_games_won",
        "previous_set_won": "previous_set_won",
        "previous_set_margin": "previous_set_margin",
        "won_previous_set_tiebreak": "won_previous_set_tiebreak",
    }
    for server_suffix, receiver_suffix in pairs.items():
        server_values, receiver_values = orient(server_suffix, receiver_suffix)
        sequential[f"server_{server_suffix}"] = server_values
        sequential[f"receiver_{receiver_suffix}"] = receiver_values
    for window in POINT_WINDOWS:
        server_values, receiver_values = orient(f"points_won_last_{window}")
        sequential[f"server_points_won_last_{window}"] = server_values
        sequential[f"receiver_points_won_last_{window}"] = receiver_values
    for window in SERVICE_WINDOWS:
        server_values, _ = orient(f"service_points_won_last_{window}")
        _, receiver_values = orient(
            f"return_points_won_last_{window}", f"return_points_won_last_{window}"
        )
        sequential[f"server_service_points_won_last_{window}"] = server_values
        sequential[f"receiver_return_points_won_last_{window}"] = receiver_values

    for weight in PRIOR_WEIGHTS:
        serve_attempts = sequential["server_service_points_played_match"]
        serve_wins = sequential["server_service_points_won_match"]
        return_attempts = sequential["receiver_return_points_played_match"]
        return_wins = sequential["receiver_return_points_won_match"]
        sequential[f"dynamic_serve_w{weight}"] = (
            weight * points["server_serve_rate"].to_numpy() + serve_wins
        ) / (weight + serve_attempts)
        sequential[f"dynamic_return_w{weight}"] = (
            weight * points["receiver_return_rate"].to_numpy() + return_wins
        ) / (weight + return_attempts)
        sequential[f"serve_today_delta_w{weight}"] = (
            sequential[f"dynamic_serve_w{weight}"] - points["server_serve_rate"].to_numpy()
        )
        sequential[f"return_today_delta_w{weight}"] = (
            sequential[f"dynamic_return_w{weight}"]
            - points["receiver_return_rate"].to_numpy()
        )
    return pd.concat([points, sequential], axis=1)


def _match_clutch_contributions(
    points: pd.DataFrame, probability: np.ndarray
) -> dict[str, Any]:
    residual = points["server_won_point"].to_numpy(dtype=float) - probability
    deciding = (points["sets_A"] + points["sets_B"]).eq(points["best_of"] - 1).to_numpy()
    categories = {
        "break": points["is_break_point"].astype(bool).to_numpy(),
        "tiebreak": points["is_tiebreak"].astype(bool).to_numpy(),
        "deciding": deciding,
    }
    record: dict[str, Any] = {
        "match_id": str(points.iloc[0]["match_id"]),
        "date": pd.Timestamp(points.iloc[0]["date"]),
        "player_A_id": str(points.iloc[0]["player_A_id"]),
        "player_B_id": str(points.iloc[0]["player_B_id"]),
    }
    server_a = points["server"].eq("A").to_numpy()
    for category, mask in categories.items():
        for side in ("A", "B"):
            as_server = mask & (server_a == (side == "A"))
            as_receiver = mask & (server_a != (side == "A"))
            values = np.concatenate((residual[as_server], -residual[as_receiver]))
            record[f"{side}_{category}_sum"] = float(values.sum())
            record[f"{side}_{category}_count"] = len(values)
    return record


def _build_clutch_snapshots(records: list[dict[str, Any]]) -> pd.DataFrame:
    matches = pd.DataFrame(records).sort_values(["date", "match_id"], kind="stable")
    state: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    snapshots: list[dict[str, Any]] = []
    for _, batch in matches.groupby("date", sort=False):
        for match in batch.itertuples(index=False):
            row: dict[str, Any] = {"match_id": match.match_id}
            for side in ("A", "B"):
                player = str(getattr(match, f"player_{side}_id"))
                for category in ("break", "tiebreak", "deciding"):
                    total, count = state[(player, category)]
                    row[f"{side}_{category}_clutch_history"] = total / (
                        count + CLUTCH_PRIOR_WEIGHT
                    )
                    row[f"{side}_{category}_clutch_samples"] = count
            snapshots.append(row)
        for match in batch.itertuples(index=False):
            for side in ("A", "B"):
                player = str(getattr(match, f"player_{side}_id"))
                for category in ("break", "tiebreak", "deciding"):
                    current = state[(player, category)]
                    current[0] += float(getattr(match, f"{side}_{category}_sum"))
                    current[1] += float(getattr(match, f"{side}_{category}_count"))
    return pd.DataFrame(snapshots)


def _add_clutch_features(points: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    points = points.merge(snapshots, on="match_id", how="left", validate="many_to_one")
    server_a = points["server"].eq("A")
    deciding = (points["sets_A"] + points["sets_B"]).eq(points["best_of"] - 1)
    for category, active in (
        ("break", points["is_break_point"].astype(int)),
        ("tiebreak", points["is_tiebreak"].astype(int)),
        ("deciding", deciding.astype(int)),
    ):
        server = np.where(
            server_a,
            points[f"A_{category}_clutch_history"],
            points[f"B_{category}_clutch_history"],
        )
        receiver = np.where(
            server_a,
            points[f"B_{category}_clutch_history"],
            points[f"A_{category}_clutch_history"],
        )
        points[f"server_{category}_clutch_history"] = server
        points[f"receiver_{category}_clutch_history"] = receiver
        points[f"{category}_clutch_edge_active"] = (server - receiver) * active
    return points


def build_point_features_v2() -> dict[str, Any]:
    freeze = write_phase7b_freeze()
    artifact = joblib.load(V1_MODEL_DIR / "model.joblib")
    V2_PARTS_DIR.mkdir(parents=True, exist_ok=True)
    for old in V2_PARTS_DIR.glob("*.parquet"):
        old.unlink()
    intermediate = V2_PARTS_DIR / "intermediate"
    intermediate.mkdir(parents=True, exist_ok=True)
    for old in intermediate.glob("*.parquet"):
        old.unlink()

    records: list[dict[str, Any]] = []
    for path in sorted(V1_PARTS_DIR.glob("*.parquet")):
        points = pd.read_parquet(path)
        enriched = add_sequential_features(points)
        probability = artifact["pipeline"].predict_proba(
            enriched[artifact["feature_columns"]]
        )[:, 1]
        enriched["v1_probability_server"] = probability
        clutch_expectation = np.clip(
            0.5
            * (
                enriched["server_serve_rate"].to_numpy()
                + 1.0
                - enriched["receiver_return_rate"].to_numpy()
            ),
            1e-6,
            1 - 1e-6,
        )
        for _, match in enriched.groupby("match_id", sort=False, observed=True):
            records.append(
                _match_clutch_contributions(match, clutch_expectation[match.index])
            )
        enriched.to_parquet(intermediate / path.name, index=False)

    snapshots = _build_clutch_snapshots(records)
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    try:
        for path in sorted(intermediate.glob("*.parquet")):
            points = _add_clutch_features(pd.read_parquet(path), snapshots)
            output = V2_PARTS_DIR / path.name
            points.to_parquet(output, index=False)
            table = pa.Table.from_pandas(points, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(V2_PATH, table.schema, compression="snappy")
            writer.write_table(table)
            total_rows += len(points)
    finally:
        if writer is not None:
            writer.close()

    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "rows": total_rows,
        "matches": len(records),
        "prior_weights": list(PRIOR_WEIGHTS),
        "clutch_prior_weight": CLUTCH_PRIOR_WEIGHT,
        "state_timing": "strictly before current point",
        "phase7b_artifact_count": len(freeze["artifacts"]),
        "output": str(V2_PATH),
    }
    (REPORTS_DIR / "phase8_feature_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary
