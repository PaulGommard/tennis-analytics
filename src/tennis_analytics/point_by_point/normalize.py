from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from tennis_analytics.modeling.logistic import apply_platt
from tennis_analytics.paths import EXTERNAL_DIR, PROCESSED_DIR, PROJECT_ROOT, REPORTS_DIR
from tennis_analytics.scoring.engine import ScoreSnapshot, TennisScore

PARTS_DIR = PROCESSED_DIR / "point_events_v1_parts"
MATCH_AUDIT_PATH = PROCESSED_DIR / "point_matches_v1.parquet"
FEATURE_PATH = PROCESSED_DIR / "point_features_v1.parquet"

SLAM_SURFACE = {
    "ausopen": "Hard",
    "frenchopen": "Clay",
    "wimbledon": "Grass",
    "usopen": "Hard",
}
SLAM_DATE = {
    "ausopen": (1, 15),
    "frenchopen": (5, 25),
    "wimbledon": (7, 1),
    "usopen": (8, 25),
}
SLAM_TOURNAMENT = {
    "ausopen": "australian open",
    "frenchopen": "roland garros",
    "wimbledon": "wimbledon",
    "usopen": "us open",
}


def normalize_name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _safe_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _score_token(value: Any) -> str:
    if pd.isna(value):
        return ""
    token = str(value).strip().upper()
    token = token.removesuffix(".0")
    return {"A": "AD", "ADV": "AD", "GAME": "0"}.get(token, token)


def _bool_flag(value: Any) -> bool | None:
    if pd.isna(value):
        return None
    if value in (1, 1.0, True, "1", "TRUE", "True"):
        return True
    if value in (0, 0.0, False, "0", "FALSE", "False"):
        return False
    return None


@dataclass(frozen=True)
class Rules:
    best_of: int
    final_set_tiebreak: bool
    final_set_tiebreak_at: int = 6
    final_set_tiebreak_points: int = 7
    rule_label: str = "standard"


def slam_rules(year: int, slam: str, gender: str) -> Rules:
    best_of = 5 if gender == "men" else 3
    if year >= 2022:
        return Rules(best_of, True, 6, 10, "grand_slam_unified_10pt")
    if slam == "usopen":
        return Rules(best_of, True, 6, 7, "us_open_standard_tb")
    if slam == "ausopen" and year >= 2019:
        return Rules(best_of, True, 6, 10, "australian_open_10pt")
    if slam == "wimbledon" and year >= 2019:
        return Rules(best_of, True, 12, 7, "wimbledon_12_all")
    return Rules(best_of, False, 6, 7, "advantage_final_set")


def mcp_rules(row: pd.Series) -> Rules | None:
    best_of = _safe_int(row.get("Best of"))
    if best_of not in (3, 5):
        return None
    tournament = normalize_name(row.get("Tournament"))
    date_value = pd.to_numeric(pd.Series([row.get("Date")]), errors="coerce").iloc[0]
    year = int(date_value) // 10000 if pd.notna(date_value) else 0
    if "australian open" in tournament:
        if year >= 2022:
            return Rules(best_of, True, 6, 10, "australian_open_10pt")
        if year >= 2019:
            return Rules(best_of, True, 6, 10, "australian_open_10pt")
        return Rules(best_of, False, rule_label="advantage_final_set")
    if "roland garros" in tournament or "french open" in tournament:
        return (
            Rules(best_of, True, 6, 10, "grand_slam_unified_10pt")
            if year >= 2022
            else Rules(best_of, False, rule_label="advantage_final_set")
        )
    if "wimbledon" in tournament:
        if year >= 2022:
            return Rules(best_of, True, 6, 10, "grand_slam_unified_10pt")
        if year >= 2019:
            return Rules(best_of, True, 12, 7, "wimbledon_12_all")
        return Rules(best_of, False, rule_label="advantage_final_set")
    if "us open" in tournament:
        return (
            Rules(best_of, True, 6, 10, "grand_slam_unified_10pt")
            if year >= 2022
            else Rules(best_of, True, 6, 7, "us_open_standard_tb")
        )
    code = str(row.get("Final TB?", "")).strip()
    if code == "0":
        return Rules(best_of, False, rule_label="source_no_final_tb")
    return Rules(best_of, True, 6, 7, "source_standard_tb")


class IdentityResolver:
    def __init__(self) -> None:
        players = pd.read_parquet(PROCESSED_DIR / "atp" / "players.parquet")
        players["normalized"] = players["full_name"].map(normalize_name)
        grouped = players.groupby("normalized")["player_id"].agg(list)
        self.unique = {name: ids[0] for name, ids in grouped.items() if len(ids) == 1}

    def resolve(self, name: str, source: str) -> tuple[str, str]:
        normalized = normalize_name(name)
        if normalized in self.unique:
            return str(self.unique[normalized]), "atp_exact_normalized_name"
        digest = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        return f"{source}:name:{digest}", "source_local_normalized_name"


class PrematchLookup:
    def __init__(self) -> None:
        features = pd.read_parquet(PROCESSED_DIR / "match_features_v3.parquet")
        timeline = pd.read_parquet(
            PROCESSED_DIR / "analytics" / "match_timeline.parquet",
            columns=["match_id", "tournament"],
        )
        players = pd.read_parquet(
            PROCESSED_DIR / "atp" / "players.parquet", columns=["player_id", "full_name"]
        ).set_index("player_id")["full_name"]
        artifact = joblib.load(PROJECT_ROOT / "models" / "prematch_v1" / "model.joblib")
        raw = artifact["pipeline"].predict_proba(features[artifact["feature_columns"]])[:, 1]
        probability = (
            apply_platt(artifact["calibrator"], raw)
            if artifact.get("calibrator") is not None
            else raw
        )
        frame = features[["match_id", "date", "surface", "player_A_id", "player_B_id"]].copy()
        frame["probability_A"] = probability
        frame["name_A"] = frame["player_A_id"].map(players).map(normalize_name)
        frame["name_B"] = frame["player_B_id"].map(players).map(normalize_name)
        frame["pair"] = frame.apply(lambda row: "|".join(sorted((row.name_A, row.name_B))), axis=1)
        frame["date_key"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
        frame["year"] = pd.to_datetime(frame["date"]).dt.year
        frame = frame.merge(timeline, on="match_id", how="left")
        frame["tournament_key"] = frame["tournament"].map(normalize_name)
        self.date_lookup = self._unique_lookup(frame, ["date_key", "pair"])
        self.slam_lookup = self._unique_lookup(frame, ["year", "tournament_key", "pair"])

    @staticmethod
    def _unique_lookup(
        frame: pd.DataFrame, columns: list[str]
    ) -> dict[tuple[Any, ...], dict[str, Any]]:
        result: dict[tuple[Any, ...], dict[str, Any]] = {}
        for key, group in frame.groupby(columns, dropna=False, sort=False):
            if len(group) == 1:
                result[key if isinstance(key, tuple) else (key,)] = group.iloc[0].to_dict()
        return result

    @staticmethod
    def _orient(record: dict[str, Any] | None, name_a: str) -> float | None:
        if record is None:
            return None
        probability = float(record["probability_A"])
        return probability if record["name_A"] == normalize_name(name_a) else 1 - probability

    def for_mcp(
        self,
        date_value: pd.Timestamp,
        tournament: str,
        name_a: str,
        name_b: str,
    ) -> float | None:
        pair = "|".join(sorted((normalize_name(name_a), normalize_name(name_b))))
        record = self.date_lookup.get((date_value.strftime("%Y-%m-%d"), pair))
        if record is None:
            record = self.slam_lookup.get(
                (date_value.year, normalize_name(tournament), pair)
            )
        return self._orient(record, name_a)

    def for_slam(self, year: int, slam: str, name_a: str, name_b: str) -> float | None:
        pair = "|".join(sorted((normalize_name(name_a), normalize_name(name_b))))
        record = self.slam_lookup.get((year, SLAM_TOURNAMENT[slam], pair))
        return self._orient(record, name_a)


def _point_flags(snapshot: ScoreSnapshot, rules: Rules) -> dict[str, int]:
    points = {"A": snapshot.points_a, "B": snapshot.points_b}
    games = {"A": snapshot.games_a, "B": snapshot.games_b}
    sets = {"A": snapshot.sets_a, "B": snapshot.sets_b}
    flags: dict[str, bool] = {}
    for player in ("A", "B"):
        opponent = "B" if player == "A" else "A"
        if snapshot.is_tiebreak:
            target = (
                rules.final_set_tiebreak_points
                if snapshot.sets_a + snapshot.sets_b == rules.best_of - 1
                else 7
            )
            game_point = points[player] + 1 >= target and points[player] + 1 - points[opponent] >= 2
            set_point = game_point
        else:
            game_point = points[player] + 1 >= 4 and points[player] + 1 - points[opponent] >= 2
            next_games = games[player] + 1
            set_point = game_point and next_games >= 6 and next_games - games[opponent] >= 2
        match_point = set_point and sets[player] + 1 == rules.best_of // 2 + 1
        flags[f"game_{player}"] = game_point
        flags[f"set_{player}"] = set_point
        flags[f"match_{player}"] = match_point
    receiver = "B" if snapshot.server == "A" else "A"
    return {
        "is_break_point": int(flags[f"game_{receiver}"]),
        "is_game_point": int(flags["game_A"] or flags["game_B"]),
        "is_set_point": int(flags["set_A"] or flags["set_B"]),
        "is_match_point": int(flags["match_A"] or flags["match_B"]),
        "deuce": int(
            not snapshot.is_tiebreak
            and snapshot.points_a >= 3
            and snapshot.points_a == snapshot.points_b
        ),
    }


def _new_score(rules: Rules, first_server: str) -> TennisScore:
    return TennisScore(
        best_of=rules.best_of,
        first_server=first_server,
        final_set_tiebreak=rules.final_set_tiebreak,
        final_set_tiebreak_at=rules.final_set_tiebreak_at,
        final_set_tiebreak_points=rules.final_set_tiebreak_points,
    )


def _base_point(
    *,
    match_id: str,
    source: str,
    date_value: pd.Timestamp,
    date_precision: str,
    surface: str,
    name_a: str,
    name_b: str,
    id_a: str,
    id_b: str,
    identity_a: str,
    identity_b: str,
    gender: str,
    rules: Rules,
    point_number: int,
    before: ScoreSnapshot,
    winner: str,
    ace: bool | None,
    double_fault: bool | None,
    prematch: float | None,
) -> dict[str, Any]:
    server_id = id_a if before.server == "A" else id_b
    receiver_id = id_b if before.server == "A" else id_a
    winner_id = id_a if winner == "A" else id_b
    flags = _point_flags(before, rules)
    return {
        "match_id": f"{source}:{match_id}",
        "source_match_id": match_id,
        "source": source,
        "date": date_value,
        "date_precision": date_precision,
        "surface": surface,
        "gender": gender,
        "player_A_id": id_a,
        "player_B_id": id_b,
        "player_A_name": name_a,
        "player_B_name": name_b,
        "identity_method_A": identity_a,
        "identity_method_B": identity_b,
        "best_of": rules.best_of,
        "rule_label": rules.rule_label,
        "set_number": before.sets_a + before.sets_b + 1,
        "game_number": before.games_a + before.games_b + 1,
        "point_number": point_number,
        "server": before.server,
        "server_id": server_id,
        "receiver_id": receiver_id,
        "sets_A": before.sets_a,
        "sets_B": before.sets_b,
        "games_A": before.games_a,
        "games_B": before.games_b,
        "points_A": before.points_a,
        "points_B": before.points_b,
        "point_score_A": before.display_a,
        "point_score_B": before.display_b,
        "is_tiebreak": before.is_tiebreak,
        "point_winner": winner,
        "point_winner_id": winner_id,
        "server_won_point": int(before.server == winner),
        "A_wins_next_point": int(winner == "A"),
        "ace": ace,
        "double_fault": double_fault,
        "prematch_probability_A": prematch,
        "sets_diff": before.sets_a - before.sets_b,
        "games_diff": before.games_a - before.games_b,
        "server_leading_in_game": int(
            (before.server == "A" and before.points_a > before.points_b)
            or (before.server == "B" and before.points_b > before.points_a)
        ),
        **flags,
    }


def _finalize_match_rows(rows: list[dict[str, Any]], score: TennisScore) -> None:
    game_winners: dict[tuple[int, int], str] = {}
    set_winners: dict[int, str] = {}
    for row in rows:
        if row["game_ended_after"]:
            game_winners[(row["set_number"], row["game_number"])] = row["point_winner"]
        if row["set_ended_after"]:
            set_winners[row["set_number"]] = row["point_winner"]
    for row in rows:
        row["current_game_winner_A"] = int(
            game_winners.get((row["set_number"], row["game_number"])) == "A"
        )
        row["current_set_winner_A"] = int(set_winners.get(row["set_number"]) == "A")
        row["final_match_winner_A"] = int(score.winner == "A") if score.match_over else np.nan


def _validate_server(before: ScoreSnapshot, value: Any) -> bool:
    server = _safe_int(value)
    return server in (1, 2) and before.server == ("A" if server == 1 else "B")


def replay_mcp_match(
    points: pd.DataFrame,
    metadata: pd.Series,
    resolver: IdentityResolver,
    prematch_lookup: PrematchLookup,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_id = str(metadata["match_id"])
    name_a, name_b = str(metadata["Player 1"]), str(metadata["Player 2"])
    id_a, identity_a = resolver.resolve(name_a, "mcp")
    id_b, identity_b = resolver.resolve(name_b, "mcp")
    date_numeric = pd.to_numeric(pd.Series([metadata.get("Date")]), errors="coerce").iloc[0]
    date_value = (
        pd.to_datetime(str(int(date_numeric)), format="%Y%m%d", errors="coerce")
        if pd.notna(date_numeric)
        else pd.NaT
    )
    surface = str(metadata.get("Surface"))
    rules = mcp_rules(metadata)
    audit = {
        "match_id": f"mcp:{source_id}",
        "source_match_id": source_id,
        "source": "mcp",
        "date": date_value,
        "date_precision": "day",
        "surface": surface,
        "gender": "men",
        "player_A_id": id_a,
        "player_B_id": id_b,
        "player_A_name": name_a,
        "player_B_name": name_b,
        "identity_method_A": identity_a,
        "identity_method_B": identity_b,
        "source_status": None,
        "rule_label": rules.rule_label if rules else "unsupported_format",
        "best_of": rules.best_of if rules else np.nan,
    }
    if rules is None or pd.isna(date_value) or surface not in {"Hard", "Clay", "Grass", "Carpet"}:
        return [], {
            **audit,
            "score_valid": False,
            "scoring_status": "invalid_metadata",
            "points": 0,
        }
    ordered = points.sort_values("Pt", kind="stable")
    if ordered["Pt"].duplicated().any() or list(ordered["Pt"].astype(int)) != list(
        range(1, len(ordered) + 1)
    ):
        return [], {
            **audit,
            "score_valid": False,
            "scoring_status": "duplicate_or_gap",
            "points": len(ordered),
        }
    first_server_value = _safe_int(ordered.iloc[0]["Svr"])
    if first_server_value not in (1, 2):
        return [], {
            **audit,
            "score_valid": False,
            "scoring_status": "invalid_server",
            "points": len(ordered),
        }
    score = _new_score(rules, "A" if first_server_value == 1 else "B")
    prematch = prematch_lookup.for_mcp(
        date_value, str(metadata.get("Tournament")), name_a, name_b
    )
    rows: list[dict[str, Any]] = []
    divergences: Counter[str] = Counter()
    for source_row in ordered.itertuples(index=False):
        winner_value = _safe_int(source_row.PtWinner)
        if winner_value not in (1, 2) or score.match_over:
            divergences["invalid_winner_or_extra_point"] += 1
            break
        before = score.snapshot()
        if not _validate_server(before, source_row.Svr):
            divergences["server"] += 1
        if (
            _safe_int(source_row.Set1) != before.sets_a
            or _safe_int(source_row.Set2) != before.sets_b
        ):
            divergences["sets"] += 1
        if (
            _safe_int(source_row.Gm1) != before.games_a
            or _safe_int(source_row.Gm2) != before.games_b
        ):
            divergences["games"] += 1
        score_tokens = str(source_row.Pts).split("-")
        expected_score = (
            (before.display_a, before.display_b)
            if before.server == "A"
            else (before.display_b, before.display_a)
        )
        if len(score_tokens) != 2 or (
            _score_token(score_tokens[0]) != expected_score[0]
            or _score_token(score_tokens[1]) != expected_score[1]
        ):
            divergences["point_score"] += 1
        winner = "A" if winner_value == 1 else "B"
        row = _base_point(
            match_id=source_id,
            source="mcp",
            date_value=date_value,
            date_precision="day",
            surface=surface,
            name_a=name_a,
            name_b=name_b,
            id_a=id_a,
            id_b=id_b,
            identity_a=identity_a,
            identity_b=identity_b,
            gender="men",
            rules=rules,
            point_number=int(source_row.Pt),
            before=before,
            winner=winner,
            ace=None,
            double_fault=None,
            prematch=prematch,
        )
        after = score.point_won_by(winner)
        row["game_ended_after"] = int(
            after.games_a + after.games_b != before.games_a + before.games_b
            or after.sets_a + after.sets_b != before.sets_a + before.sets_b
        )
        row["set_ended_after"] = int(after.sets_a + after.sets_b > before.sets_a + before.sets_b)
        rows.append(row)
    valid = not divergences
    if valid:
        _finalize_match_rows(rows, score)
    status = "completed" if valid and score.match_over else "incomplete_or_retirement"
    return (rows if valid else []), {
        **audit,
        "score_valid": valid,
        "scoring_status": status if valid else "divergence",
        "match_completed": bool(valid and score.match_over),
        "match_winner": score.winner if valid else None,
        "points": len(ordered),
        "prematch_available": prematch is not None,
        "divergences_json": json.dumps(divergences, sort_keys=True),
    }


def replay_slam_match(
    points: pd.DataFrame,
    metadata: pd.Series,
    resolver: IdentityResolver,
    prematch_lookup: PrematchLookup,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_id = str(metadata["match_id"])
    year, slam = int(metadata["year"]), str(metadata["slam"])
    name_a, name_b = str(metadata["player1"]), str(metadata["player2"])
    id_a, identity_a = resolver.resolve(name_a, "slam")
    id_b, identity_b = resolver.resolve(name_b, "slam")
    match_num = _safe_int(metadata.get("match_num")) or 0
    gender = {1: "men", 2: "women"}.get(match_num // 1000, "unknown")
    surface = SLAM_SURFACE.get(slam, "Unknown")
    month, day = SLAM_DATE.get(slam, (7, 1))
    date_value = pd.Timestamp(year=year, month=month, day=day)
    rules = slam_rules(year, slam, gender)
    status = metadata.get("status")
    audit = {
        "match_id": f"slam:{source_id}",
        "source_match_id": source_id,
        "source": "slam",
        "date": date_value,
        "date_precision": "tournament_approximation",
        "surface": surface,
        "gender": gender,
        "player_A_id": id_a,
        "player_B_id": id_b,
        "player_A_name": name_a,
        "player_B_name": name_b,
        "identity_method_A": identity_a,
        "identity_method_B": identity_b,
        "source_status": None if pd.isna(status) else str(status),
        "rule_label": rules.rule_label,
        "best_of": rules.best_of,
    }
    ordered = points.copy()
    ordered["point_numeric"] = pd.to_numeric(ordered["PointNumber"], errors="coerce")
    ordered = ordered.loc[ordered["point_numeric"] > 0].sort_values("point_numeric", kind="stable")
    numbers = ordered["point_numeric"].astype(int).tolist()
    if (
        not numbers
        or ordered["point_numeric"].duplicated().any()
        or numbers != list(range(1, len(numbers) + 1))
    ):
        return [], {
            **audit,
            "score_valid": False,
            "scoring_status": "duplicate_or_gap",
            "points": len(ordered),
        }
    first_server_value = _safe_int(ordered.iloc[0]["PointServer"])
    if first_server_value not in (1, 2):
        return [], {
            **audit,
            "score_valid": False,
            "scoring_status": "invalid_server",
            "points": len(ordered),
        }
    score = _new_score(rules, "A" if first_server_value == 1 else "B")
    prematch = prematch_lookup.for_slam(year, slam, name_a, name_b)
    rows: list[dict[str, Any]] = []
    divergences: Counter[str] = Counter()
    for source_row in ordered.itertuples(index=False):
        winner_value = _safe_int(source_row.PointWinner)
        if winner_value not in (1, 2) or score.match_over:
            divergences["invalid_winner_or_extra_point"] += 1
            break
        before = score.snapshot()
        if not _validate_server(before, source_row.PointServer):
            divergences["server"] += 1
        if _safe_int(source_row.SetNo) != before.sets_a + before.sets_b + 1:
            divergences["set_number"] += 1
        winner = "A" if winner_value == 1 else "B"
        ace_column = source_row.P1Ace if winner == "A" else source_row.P2Ace
        server_df = source_row.P1DoubleFault if before.server == "A" else source_row.P2DoubleFault
        row = _base_point(
            match_id=source_id,
            source="slam",
            date_value=date_value,
            date_precision="tournament_approximation",
            surface=surface,
            name_a=name_a,
            name_b=name_b,
            id_a=id_a,
            id_b=id_b,
            identity_a=identity_a,
            identity_b=identity_b,
            gender=gender,
            rules=rules,
            point_number=int(source_row.point_numeric),
            before=before,
            winner=winner,
            ace=_bool_flag(ace_column),
            double_fault=_bool_flag(server_df),
            prematch=prematch,
        )
        after = score.point_won_by(winner)
        game_ended = (
            after.games_a + after.games_b != before.games_a + before.games_b
            or after.sets_a + after.sets_b != before.sets_a + before.sets_b
        )
        set_ended = after.sets_a + after.sets_b > before.sets_a + before.sets_b
        expected_games_a = before.games_a + int(game_ended and winner == "A")
        expected_games_b = before.games_b + int(game_ended and winner == "B")
        if not set_ended:
            expected_games_a, expected_games_b = after.games_a, after.games_b
        if (
            _safe_int(source_row.P1GamesWon) != expected_games_a
            or _safe_int(source_row.P2GamesWon) != expected_games_b
        ):
            divergences["games"] += 1
        if (
            _score_token(source_row.P1Score) != after.display_a
            or _score_token(source_row.P2Score) != after.display_b
        ):
            divergences["point_score"] += 1
        row["game_ended_after"] = int(game_ended)
        row["set_ended_after"] = int(set_ended)
        rows.append(row)
    valid = not divergences
    if valid:
        _finalize_match_rows(rows, score)
    source_retired = str(status).lower() == "retired"
    completed = valid and score.match_over and not source_retired
    scoring_status = "retired" if source_retired else ("completed" if completed else "incomplete")
    return (rows if valid else []), {
        **audit,
        "score_valid": valid,
        "scoring_status": scoring_status if valid else "divergence",
        "match_completed": bool(completed),
        "match_winner": score.winner if valid else None,
        "points": len(ordered),
        "prematch_available": prematch is not None,
        "divergences_json": json.dumps(divergences, sort_keys=True),
    }


def _write_part(rows: list[dict[str, Any]], path: Path) -> None:
    if rows:
        pd.DataFrame(rows).to_parquet(path, index=False)


def _flush_rows(rows: list[dict[str, Any]], stem: str, index: int) -> int:
    if not rows:
        return index
    _write_part(rows, PARTS_DIR / f"{stem}_{index:03d}.parquet")
    rows.clear()
    return index + 1


def normalize_sources() -> pd.DataFrame:
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    for old in PARTS_DIR.glob("*.parquet"):
        old.unlink()
    resolver = IdentityResolver()
    prematch = PrematchLookup()
    audits: list[dict[str, Any]] = []

    mcp_metadata = pd.read_csv(
        EXTERNAL_DIR / "match_charting_project" / "charting-m-matches.csv", low_memory=False
    ).set_index("match_id", drop=False)
    for point_path in sorted(
        (EXTERNAL_DIR / "match_charting_project").glob("charting-m-points-*.csv")
    ):
        points = pd.read_csv(point_path, low_memory=False)
        part_rows: list[dict[str, Any]] = []
        part_index = 0
        for match_id, group in points.groupby("match_id", sort=False):
            if match_id not in mcp_metadata.index:
                audits.append(
                    {
                        "match_id": f"mcp:{match_id}",
                        "source_match_id": match_id,
                        "source": "mcp",
                        "score_valid": False,
                        "scoring_status": "missing_metadata",
                        "points": len(group),
                    }
                )
                continue
            rows, audit = replay_mcp_match(group, mcp_metadata.loc[match_id], resolver, prematch)
            part_rows.extend(rows)
            audits.append(audit)
            if len(part_rows) >= 100_000:
                part_index = _flush_rows(part_rows, f"mcp_{point_path.stem}", part_index)
        _flush_rows(part_rows, f"mcp_{point_path.stem}", part_index)

    slam_root = EXTERNAL_DIR / "slam_pointbypoint"
    for point_path in sorted(slam_root.rglob("*-points.csv")):
        match_path = point_path.with_name(point_path.name.replace("-points.csv", "-matches.csv"))
        metadata = pd.read_csv(match_path, low_memory=False).set_index("match_id", drop=False)
        points = pd.read_csv(point_path, low_memory=False)
        part_rows = []
        part_index = 0
        for match_id, group in points.groupby("match_id", sort=False):
            if match_id not in metadata.index:
                audits.append(
                    {
                        "match_id": f"slam:{match_id}",
                        "source_match_id": match_id,
                        "source": "slam",
                        "score_valid": False,
                        "scoring_status": "missing_metadata",
                        "points": len(group),
                    }
                )
                continue
            rows, audit = replay_slam_match(group, metadata.loc[match_id], resolver, prematch)
            part_rows.extend(rows)
            audits.append(audit)
            if len(part_rows) >= 100_000:
                part_index = _flush_rows(
                    part_rows,
                    f"slam_{point_path.parent.name}_{point_path.stem}",
                    part_index,
                )
        _flush_rows(
            part_rows,
            f"slam_{point_path.parent.name}_{point_path.stem}",
            part_index,
        )

    audit_frame = pd.DataFrame(audits)
    audit_frame.to_parquet(MATCH_AUDIT_PATH, index=False)
    return audit_frame


@dataclass
class RateState:
    serve_wins: float = 0.0
    serve_points: float = 0.0
    return_wins: float = 0.0
    return_points: float = 0.0


def _rate(numerator: float, denominator: float, prior: float, strength: float) -> float:
    return (numerator + strength * prior) / (denominator + strength)


def build_historical_context(audit: pd.DataFrame) -> pd.DataFrame:
    eligible = audit.loc[
        audit["score_valid"].eq(True) & audit["match_completed"].eq(True)
    ].copy()
    eligible["date"] = pd.to_datetime(eligible["date"])
    aggregates: list[dict[str, Any]] = []
    for path in sorted(PARTS_DIR.glob("*.parquet")):
        points = pd.read_parquet(
            path,
            columns=["match_id", "server", "server_won_point", "player_A_id", "player_B_id"],
        )
        if points.empty:
            continue
        for match_id, group in points.groupby("match_id", sort=False):
            a_serve = group["server"] == "A"
            b_serve = ~a_serve
            aggregates.append(
                {
                    "match_id": match_id,
                    "A_serve_wins": int(group.loc[a_serve, "server_won_point"].sum()),
                    "A_serve_points": int(a_serve.sum()),
                    "B_serve_wins": int(group.loc[b_serve, "server_won_point"].sum()),
                    "B_serve_points": int(b_serve.sum()),
                }
            )
    aggregate = pd.DataFrame(aggregates)
    eligible = eligible.merge(aggregate, on="match_id", how="inner", validate="one_to_one")
    eligible = eligible.sort_values(["date", "source", "match_id"], kind="stable")
    global_wins = 0.0
    global_points = 0.0
    career: dict[str, RateState] = {}
    surface_state: dict[tuple[str, str], RateState] = {}
    recent_events: dict[str, list[tuple[pd.Timestamp, RateState]]] = {}
    rows: list[dict[str, Any]] = []
    for _, batch in eligible.groupby("date", sort=False):
        global_prior = global_wins / global_points if global_points else 0.60
        for match in batch.itertuples(index=False):
            snapshot: dict[str, Any] = {"match_id": match.match_id}
            for side, player, opponent in (
                ("A", match.player_A_id, match.player_B_id),
                ("B", match.player_B_id, match.player_A_id),
            ):
                own = career.get(player, RateState())
                other = career.get(opponent, RateState())
                own_surface = surface_state.get((player, match.surface), RateState())
                other_surface = surface_state.get((opponent, match.surface), RateState())
                cutoff = match.date - pd.Timedelta(days=365)
                recent_own = RateState()
                for event_date, event in recent_events.get(player, []):
                    if event_date >= cutoff:
                        recent_own.serve_wins += event.serve_wins
                        recent_own.serve_points += event.serve_points
                        recent_own.return_wins += event.return_wins
                        recent_own.return_points += event.return_points
                recent_other = RateState()
                for event_date, event in recent_events.get(opponent, []):
                    if event_date >= cutoff:
                        recent_other.serve_wins += event.serve_wins
                        recent_other.serve_points += event.serve_points
                        recent_other.return_wins += event.return_wins
                        recent_other.return_points += event.return_points
                snapshot[f"{side}_serve_rate"] = _rate(
                    own.serve_wins, own.serve_points, global_prior, 200
                )
                snapshot[f"{side}_return_rate"] = _rate(
                    other.return_wins, other.return_points, 1 - global_prior, 200
                )
                snapshot[f"{side}_surface_serve_rate"] = _rate(
                    own_surface.serve_wins,
                    own_surface.serve_points,
                    snapshot[f"{side}_serve_rate"],
                    200,
                )
                snapshot[f"{side}_surface_return_rate"] = _rate(
                    other_surface.return_wins,
                    other_surface.return_points,
                    snapshot[f"{side}_return_rate"],
                    200,
                )
                snapshot[f"{side}_serve_rate_365d"] = _rate(
                    recent_own.serve_wins, recent_own.serve_points, global_prior, 200
                )
                snapshot[f"{side}_return_rate_365d"] = _rate(
                    recent_other.return_wins, recent_other.return_points, 1 - global_prior, 200
                )
            snapshot["global_server_rate"] = global_prior
            rows.append(snapshot)
        for match in batch.itertuples(index=False):
            event_a = RateState(
                serve_wins=match.A_serve_wins,
                serve_points=match.A_serve_points,
                return_wins=match.B_serve_points - match.B_serve_wins,
                return_points=match.B_serve_points,
            )
            event_b = RateState(
                serve_wins=match.B_serve_wins,
                serve_points=match.B_serve_points,
                return_wins=match.A_serve_points - match.A_serve_wins,
                return_points=match.A_serve_points,
            )
            for player, event in ((match.player_A_id, event_a), (match.player_B_id, event_b)):
                state = career.setdefault(player, RateState())
                surface = surface_state.setdefault((player, match.surface), RateState())
                for target in (state, surface):
                    target.serve_wins += event.serve_wins
                    target.serve_points += event.serve_points
                    target.return_wins += event.return_wins
                    target.return_points += event.return_points
                recent_events.setdefault(player, []).append((match.date, event))
            global_wins += match.A_serve_wins + match.B_serve_wins
            global_points += match.A_serve_points + match.B_serve_points
    return pd.DataFrame(rows)


def _coerce_feature_types(frame: pd.DataFrame) -> pd.DataFrame:
    bool_columns = ["is_tiebreak", "ace", "double_fault"]
    for column in bool_columns:
        frame[column] = frame[column].astype("boolean")
    string_columns = [
        "match_id",
        "source_match_id",
        "source",
        "date_precision",
        "surface",
        "gender",
        "player_A_id",
        "player_B_id",
        "player_A_name",
        "player_B_name",
        "identity_method_A",
        "identity_method_B",
        "rule_label",
        "server",
        "server_id",
        "receiver_id",
        "point_score_A",
        "point_score_B",
        "point_winner",
        "point_winner_id",
    ]
    for column in string_columns:
        frame[column] = frame[column].astype("string")
    for column in ("prematch_probability_A", "prematch_server", "final_match_winner_A"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    return frame


def build_point_features(audit: pd.DataFrame) -> pd.DataFrame:
    context = build_historical_context(audit)
    eligible_ids = set(
        audit.loc[
            audit["score_valid"].eq(True) & audit["match_completed"].eq(True),
            "match_id",
        ]
    )
    feature_parts = PROCESSED_DIR / "point_features_v1_parts"
    feature_parts.mkdir(parents=True, exist_ok=True)
    for old in feature_parts.glob("*.parquet"):
        old.unlink()
    written: list[Path] = []
    for event_path in sorted(PARTS_DIR.glob("*.parquet")):
        points = pd.read_parquet(event_path)
        points = points.loc[points["match_id"].isin(eligible_ids)].copy()
        if points.empty:
            continue
        points = points.merge(context, on="match_id", how="left", validate="many_to_one")
        server_a = points["server"] == "A"
        points["server_serve_rate"] = np.where(
            server_a, points["A_serve_rate"], points["B_serve_rate"]
        )
        points["receiver_return_rate"] = np.where(
            server_a, points["A_return_rate"], points["B_return_rate"]
        )
        points["server_surface_serve_rate"] = np.where(
            server_a, points["A_surface_serve_rate"], points["B_surface_serve_rate"]
        )
        points["receiver_surface_return_rate"] = np.where(
            server_a, points["A_surface_return_rate"], points["B_surface_return_rate"]
        )
        points["server_serve_rate_365d"] = np.where(
            server_a, points["A_serve_rate_365d"], points["B_serve_rate_365d"]
        )
        points["receiver_return_rate_365d"] = np.where(
            server_a, points["A_return_rate_365d"], points["B_return_rate_365d"]
        )
        points["prematch_server"] = np.where(
            server_a, points["prematch_probability_A"], 1 - points["prematch_probability_A"]
        )
        points["surface_clay"] = (points["surface"] == "Clay").astype(int)
        points["surface_grass"] = (points["surface"] == "Grass").astype(int)
        points["sets_diff_server"] = np.where(server_a, points["sets_diff"], -points["sets_diff"])
        points["games_diff_server"] = np.where(
            server_a, points["games_diff"], -points["games_diff"]
        )
        part_path = feature_parts / event_path.name
        _coerce_feature_types(points).to_parquet(part_path, index=False)
        written.append(part_path)
    writer: pq.ParquetWriter | None = None
    try:
        for path in written:
            table = pq.read_table(path)
            if writer is None:
                writer = pq.ParquetWriter(FEATURE_PATH, table.schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    return context


def write_scoring_audit(audit: pd.DataFrame) -> None:
    total = len(audit)
    valid = int(audit["score_valid"].eq(True).sum())
    completed = int(audit["match_completed"].eq(True).sum())
    summary = {
        "matches_total": total,
        "score_valid": valid,
        "completed_eligible": completed,
        "by_source": audit.groupby("source", dropna=False)
        .agg(
            matches=("match_id", "size"),
            score_valid=("score_valid", "sum"),
            completed=("match_completed", "sum"),
            points=("points", "sum"),
        )
        .reset_index()
        .to_dict("records"),
        "status": audit["scoring_status"].value_counts(dropna=False).to_dict(),
        "rules": audit["rule_label"].value_counts(dropna=False).to_dict(),
        "prematch_coverage_completed": float(
            audit.loc[audit["match_completed"].eq(True), "prematch_available"]
            .fillna(False)
            .mean()
        ),
    }
    (REPORTS_DIR / "live_scoring_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    source_rows = "\n".join(
        f"| {row['source']} | {row['matches']:,} | {int(row['score_valid']):,} | {int(row['completed']):,} | {int(row['points']):,} |"
        for row in summary["by_source"]
    )
    text = f"""# Phase 7B — audit de reconstruction du score

Chaque match est rejoué depuis 0–0 avec `TennisScore`. L'état canonique est capturé **avant**
le point, puis le gagnant du point est appliqué. MCP est comparé à son score source avant point ;
Grand Slam à son score source après point. Une divergence exclut le match entier du dataset ML.

| Source | Matchs bruts | Score cohérent | Terminés éligibles | Points bruts |
|---|---:|---:|---:|---:|
{source_rows}

Total : {total:,} matchs, {valid:,} séquences cohérentes, {completed:,} matchs terminés et
éligibles au modèle point. Couverture PRE-MATCH V1 parmi les matchs éligibles :
{summary["prematch_coverage_completed"]:.2%}.

Les exclusions distinguent doublons/gaps, métadonnées invalides, score/serveur divergent,
match incomplet ou abandon. Les formats Grand Chelem historiques sont versionnés : avantage,
Wimbledon 12–12, Australian Open 10 points et règle unifiée 2022. Les super tie-breaks et
formats non standard non représentables sont exclus par divergence, jamais corrigés en silence.

Le détail match par match est conservé dans `data/processed/point_matches_v1.parquet`.
"""
    (REPORTS_DIR / "live_scoring_audit.md").write_text(text, encoding="utf-8")


def run_normalization() -> dict[str, Any]:
    audit = normalize_sources()
    write_scoring_audit(audit)
    context = build_point_features(audit)
    return {
        "matches": len(audit),
        "valid": int(audit["score_valid"].eq(True).sum()),
        "completed": int(audit["match_completed"].eq(True).sum()),
        "contexts": len(context),
        "feature_path": str(FEATURE_PATH),
    }
