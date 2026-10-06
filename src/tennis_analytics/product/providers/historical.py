from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tennis_analytics.paths import PROCESSED_DIR
from tennis_analytics.product.providers.base import TennisDataProvider
from tennis_analytics.tournament.replay import ROUND_ORDER


def _normalize(value: Any) -> str:
    folded = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", folded.lower()).strip()


def _json_value(value: Any) -> Any:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


class HistoricalTennisDataProvider(TennisDataProvider):
    provider_name = "historical_replay"

    def __init__(self) -> None:
        timeline_columns = [
            "match_id",
            "date",
            "tournament",
            "tournament_id",
            "tournament_edition_id",
            "surface",
            "level",
            "round",
            "source_match_num",
            "score",
            "player_A_id",
            "player_B_id",
            "player_A_name",
            "player_B_name",
            "winner_id",
            "player_A_win",
            "rank_A",
            "rank_B",
            "rank_points_A",
            "rank_points_B",
            "is_elo_eligible",
        ]
        self.timeline = pd.read_parquet(
            PROCESSED_DIR / "analytics" / "match_timeline.parquet",
            columns=timeline_columns,
        )
        self.timeline = self.timeline.copy()
        self.timeline["date"] = pd.to_datetime(self.timeline["date"])
        self.timeline["round_order"] = self.timeline["round"].map(ROUND_ORDER).fillna(-1).astype(int)
        matches = pd.read_parquet(
            PROCESSED_DIR / "atp" / "matches.parquet",
            columns=["match_id", "best_of", "minutes"],
        )
        self.timeline = self.timeline.merge(matches, on="match_id", how="left", validate="one_to_one")
        self.timeline_by_id = self.timeline.set_index("match_id", drop=False)
        point_matches_path = PROCESSED_DIR / "point_matches_v1.parquet"
        if Path(point_matches_path).exists():
            self.point_matches = pd.read_parquet(point_matches_path)
        else:
            # Point-by-point data is optional for the pre-match product.  A clean
            # clone can therefore serve ATP history and upcoming predictions after
            # the smaller pre-match data setup, while live replay stays unavailable.
            self.point_matches = pd.DataFrame(
                columns=[
                    "match_id",
                    "date",
                    "gender",
                    "prematch_available",
                    "score_valid",
                    "player_A_name",
                    "player_B_name",
                    "surface",
                ]
            )
        self.point_matches["date"] = pd.to_datetime(self.point_matches["date"])
        self.point_matches = self.point_matches.loc[
            self.point_matches["gender"].eq("men")
            & self.point_matches["prematch_available"].eq(True)
            & self.point_matches["score_valid"].eq(True)
        ].copy()
        self._point_candidates = self._build_point_candidates()
        self._link_cache: dict[str, str | None] = {}
        self._point_frame_cache: dict[str, pd.DataFrame] = {}

    @staticmethod
    def _pair_key(name_a: str, name_b: str, year: int, surface: str) -> str:
        players = "|".join(sorted((_normalize(name_a), _normalize(name_b))))
        return f"{year}|{surface}|{players}"

    def _build_point_candidates(self) -> dict[str, list[tuple[pd.Timestamp, str]]]:
        candidates: dict[str, list[tuple[pd.Timestamp, str]]] = {}
        for row in self.point_matches.itertuples(index=False):
            key = self._pair_key(
                str(row.player_A_name),
                str(row.player_B_name),
                int(row.date.year),
                str(row.surface),
            )
            candidates.setdefault(key, []).append((row.date, str(row.match_id)))
        return candidates

    def point_match_id(self, match_id: str) -> str | None:
        if match_id in self._link_cache:
            return self._link_cache[match_id]
        if match_id not in self.timeline_by_id.index:
            raise KeyError(match_id)
        row = self.timeline_by_id.loc[match_id]
        key = self._pair_key(
            str(row["player_A_name"]),
            str(row["player_B_name"]),
            int(row["date"].year),
            str(row["surface"]),
        )
        candidates = self._point_candidates.get(key, [])
        if not candidates:
            linked = None
        else:
            distance, linked = min(
                ((abs((date - row["date"]).days), candidate) for date, candidate in candidates),
                key=lambda item: item[0],
            )
            if distance > 35:
                linked = None
        self._link_cache[match_id] = linked
        return linked

    def point_orientation(self, match_id: str) -> int:
        """Return +1 when point A is ATP A, otherwise -1."""
        match = self.timeline_by_id.loc[match_id]
        point_id = self.point_match_id(match_id)
        if point_id is None:
            raise KeyError(f"No point-by-point replay for {match_id}")
        point = self.point_matches.loc[self.point_matches["match_id"] == point_id].iloc[0]
        return (
            1
            if _normalize(point["player_A_name"]) == _normalize(match["player_A_name"])
            else -1
        )

    def _coverage_count(self, edition: pd.DataFrame) -> int:
        return sum(self.point_match_id(str(match_id)) is not None for match_id in edition["match_id"])

    def get_tournaments(self, *, limit: int = 24) -> list[dict[str, Any]]:
        recent = self.timeline.loc[self.timeline["date"].dt.year >= 2023]
        rows = []
        for edition_id, edition in recent.groupby("tournament_edition_id", sort=False):
            first = edition.iloc[0]
            rows.append(
                {
                    "id": str(edition_id),
                    "name": str(first["tournament"]),
                    "start_date": first["date"].date().isoformat(),
                    "surface": str(first["surface"]),
                    "level": str(first["level"]),
                    "matches": len(edition),
                    "players": int(
                        pd.concat([edition["player_A_id"], edition["player_B_id"]]).nunique()
                    ),
                    "replay_matches": self._coverage_count(edition),
                    "status": "historical",
                    "provider": self.provider_name,
                }
            )
        rows.sort(key=lambda row: (row["replay_matches"], row["start_date"]), reverse=True)
        return rows[:limit]

    def get_tournament(self, tournament_id: str) -> dict[str, Any]:
        edition = self.timeline.loc[self.timeline["tournament_edition_id"] == tournament_id]
        if edition.empty:
            raise KeyError(tournament_id)
        first = edition.iloc[0]
        rounds = sorted(
            edition["round"].dropna().unique(), key=lambda value: ROUND_ORDER.get(str(value), -1)
        )
        return {
            "id": tournament_id,
            "name": str(first["tournament"]),
            "start_date": first["date"].date().isoformat(),
            "surface": str(first["surface"]),
            "level": str(first["level"]),
            "matches": len(edition),
            "players": int(
                pd.concat([edition["player_A_id"], edition["player_B_id"]]).nunique()
            ),
            "rounds": [str(value) for value in rounds],
            "replay_matches": self._coverage_count(edition),
            "provider": self.provider_name,
        }

    def _match_payload(self, row: pd.Series) -> dict[str, Any]:
        winner = "A" if int(row["player_A_win"]) else "B"
        point_id = self.point_match_id(str(row["match_id"]))
        return {
            "id": str(row["match_id"]),
            "tournament_id": str(row["tournament_edition_id"]),
            "tournament_name": str(row["tournament"]),
            "date": row["date"].date().isoformat(),
            "surface": str(row["surface"]),
            "level": str(row["level"]),
            "round": str(row["round"]),
            "round_order": int(row["round_order"]),
            "source_match_num": int(row["source_match_num"]),
            "best_of": int(row["best_of"]),
            "duration_minutes": _json_value(row["minutes"]),
            "score": _json_value(row["score"]),
            "winner": winner,
            "player_a": {
                "id": str(row["player_A_id"]),
                "name": str(row["player_A_name"]),
                "rank": _json_value(row["rank_A"]),
                "rank_points": _json_value(row["rank_points_A"]),
            },
            "player_b": {
                "id": str(row["player_B_id"]),
                "name": str(row["player_B_name"]),
                "rank": _json_value(row["rank_B"]),
                "rank_points": _json_value(row["rank_points_B"]),
            },
            "live_replay_available": point_id is not None,
            "prediction_available": bool(row["is_elo_eligible"]),
            "point_match_id": point_id,
        }

    def get_matches(self, tournament_id: str) -> list[dict[str, Any]]:
        edition = self.timeline.loc[self.timeline["tournament_edition_id"] == tournament_id]
        if edition.empty:
            raise KeyError(tournament_id)
        edition = edition.sort_values(["round_order", "source_match_num"], kind="stable")
        return [self._match_payload(row) for _, row in edition.iterrows()]

    def get_bracket(self, tournament_id: str) -> list[dict[str, Any]]:
        matches = self.get_matches(tournament_id)
        rounds: dict[str, list[dict[str, Any]]] = {}
        for match in matches:
            rounds.setdefault(match["round"], []).append(match)
        return [
            {"round": name, "round_order": ROUND_ORDER.get(name, -1), "matches": values}
            for name, values in sorted(rounds.items(), key=lambda item: ROUND_ORDER.get(item[0], -1))
        ]

    def get_match(self, match_id: str) -> dict[str, Any]:
        if match_id not in self.timeline_by_id.index:
            raise KeyError(match_id)
        return self._match_payload(self.timeline_by_id.loc[match_id])

    def point_frame(self, match_id: str) -> pd.DataFrame:
        if match_id in self._point_frame_cache:
            return self._point_frame_cache[match_id]
        point_match_id = self.point_match_id(match_id)
        if point_match_id is None:
            raise KeyError(f"No point-by-point replay for {match_id}")
        frame = pd.read_parquet(
            PROCESSED_DIR / "point_features_v2.parquet",
            filters=[("match_id", "==", point_match_id)],
        )
        if frame.empty:
            raise KeyError(f"Point features missing for {point_match_id}")
        result = frame.reset_index(drop=True)
        if len(self._point_frame_cache) >= 128:
            self._point_frame_cache.pop(next(iter(self._point_frame_cache)))
        self._point_frame_cache[match_id] = result
        return result

    def get_live_score(self, match_id: str, point_index: int) -> dict[str, Any]:
        frame = self.point_frame(match_id)
        index = min(max(int(point_index), 0), len(frame) - 1)
        row = frame.iloc[index]
        orientation = self.point_orientation(match_id)
        sets = [int(row["sets_A"]), int(row["sets_B"])]
        games = [int(row["games_A"]), int(row["games_B"])]
        points = [str(row["point_score_A"]), str(row["point_score_B"])]
        server = str(row["server"])
        winner = str(row["point_winner"])
        if orientation == -1:
            sets.reverse()
            games.reverse()
            points.reverse()
            server = "B" if server == "A" else "A"
            winner = "B" if winner == "A" else "A"
        return {
            "point_index": index,
            "total_points": len(frame),
            "sets": sets,
            "games": games,
            "points": points,
            "server": server,
            "is_tiebreak": bool(row["is_tiebreak"]),
            "next_point_winner": winner,
        }

    def get_point_by_point(self, match_id: str) -> list[dict[str, Any]]:
        frame = self.point_frame(match_id)
        columns = [
            "point_number",
            "sets_A",
            "sets_B",
            "games_A",
            "games_B",
            "point_score_A",
            "point_score_B",
            "server",
            "point_winner",
            "ace",
            "double_fault",
        ]
        return [
            {column: _json_value(value) for column, value in row.items()}
            for row in frame[columns].to_dict("records")
        ]
