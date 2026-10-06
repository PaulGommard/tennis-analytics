from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from tennis_analytics.paths import PROCESSED_DIR

LOGGER = logging.getLogger(__name__)
NON_PLAYED_PATTERNS = (
    ("walkover", r"WALKOVER|(?:^|\s)W/O(?:$|\s)|(?:^|\s)WO(?:$|\s)"),
    ("default", r"DEFAULT|(?:^|\s)DEF\.?(?:$|\s)"),
    ("abandoned", r"ABANDONED|UNFINISHED|(?:^|\s)ABD(?:$|\s)|(?:^|\s)ABN(?:$|\s)"),
)


def _read_tables(input_dir: Path) -> dict[str, pd.DataFrame]:
    names = ("matches", "match_participants", "players", "tournament_editions", "tournaments")
    missing = [name for name in names if not (input_dir / f"{name}.parquet").exists()]
    if missing:
        raise FileNotFoundError(f"Missing normalized table(s): {', '.join(missing)}")
    return {name: pd.read_parquet(input_dir / f"{name}.parquet") for name in names}


def _exclusion_reason(row: pd.Series) -> str | None:
    if pd.isna(row["date"]):
        return "invalid_date"
    if pd.isna(row["player_A_id"]) or pd.isna(row["player_B_id"]):
        return "missing_participant"
    if row["player_A_id"] == row["player_B_id"]:
        return "self_match"
    if row["winner_id"] not in (row["player_A_id"], row["player_B_id"]):
        return "winner_not_participant"
    score = "" if pd.isna(row["score"]) else str(row["score"]).upper()
    for reason, pattern in NON_PLAYED_PATTERNS:
        if pd.Series([score]).str.contains(pattern, regex=True).iloc[0]:
            return reason
    return None


def build_timeline(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    matches = tables["matches"].copy()
    participants = tables["match_participants"].copy()
    players = tables["players"][["player_id", "full_name"]]
    editions = tables["tournament_editions"]
    tournaments = tables["tournaments"]

    duplicate_slots = participants.duplicated(["match_id", "slot"], keep=False)
    if duplicate_slots.any():
        raise ValueError("Duplicate match/slot rows prevent a deterministic A/B timeline")
    slot_counts = participants.groupby("match_id")["slot"].agg(set)
    if not slot_counts.map(lambda slots: slots == {"A", "B"}).all():
        raise ValueError("Every match must contain exactly slots A and B")

    participant_columns = ["match_id", "player_id", "rank", "rank_points", "age_years"]
    slot_a = participants.loc[participants["slot"] == "A", participant_columns].rename(
        columns={column: f"{column}_A" for column in participant_columns if column != "match_id"}
    )
    slot_b = participants.loc[participants["slot"] == "B", participant_columns].rename(
        columns={column: f"{column}_B" for column in participant_columns if column != "match_id"}
    )
    names = players.set_index("player_id")["full_name"]

    timeline = (
        matches.merge(editions, on="tournament_edition_id", how="left", validate="many_to_one")
        .merge(tournaments, on="tournament_id", how="left", validate="many_to_one")
        .merge(slot_a, on="match_id", how="left", validate="one_to_one")
        .merge(slot_b, on="match_id", how="left", validate="one_to_one")
        .rename(
            columns={
                "name": "tournament",
                "winner_player_id": "winner_id",
                "player_id_A": "player_A_id",
                "player_id_B": "player_B_id",
                "rank_A": "rank_A",
                "rank_B": "rank_B",
            }
        )
    )
    timeline["player_A_name"] = timeline["player_A_id"].map(names)
    timeline["player_B_name"] = timeline["player_B_id"].map(names)
    timeline["player_A_win"] = (timeline["winner_id"] == timeline["player_A_id"]).astype("int8")
    timeline["date"] = pd.to_datetime(timeline["date"], errors="coerce")
    timeline["exclusion_reason"] = timeline.apply(_exclusion_reason, axis=1)
    timeline["is_elo_eligible"] = timeline["exclusion_reason"].isna()
    timeline["temporal_batch_size"] = (
        timeline.groupby("date")["match_id"].transform("size").astype("Int64")
    )
    timeline = timeline.sort_values(["date", "match_id"], kind="stable", ignore_index=True)
    output_columns = [
        "match_id",
        "date",
        "date_precision",
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
        "age_years_A",
        "age_years_B",
        "temporal_batch_size",
        "is_elo_eligible",
        "exclusion_reason",
    ]
    return timeline[output_columns]


def write_timeline(timeline: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".parquet.tmp")
    timeline.to_parquet(temporary, index=False)
    temporary.replace(output_path)
    metadata = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rows": len(timeline),
        "eligible_rows": int(timeline["is_elo_eligible"].sum()),
        "ab_slot_rule": "lexical ascending canonical player_id; inherited from normalized data",
        "chronology": (
            "date is tournament start; all matches with the same date are an unordered batch"
        ),
        "exclusions": timeline["exclusion_reason"].value_counts(dropna=True).to_dict(),
    }
    output_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description="Build the deterministic chronological match table"
    )
    command.add_argument("--input-dir", type=Path, default=PROCESSED_DIR / "atp")
    command.add_argument(
        "--output", type=Path, default=PROCESSED_DIR / "analytics" / "match_timeline.parquet"
    )
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    timeline = build_timeline(_read_tables(args.input_dir))
    write_timeline(timeline, args.output)
    LOGGER.info("Timeline written: %d rows", len(timeline))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
