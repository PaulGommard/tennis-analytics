from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from tennis_analytics.paths import INTERIM_DIR, PROCESSED_DIR, RAW_DIR, ensure_data_directories
from tennis_analytics.preprocessing.parsing import (
    canonical_player_id,
    normalize_name,
    normalize_surface,
    parse_birth_date,
    parse_tourney_date,
)

LOGGER = logging.getLogger(__name__)

SIDE_FIELDS = {
    "source_player_id": "id",
    "full_name": "name",
    "hand": "hand",
    "height_cm": "ht",
    "country_code": "ioc",
    "age_years": "age",
    "rank": "rank",
    "rank_points": "rank_points",
    "aces": "ace",
    "double_faults": "df",
    "service_points": "svpt",
    "first_serve_in": "1stIn",
    "first_serve_won": "1stWon",
    "second_serve_won": "2ndWon",
    "service_games": "SvGms",
    "break_points_saved": "bpSaved",
    "break_points_faced": "bpFaced",
}
INTEGER_FIELDS = (
    "height_cm",
    "rank",
    "rank_points",
    "aces",
    "double_faults",
    "service_points",
    "first_serve_in",
    "first_serve_won",
    "second_serve_won",
    "service_games",
    "break_points_saved",
    "break_points_faced",
)
STAT_FIELDS = {
    "aces",
    "double_faults",
    "service_points",
    "first_serve_in",
    "first_serve_won",
    "second_serve_won",
    "service_games",
    "break_points_saved",
    "break_points_faced",
}


def _read_matches(paths: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(paths):
        LOGGER.info("Reading %s", path)
        frame = pd.read_csv(path, low_memory=False)
        frame["_source_file"] = path.name
        frame["_source_row"] = range(2, len(frame) + 2)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No atp_matches_YYYY.csv files found in the selected raw directory")
    return pd.concat(frames, ignore_index=True, sort=False)


def _participant_frame(raw: pd.DataFrame, side: str) -> pd.DataFrame:
    prefix = f"{side}_"
    stats_prefix = "w_" if side == "winner" else "l_"
    data: dict[str, Any] = {}
    for target, suffix in SIDE_FIELDS.items():
        source = f"{stats_prefix if target in STAT_FIELDS else prefix}{suffix}"
        data[target] = raw[source] if source in raw else pd.Series(pd.NA, index=raw.index)
    participants = pd.DataFrame(data)
    participants["outcome_role"] = side
    participants["is_winner"] = side == "winner"
    participants["_raw_index"] = raw.index
    participants["full_name"] = participants["full_name"].map(normalize_name)
    participants["hand"] = participants["hand"].astype("string").str.strip().str.upper()
    participants["country_code"] = (
        participants["country_code"].astype("string").str.strip().str.upper()
    )
    participants["age_years"] = pd.to_numeric(participants["age_years"], errors="coerce")
    for column in INTEGER_FIELDS:
        participants[column] = pd.to_numeric(participants[column], errors="coerce").astype("Int64")
    identities = [
        canonical_player_id(source_id, name, None, country)
        for source_id, name, country in zip(
            participants["source_player_id"],
            participants["full_name"],
            participants["country_code"],
            strict=True,
        )
    ]
    participants["player_id"] = [identity[0] for identity in identities]
    participants["id_source"] = [identity[1] for identity in identities]
    participants["source_player_id"] = pd.to_numeric(
        participants["source_player_id"], errors="coerce"
    ).astype("Int64")
    return participants


def _first_present(values: pd.Series) -> Any:
    available = values.dropna()
    return available.iloc[0] if not available.empty else pd.NA


def _load_reference_players(path: Path) -> pd.DataFrame:
    columns = [
        "player_id",
        "source_player_id",
        "full_name",
        "hand",
        "birth_date",
        "height_cm",
        "country_code",
        "id_source",
    ]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    reference = pd.read_csv(path, low_memory=False)
    first = reference.get("name_first", pd.Series("", index=reference.index)).fillna("")
    last = reference.get("name_last", pd.Series("", index=reference.index)).fillna("")
    full_names = (first.astype(str).str.strip() + " " + last.astype(str).str.strip()).str.strip()
    records = []
    for index, row in reference.iterrows():
        player_id, id_source = canonical_player_id(
            row.get("player_id"), full_names.iloc[index], row.get("dob"), row.get("ioc")
        )
        records.append(
            {
                "player_id": player_id,
                "source_player_id": row.get("player_id"),
                "full_name": normalize_name(full_names.iloc[index]),
                "hand": row.get("hand"),
                "birth_date": parse_birth_date(row.get("dob")),
                "height_cm": row.get("height"),
                "country_code": row.get("ioc"),
                "id_source": id_source,
            }
        )
    result = pd.DataFrame(records, columns=columns)
    result["source_player_id"] = pd.to_numeric(result["source_player_id"], errors="coerce").astype(
        "Int64"
    )
    result["height_cm"] = pd.to_numeric(result["height_cm"], errors="coerce").astype("Int64")
    result["hand"] = result["hand"].astype("string").str.strip().str.upper()
    result["country_code"] = result["country_code"].astype("string").str.strip().str.upper()
    return result


def _build_players(
    participants: pd.DataFrame, reference: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    match_players = participants[
        [
            "player_id",
            "source_player_id",
            "full_name",
            "hand",
            "height_cm",
            "country_code",
            "id_source",
        ]
    ].copy()
    match_players["birth_date"] = None
    # The dimension describes players in this selected match corpus, enriched
    # by the reference file; it is not the full all-era ATP player registry.
    if "player_id" in reference:
        reference = reference[reference["player_id"].isin(match_players["player_id"])].copy()
    else:
        reference = match_players.iloc[0:0].copy()
    combined = pd.concat([reference, match_players], ignore_index=True, sort=False)

    issues: list[dict[str, Any]] = []
    for player_id, group in combined.groupby("player_id", dropna=False):
        for field in ("source_player_id", "full_name", "birth_date"):
            values = group[field].dropna().astype(str).unique()
            if len(values) > 1:
                issues.append(
                    {
                        "issue_type": f"conflicting_{field}",
                        "player_id": player_id,
                        "values": " | ".join(sorted(values)),
                    }
                )

    players = (
        combined.groupby("player_id", as_index=False, sort=True)
        .agg(
            source_player_id=("source_player_id", _first_present),
            full_name=("full_name", _first_present),
            hand=("hand", _first_present),
            birth_date=("birth_date", _first_present),
            height_cm=("height_cm", _first_present),
            country_code=("country_code", _first_present),
            id_source=("id_source", _first_present),
        )
        .reset_index(drop=True)
    )
    players["source_player_id"] = pd.to_numeric(
        players["source_player_id"], errors="coerce"
    ).astype("Int64")
    players["height_cm"] = pd.to_numeric(players["height_cm"], errors="coerce").astype("Int64")
    return players, pd.DataFrame(issues, columns=["issue_type", "player_id", "values"])


def normalize(raw: pd.DataFrame, reference_players: pd.DataFrame) -> dict[str, pd.DataFrame]:
    raw = raw.copy()
    raw["tournament_start_date"] = raw["tourney_date"].map(parse_tourney_date)
    raw["surface"] = raw.get("surface", pd.Series(pd.NA, index=raw.index)).map(normalize_surface)
    raw["tourney_name"] = raw["tourney_name"].map(normalize_name)
    raw["source_tourney_id"] = raw["tourney_id"].astype("string").str.strip()
    fallback_tourney = (
        raw["tourney_name"].fillna("unknown").str.casefold().str.replace(r"\W+", "-", regex=True)
    )
    raw["tournament_id"] = "atp:" + raw["source_tourney_id"].fillna(fallback_tourney)
    date_text = raw["tournament_start_date"].map(
        lambda value: value.isoformat() if value else "invalid"
    )
    raw["tournament_edition_id"] = raw["tournament_id"] + ":" + date_text

    winner = _participant_frame(raw, "winner")
    loser = _participant_frame(raw, "loser")
    participants = pd.concat([winner, loser], ignore_index=True)
    winner_ids = winner.set_index("_raw_index")["player_id"]
    loser_ids = loser.set_index("_raw_index")["player_id"]

    match_number = pd.to_numeric(
        raw.get("match_num", pd.Series(pd.NA, index=raw.index)), errors="coerce"
    ).astype("Int64")
    match_number_text = match_number.astype("string").fillna(
        "row-" + raw["_source_row"].astype(str)
    )
    natural_key = (
        "atp:"
        + raw["source_tourney_id"].fillna("unknown")
        + ":"
        + date_text
        + ":"
        + match_number_text
    )
    occurrence = natural_key.groupby(natural_key).cumcount()
    raw["natural_match_key"] = natural_key
    raw["match_id"] = natural_key + occurrence.map(
        lambda value: "" if value == 0 else f":dup{value}"
    )

    matches = pd.DataFrame(
        {
            "match_id": raw["match_id"],
            "natural_match_key": raw["natural_match_key"],
            "tournament_edition_id": raw["tournament_edition_id"],
            # Source only supplies tournament start date, not the exact match day.
            "date": raw["tournament_start_date"],
            "date_precision": "tournament_start",
            "round": raw.get("round"),
            "best_of": pd.to_numeric(raw.get("best_of"), errors="coerce").astype("Int64"),
            "minutes": pd.to_numeric(raw.get("minutes"), errors="coerce").astype("Int64"),
            "score": raw.get("score"),
            "score_perspective": "winner",
            "winner_player_id": raw.index.to_series().map(winner_ids).to_numpy(),
            "source_match_num": match_number,
            "source_file": raw["_source_file"],
            "source_row": raw["_source_row"].astype("Int64"),
        }
    )

    participant_match_ids = participants["_raw_index"].map(raw["match_id"])
    opposing_winner = participants["_raw_index"].map(winner_ids)
    opposing_loser = participants["_raw_index"].map(loser_ids)
    other_ids = opposing_winner.where(~participants["is_winner"], opposing_loser)
    # Slot assignment is based only on canonical IDs, never on outcome.
    participants["slot"] = "A"
    participants.loc[participants["player_id"] > other_ids.to_numpy(), "slot"] = "B"
    participants["match_id"] = participant_match_ids.to_numpy()

    match_participants = participants[
        [
            "match_id",
            "player_id",
            "slot",
            "is_winner",
            "rank",
            "rank_points",
            "age_years",
            "id_source",
            "source_player_id",
        ]
    ].sort_values(["match_id", "slot"], ignore_index=True)
    match_stats = participants[
        [
            "match_id",
            "player_id",
            "aces",
            "double_faults",
            "service_points",
            "first_serve_in",
            "first_serve_won",
            "second_serve_won",
            "service_games",
            "break_points_saved",
            "break_points_faced",
        ]
    ].sort_values(["match_id", "player_id"], ignore_index=True)

    tournaments = raw[["tournament_id", "source_tourney_id", "tourney_name"]].rename(
        columns={"tourney_name": "name"}
    )
    tournaments = (
        tournaments.groupby("tournament_id", as_index=False, sort=True)
        .agg(source_tourney_id=("source_tourney_id", _first_present), name=("name", _first_present))
        .reset_index(drop=True)
    )
    editions_source = pd.DataFrame(
        {
            "tournament_edition_id": raw["tournament_edition_id"],
            "tournament_id": raw["tournament_id"],
            "start_date": raw["tournament_start_date"],
            "surface": raw["surface"],
            "level": raw.get("tourney_level"),
            "draw_size": pd.to_numeric(raw.get("draw_size"), errors="coerce").astype("Int64"),
        }
    )
    editions = (
        editions_source.groupby("tournament_edition_id", as_index=False, sort=True)
        .agg(
            tournament_id=("tournament_id", _first_present),
            start_date=("start_date", _first_present),
            surface=("surface", _first_present),
            level=("level", _first_present),
            draw_size=("draw_size", _first_present),
        )
        .reset_index(drop=True)
    )

    players, identity_issues = _build_players(participants, reference_players)
    return {
        "players": players,
        "tournaments": tournaments,
        "tournament_editions": editions,
        "matches": matches,
        "match_participants": match_participants,
        "match_stats": match_stats,
        "identity_issues": identity_issues,
    }


def write_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        destination = (INTERIM_DIR if name == "identity_issues" else output_dir) / f"{name}.parquet"
        temporary = destination.with_suffix(".parquet.tmp")
        LOGGER.info("Writing %s (%d rows)", destination, len(frame))
        frame.to_parquet(temporary, index=False, engine="pyarrow")
        temporary.replace(destination)
    metadata = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "tables": {name: len(frame) for name, frame in tables.items()},
        "notes": {
            "date": "ATP match files provide tournament start date, not exact match date.",
            "score": "The score text remains in the source winner's perspective.",
            "slots": "A/B is assigned by canonical player_id ordering, independently of outcome.",
        },
    }
    (output_dir / "_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Normalize raw ATP CSV files to Parquet tables")
    command.add_argument("--start-year", type=int)
    command.add_argument("--end-year", type=int)
    command.add_argument("--raw-dir", type=Path, default=RAW_DIR / "atp_archive")
    command.add_argument("--output-dir", type=Path, default=PROCESSED_DIR / "atp")
    command.add_argument("--verbose", action="store_true")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    ensure_data_directories()
    paths = list((args.raw_dir / "matches").glob("atp_matches_*.csv"))
    if args.start_year is not None:
        paths = [path for path in paths if int(path.stem.rsplit("_", 1)[1]) >= args.start_year]
    if args.end_year is not None:
        paths = [path for path in paths if int(path.stem.rsplit("_", 1)[1]) <= args.end_year]
    raw = _read_matches(paths)
    reference = _load_reference_players(args.raw_dir / "metadata" / "atp_players.csv")
    tables = normalize(raw, reference)
    write_tables(tables, args.output_dir)
    LOGGER.info("Normalization complete: %d matches", len(tables["matches"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
