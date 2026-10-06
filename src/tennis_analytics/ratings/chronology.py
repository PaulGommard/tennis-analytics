from __future__ import annotations

import json
import tomllib
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from tennis_analytics.paths import PROJECT_ROOT

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "elo_phase3.toml"


def load_chronology_config(path: Path = DEFAULT_CONFIG) -> tuple[dict[str, int], set[str]]:
    with path.open("rb") as stream:
        values = tomllib.load(stream)["chronology"]
    order = {label: index for index, label in enumerate(values["ordered_rounds"])}
    # Bronze and championship matches depend on semifinals but not on each other.
    order["BR"] = order["F"]
    return order, set(values["conservative_levels"])


def _overlap_dates(timeline: pd.DataFrame) -> tuple[set[pd.Timestamp], pd.DataFrame]:
    appearances = pd.concat(
        [
            timeline[["date", "tournament_edition_id", "player_A_id"]].rename(
                columns={"player_A_id": "player_id"}
            ),
            timeline[["date", "tournament_edition_id", "player_B_id"]].rename(
                columns={"player_B_id": "player_id"}
            ),
        ],
        ignore_index=True,
    ).drop_duplicates()
    counts = (
        appearances.groupby(["date", "player_id"])["tournament_edition_id"]
        .nunique()
        .rename("edition_count")
        .reset_index()
    )
    overlaps = counts.loc[counts["edition_count"] > 1]
    return set(overlaps["date"]), overlaps


def apply_round_chronology(
    timeline: pd.DataFrame,
    *,
    config_path: Path = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Assign leakage-safe batches using only structural round information."""
    order, conservative_levels = load_chronology_config(config_path)
    data = timeline.loc[timeline["is_elo_eligible"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    overlap_dates, _ = _overlap_dates(data)
    data["chronology_strategy"] = "round"
    data["chronology_stage"] = data["round"].astype("string")
    data.loc[data["round"].isin(["F", "BR"]), "chronology_stage"] = "FINAL"
    data["round_order"] = data["round"].map(order).astype("Int64")

    unknown_editions = set(
        data.loc[data["round_order"].isna(), "tournament_edition_id"].astype(str)
    )
    conservative_editions = (
        set(data.loc[data["level"].isin(conservative_levels), "tournament_edition_id"].astype(str))
        | unknown_editions
    )
    conservative_mask = data["tournament_edition_id"].astype(str).isin(conservative_editions)
    data.loc[conservative_mask, "chronology_strategy"] = "tournament_conservative"
    data.loc[conservative_mask, "chronology_stage"] = "ALL"
    data.loc[conservative_mask, "round_order"] = 0

    overlap_mask = data["date"].isin(overlap_dates)
    data.loc[overlap_mask, "chronology_strategy"] = "date_overlap_conservative"
    data.loc[overlap_mask, "chronology_stage"] = "ALL"
    data.loc[overlap_mask, "round_order"] = 0

    batch_id = 0
    batch_ids = pd.Series(index=data.index, dtype="int64")
    chronology_order = pd.Series(index=data.index, dtype="int64")
    sequence = 0
    for date, date_frame in data.groupby("date", sort=True):
        if date in overlap_dates:
            indexes = date_frame.index
            batch_ids.loc[indexes] = batch_id
            chronology_order.loc[indexes] = sequence
            batch_id += 1
            sequence += 1
            continue
        # Player sets are disjoint across editions on this date, so edition
        # ordering cannot alter any participant's pre-match state.
        for _, edition in date_frame.groupby("tournament_edition_id", sort=True):
            for _, stage in edition.groupby("round_order", sort=True, dropna=False):
                indexes = stage.index
                batch_ids.loc[indexes] = batch_id
                chronology_order.loc[indexes] = sequence
                batch_id += 1
                sequence += 1
    data["chronology_batch_id"] = batch_ids.astype("int64")
    data["chronology_order"] = chronology_order.astype("int64")
    return data.sort_values(["chronology_order", "match_id"], kind="stable", ignore_index=True)


def chronology_audit(timeline: pd.DataFrame) -> dict[str, Any]:
    data = timeline.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    editions = data.groupby("tournament_edition_id").agg(
        matches=("match_id", "size"), dates=("date", "nunique")
    )
    eligible = data.loc[data["is_elo_eligible"]].copy()
    overlap_dates, overlaps = _overlap_dates(eligible)
    round_sets = eligible.groupby("tournament_edition_id")["round"].agg(
        lambda values: "|".join(sorted(set(values.dropna().astype(str))))
    )
    special = {
        "qualifying_matches": int(eligible["round"].astype(str).str.match(r"^Q[1-4]$").sum()),
        "round_robin_matches": int((eligible["round"] == "RR").sum()),
        "early_round_matches": int((eligible["round"] == "ER").sum()),
        "bronze_matches": int((eligible["round"] == "BR").sum()),
        "davis_cup_matches": int((eligible["level"] == "D").sum()),
        "olympics_name_matches": int(
            eligible["tournament"].astype(str).str.contains("Olympic", case=False).sum()
        ),
        "tour_finals_name_matches": int(
            eligible["tournament"].astype(str).str.contains("Finals|Masters Cup", case=False).sum()
        ),
    }
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "editions": len(editions),
        "editions_all_matches_one_date": int((editions["dates"] == 1).sum()),
        "matches_in_single_date_editions": int(
            editions.loc[editions["dates"] == 1, "matches"].sum()
        ),
        "round_counts": {
            str(key): int(value) for key, value in data["round"].value_counts(dropna=False).items()
        },
        "round_set_counts": dict(Counter(round_sets).most_common()),
        "special_formats": special,
        "overlap_player_date_pairs": len(overlaps),
        "overlap_dates": sorted(date.date().isoformat() for date in overlap_dates),
        "overlap_details": overlaps.assign(date=overlaps["date"].dt.date.astype(str)).to_dict(
            "records"
        ),
        "decision": {
            "standard_order": "Q1,Q2,Q3,Q4,RR,ER,R128,R64,R32,R16,QF,SF,FINAL",
            "same_round": "batch",
            "F_and_BR": "same FINAL batch",
            "Davis_Cup": "whole tournament edition remains one conservative batch",
            "unknown_round": "whole tournament edition remains one conservative batch",
            "cross_edition_player_overlap": "whole date remains one conservative batch",
        },
    }


def write_chronology_audit(audit: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "round_chronology_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    rounds = "\n".join(f"- `{label}`: {count:,}" for label, count in audit["round_counts"].items())
    special = "\n".join(
        f"- {label}: {count:,}" for label, count in audit["special_formats"].items()
    )
    text = f"""# Intra-tournament chronology audit

- Editions: {audit["editions"]:,}
- Editions with all matches on one date: {audit["editions_all_matches_one_date"]:,}
- Matches concerned: {audit["matches_in_single_date_editions"]:,}
- Cross-edition player/date overlaps: {audit["overlap_player_date_pairs"]:,}
- Overlap dates: {", ".join(audit["overlap_dates"]) or "none"}

## Round labels

{rounds}

No qualifying labels Q1–Q4 occur in this main-tour collection.

## Special formats

{special}

`RR` includes Davis Cup ties, Tour Finals, Next Gen, Laver Cup, ATP Cup,
United Cup and the former Dusseldorf team event. `ER` occurs only in four 2007
round-robin experiments. `BR` is a bronze/third-place match.

## Safe policy

- Standard structural order: Q1 → Q2 → Q3 → Q4 → RR → ER → R128 → R64 → R32 → R16 → QF → SF → FINAL.
- Every round remains a batch; F and BR share FINAL.
- Davis Cup and unknown labels retain a whole-edition conservative batch.
- Any date with a player in multiple editions retains the whole-date batch.
"""
    (report_dir / "round_chronology_audit.md").write_text(text, encoding="utf-8")
