from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

_SET = re.compile(r"^(?P<a>\d{1,2})-(?P<b>\d{1,2})(?:\(\d{1,2}\))?$")
_MATCH_TIEBREAK = re.compile(r"^\[(?P<a>\d{1,2})-(?P<b>\d{1,2})\]$")
_NON_PLAYED = {"W/O", "WO", "WALKOVER", "DEF", "DEF.", "DEFAULT"}
_STOP = {"RET", "ABD", "ABN", "ABANDONED", "UNFINISHED"}


@dataclass(frozen=True)
class ScoreWorkload:
    sets: int | None
    games: int | None
    match_tiebreaks: int
    status: str


def parse_score_workload(value: Any) -> ScoreWorkload:
    """Parse only workload explicitly present in a winner-perspective score.

    A bracketed match tie-break counts as one played set but zero standard
    games. Partial sets before RET count their explicitly recorded games.
    Ambiguous tokens make both workload totals unavailable.
    """
    if value is None or pd.isna(value) or not str(value).strip():
        return ScoreWorkload(None, None, 0, "missing")
    text = str(value).strip().upper().replace(";", " ")
    for verbose, compact in (
        ("PLAYED AND UNFINISHED", "ABD"),
        ("PLAYED AND ABANDONED", "ABD"),
    ):
        text = text.replace(verbose, compact)
    tokens = text.split()
    if any(token in _NON_PLAYED for token in tokens):
        return ScoreWorkload(0, 0, 0, "not_played")

    sets = 0
    games = 0
    match_tiebreaks = 0
    stopped = False
    for token in tokens:
        if token in _STOP:
            stopped = True
            continue
        set_match = _SET.fullmatch(token)
        if set_match:
            sets += 1
            games += int(set_match["a"]) + int(set_match["b"])
            continue
        tiebreak_match = _MATCH_TIEBREAK.fullmatch(token)
        if tiebreak_match:
            sets += 1
            match_tiebreaks += 1
            continue
        return ScoreWorkload(None, None, match_tiebreaks, "ambiguous")
    if not sets:
        return ScoreWorkload(None, None, match_tiebreaks, "ambiguous")
    return ScoreWorkload(sets, games, match_tiebreaks, "retired" if stopped else "complete")
