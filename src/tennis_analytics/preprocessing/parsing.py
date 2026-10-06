from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import date
from typing import Any

import pandas as pd

SURFACES = {
    "hard": "Hard",
    "clay": "Clay",
    "grass": "Grass",
    "carpet": "Carpet",
}
TERMINAL_SCORE_TOKENS = {"RET", "W/O", "WO", "DEF", "ABD", "ABN"}
SET_SCORE = re.compile(r"^(?:\d{1,2}-\d{1,2}(?:\(\d{1,2}\))?|\[\d{1,2}-\d{1,2}\])$")


def parse_tourney_date(value: Any) -> date | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    text = text.removesuffix(".0")
    if not re.fullmatch(r"\d{8}", text):
        return None
    try:
        return date(int(text[:4]), int(text[4:6]), int(text[6:8]))
    except ValueError:
        return None


def parse_birth_date(value: Any) -> date | None:
    return parse_tourney_date(value)


def normalize_surface(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return SURFACES.get(text.casefold(), text.title())


def normalize_name(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return " ".join(str(value).strip().split()) or None


def name_key(value: Any) -> str:
    name = normalize_name(value) or "unknown"
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", ascii_name.casefold()).strip()


def canonical_player_id(
    source_player_id: Any,
    name: Any,
    birth_date: Any = None,
    country_code: Any = None,
) -> tuple[str, str]:
    numeric = pd.to_numeric(pd.Series([source_player_id]), errors="coerce").iloc[0]
    if pd.notna(numeric) and float(numeric).is_integer() and int(numeric) > 0:
        return f"atp:{int(numeric)}", "atp"
    identity = "|".join(
        (
            name_key(name),
            str(birth_date or ""),
            str(country_code or "").strip().upper(),
        )
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"synthetic:{digest}", "synthetic"


def valid_score(value: Any) -> bool:
    """Conservative syntax validation; it does not reconstruct tennis rules."""
    if value is None or pd.isna(value) or not str(value).strip():
        return True
    text = str(value).strip().upper().replace(";", " ")
    for verbose, compact in (
        ("PLAYED AND UNFINISHED", "ABD"),
        ("PLAYED AND ABANDONED", "ABD"),
        ("WALKOVER", "W/O"),
        ("DEFAULT", "DEF"),
        ("DEF.", "DEF"),
    ):
        text = text.replace(verbose, compact)
    tokens = text.split()
    if not tokens:
        return True
    seen_set = False
    for token in tokens:
        if token in TERMINAL_SCORE_TOKENS:
            continue
        if SET_SCORE.fullmatch(token):
            seen_set = True
            continue
        return False
    return seen_set or any(token in TERMINAL_SCORE_TOKENS for token in tokens)
