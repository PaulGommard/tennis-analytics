from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

SHOT_TYPES = {
    "f": ("forehand_groundstroke", "forehand"),
    "b": ("backhand_groundstroke", "backhand"),
    "r": ("forehand_slice", "forehand"),
    "s": ("backhand_slice", "backhand"),
    "v": ("forehand_volley", "forehand"),
    "z": ("backhand_volley", "backhand"),
    "o": ("overhead", "forehand"),
    "p": ("backhand_overhead", "backhand"),
    "u": ("forehand_dropshot", "forehand"),
    "y": ("backhand_dropshot", "backhand"),
    "l": ("forehand_lob", "forehand"),
    "m": ("backhand_lob", "backhand"),
    "h": ("forehand_half_volley", "forehand"),
    "i": ("backhand_half_volley", "backhand"),
    "j": ("forehand_swinging_volley", "forehand"),
    "k": ("backhand_swinging_volley", "backhand"),
    "t": ("trick_shot", "unknown"),
    "q": ("unknown_shot", "unknown"),
}
SHOT_DIRECTIONS = {"0": "unknown", "1": "zone_1", "2": "middle", "3": "zone_3"}
SERVE_DIRECTIONS = {"0": "unknown", "4": "wide", "5": "body", "6": "t"}
RETURN_DEPTHS = {"0": "unknown", "7": "shallow", "8": "medium", "9": "deep"}
ERROR_TYPES = {
    "n": "net",
    "w": "wide",
    "d": "deep",
    "x": "wide_and_deep",
    "e": "unknown",
    "!": "shank",
    "g": "foot_fault",
}
POSITION_CODES = {"+": "approach", "-": "net", "=": "baseline"}
SPECIAL_POINTS = {
    "P": "penalty_against_server",
    "Q": "penalty_against_returner",
    "R": "unobserved_returner_won",
    "S": "unobserved_server_won",
}


@dataclass(frozen=True)
class ShotEvent:
    shot_index: int
    active_shot_index: int | None
    hitter: str
    is_serve: bool
    serve_number: int | None
    serve_direction: str | None
    shot_code: str
    shot_type: str
    wing: str
    direction: str | None
    depth: str | None
    position: str
    is_approach: bool
    at_net: bool
    net_cord: bool
    stop_volley: bool
    is_shank: bool
    outcome: str
    error_type: str | None
    is_terminal: bool
    raw_token: str
    has_explicit_unknown: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParseResult:
    valid: bool
    events: tuple[ShotEvent, ...]
    active_serve_number: int | None
    terminal_outcome: str | None
    derived_winner: str | None
    reason: str | None
    special_code: str | None = None


def _relation(wing: str, direction: str | None, hand: str | None) -> str | None:
    if direction in (None, "unknown") or wing == "unknown":
        return "unknown"
    if direction == "middle":
        return "middle"
    zone = 1 if direction == "zone_1" else 3
    cross_zone = 1 if (hand or "R").upper() == "R" else 3
    if wing == "backhand":
        cross_zone = 3 if cross_zone == 1 else 1
    return "crosscourt" if zone == cross_zone else "down_the_line"


def direction_relation(event: ShotEvent, hitter_hand: str | None) -> str | None:
    """Return crosscourt/line semantics without mutating the lossless raw direction."""
    return _relation(event.wing, event.direction, hitter_hand)


def _serve_event(
    raw: str,
    serve_number: int,
    shot_index: int,
    active: bool,
) -> tuple[ShotEvent | None, str, str | None]:
    pos = 0
    lets = 0
    while pos < len(raw) and raw[pos] == "c":
        lets += 1
        pos += 1
    if pos >= len(raw) or raw[pos] not in SERVE_DIRECTIONS:
        return None, raw, "serve_direction_missing"
    direction_code = raw[pos]
    pos += 1
    serve_and_volley = False
    net_cord = lets > 0
    caret_modifier = False
    while pos < len(raw) and raw[pos] in "+;^":
        serve_and_volley |= raw[pos] == "+"
        net_cord |= raw[pos] == ";"
        caret_modifier |= raw[pos] == "^"
        pos += 1
    outcome = "in_play"
    error_type = None
    terminal = False
    if pos < len(raw) and raw[pos] == "*":
        outcome, terminal, pos = "ace", True, pos + 1
    elif pos < len(raw) and raw[pos] == "#":
        outcome, terminal, pos = "unreturned", True, pos + 1
    elif pos < len(raw) and raw[pos] in ERROR_TYPES:
        error_type = ERROR_TYPES[raw[pos]]
        outcome, pos = "fault", pos + 1
        terminal = serve_number == 2
        while pos < len(raw) and raw[pos] in ";^":
            net_cord |= raw[pos] == ";"
            caret_modifier |= raw[pos] == "^"
            pos += 1
    token = raw[:pos]
    active_index = 1 if active else None
    event = ShotEvent(
        shot_index=shot_index,
        active_shot_index=active_index,
        hitter="server",
        is_serve=True,
        serve_number=serve_number,
        serve_direction=SERVE_DIRECTIONS[direction_code],
        shot_code="serve",
        shot_type="serve",
        wing="unknown",
        direction=None,
        depth=None,
        position="approach" if serve_and_volley else "baseline",
        is_approach=serve_and_volley,
        at_net=False,
        net_cord=net_cord,
        stop_volley=False,
        is_shank=error_type == "shank",
        outcome=outcome,
        error_type=error_type,
        is_terminal=terminal,
        raw_token=token,
        has_explicit_unknown=(
            direction_code == "0" or error_type == "unknown" or caret_modifier
        ),
    )
    return event, raw[pos:], None


def _shot_event(
    raw: str,
    shot_index: int,
    active_index: int,
    hitter: str,
    is_return: bool,
) -> tuple[ShotEvent | None, str, str | None]:
    if not raw or raw[0] not in SHOT_TYPES:
        return None, raw, f"unknown_shot_code:{raw[:1] or 'eof'}"
    code = raw[0]
    shot_type, wing = SHOT_TYPES[code]
    pos = 1
    position = "net" if code in "vzop hijk".replace(" ", "") else "baseline"
    approach = False
    net_cord = False
    stop_volley = False
    direction = None
    depth = None
    error_codes: list[str] = []
    while pos < len(raw):
        char = raw[pos]
        if char in SHOT_TYPES or char in "*@#C":
            break
        if char in "+-=;^c":
            approach |= char == "+"
            net_cord |= char in ";c"
            stop_volley |= char == "^"
            if char in POSITION_CODES:
                position = POSITION_CODES[char]
        elif char in SHOT_DIRECTIONS and direction is None:
            direction = SHOT_DIRECTIONS[char]
        elif char in RETURN_DEPTHS and depth is None:
            # Recent MCP charts use the original return-depth extension on all strokes.
            depth = RETURN_DEPTHS[char]
        elif char in ERROR_TYPES:
            error_codes.append(ERROR_TYPES[char])
        else:
            return None, raw, f"unknown_shot_attribute:{char}"
        pos += 1
    outcome = "in_play"
    error_type = "+".join(dict.fromkeys(error_codes)) or None
    terminal = False
    if pos < len(raw) and raw[pos] == "*":
        if error_codes:
            return None, raw, "error_code_on_winner"
        outcome, terminal, pos = "winner", True, pos + 1
    elif pos < len(raw) and raw[pos] in "@#":
        outcome = "unforced_error" if raw[pos] == "@" else "forced_error"
        terminal, pos = True, pos + 1
    elif pos < len(raw) and raw[pos] == "C":
        return None, raw, "incorrect_challenge"
    elif error_codes:
        return None, raw, "error_without_terminal_marker"
    token = raw[:pos]
    event = ShotEvent(
        shot_index=shot_index,
        active_shot_index=active_index,
        hitter=hitter,
        is_serve=False,
        serve_number=None,
        serve_direction=None,
        shot_code=code,
        shot_type=shot_type,
        wing=wing,
        direction=direction,
        depth=depth,
        position=position,
        is_approach=approach,
        at_net=position == "net",
        net_cord=net_cord,
        stop_volley=stop_volley,
        is_shank="shank" in error_codes,
        outcome=outcome,
        error_type=error_type,
        is_terminal=terminal,
        raw_token=token,
        has_explicit_unknown=(
            code == "q"
            or direction == "unknown"
            or depth == "unknown"
            or (error_type is not None and "unknown" in error_type)
        ),
    )
    return event, raw[pos:], None


def parse_point(first: Any, second: Any = None) -> ParseResult:
    """Parse MCP cells strictly; unsupported or ambiguous points are explicit failures."""
    first_raw = "" if first is None else str(first).strip()
    second_raw = "" if second is None else str(second).strip()
    if first_raw.lower() == "nan":
        first_raw = ""
    if second_raw.lower() == "nan":
        second_raw = ""
    if not first_raw:
        return ParseResult(False, (), None, None, None, "missing_first_serve")
    if first_raw in SPECIAL_POINTS and not second_raw:
        return ParseResult(
            False, (), None, None, None, "special_point", SPECIAL_POINTS[first_raw]
        )
    active_number = 2 if second_raw else 1
    events: list[ShotEvent] = []
    cells = [(1, first_raw, active_number == 1)]
    if second_raw:
        cells.append((2, second_raw, True))
    for serve_number, cell, active in cells:
        event, remainder, error = _serve_event(cell, serve_number, len(events) + 1, active)
        if error or event is None:
            return ParseResult(False, tuple(events), active_number, None, None, error)
        events.append(event)
        if not active:
            if event.outcome != "fault" or remainder:
                return ParseResult(
                    False, tuple(events), active_number, None, None, "first_serve_not_fault"
                )
            continue
        if event.is_terminal:
            if remainder:
                return ParseResult(
                    False, tuple(events), active_number, event.outcome, None, "trailing_after_terminal"
                )
            winner = "server" if event.outcome in ("ace", "unreturned") else "receiver"
            terminal = "double_fault" if serve_number == 2 and event.outcome == "fault" else event.outcome
            return ParseResult(True, tuple(events), active_number, terminal, winner, None)
        if event.outcome == "fault":
            if serve_number == 2 and not remainder:
                return ParseResult(True, tuple(events), active_number, "double_fault", "receiver", None)
            return ParseResult(False, tuple(events), active_number, None, None, "active_serve_fault")
        hitter = "receiver"
        active_index = 2
        while remainder:
            shot, new_remainder, error = _shot_event(
                remainder, len(events) + 1, active_index, hitter, active_index == 2
            )
            if error or shot is None:
                return ParseResult(False, tuple(events), active_number, None, None, error)
            if new_remainder == remainder:
                return ParseResult(False, tuple(events), active_number, None, None, "parser_stalled")
            events.append(shot)
            remainder = new_remainder
            if shot.is_terminal:
                if remainder:
                    return ParseResult(
                        False, tuple(events), active_number, shot.outcome, None, "trailing_after_terminal"
                    )
                winner = hitter if shot.outcome == "winner" else (
                    "server" if hitter == "receiver" else "receiver"
                )
                return ParseResult(True, tuple(events), active_number, shot.outcome, winner, None)
            hitter = "server" if hitter == "receiver" else "receiver"
            active_index += 1
    return ParseResult(False, tuple(events), active_number, None, None, "missing_terminal")
