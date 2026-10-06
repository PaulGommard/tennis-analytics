#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis_analytics.live.predictor import LiveInputError, predict_live_state


def pair(value: str) -> tuple[str, str]:
    parts = value.upper().split("-", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected A-B")
    return parts[0], parts[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="LIVE TENNIS V1")
    parser.add_argument("--player-a", required=True)
    parser.add_argument("--player-b", required=True)
    parser.add_argument("--surface", required=True)
    parser.add_argument("--sets", type=pair, required=True)
    parser.add_argument("--games", type=pair, required=True)
    parser.add_argument("--points", type=pair, required=True)
    parser.add_argument("--server", choices=("A", "B", "a", "b"), required=True)
    parser.add_argument(
        "--first-server",
        choices=("A", "B", "a", "b"),
        help="Server of the first match point; current server is assumed if omitted",
    )
    parser.add_argument("--tiebreak", action="store_true")
    parser.add_argument("--best-of", type=int, choices=(3, 5), default=3)
    parser.add_argument("--final-tiebreak-at", type=int, default=6)
    parser.add_argument("--final-tiebreak-points", type=int, default=7)
    parser.add_argument("--advantage-final-set", action="store_true")
    parser.add_argument(
        "--prematch-a", type=float, help="Frozen prior; otherwise PRE-MATCH V1 runs"
    )
    parser.add_argument("--date", help="ISO date used only when PRE-MATCH V1 must run")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        prediction = predict_live_state(
            player_a=args.player_a,
            player_b=args.player_b,
            surface=args.surface,
            sets_a=int(args.sets[0]),
            sets_b=int(args.sets[1]),
            games_a=int(args.games[0]),
            games_b=int(args.games[1]),
            points_a=args.points[0],
            points_b=args.points[1],
            server=args.server,
            first_server=args.first_server,
            is_tiebreak=args.tiebreak,
            best_of=args.best_of,
            final_set_tiebreak=not args.advantage_final_set,
            final_set_tiebreak_at=args.final_tiebreak_at,
            final_set_tiebreak_points=args.final_tiebreak_points,
            prematch_probability_a=args.prematch_a,
            match_date=args.date,
        )
    except (LiveInputError, ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(prediction.to_dict(), indent=2, ensure_ascii=False))
        return
    print("LIVE TENNIS V1\n")
    print(f"{prediction.player_a} vs {prediction.player_b} — {prediction.surface}\n")
    print(f"Next point — A:   {prediction.p_next_point_a:.1%}")
    print(f"Current game — A: {prediction.p_current_game_a:.1%}")
    print(f"Current set — A:  {prediction.p_current_set_a:.1%}")
    print(f"Match — A:        {prediction.p_match_a:.1%}")
    print(f"\nPRE-MATCH A:      {prediction.prematch_probability_a:.1%}")
    if prediction.warnings:
        print("\nWarnings:")
        for warning in prediction.warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
