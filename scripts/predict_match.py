#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis_analytics.prematch.predictor import PredictionInputError, PreMatchPredictor
from tennis_analytics.prematch.registry import append_prediction


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen PRE-MATCH V1 inference")
    parser.add_argument("--player-a", required=True)
    parser.add_argument("--player-b", required=True)
    parser.add_argument("--surface", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument(
        "--record", action="store_true", help="Append to immutable prospective ledger"
    )
    parser.add_argument("--snapshot", action="store_true", help="Include all model features")
    args = parser.parse_args()
    try:
        prediction = PreMatchPredictor().predict(
            player_a=args.player_a,
            player_b=args.player_b,
            surface=args.surface,
            match_date=args.date,
        )
        payload = prediction.to_dict(include_features=args.snapshot)
        if args.record:
            payload["recorded"] = append_prediction(prediction)
        print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
    except (PredictionInputError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
