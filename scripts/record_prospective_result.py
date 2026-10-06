#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis_analytics.prematch.registry import record_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Record one immutable prospective result")
    parser.add_argument("prediction_id")
    parser.add_argument("result_a", type=int, choices=(0, 1))
    args = parser.parse_args()
    record_result(args.prediction_id, args.result_a)
    print(f"Recorded result for {args.prediction_id}")


if __name__ == "__main__":
    main()
