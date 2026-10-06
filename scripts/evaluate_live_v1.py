#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis_analytics.live.evaluation import finalize_existing_evaluation, run_live_evaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Historical evaluation for LIVE TENNIS V1")
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="Regenerate summary/plot from existing replay artifacts",
    )
    args = parser.parse_args()
    result = finalize_existing_evaluation() if args.finalize_existing else run_live_evaluation()
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
