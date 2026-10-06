#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis_analytics.live.phase8c import finalize_phase8c, run_phase8c

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 8C stabilization study")
    parser.add_argument(
        "--resume-test",
        action="store_true",
        help="Resume after blend/calibration selection using saved validation artifacts",
    )
    args = parser.parse_args()
    result = finalize_phase8c() if args.resume_test else run_phase8c()
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
