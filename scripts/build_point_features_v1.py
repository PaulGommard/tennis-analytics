#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis_analytics.point_by_point.normalize import run_normalization

if __name__ == "__main__":
    print(json.dumps(run_normalization(), indent=2, ensure_ascii=False))
