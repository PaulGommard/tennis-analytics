#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis_analytics.point_by_point.audit import run_audit

if __name__ == "__main__":
    print(json.dumps(run_audit(), indent=2, ensure_ascii=False))
