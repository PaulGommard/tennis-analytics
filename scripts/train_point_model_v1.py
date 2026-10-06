#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis_analytics.modeling.point_model import run_point_model

if __name__ == "__main__":
    result = run_point_model()
    printable = {
        key: value
        for key, value in result.items()
        if key not in {"results", "segments", "coefficients", "score_effect"}
    }
    print(json.dumps(printable, indent=2, ensure_ascii=False))
