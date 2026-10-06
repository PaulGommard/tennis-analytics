#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis_analytics.prematch.registry import evaluate_ledger

if __name__ == "__main__":
    print(json.dumps(evaluate_ledger(), indent=2, ensure_ascii=False, allow_nan=False))
