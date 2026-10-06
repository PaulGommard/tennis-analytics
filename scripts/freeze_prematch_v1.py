#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    source = ROOT / "data" / "processed" / "models" / "logistic_phase5.joblib"
    destination = ROOT / "models" / "prematch_v1"
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "model.joblib"
    shutil.copy2(source, target)
    artifact = joblib.load(target)
    configs = {}
    for name in ("elo_phase3.toml", "phase4.toml", "phase5.toml"):
        path = ROOT / "config" / name
        configs[name] = path.read_text(encoding="utf-8")
    (destination / "config_snapshot.json").write_text(
        json.dumps(configs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    schema = {
        "schema_version": 1,
        "feature_count": len(artifact["feature_columns"]),
        "ordered_features": artifact["feature_columns"],
        "input_contract": {
            "player_a": "canonical player id or exact normalized full name",
            "player_b": "canonical player id or exact normalized full name",
            "surface": ["Hard", "Clay", "Grass", "Carpet"],
            "date": "ISO-8601 date; history uses strictly earlier dates",
        },
        "missing_values": "preserved as null/NaN then median-imputed by frozen pipeline",
    }
    (destination / "feature_schema.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    metadata = {
        "model_version": "prematch_v1",
        "status": "official_production_baseline",
        "frozen_at": datetime.now(UTC).isoformat(),
        "source_artifact": str(source.relative_to(ROOT)),
        "source_sha256": sha256(source),
        "frozen_sha256": sha256(target),
        "model_name": artifact["model_name"],
        "features_expected": artifact["feature_columns"],
        "feature_count": len(artifact["feature_columns"]),
        "trained_through": artifact["trained_through"],
        "calibration_period": artifact["calibration_period"],
        "research_test_period": ["2023-01-01", "2026-05-25"],
        "research_test_metrics": {
            "accuracy": 0.657382,
            "log_loss": 0.609876,
            "brier": 0.211585,
            "auc": 0.724285,
        },
        "temporal_contract": "only completed matches in earlier chronology batches",
        "prospective_registry_start": "2026-10-06",
        "dataset_rows": 78845,
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Frozen {metadata['model_version']} at {destination}")


if __name__ == "__main__":
    main()
