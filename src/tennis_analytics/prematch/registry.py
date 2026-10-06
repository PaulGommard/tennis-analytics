from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from tennis_analytics.paths import DATA_DIR
from tennis_analytics.prematch.predictor import Prediction

DEFAULT_LEDGER = DATA_DIR / "prospective" / "prospective_predictions.parquet"
PROSPECTIVE_START = pd.Timestamp("2026-10-06")
LEDGER_COLUMNS = [
    "prediction_id",
    "match_id",
    "model_version",
    "prediction_created_at",
    "date",
    "surface",
    "player_A_id",
    "player_A",
    "player_B_id",
    "player_B",
    "prematch_probability_A",
    "prematch_probability_B",
    "raw_probability_A",
    "favorite",
    "feature_snapshot_hash",
    "warnings_json",
    "features_json",
    "result_a",
    "result_recorded_at",
]

LEDGER_DTYPES = {
    "prediction_id": "string",
    "match_id": "string",
    "model_version": "string",
    "prediction_created_at": "string",
    "date": "string",
    "surface": "string",
    "player_A_id": "string",
    "player_A": "string",
    "player_B_id": "string",
    "player_B": "string",
    "prematch_probability_A": "float64",
    "prematch_probability_B": "float64",
    "raw_probability_A": "float64",
    "favorite": "string",
    "feature_snapshot_hash": "string",
    "warnings_json": "string",
    "features_json": "string",
    "result_a": "float64",
    "result_recorded_at": "string",
}


def _empty_ledger() -> pd.DataFrame:
    return pd.DataFrame(
        {column: pd.Series(dtype=LEDGER_DTYPES[column]) for column in LEDGER_COLUMNS}
    )


def load_ledger(path: Path = DEFAULT_LEDGER) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else _empty_ledger()


def _atomic_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.stem + ".", suffix=".parquet", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        frame.to_parquet(temporary_path, index=False)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def append_prediction(prediction: Prediction, path: Path = DEFAULT_LEDGER) -> bool:
    """Append once; an existing prediction can never be recomputed or overwritten."""
    if pd.Timestamp(prediction.match_date) < PROSPECTIVE_START:
        raise ValueError(
            f"Prospective registry starts on {PROSPECTIVE_START.date()}; "
            "historical predictions cannot be recorded."
        )
    ledger = load_ledger(path)
    existing = ledger.loc[ledger["prediction_id"] == prediction.prediction_id]
    if len(existing):
        prior = existing.iloc[0]
        immutable = {
            "prematch_probability_A": prediction.probability_a,
            "prematch_probability_B": prediction.probability_b,
            "feature_snapshot_hash": prediction.feature_snapshot_hash,
        }
        for column, expected in immutable.items():
            if str(prior[column]) != str(expected):
                raise ValueError(f"Immutable prediction collision on {column}.")
        return False
    payload = prediction.to_dict(include_features=True)
    row = {
        "prediction_id": prediction.prediction_id,
        "match_id": payload["match_id"],
        "model_version": prediction.model_version,
        "prediction_created_at": prediction.generated_at,
        "date": prediction.match_date,
        "surface": prediction.surface,
        "player_A_id": prediction.player_a_id,
        "player_A": prediction.player_a_name,
        "player_B_id": prediction.player_b_id,
        "player_B": prediction.player_b_name,
        "prematch_probability_A": prediction.probability_a,
        "prematch_probability_B": prediction.probability_b,
        "raw_probability_A": prediction.raw_probability_a,
        "favorite": prediction.favorite,
        "feature_snapshot_hash": prediction.feature_snapshot_hash,
        "warnings_json": json.dumps(prediction.warnings, ensure_ascii=False),
        "features_json": json.dumps(prediction.features, sort_keys=True, allow_nan=False),
        "result_a": np.nan,
        "result_recorded_at": None,
    }
    updated = pd.concat([ledger, pd.DataFrame([row], columns=LEDGER_COLUMNS)], ignore_index=True)
    _atomic_write(updated, path)
    return True


def record_result(prediction_id: str, result_a: int, path: Path = DEFAULT_LEDGER) -> None:
    if result_a not in (0, 1):
        raise ValueError("result_a must be 0 or 1.")
    ledger = load_ledger(path)
    mask = ledger["prediction_id"] == prediction_id
    if int(mask.sum()) != 1:
        raise ValueError(f"Expected one prediction '{prediction_id}', found {int(mask.sum())}.")
    current = ledger.loc[mask, "result_a"].iloc[0]
    if pd.notna(current):
        if int(current) == result_a:
            return
        raise ValueError("A recorded result is immutable and cannot be changed.")
    ledger.loc[mask, "result_a"] = result_a
    ledger.loc[mask, "result_recorded_at"] = datetime.now(UTC).isoformat()
    _atomic_write(ledger, path)


def evaluate_ledger(path: Path = DEFAULT_LEDGER) -> dict[str, Any]:
    ledger = load_ledger(path)
    known = ledger.dropna(subset=["result_a"]).copy()
    if known.empty:
        return {"predictions": len(ledger), "evaluated": 0, "status": "no_results"}
    y = known["result_a"].astype(int).to_numpy()
    probability = known["prematch_probability_A"].astype(float).to_numpy()
    result: dict[str, Any] = {
        "predictions": len(ledger),
        "evaluated": len(known),
        "accuracy": float(accuracy_score(y, probability >= 0.5)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(y, probability)),
        "auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
    }
    edges = np.linspace(0, 1, 11)
    known["bin"] = pd.cut(known["prematch_probability_A"], edges, include_lowest=True)
    calibration = (
        known.groupby("bin", observed=False)
        .agg(
            count=("result_a", "size"),
            predicted=("prematch_probability_A", "mean"),
            observed=("result_a", "mean"),
        )
        .reset_index()
    )
    result["calibration"] = [
        {
            "bin": str(row.bin),
            "count": int(row.count),
            "predicted": None if pd.isna(row.predicted) else float(row.predicted),
            "observed": None if pd.isna(row.observed) else float(row.observed),
        }
        for row in calibration.itertuples()
    ]
    return result
