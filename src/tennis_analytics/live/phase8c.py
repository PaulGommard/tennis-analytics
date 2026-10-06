from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from tennis_analytics.live.evaluation import match_format
from tennis_analytics.live.evaluation_v2 import (
    _neutral_features,
    _required_columns,
    _sigmoid_shift,
)
from tennis_analytics.live.markov import (
    align_serve_probabilities_to_prematch,
    clear_probability_caches,
    infer_tiebreak_first_server,
    live_probabilities,
    regular_game_probability_a,
    tiebreak_probability_a,
)
from tennis_analytics.modeling.point_model import FEATURE_BLOCKS, make_pipeline
from tennis_analytics.paths import PROCESSED_DIR, PROJECT_ROOT, REPORTS_DIR
from tennis_analytics.point_by_point.in_match import V2_PATH

LIVE_V3_DIR = PROJECT_ROOT / "models" / "live_v3"
VALIDATION_BASE_PATH = PROCESSED_DIR / "phase8c_validation_states.parquet"
TEST_OUTPUT_PATH = PROCESSED_DIR / "live_v3_probabilities_test.parquet"
V1_MODEL_PATH = PROJECT_ROOT / "models" / "point_v1" / "model.joblib"
V2_MODEL_PATH = PROJECT_ROOT / "models" / "point_v2" / "model.joblib"
V1_TEST_PATH = PROCESSED_DIR / "live_probabilities_test.parquet"
V2_TEST_PATH = PROCESSED_DIR / "live_v2_probabilities_test.parquet"
BLEND_CONSTANTS = (25, 50, 75, 100, 150, 200)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_phase8() -> dict[str, Any]:
    paths = [
        V1_MODEL_PATH,
        PROJECT_ROOT / "models" / "point_v1" / "metadata.json",
        V2_MODEL_PATH,
        PROJECT_ROOT / "models" / "point_v2" / "metadata.json",
        PROCESSED_DIR / "point_features_v1.parquet",
        V2_PATH,
        REPORTS_DIR / "phase7b_report.md",
        REPORTS_DIR / "phase8_report.md",
    ]
    LIVE_V3_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "created_at": datetime.now(UTC).isoformat(),
        "policy": "Phase 8C reads but never rewrites V1/V2 artifacts",
        "artifacts": [
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in paths
        ],
    }
    (LIVE_V3_DIR / "phase8_freeze.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return snapshot


def verify_phase8_freeze() -> list[str]:
    snapshot = json.loads(
        (LIVE_V3_DIR / "phase8_freeze.json").read_text(encoding="utf-8")
    )
    mismatches = []
    for artifact in snapshot["artifacts"]:
        path = PROJECT_ROOT / artifact["path"]
        if (
            not path.exists()
            or path.stat().st_size != artifact["bytes"]
            or _sha256(path) != artifact["sha256"]
        ):
            mismatches.append(artifact["path"])
    return mismatches


def _logit(probability: np.ndarray | pd.Series) -> np.ndarray:
    values = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(values / (1 - values))


def _sigmoid(value: np.ndarray | float) -> np.ndarray:
    return 1 / (1 + np.exp(-np.asarray(value, dtype=float)))


def _blend(
    v1: np.ndarray,
    v2: np.ndarray,
    points_played: np.ndarray,
    constant: float,
    method: str,
) -> np.ndarray:
    weight = points_played / (points_played + constant)
    if method == "linear":
        return (1 - weight) * v1 + weight * v2
    if method == "tempered_logit":
        return _sigmoid(_logit(v1) + weight * (_logit(v2) - _logit(v1)))
    raise ValueError(method)


def _binary_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(target, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else math.nan,
    }


def _checkpoint_indices(group: pd.DataFrame) -> list[tuple[int, str]]:
    checkpoints: list[tuple[int, str]] = [(int(group.index[0]), "before_match")]
    for threshold in (20, 50, 100):
        found = group.loc[group["point_index"] >= threshold]
        if not found.empty:
            checkpoints.append((int(found.index[0]), f"after_{threshold}_points"))
    for sets_completed in (1, 2):
        found = group.loc[(group["sets_A"] + group["sets_B"]) >= sets_completed]
        if not found.empty:
            checkpoints.append((int(found.index[0]), f"after_{sets_completed}_sets"))
    return list(dict.fromkeys(checkpoints))


def _game_probability(row: pd.Series, p_a_serve: float, p_b_serve: float) -> float:
    if bool(row["is_tiebreak"]):
        first = infer_tiebreak_first_server(
            str(row["server"]), int(row["points_A"] + row["points_B"])
        )
        fmt = match_format(str(row["rule_label"]), int(row["best_of"]))
        final_set = int(row["sets_A"] + row["sets_B"]) == fmt.best_of - 1
        target = fmt.final_set_tiebreak_points if final_set else fmt.tiebreak_points
        return tiebreak_probability_a(
            int(row["points_A"]),
            int(row["points_B"]),
            first_server=first,
            p_a_serve=p_a_serve,
            p_b_serve=p_b_serve,
            target=target,
        )
    return regular_game_probability_a(
        int(row["points_A"]),
        int(row["points_B"]),
        server=str(row["server"]),
        p_a_serve=p_a_serve,
        p_b_serve=p_b_serve,
    )


def _score_probabilities(row: pd.Series, p_a_serve: float, p_b_serve: float) -> dict[str, float]:
    fmt = match_format(str(row["rule_label"]), int(row["best_of"]))
    probabilities = live_probabilities(
        sets_a=int(row["sets_A"]),
        sets_b=int(row["sets_B"]),
        games_a=int(row["games_A"]),
        games_b=int(row["games_B"]),
        points_a=int(row["points_A"]),
        points_b=int(row["points_B"]),
        server=str(row["server"]),
        is_tiebreak=bool(row["is_tiebreak"]),
        p_a_serve=p_a_serve,
        p_b_serve=p_b_serve,
        match_format=fmt,
    )
    clear_probability_caches()
    return {
        "p_game": probabilities.p_current_game_a,
        "p_set": probabilities.p_current_set_a,
        "p_match": probabilities.p_match_a,
    }


def _load_split(start: str, end: str | None = None) -> pd.DataFrame:
    v2_artifact = joblib.load(V2_MODEL_PATH)
    columns = _required_columns(v2_artifact["feature_columns"])
    filters: list[tuple[str, str, Any]] = [("gender", "==", "men"), ("date", ">=", pd.Timestamp(start))]
    if end is not None:
        filters.append(("date", "<=", pd.Timestamp(end)))
    frame = pd.read_parquet(V2_PATH, columns=columns, filters=filters)
    frame = frame.loc[frame["prematch_probability_A"].notna()].reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["point_index"] = frame.groupby("match_id", sort=False).cumcount()
    return frame


def _model_probabilities(
    frame: pd.DataFrame,
    artifact: dict[str, Any],
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    columns = artifact["feature_columns"]
    prior_weight = int(artifact.get("selected_prior_weight", 50))
    pipeline = artifact["pipeline"]
    raw_current = pipeline.predict_proba(frame[columns])[:, 1]
    raw_a = pipeline.predict_proba(_neutral_features(frame, "A", columns, prior_weight))[:, 1]
    raw_b = pipeline.predict_proba(_neutral_features(frame, "B", columns, prior_weight))[:, 1]
    deltas: dict[str, float] = {}
    for match_id, group in frame.groupby("match_id", sort=False, observed=True):
        index = int(group.index[0])
        fmt = match_format(str(group.iloc[0]["rule_label"]), int(group.iloc[0]["best_of"]))
        _, _, delta = align_serve_probabilities_to_prematch(
            float(raw_a[index]),
            float(raw_b[index]),
            float(group.iloc[0]["prematch_probability_A"]),
            first_server=str(group.iloc[0]["server"]),
            match_format=fmt,
        )
        deltas[str(match_id)] = delta
        clear_probability_caches()
    delta = frame["match_id"].map(deltas).to_numpy(dtype=float)
    p_a = _sigmoid_shift(raw_a, delta)
    p_b = _sigmoid_shift(raw_b, -delta)
    adjusted_server = _sigmoid_shift(
        raw_current, np.where(frame["server"].eq("A"), delta, -delta)
    )
    p_next_a = np.where(frame["server"].eq("A"), adjusted_server, 1 - adjusted_server)
    frame[f"p_A_serve_{label}"] = p_a
    frame[f"p_B_serve_{label}"] = p_b
    frame[f"p_next_A_{label}"] = p_next_a
    return p_a, p_b, p_next_a


def build_validation_states() -> pd.DataFrame:
    frame = _load_split("2019-01-01", "2022-12-31")
    v1 = joblib.load(V1_MODEL_PATH)
    v2 = joblib.load(V2_MODEL_PATH)
    _model_probabilities(frame, v1, "v1")
    _model_probabilities(frame, v2, "v2")
    checkpoints = []
    for _, group in frame.groupby("match_id", sort=False, observed=True):
        for index, depth in _checkpoint_indices(group):
            row = frame.loc[index].copy()
            row["depth"] = depth
            checkpoints.append(row)
    states = pd.DataFrame(checkpoints).reset_index(drop=True)
    for label in ("v1", "v2"):
        games = []
        sets = []
        matches = []
        for _, row in states.iterrows():
            probability = _score_probabilities(
                row, float(row[f"p_A_serve_{label}"]), float(row[f"p_B_serve_{label}"])
            )
            games.append(probability["p_game"])
            sets.append(probability["p_set"])
            matches.append(probability["p_match"])
        states[f"p_game_{label}"] = games
        states[f"p_set_{label}"] = sets
        states[f"p_match_{label}"] = matches
    keep = [
        "match_id",
        "date",
        "source",
        "surface",
        "best_of",
        "rule_label",
        "point_index",
        "depth",
        "server",
        "sets_A",
        "sets_B",
        "games_A",
        "games_B",
        "points_A",
        "points_B",
        "is_tiebreak",
        "A_wins_next_point",
        "current_game_winner_A",
        "current_set_winner_A",
        "final_match_winner_A",
        "prematch_probability_A",
        *[
            f"p_{horizon}_{label}"
            for label in ("v1", "v2")
            for horizon in ("game", "set", "match")
        ],
        *[
            f"p_{side}_serve_{label}"
            for label in ("v1", "v2")
            for side in ("A", "B")
        ],
        "p_next_A_v1",
        "p_next_A_v2",
    ]
    states[keep].to_parquet(VALIDATION_BASE_PATH, index=False)
    return states[keep]


def _candidate_probabilities(
    states: pd.DataFrame, method: str, constant: float
) -> pd.DataFrame:
    points = states["point_index"].to_numpy(dtype=float)
    p_a = _blend(
        states["p_A_serve_v1"].to_numpy(),
        states["p_A_serve_v2"].to_numpy(),
        points,
        constant,
        method,
    )
    p_b = _blend(
        states["p_B_serve_v1"].to_numpy(),
        states["p_B_serve_v2"].to_numpy(),
        points,
        constant,
        method,
    )
    p_next = _blend(
        states["p_next_A_v1"].to_numpy(),
        states["p_next_A_v2"].to_numpy(),
        points,
        constant,
        method,
    )
    result = pd.DataFrame({"p_next": p_next, "p_A_serve": p_a, "p_B_serve": p_b})
    games, sets, matches = [], [], []
    for index, row in states.iterrows():
        probability = _score_probabilities(row, float(p_a[index]), float(p_b[index]))
        games.append(probability["p_game"])
        sets.append(probability["p_set"])
        matches.append(probability["p_match"])
    result["p_game"] = games
    result["p_set"] = sets
    result["p_match"] = matches
    return result


class ProbabilityCalibrator:
    def __init__(self, method: str) -> None:
        self.method = method
        self.model: Any = None

    def fit(
        self, probability: np.ndarray, target: np.ndarray, sample_weight: np.ndarray
    ) -> ProbabilityCalibrator:
        p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
        y = np.asarray(target, dtype=int)
        if self.method == "none":
            return self
        if self.method == "platt":
            self.model = LogisticRegression(C=1e6, max_iter=500)
            self.model.fit(_logit(p).reshape(-1, 1), y, sample_weight=sample_weight)
        elif self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip")
            self.model.fit(p, y, sample_weight=sample_weight)
        elif self.method == "temperature":
            candidates = np.linspace(0.6, 2.5, 191)
            losses = [
                log_loss(y, _sigmoid(_logit(p) / temperature), sample_weight=sample_weight)
                for temperature in candidates
            ]
            self.model = float(candidates[int(np.argmin(losses))])
        else:
            raise ValueError(self.method)
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
        if self.method == "none":
            return p
        if self.method == "platt":
            return self.model.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        if self.method == "isotonic":
            return self.model.predict(p)
        return _sigmoid(_logit(p) / float(self.model))


def _equal_match_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("match_id")["match_id"].transform("size").to_numpy(dtype=float)
    return 1 / counts


def select_blend_and_calibration(states: pd.DataFrame) -> dict[str, Any]:
    approximate_rows = []
    target = states["final_match_winner_A"].to_numpy(dtype=int)
    points = states["point_index"].to_numpy(dtype=float)
    for method in ("linear", "tempered_logit"):
        for constant in BLEND_CONSTANTS:
            probability = _blend(
                states["p_match_v1"].to_numpy(),
                states["p_match_v2"].to_numpy(),
                points,
                constant,
                method,
            )
            approximate_rows.append(
                {"method": method, "constant": constant, **_binary_metrics(target, probability)}
            )
    approximate = pd.DataFrame(approximate_rows).sort_values("log_loss")
    approximate.to_csv(REPORTS_DIR / "phase8c_blend_prescreen.csv", index=False)
    top = approximate.head(4)[["method", "constant"]].to_dict("records")
    exact_rows = []
    exact_predictions: dict[tuple[str, int], pd.DataFrame] = {}
    for candidate in top:
        method = str(candidate["method"])
        constant = int(candidate["constant"])
        predictions = _candidate_probabilities(states, method, constant)
        exact_predictions[(method, constant)] = predictions
        for horizon, truth in (
            ("next", "A_wins_next_point"),
            ("game", "current_game_winner_A"),
            ("set", "current_set_winner_A"),
            ("match", "final_match_winner_A"),
        ):
            exact_rows.append(
                {
                    "method": method,
                    "constant": constant,
                    "horizon": horizon,
                    "states": len(states),
                    **_binary_metrics(
                        states[truth].to_numpy(), predictions[f"p_{horizon}"].to_numpy()
                    ),
                }
            )
    exact = pd.DataFrame(exact_rows)
    exact.to_csv(REPORTS_DIR / "phase8c_blend_validation.csv", index=False)
    selected_row = exact.loc[exact["horizon"] == "match"].sort_values("log_loss").iloc[0]
    selected_method = str(selected_row["method"])
    selected_constant = int(selected_row["constant"])
    selected_predictions = exact_predictions[(selected_method, selected_constant)]
    for column in selected_predictions:
        states[f"v3_raw_{column}"] = selected_predictions[column].to_numpy()

    calibration_rows = []
    calibrators: dict[str, ProbabilityCalibrator] = {}
    fit_mask = states["date"] <= pd.Timestamp("2020-12-31")
    selection_mask = ~fit_mask
    for horizon, truth in (
        ("game", "current_game_winner_A"),
        ("set", "current_set_winner_A"),
        ("match", "final_match_winner_A"),
    ):
        probability = states[f"v3_raw_p_{horizon}"].to_numpy()
        target_values = states[truth].to_numpy(dtype=int)
        for calibration_method in ("none", "temperature", "platt", "isotonic"):
            calibrator = ProbabilityCalibrator(calibration_method).fit(
                probability[fit_mask],
                target_values[fit_mask],
                _equal_match_weights(states.loc[fit_mask]),
            )
            predicted = calibrator.predict(probability[selection_mask])
            calibration_rows.append(
                {
                    "horizon": horizon,
                    "method": calibration_method,
                    "fit_period": "2019-2020",
                    "selection_period": "2021-2022",
                    "states": int(selection_mask.sum()),
                    **_binary_metrics(target_values[selection_mask], predicted),
                }
            )
        horizon_rows = [row for row in calibration_rows if row["horizon"] == horizon]
        selected_calibration = min(horizon_rows, key=lambda row: row["log_loss"])["method"]
        calibrators[horizon] = ProbabilityCalibrator(str(selected_calibration)).fit(
            probability,
            target_values,
            _equal_match_weights(states),
        )
    calibration = pd.DataFrame(calibration_rows)
    calibration.to_csv(REPORTS_DIR / "phase8c_calibration_validation.csv", index=False)
    states.to_parquet(VALIDATION_BASE_PATH, index=False)
    artifact = {
        "model_version": "live_v3",
        "blend_method": selected_method,
        "blend_constant": selected_constant,
        "calibrators": calibrators,
        "calibration_methods": {
            horizon: calibrator.method for horizon, calibrator in calibrators.items()
        },
        "calibration_fit_period": "2019-2022",
        "calibration_selection_protocol": "fit 2019-2020, choose on 2021-2022, refit 2019-2022",
    }
    joblib.dump(artifact, LIVE_V3_DIR / "model.joblib")
    return artifact


def _load_test_base() -> pd.DataFrame:
    v2_columns = [
        "match_id",
        "date",
        "source",
        "surface",
        "player_A_name",
        "player_B_name",
        "point_index",
        "server",
        "sets_A",
        "sets_B",
        "games_A",
        "games_B",
        "points_A",
        "points_B",
        "is_tiebreak",
        "A_wins_next_point",
        "current_game_winner_A",
        "current_set_winner_A",
        "final_match_winner_A",
        "prematch_probability_A",
        "p_next_point_A_v2",
        "p_current_game_A_v2",
        "p_current_set_A_v2",
        "p_match_A_v2",
        "p_A_serve_v2",
        "p_B_serve_v2",
        "live_sampled",
    ]
    v1_columns = [
        "match_id",
        "point_index",
        "p_next_point_A",
        "p_current_game_A",
        "p_current_set_A",
        "p_match_A",
        "p_A_serve",
        "p_B_serve",
    ]
    v2 = pd.read_parquet(V2_TEST_PATH, columns=v2_columns)
    v1 = pd.read_parquet(V1_TEST_PATH, columns=v1_columns)
    formats = pd.read_parquet(
        V2_PATH,
        columns=["match_id", "best_of", "rule_label", "date", "gender"],
        filters=[("gender", "==", "men"), ("date", ">=", pd.Timestamp("2023-01-01"))],
    ).drop_duplicates("match_id")
    return (
        v2.merge(v1, on=["match_id", "point_index"], validate="one_to_one")
        .merge(formats[["match_id", "best_of", "rule_label"]], on="match_id", validate="many_to_one")
    )


def evaluate_test(artifact: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _load_test_base()
    method = str(artifact["blend_method"])
    constant = float(artifact["blend_constant"])
    points = frame["point_index"].to_numpy(dtype=float)
    frame["p_next_point_A_v3_raw"] = _blend(
        frame["p_next_point_A"].to_numpy(),
        frame["p_next_point_A_v2"].to_numpy(),
        points,
        constant,
        method,
    )
    frame["p_A_serve_v3"] = _blend(
        frame["p_A_serve"].to_numpy(),
        frame["p_A_serve_v2"].to_numpy(),
        points,
        constant,
        method,
    )
    frame["p_B_serve_v3"] = _blend(
        frame["p_B_serve"].to_numpy(),
        frame["p_B_serve_v2"].to_numpy(),
        points,
        constant,
        method,
    )
    frame["p_current_game_A_v3_raw"] = [
        _game_probability(row, float(row["p_A_serve_v3"]), float(row["p_B_serve_v3"]))
        for _, row in frame.iterrows()
    ]
    sampled = frame["live_sampled"].astype(bool)
    frame["p_current_set_A_v3_raw"] = np.nan
    frame["p_match_A_v3_raw"] = np.nan
    for index in frame.index[sampled]:
        probability = _score_probabilities(
            frame.loc[index],
            float(frame.at[index, "p_A_serve_v3"]),
            float(frame.at[index, "p_B_serve_v3"]),
        )
        frame.at[index, "p_current_set_A_v3_raw"] = probability["p_set"]
        frame.at[index, "p_match_A_v3_raw"] = probability["p_match"]
    frame["p_current_game_A_v3_cal"] = artifact["calibrators"]["game"].predict(
        frame["p_current_game_A_v3_raw"].to_numpy()
    )
    frame.loc[sampled, "p_current_set_A_v3_cal"] = artifact["calibrators"]["set"].predict(
        frame.loc[sampled, "p_current_set_A_v3_raw"].to_numpy()
    )
    frame.loc[sampled, "p_match_A_v3_cal"] = artifact["calibrators"]["match"].predict(
        frame.loc[sampled, "p_match_A_v3_raw"].to_numpy()
    )

    rows = []
    horizons = [
        ("point", "A_wins_next_point", np.ones(len(frame), dtype=bool)),
        ("game", "current_game_winner_A", np.ones(len(frame), dtype=bool)),
        ("set", "current_set_winner_A", sampled.to_numpy()),
        ("match", "final_match_winner_A", sampled.to_numpy()),
    ]
    columns = {
        "point": {
            "LIVE_V1": "p_next_point_A",
            "LIVE_V2": "p_next_point_A_v2",
            "LIVE_V3_RAW": "p_next_point_A_v3_raw",
        },
        "game": {
            "LIVE_V1": "p_current_game_A",
            "LIVE_V2": "p_current_game_A_v2",
            "LIVE_V3_RAW": "p_current_game_A_v3_raw",
            "LIVE_V3_CAL": "p_current_game_A_v3_cal",
        },
        "set": {
            "LIVE_V1": "p_current_set_A",
            "LIVE_V2": "p_current_set_A_v2",
            "LIVE_V3_RAW": "p_current_set_A_v3_raw",
            "LIVE_V3_CAL": "p_current_set_A_v3_cal",
        },
        "match": {
            "LIVE_V1": "p_match_A",
            "LIVE_V2": "p_match_A_v2",
            "LIVE_V3_RAW": "p_match_A_v3_raw",
            "LIVE_V3_CAL": "p_match_A_v3_cal",
        },
    }
    for horizon, truth, eligible in horizons:
        for model, column in columns[horizon].items():
            rows.append(
                {
                    "horizon": horizon,
                    "model": model,
                    "scope": "all",
                    "states": int(eligible.sum()),
                    **_binary_metrics(
                        frame.loc[eligible, truth].to_numpy(),
                        frame.loc[eligible, column].to_numpy(),
                    ),
                }
            )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(REPORTS_DIR / "phase8c_test_comparison.csv", index=False)
    frame.to_parquet(TEST_OUTPUT_PATH, index=False)
    return frame, metrics


def _confidence_bucket_table(frame: pd.DataFrame) -> pd.DataFrame:
    buckets = [0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 0.95, 1.000001]
    labels = ["50-55", "55-60", "60-70", "70-80", "80-90", "90-95", "95-100"]
    rows = []
    sampled = frame["live_sampled"].astype(bool)
    for model, column in (
        ("LIVE_V1", "p_match_A"),
        ("LIVE_V2", "p_match_A_v2"),
        ("LIVE_V3_RAW", "p_match_A_v3_raw"),
        ("LIVE_V3_CAL", "p_match_A_v3_cal"),
    ):
        subset = frame.loc[sampled, [column, "final_match_winner_A"]].copy()
        subset["confidence"] = np.maximum(subset[column], 1 - subset[column])
        subset["correct"] = np.where(
            subset[column] >= 0.5,
            subset["final_match_winner_A"],
            1 - subset["final_match_winner_A"],
        )
        subset["bucket"] = pd.cut(
            subset["confidence"], buckets, labels=labels, right=False, include_lowest=True
        )
        for bucket, group in subset.groupby("bucket", observed=False):
            if group.empty:
                continue
            rows.append(
                {
                    "model": model,
                    "bucket": str(bucket),
                    "states": len(group),
                    "mean_probability": float(group["confidence"].mean()),
                    "observed_win_rate": float(group["correct"].mean()),
                    "log_loss": float(log_loss(group["correct"], group["confidence"])),
                    "brier": float(brier_score_loss(group["correct"], group["confidence"])),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(REPORTS_DIR / "phase8c_match_confidence_buckets.csv", index=False)
    return result


def _segment_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    sampled = frame.loc[frame["live_sampled"]].copy()
    sampled["points_segment"] = pd.cut(
        sampled["point_index"],
        [-1, 19, 49, 99, 149, np.inf],
        labels=["1-20", "21-50", "51-100", "101-150", "150+"],
    )
    set_number = sampled["sets_A"] + sampled["sets_B"] + 1
    sampled["set_segment"] = np.where(
        set_number == 1, "set_1", np.where(set_number == 2, "set_2", "set_3_plus")
    )
    games_played = sampled["games_A"] + sampled["games_B"]
    sampled["set_phase"] = np.where(
        games_played <= 3,
        "set_beginning",
        np.where((games_played <= 7) & ~sampled["is_tiebreak"], "set_middle", "set_end"),
    )
    rows = []
    models = {
        "LIVE_V1": "p_match_A",
        "LIVE_V2": "p_match_A_v2",
        "LIVE_V3_RAW": "p_match_A_v3_raw",
        "LIVE_V3_CAL": "p_match_A_v3_cal",
    }
    for segment_type in ("points_segment", "set_segment", "set_phase"):
        for segment, group in sampled.groupby(segment_type, observed=True):
            for model, column in models.items():
                rows.append(
                    {
                        "segment_type": segment_type,
                        "segment": str(segment),
                        "model": model,
                        "states": len(group),
                        **_binary_metrics(
                            group["final_match_winner_A"].to_numpy(), group[column].to_numpy()
                        ),
                    }
                )
    depth_frames = []
    for _, group in sampled.groupby("match_id", sort=False, observed=True):
        candidates: list[tuple[str, pd.DataFrame]] = [("before_match", group.iloc[[0]])]
        for threshold in (20, 50, 100):
            found = group.loc[group["point_index"] >= threshold]
            if not found.empty:
                candidates.append((f"after_{threshold}_points", found.iloc[[0]]))
        for sets_completed in (1, 2):
            found = group.loc[(group["sets_A"] + group["sets_B"]) >= sets_completed]
            if not found.empty:
                candidates.append((f"after_{sets_completed}_sets", found.iloc[[0]]))
        for depth, selected in candidates:
            selected = selected.copy()
            selected["depth"] = depth
            depth_frames.append(selected)
    depths = pd.concat(depth_frames, ignore_index=True)
    for depth, group in depths.groupby("depth", sort=True):
        for model, column in models.items():
            rows.append(
                {
                    "segment_type": "depth",
                    "segment": depth,
                    "model": model,
                    "states": len(group),
                    **_binary_metrics(
                        group["final_match_winner_A"].to_numpy(), group[column].to_numpy()
                    ),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(REPORTS_DIR / "phase8c_match_segments.csv", index=False)
    return result


def _large_change_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    sampled = frame.loc[frame["live_sampled"]].copy()
    sampled["absolute_change"] = np.abs(sampled["p_match_A_v2"] - sampled["p_match_A"])
    y = sampled["final_match_winner_A"].to_numpy(dtype=int)
    loss_v1 = -(y * np.log(np.clip(sampled["p_match_A"], 1e-6, 1 - 1e-6)) + (1 - y) * np.log(np.clip(1 - sampled["p_match_A"], 1e-6, 1 - 1e-6)))
    loss_v2 = -(y * np.log(np.clip(sampled["p_match_A_v2"], 1e-6, 1 - 1e-6)) + (1 - y) * np.log(np.clip(1 - sampled["p_match_A_v2"], 1e-6, 1 - 1e-6)))
    sampled["v2_loss_improvement"] = loss_v1 - loss_v2
    rows = []
    for threshold in (0.10, 0.20, 0.30):
        group = sampled.loc[sampled["absolute_change"] >= threshold]
        rows.append(
            {
                "threshold": threshold,
                "states": len(group),
                "matches": int(group["match_id"].nunique()),
                "mean_absolute_change": float(group["absolute_change"].mean()),
                "share_v2_improved_log_loss": float((group["v2_loss_improvement"] > 0).mean()),
                "mean_v2_log_loss_improvement": float(group["v2_loss_improvement"].mean()),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(REPORTS_DIR / "phase8c_large_changes.csv", index=False)
    sampled.sort_values("absolute_change", ascending=False).head(200).to_csv(
        REPORTS_DIR / "phase8c_overreaction_examples.csv", index=False
    )
    return result


def _component_point_models() -> pd.DataFrame:
    base = FEATURE_BLOCKS["P5_RECENT_SCORE"]
    variants = {
        "V1_BASE": base,
        "DYNAMIC_SERVE": base + ["dynamic_serve_w50", "serve_today_delta_w50"],
        "DYNAMIC_RETURN": base + ["dynamic_return_w50", "return_today_delta_w50"],
        "DYNAMIC_BOTH": base
        + [
            "dynamic_serve_w50",
            "serve_today_delta_w50",
            "dynamic_return_w50",
            "return_today_delta_w50",
        ],
    }
    v2 = joblib.load(V2_MODEL_PATH)
    variants["V2_NO_SHORT_RECENCY"] = [
        feature
        for feature in v2["feature_columns"]
        if "_last_3" not in feature and "_last_5" not in feature
    ]
    columns = sorted(
        {
            "date",
            "gender",
            "server_won_point",
            *[feature for features in variants.values() for feature in features],
        }
    )
    frame = pd.read_parquet(V2_PATH, columns=columns, filters=[("gender", "==", "men")])
    frame["date"] = pd.to_datetime(frame["date"])
    train = frame["date"] <= pd.Timestamp("2018-12-31")
    validation = frame["date"].between("2019-01-01", "2022-12-31")
    test = frame["date"] >= pd.Timestamp("2023-01-01")
    rows = []
    for name, features in variants.items():
        pipeline = make_pipeline()
        pipeline.fit(frame.loc[train, features], frame.loc[train, "server_won_point"])
        for split_name, mask in (("validation", validation), ("test", test)):
            probability = pipeline.predict_proba(frame.loc[mask, features])[:, 1]
            rows.append(
                {
                    "model": name,
                    "split": split_name,
                    "points": int(mask.sum()),
                    **_binary_metrics(frame.loc[mask, "server_won_point"].to_numpy(), probability),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(REPORTS_DIR / "phase8c_point_components.csv", index=False)
    return result


def run_phase8c() -> dict[str, Any]:
    freeze_phase8()
    components = _component_point_models()
    validation = build_validation_states()
    artifact = select_blend_and_calibration(validation)
    return finalize_phase8c(artifact, validation, components)


def finalize_phase8c(
    artifact: dict[str, Any] | None = None,
    validation: pd.DataFrame | None = None,
    components: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if artifact is None:
        artifact = joblib.load(LIVE_V3_DIR / "model.joblib")
    if validation is None:
        validation = pd.read_parquet(VALIDATION_BASE_PATH)
    if components is None:
        components = pd.read_csv(REPORTS_DIR / "phase8c_point_components.csv")
    test, metrics = evaluate_test(artifact)
    confidence = _confidence_bucket_table(test)
    segments = _segment_analysis(test)
    changes = _large_change_analysis(test)
    metadata = {
        key: value for key, value in artifact.items() if key != "calibrators"
    }
    match_rows = metrics.loc[metrics["horizon"] == "match"].set_index("model")
    best_model = str(match_rows["log_loss"].idxmin())
    metadata.update(
        {
            "created_at": datetime.now(UTC).isoformat(),
            "validation_states": len(validation),
            "test_states": len(test),
            "test_sampled_match_states": int(test["live_sampled"].sum()),
            "best_test_match_model": best_model,
            "promote_live_v3": best_model.startswith("LIVE_V3"),
            "phase8_freeze_verified": not verify_phase8_freeze(),
        }
    )
    (LIVE_V3_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {
        "metadata": metadata,
        "test_metrics": metrics.to_dict("records"),
        "component_rows": len(components),
        "confidence_rows": len(confidence),
        "segment_rows": len(segments),
        "change_rows": len(changes),
    }
