from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tennis_analytics.paths import PROCESSED_DIR, PROJECT_ROOT, REPORTS_DIR
from tennis_analytics.shots.build import CHECKPOINTS_PATH, POINTS_PATH

PREDICTIONS_PATH = PROCESSED_DIR / "shot_level" / "point_v2_mcp_predictions.parquet"
ENRICHED_CHECKPOINTS_PATH = PROCESSED_DIR / "shot_level" / "shot_checkpoints_v1_enriched.parquet"
MODEL_DIR = PROJECT_ROOT / "models" / "shot_v1"
POINT_FEATURES_PATH = PROCESSED_DIR / "point_features_v2.parquet"


def _metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(y, probability)),
        "auc": float(roc_auc_score(y, probability)),
        "accuracy": float(accuracy_score(y, probability >= 0.5)),
    }


def build_point_v2_predictions() -> dict[str, int]:
    """Replay the frozen POINT MODEL V2 on MCP rows, in bounded-memory batches."""
    artifact = joblib.load(PROJECT_ROOT / "models" / "point_v2" / "model.joblib")
    feature_columns = list(artifact["feature_columns"])
    columns = list(
        dict.fromkeys(
            [
                "source_match_id",
                "point_number",
                "date",
                "is_break_point",
                "is_set_point",
                "is_match_point",
                "is_tiebreak",
                *feature_columns,
            ]
        )
    )
    dataset = ds.dataset(POINT_FEATURES_PATH, format="parquet")
    scanner = dataset.scanner(
        columns=columns,
        filter=ds.field("source") == "mcp",
        batch_size=50_000,
    )
    schema = pa.schema(
        [
            ("point_id", pa.string()),
            ("match_id", pa.string()),
            ("point_number", pa.int32()),
            ("date", pa.date32()),
            ("point_v2_probability_server", pa.float64()),
            ("is_break_point", pa.int8()),
            ("is_set_point", pa.int8()),
            ("is_match_point", pa.int8()),
            ("is_tiebreak", pa.int8()),
        ]
    )
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(PREDICTIONS_PATH, schema, compression="zstd")
    rows = 0
    try:
        for batch in scanner.to_batches():
            frame = batch.to_pandas()
            probability = artifact["pipeline"].predict_proba(frame[feature_columns])[:, 1]
            point_number = frame["point_number"].astype("int32")
            output = pa.Table.from_pydict(
                {
                    "point_id": (
                        frame["source_match_id"].astype(str)
                        + ":"
                        + point_number.astype(str)
                    ).tolist(),
                    "match_id": frame["source_match_id"].astype(str).tolist(),
                    "point_number": point_number.tolist(),
                    "date": pd.to_datetime(frame["date"]).dt.date.tolist(),
                    "point_v2_probability_server": probability.tolist(),
                    "is_break_point": frame["is_break_point"].astype("int8").tolist(),
                    "is_set_point": frame["is_set_point"].astype("int8").tolist(),
                    "is_match_point": frame["is_match_point"].astype("int8").tolist(),
                    "is_tiebreak": frame["is_tiebreak"].astype("int8").tolist(),
                },
                schema=schema,
            )
            writer.write_table(output)
            rows += len(frame)
    finally:
        writer.close()
    return {"rows": rows}


def enrich_checkpoints() -> dict[str, int]:
    predictions = pq.read_table(PREDICTIONS_PATH).to_pandas()
    lookup = dict(
        zip(
            predictions["point_id"],
            zip(
                predictions["point_v2_probability_server"],
                predictions["is_break_point"],
                predictions["is_tiebreak"],
                strict=True,
            ),
            strict=True,
        )
    )
    source = pq.ParquetFile(CHECKPOINTS_PATH)
    writer: pq.ParquetWriter | None = None
    rows = matched = 0
    try:
        for batch in source.iter_batches(batch_size=100_000):
            table = pa.Table.from_batches([batch])
            records = [lookup.get(value.as_py()) for value in table["point_id"]]
            values = [record[0] if record is not None else None for record in records]
            matched += sum(record is not None for record in records)
            for field, position in (("is_break_point", 1), ("is_tiebreak", 2)):
                source_values = table[field].to_pylist()
                exact = [
                    bool(record[position]) if record is not None else source_value
                    for record, source_value in zip(records, source_values, strict=True)
                ]
                index = table.schema.get_field_index(field)
                table = table.set_column(index, field, pa.array(exact, type=pa.bool_()))
            table = table.append_column(
                "point_v2_probability_server", pa.array(values, type=pa.float64())
            )
            if writer is None:
                writer = pq.ParquetWriter(ENRICHED_CHECKPOINTS_PATH, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(table)
    finally:
        if writer is not None:
            writer.close()
    return {"rows": rows, "matched": matched}


def _pipeline(categorical: list[str], numeric: list[str]) -> Pipeline:
    transformers: list[tuple[str, Any, list[str]]] = []
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    transformers.append(
        (
            "numeric",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]
            ),
            numeric,
        )
    )
    transformer = ColumnTransformer(transformers)
    return Pipeline(
        [
            ("features", transformer),
            ("logistic", LogisticRegression(max_iter=300, C=1.0)),
        ]
    )


def _clustered_effect(
    frame: pd.DataFrame, baseline: np.ndarray, candidate: np.ndarray
) -> dict[str, float]:
    target = frame["server_won_point"].to_numpy(dtype=int)
    base_loss = -(target * np.log(baseline) + (1 - target) * np.log(1 - baseline))
    model_loss = -(target * np.log(candidate) + (1 - target) * np.log(1 - candidate))
    grouped = pd.DataFrame(
        {"match_id": frame["match_id"], "gain": base_loss - model_loss}
    ).groupby("match_id")["gain"].mean()
    se = grouped.std(ddof=1) / math.sqrt(len(grouped))
    mean = grouped.mean()
    return {
        "mean_match_log_loss_gain": float(mean),
        "ci95_low": float(mean - 1.96 * se),
        "ci95_high": float(mean + 1.96 * se),
        "share_matches_improved": float((grouped > 0).mean()),
    }


def train_shot_models() -> dict[str, Any]:
    categorical_by_checkpoint = {
        1: ["surface", "serve_number", "serve_direction", "is_tiebreak", "is_break_point"],
        2: [
            "surface",
            "serve_number",
            "serve_direction",
            "return_type",
            "return_direction",
            "return_depth",
            "is_tiebreak",
            "is_break_point",
        ],
        3: [
            "surface",
            "serve_number",
            "serve_direction",
            "return_type",
            "return_direction",
            "return_depth",
            "last_shot_type",
            "last_shot_direction",
            "last_shot_relation",
            "next_hitter_role",
            "is_tiebreak",
            "is_break_point",
        ],
        5: [
            "surface",
            "serve_number",
            "serve_direction",
            "return_type",
            "return_direction",
            "return_depth",
            "last_shot_type",
            "last_shot_direction",
            "last_shot_relation",
            "next_hitter_role",
            "is_tiebreak",
            "is_break_point",
        ],
    }
    for checkpoint, features in categorical_by_checkpoint.items():
        categorical_by_checkpoint[checkpoint] = [
            "server_id",
            "receiver_id",
            *features,
        ]
    rows: list[dict[str, Any]] = []
    effects: list[dict[str, Any]] = []
    artifacts: dict[int, Pipeline] = {}
    for checkpoint, categorical in categorical_by_checkpoint.items():
        table = pq.read_table(
            ENRICHED_CHECKPOINTS_PATH,
            filters=[("checkpoint", "=", checkpoint)],
        )
        frame = table.to_pandas()
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.dropna(subset=["point_v2_probability_server"])
        train = frame["date"] <= "2018-12-31"
        validation = frame["date"].between("2019-01-01", "2022-12-31")
        test = frame["date"] >= "2023-01-01"
        numeric = ["point_v2_probability_server", "observed_shots"]
        prior_pipeline = _pipeline([], ["point_v2_probability_server"])
        prior_pipeline.fit(
            frame.loc[train, ["point_v2_probability_server"]],
            frame.loc[train, "server_won_point"],
        )
        pipeline = _pipeline(categorical, numeric)
        pipeline.fit(frame.loc[train, categorical + numeric], frame.loc[train, "server_won_point"])
        artifacts[checkpoint] = pipeline
        for split_name, mask in (("validation", validation), ("test", test)):
            evaluated = frame.loc[mask]
            target = evaluated["server_won_point"].to_numpy(dtype=int)
            baseline = np.clip(
                evaluated["point_v2_probability_server"].to_numpy(dtype=float), 1e-6, 1 - 1e-6
            )
            calibrated = np.clip(
                prior_pipeline.predict_proba(evaluated[["point_v2_probability_server"]])[:, 1],
                1e-6,
                1 - 1e-6,
            )
            shot_probability = np.clip(
                pipeline.predict_proba(evaluated[categorical + numeric])[:, 1],
                1e-6,
                1 - 1e-6,
            )
            for model_name, values in (
                ("POINT_MODEL_V2", baseline),
                ("PRIOR_RECALIBRATED", calibrated),
                (f"SHOT_LOGISTIC_AFTER_{checkpoint}", shot_probability),
            ):
                rows.append(
                    {
                        "checkpoint": checkpoint,
                        "split": split_name,
                        "model": model_name,
                        "points": len(evaluated),
                        "matches": evaluated["match_id"].nunique(),
                        **_metrics(target, values),
                    }
                )
            effects.append(
                {
                    "checkpoint": checkpoint,
                    "split": split_name,
                    "comparison": "shot_minus_point_v2",
                    **_clustered_effect(evaluated, baseline, shot_probability),
                }
            )
            effects.append(
                {
                    "checkpoint": checkpoint,
                    "split": split_name,
                    "comparison": "shot_minus_recalibrated_prior",
                    **_clustered_effect(evaluated, calibrated, shot_probability),
                }
            )
    comparison = pd.DataFrame(rows)
    effects_frame = pd.DataFrame(effects)
    comparison.to_csv(REPORTS_DIR / "shot_model_comparison.csv", index=False)
    effects_frame.to_csv(REPORTS_DIR / "shot_model_clustered_effects.csv", index=False)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_version": "shot_v1_logistic",
            "checkpoint_models": artifacts,
            "temporal_split": {
                "train_end": "2018-12-31",
                "validation": "2019-01-01..2022-12-31",
                "test_start": "2023-01-01",
            },
            "outcome": "server_won_point",
            "leakage_guard": "only non-terminal observed prefix fields",
            "categorical_features": categorical_by_checkpoint,
            "numeric_features": ["point_v2_probability_server", "observed_shots"],
        },
        MODEL_DIR / "model.joblib",
    )
    metadata = {
        "model_version": "shot_v1_logistic",
        "checkpoints": sorted(artifacts),
        "comparison_rows": len(comparison),
        "test_points_by_checkpoint": {
            str(row.checkpoint): int(row.points)
            for row in comparison.query("split == 'test' and model == 'POINT_MODEL_V2'").itertuples()
        },
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (MODEL_DIR / "feature_schema.json").write_text(
        json.dumps(
            {
                "categorical_by_checkpoint": categorical_by_checkpoint,
                "numeric": ["point_v2_probability_server", "observed_shots"],
                "target": "server_won_point",
                "forbidden": [
                    "final_rally_length",
                    "terminal_outcome",
                    "point_winner_number",
                    "future_shots",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    return metadata


def evaluate_next_shot() -> dict[str, Any]:
    """Evaluate leakage-safe unigram/Markov/player-shrunk next-shot baselines."""
    train_unigram: Counter[str] = Counter()
    train_context: defaultdict[str, Counter[str]] = defaultdict(Counter)
    player_context: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    player_samples: Counter[str] = Counter()
    parquet = pq.ParquetFile(POINTS_PATH)
    for batch in parquet.iter_batches(
        columns=["date", "server_id", "receiver_id", "sequence_tokens"],
        batch_size=100_000,
    ):
        frame = pa.Table.from_batches([batch]).to_pandas()
        for row in frame.itertuples(index=False):
            tokens = str(row.sequence_tokens).split()
            if len(tokens) < 2:
                continue
            date = pd.Timestamp(row.date)
            for index in range(len(tokens) - 1):
                context, target = tokens[index], tokens[index + 1]
                next_player = row.receiver_id if (index + 1) % 2 else row.server_id
                if date <= pd.Timestamp("2018-12-31"):
                    train_unigram[target] += 1
                    train_context[context][target] += 1
                    player_context[(next_player, context)][target] += 1
                    player_samples[next_player] += 1
    vocabulary = sorted(train_unigram)
    total = sum(train_unigram.values())
    alpha = 0.5
    global_cache: dict[str, np.ndarray] = {}
    player_cache: dict[tuple[str, str], np.ndarray] = {}

    def distribution(context: str, player: str | None, strength: int = 100) -> np.ndarray:
        if player is not None and (player, context) in player_cache:
            return player_cache[(player, context)]
        if player is None and context in global_cache:
            return global_cache[context]
        global_context = train_context.get(context, Counter())
        context_total = sum(global_context.values())
        if context_total:
            base = np.array(
                [
                    (global_context[token] + alpha)
                    / (context_total + alpha * len(vocabulary))
                    for token in vocabulary
                ]
            )
        else:
            base = np.array(
                [
                    (train_unigram[token] + alpha) / (total + alpha * len(vocabulary))
                    for token in vocabulary
                ]
            )
        global_cache[context] = base
        if player is None or player_samples[player] < 200:
            return base
        personal = player_context.get((player, context), Counter())
        personal_total = sum(personal.values())
        if not personal_total:
            return base
        counts = np.array([personal[token] for token in vocabulary], dtype=float)
        result = (counts + strength * base) / (personal_total + strength)
        player_cache[(player, context)] = result
        return result

    token_index = {token: index for index, token in enumerate(vocabulary)}
    unigram_values = np.array(
        [
            (train_unigram[token] + alpha) / (total + alpha * len(vocabulary))
            for token in vocabulary
        ]
    )
    accumulators: dict[tuple[str, str], dict[str, float]] = {
        (split, model): {"loss": 0.0, "correct": 0.0, "count": 0.0}
        for split in ("validation", "test")
        for model in ("unigram", "markov_1", "player_markov")
    }
    for batch in parquet.iter_batches(
        columns=["date", "server_id", "receiver_id", "sequence_tokens"],
        batch_size=100_000,
    ):
        frame = pa.Table.from_batches([batch]).to_pandas()
        for row in frame.itertuples(index=False):
            date = pd.Timestamp(row.date)
            if date <= pd.Timestamp("2018-12-31"):
                continue
            split = "validation" if date <= pd.Timestamp("2022-12-31") else "test"
            tokens = str(row.sequence_tokens).split()
            for index in range(len(tokens) - 1):
                context, target = tokens[index], tokens[index + 1]
                if target not in token_index:
                    continue
                next_player = row.receiver_id if (index + 1) % 2 else row.server_id
                candidates = {
                    "unigram": unigram_values,
                    "markov_1": distribution(context, None),
                    "player_markov": distribution(context, next_player),
                }
                for model, values in candidates.items():
                    accumulator = accumulators[(split, model)]
                    probability = max(values[token_index[target]], 1e-12)
                    accumulator["loss"] += -math.log(probability)
                    accumulator["correct"] += int(
                        vocabulary[int(np.argmax(values))] == target
                    )
                    accumulator["count"] += 1
    rows = []
    for (split, model), values in accumulators.items():
        count = int(values["count"])
        rows.append(
            {
                "split": split,
                "model": model,
                "transitions": count,
                "log_loss": values["loss"] / count,
                "top1_accuracy": values["correct"] / count,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(REPORTS_DIR / "next_shot_comparison.csv", index=False)
    joblib.dump(
        {
            "model_version": "shot_v1_markov",
            "vocabulary": vocabulary,
            "unigram": dict(train_unigram),
            "context": {key: dict(value) for key, value in train_context.items()},
            "player_context": {key: dict(value) for key, value in player_context.items()},
            "shrinkage_strength": 100,
            "minimum_player_samples": 200,
        },
        MODEL_DIR / "next_shot_markov.joblib",
    )
    return {
        "vocabulary": len(vocabulary),
        "train_transitions": total,
        "validation_transitions": int(accumulators[("validation", "unigram")]["count"]),
        "test_transitions": int(accumulators[("test", "unigram")]["count"]),
    }
