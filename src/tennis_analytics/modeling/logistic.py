from __future__ import annotations

import argparse
import json
import logging
import math
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tennis_analytics.features.stat_history import (
    CAREER_ONLY_METRICS,
    CORE_METRICS,
    build_stat_history_base,
    materialize_features,
    write_feature_dataset,
)
from tennis_analytics.paths import PROCESSED_DIR, PROJECT_ROOT, REPORTS_DIR
from tennis_analytics.validation.stats_audit import build_stats_audit, write_stats_audit

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "phase4.toml"


@dataclass(frozen=True)
class Phase4Config:
    point_prior_strengths: tuple[float, ...]
    surface_prior_strengths: tuple[float, ...]
    logistic_c_values: tuple[float, ...]
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp

    @classmethod
    def from_toml(cls, path: Path = DEFAULT_CONFIG) -> Phase4Config:
        with path.open("rb") as stream:
            values = tomllib.load(stream)
        search = values["search"]
        evaluation = values["evaluation"]
        return cls(
            point_prior_strengths=tuple(map(float, search["point_prior_strengths"])),
            surface_prior_strengths=tuple(map(float, search["surface_prior_strengths"])),
            logistic_c_values=tuple(map(float, search["logistic_c_values"])),
            train_end=pd.Timestamp(evaluation["train_end"]),
            validation_start=pd.Timestamp(evaluation["validation_start"]),
            validation_end=pd.Timestamp(evaluation["validation_end"]),
            test_start=pd.Timestamp(evaluation["test_start"]),
        )


def feature_blocks() -> dict[str, list[str]]:
    service_career_metrics = [
        metric
        for metric in CORE_METRICS + CAREER_ONLY_METRICS
        if "return" not in metric and metric != "break_points_converted_rate"
    ]
    return_career_metrics = [
        metric
        for metric in CORE_METRICS + CAREER_ONLY_METRICS
        if "return" in metric or metric == "break_points_converted_rate"
    ]
    serve = [f"{metric}_career_diff" for metric in service_career_metrics]
    returns = [f"{metric}_career_diff" for metric in return_career_metrics]
    rolling = [
        f"{metric}_{horizon}_diff" for metric in CORE_METRICS for horizon in ("365d", "90d")
    ] + [
        "service_points_won_rate_surface_diff",
        "return_points_won_rate_surface_diff",
        "win_rate_career_diff",
        "win_rate_365d_diff",
        "win_rate_90d_diff",
    ]
    fatigue = [
        f"{measure}_{window}d_diff"
        for window in (7, 14, 30)
        for measure in ("matches", "sets", "games", "minutes")
    ] + ["rest_days_diff"]
    ranking = ["rank_diff", "rank_points_diff"]
    experience = ["career_matches_diff", "surface_matches_diff", "age_years_diff"]
    elo = ["elo_logit"]
    return {
        "MODEL_0_ELO": elo,
        "MODEL_1_ELO_RANKING": elo + ranking,
        "MODEL_2_ELO_SERVE": elo + serve,
        "MODEL_RETURN_ABLATION": elo + returns,
        "MODEL_3_ELO_SERVE_RETURN": elo + serve + returns,
        "MODEL_RANKING_SERVE_RETURN": elo + ranking + serve + returns,
        "MODEL_4_PLUS_FORM_ROLLING": elo + serve + returns + rolling,
        "MODEL_RANKING_STATS_FORM": elo + ranking + serve + returns + rolling,
        "MODEL_FATIGUE_ABLATION": elo + fatigue,
        "MODEL_5_ALL": elo + ranking + serve + returns + rolling + fatigue + experience,
    }


def temporal_masks(
    frame: pd.DataFrame, config: Phase4Config
) -> tuple[pd.Series, pd.Series, pd.Series]:
    dates = pd.to_datetime(frame["date"])
    train = dates <= config.train_end
    validation = dates.between(config.validation_start, config.validation_end)
    test = dates >= config.test_start
    if (train & validation).any() or (validation & test).any() or (train & test).any():
        raise ValueError("Temporal splits overlap")
    return train, validation, test


def make_pipeline(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("logistic", LogisticRegression(C=c_value, max_iter=2000)),
        ]
    )


def probability_metrics(
    target: pd.Series, probability: np.ndarray | pd.Series
) -> dict[str, float | int]:
    y = target.astype(int).to_numpy()
    p = np.asarray(probability, dtype=float)
    return {
        "matches": len(y),
        "accuracy": float(((p >= 0.5).astype(int) == y).mean()),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(np.mean((p - y) ** 2)),
        "roc_auc": float(roc_auc_score(y, p)),
    }


def select_best(trials: pd.DataFrame) -> pd.Series:
    return trials.sort_values(
        ["validation_log_loss", "validation_brier", "validation_accuracy"],
        ascending=[True, True, False],
        kind="stable",
    ).iloc[0]


def tune_shrinkage(base: pd.DataFrame, config: Phase4Config) -> tuple[pd.DataFrame, float, float]:
    train, validation, _ = temporal_masks(base, config)
    columns = feature_blocks()["MODEL_3_ELO_SERVE_RETURN"] + [
        "service_points_won_rate_surface_diff",
        "return_points_won_rate_surface_diff",
    ]
    rows = []
    for point_strength in config.point_prior_strengths:
        for surface_strength in config.surface_prior_strengths:
            features = materialize_features(
                base,
                point_prior_strength=point_strength,
                surface_prior_strength=surface_strength,
            )
            model = make_pipeline(1.0)
            model.fit(features.loc[train, columns], features.loc[train, "player_A_win"])
            probability = model.predict_proba(features.loc[validation, columns])[:, 1]
            metrics = probability_metrics(features.loc[validation, "player_A_win"], probability)
            rows.append(
                {
                    "point_prior_strength": point_strength,
                    "surface_prior_strength": surface_strength,
                    **{f"validation_{key}": value for key, value in metrics.items()},
                }
            )
    trials = pd.DataFrame(rows)
    selected = select_best(trials)
    return (
        trials,
        float(selected["point_prior_strength"]),
        float(selected["surface_prior_strength"]),
    )


def tune_models(
    features: pd.DataFrame, config: Phase4Config
) -> tuple[pd.DataFrame, dict[str, tuple[Pipeline, float]]]:
    train, validation, _ = temporal_masks(features, config)
    rows = []
    selected: dict[str, tuple[Pipeline, float]] = {}
    for model_name, columns in feature_blocks().items():
        model_rows = []
        for c_value in config.logistic_c_values:
            model = make_pipeline(c_value)
            model.fit(features.loc[train, columns], features.loc[train, "player_A_win"])
            probability = model.predict_proba(features.loc[validation, columns])[:, 1]
            metrics = probability_metrics(features.loc[validation, "player_A_win"], probability)
            row = {
                "model": model_name,
                "c_value": c_value,
                "features": len(columns),
                **{f"validation_{key}": value for key, value in metrics.items()},
            }
            rows.append(row)
            model_rows.append(row)
        best = select_best(pd.DataFrame(model_rows))
        final_model = make_pipeline(float(best["c_value"]))
        final_model.fit(features.loc[train, columns], features.loc[train, "player_A_win"])
        selected[model_name] = (final_model, float(best["c_value"]))
    return pd.DataFrame(rows), selected


def fit_platt(probability: np.ndarray, target: pd.Series) -> LogisticRegression:
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, max_iter=1000)
    model.fit(logits, target.astype(int))
    return model


def apply_platt(model: LogisticRegression, probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return model.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]


def calibration_table(
    target: pd.Series, variants: dict[str, np.ndarray], indexes: pd.Index
) -> pd.DataFrame:
    edges = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.0000001]
    labels = ["50-55%", "55-60%", "60-65%", "65-70%", "70-75%", "75-80%", "80-90%", "90-100%"]
    rows = []
    y = target.loc[indexes].astype(int).to_numpy()
    for name, probability in variants.items():
        favorite_probability = np.maximum(probability, 1 - probability)
        favorite_won = np.where(probability >= 0.5, y, 1 - y)
        frame = pd.DataFrame(
            {
                "bin": pd.cut(favorite_probability, edges, labels=labels, right=False),
                "probability": favorite_probability,
                "favorite_won": favorite_won,
            }
        )
        grouped = (
            frame.groupby("bin", observed=False)
            .agg(
                matches=("favorite_won", "size"),
                mean_probability=("probability", "mean"),
                actual_win_rate=("favorite_won", "mean"),
            )
            .reset_index()
        )
        grouped.insert(0, "model", name)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def _baseline_rows(
    features: pd.DataFrame, ratings: pd.DataFrame, test: pd.Series
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    y = features.loc[test, "player_A_win"]
    for name, column in (
        ("Elo Phase 3 raw", "final_raw_expected_A"),
        ("Elo Phase 3 calibrated", "final_calibrated_expected_A"),
    ):
        probability = ratings.loc[test, column].to_numpy()
        rows.append({"model": name, **probability_metrics(y, probability)})
    ranking = (
        test
        & features["rank_A"].notna()
        & features["rank_B"].notna()
        & (features["rank_A"] != features["rank_B"])
    )
    prediction = features.loc[ranking, "rank_A"] < features.loc[ranking, "rank_B"]
    rows.append(
        {
            "model": "ATP Ranking",
            "matches": int(ranking.sum()),
            "accuracy": float(
                (prediction.astype(int) == features.loc[ranking, "player_A_win"].astype(int)).mean()
            ),
            "log_loss": np.nan,
            "brier": np.nan,
            "roc_auc": np.nan,
        }
    )
    return rows


def _segment_rows(
    frame: pd.DataFrame,
    mask: pd.Series,
    probabilities: dict[str, pd.Series],
    segment_type: str,
    segment: str,
) -> list[dict[str, Any]]:
    return [
        {
            "segment_type": segment_type,
            "segment": segment,
            "model": name,
            **probability_metrics(frame.loc[mask, "player_A_win"], probability.loc[mask]),
        }
        for name, probability in probabilities.items()
    ]


def standardized_coefficients(model: Pipeline, columns: list[str]) -> pd.DataFrame:
    coefficients = model.named_steps["logistic"].coef_[0]
    return (
        pd.DataFrame({"feature": columns, "standardized_coefficient": coefficients})
        .assign(absolute_coefficient=lambda frame: frame["standardized_coefficient"].abs())
        .sort_values("absolute_coefficient", ascending=False)
    )


def feature_audit(features: pd.DataFrame, report_dir: Path) -> None:
    numeric = features.select_dtypes(include="number").drop(columns=["player_A_win"])
    coverage = pd.DataFrame(
        {
            "feature": numeric.columns,
            "non_missing": numeric.notna().sum().to_numpy(),
            "missing_rate": numeric.isna().mean().to_numpy(),
            "minimum": numeric.min().to_numpy(),
            "p01": numeric.quantile(0.01).to_numpy(),
            "median": numeric.median().to_numpy(),
            "p99": numeric.quantile(0.99).to_numpy(),
            "maximum": numeric.max().to_numpy(),
        }
    )
    coverage.to_csv(report_dir / "feature_coverage_v2.csv", index=False)
    diff_columns = [column for column in numeric if column.endswith("_diff")]
    correlation = numeric[diff_columns].corr()
    pairs = []
    for index, first in enumerate(diff_columns):
        for second in diff_columns[index + 1 :]:
            value = correlation.at[first, second]
            if pd.notna(value) and abs(value) >= 0.80:
                pairs.append({"feature_1": first, "feature_2": second, "correlation": value})
    correlations = pd.DataFrame(pairs).sort_values("correlation", key=abs, ascending=False)
    correlations.to_csv(report_dir / "feature_correlations_v2.csv", index=False)
    years = pd.to_datetime(features["date"]).dt.year
    key_columns = [
        "service_points_won_rate_career_diff",
        "return_points_won_rate_career_diff",
        "minutes_14d_diff",
        "rank_diff",
    ]
    annual = features[key_columns].notna().groupby(years).mean().reset_index(names="year")
    annual.to_csv(report_dir / "feature_coverage_v2_by_year.csv", index=False)
    text = f"""# Audit des features V2

- Matchs : {len(features):,}
- Colonnes totales : {len(features.columns):,}
- Features numériques hors cible : {len(numeric.columns):,}
- Paires de différences avec |corrélation| ≥ 0,80 : {len(correlations):,}

Les distributions et taux de valeurs manquantes complets sont dans
`feature_coverage_v2.csv`. Les couvertures annuelles des statistiques service,
retour, durée et ranking sont dans `feature_coverage_v2_by_year.csv`.

Les taux sont bornés par construction lorsque les comptes source sont cohérents.
Les valeurs incohérentes sont transformées en données manquantes pour le ratio
concerné. L'imputation médiane est apprise uniquement sur 2000–2018.

Les corrélations fortes sont conservées dans le dataset pour l'ablation mais
leurs coefficients ne doivent pas être interprétés isolément.
"""
    (report_dir / "feature_audit_v2.md").write_text(text, encoding="utf-8")


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame[columns].to_dict("records"):
        values = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                values.append("—")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.4f}")
            elif isinstance(value, (int, np.integer)):
                values.append(f"{int(value):,}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_feature_dictionary(features: pd.DataFrame, path: Path) -> None:
    text = """# Dictionnaire de features V2

Toutes les variables sont des snapshots strictement pré-match. Le suffixe `_A`
ou `_B` désigne le joueur; `_diff` vaut A−B, sauf `rank_diff`, qui conserve la
convention historique B−A afin qu'une valeur positive favorise A.

## Ratios

- `ace_rate`: aces / points de service.
- `double_fault_rate`: doubles fautes / points de service.
- `first_serve_in_rate`: premières balles en jeu / points de service.
- `first_serve_won_rate`: points gagnés sur première / premières en jeu.
- `second_serve_won_rate`: points gagnés sur seconde / (points de service − premières en jeu).
- `service_points_won_rate`: (premières gagnées + secondes gagnées) / points de service.
- `return_points_won_rate`: 1 − taux de points de service gagnés par les adversaires.
- `first_return_points_won_rate`, `second_return_points_won_rate`: compléments des taux adverses.
- `break_points_saved_rate`: balles sauvées / balles à sauver.
- `break_points_converted_rate`: breaks obtenus / opportunités adverses.
- `hold_rate`: (jeux de service − breaks concédés) / jeux de service.
- `win_rate`: victoires / matchs, avec prior Beta équivalent à 10 matchs à 50 %.

`career`, `365d` et `90d` indiquent l'horizon. Chaque ratio est calculé sur les
sommes de numérateurs et dénominateurs, puis shrinké vers la moyenne historique
du circuit disponible avant le match. Les colonnes `opportunities` exposent le
dénominateur pré-match. Les variantes `surface` sont shrinkées vers la carrière.

## Charge et expérience

- `matches/sets/games/minutes_{7,14,30}d`: charge des matchs des batches antérieurs.
- `rest_days`: jours depuis le dernier batch source; zéro peut signifier un round antérieur du même tournoi.
- `career_matches`, `surface_matches`, `age_years`: expérience pré-match.
- `elo_logit`, `elo_raw_probability`, `elo_diff`, `surface_elo_blended_diff`: baseline Phase 3 figée.
- `rank`, `rank_points`: observation ATP du match; `rank_diff = rank_B - rank_A`.

## Colonnes présentes

"""
    text += "\n".join(f"- `{column}`" for column in features.columns) + "\n"
    path.write_text(text, encoding="utf-8")


def run_phase4(
    *,
    config: Phase4Config,
    ratings: pd.DataFrame,
    timeline: pd.DataFrame,
    matches: pd.DataFrame,
    stats: pd.DataFrame,
    editions: pd.DataFrame,
    report_dir: Path,
    output_path: Path,
    model_path: Path,
) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    audit, by_year, by_level, by_surface = build_stats_audit(matches, stats, editions)
    write_stats_audit(report_dir, audit, by_year, by_level, by_surface)

    LOGGER.info("Building strictly pre-match sufficient statistics")
    base = build_stat_history_base(ratings, timeline, matches, stats)
    LOGGER.info("Selecting 9 shrinkage configurations on validation")
    shrinkage_trials, point_strength, surface_strength = tune_shrinkage(base, config)
    shrinkage_trials.to_csv(report_dir / "phase4_shrinkage_tuning.csv", index=False)
    features = materialize_features(
        base,
        point_prior_strength=point_strength,
        surface_prior_strength=surface_strength,
    )
    write_feature_dataset(
        features,
        output_path,
        point_prior_strength=point_strength,
        surface_prior_strength=surface_strength,
    )
    write_feature_dictionary(features, PROJECT_ROOT / "docs" / "features_v2.md")
    feature_audit(features, report_dir)

    LOGGER.info("Tuning logistic L2 models on validation")
    model_trials, selected_models = tune_models(features, config)
    model_trials.to_csv(report_dir / "phase4_logistic_tuning.csv", index=False)
    _, validation, test = temporal_masks(features, config)
    validation_rows = []
    for model_name, (model, c_value) in selected_models.items():
        columns = feature_blocks()[model_name]
        probability = model.predict_proba(features.loc[validation, columns])[:, 1]
        validation_rows.append(
            {
                "model": model_name,
                "c_value": c_value,
                "features": len(columns),
                **probability_metrics(features.loc[validation, "player_A_win"], probability),
            }
        )
    validation_comparison = pd.DataFrame(validation_rows).sort_values("log_loss")
    validation_comparison.to_csv(report_dir / "phase4_ablation_validation.csv", index=False)
    final_name = str(validation_comparison.iloc[0]["model"])
    final_model = selected_models[final_name][0]
    final_columns = feature_blocks()[final_name]

    validation_raw = final_model.predict_proba(features.loc[validation, final_columns])[:, 1]
    calibrator = fit_platt(validation_raw, features.loc[validation, "player_A_win"])
    validation_calibrated = apply_platt(calibrator, validation_raw)
    raw_validation_metrics = probability_metrics(
        features.loc[validation, "player_A_win"], validation_raw
    )
    calibrated_validation_metrics = probability_metrics(
        features.loc[validation, "player_A_win"], validation_calibrated
    )
    use_calibration = (
        calibrated_validation_metrics["log_loss"] < raw_validation_metrics["log_loss"]
        and calibrated_validation_metrics["brier"] <= raw_validation_metrics["brier"]
    )
    calibration_validation = pd.DataFrame(
        [
            {"model": "Logistic raw", **raw_validation_metrics},
            {"model": "Logistic Platt", **calibrated_validation_metrics},
        ]
    )
    calibration_validation.to_csv(report_dir / "phase4_calibration_validation.csv", index=False)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(
        {
            "pipeline": final_model,
            "calibrator": calibrator if use_calibration else None,
            "feature_columns": final_columns,
            "model_name": final_name,
            "trained_through": config.train_end.date().isoformat(),
            "calibration_period": [
                config.validation_start.date().isoformat(),
                config.validation_end.date().isoformat(),
            ],
        },
        model_path,
    )

    # The final test is opened only after shrinkage, blocks, C and calibration are frozen.
    test_raw_values = final_model.predict_proba(features.loc[test, final_columns])[:, 1]
    test_calibrated_values = apply_platt(calibrator, test_raw_values)
    test_final_values = test_calibrated_values if use_calibration else test_raw_values
    test_index = features.index[test]
    final_probability = pd.Series(np.nan, index=features.index, dtype=float)
    final_probability.loc[test_index] = test_final_values
    raw_probability = pd.Series(np.nan, index=features.index, dtype=float)
    raw_probability.loc[test_index] = test_raw_values

    comparison_rows = _baseline_rows(features, ratings, test)
    for model_name, (model, _) in selected_models.items():
        columns = feature_blocks()[model_name]
        probability = model.predict_proba(features.loc[test, columns])[:, 1]
        comparison_rows.append(
            {
                "model": model_name,
                **probability_metrics(features.loc[test, "player_A_win"], probability),
            }
        )
    comparison_rows.append(
        {
            "model": f"{final_name} CALIBRATED" if use_calibration else f"{final_name} FINAL",
            **probability_metrics(features.loc[test, "player_A_win"], test_final_values),
        }
    )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(report_dir / "phase4_test_comparison.csv", index=False)
    ranking_overlap = (
        test
        & features["rank_A"].notna()
        & features["rank_B"].notna()
        & (features["rank_A"] != features["rank_B"])
    )
    ranking_prediction = (
        features.loc[ranking_overlap, "rank_A"] < features.loc[ranking_overlap, "rank_B"]
    )
    atp_overlap = pd.DataFrame(
        [
            {
                "model": "ATP Ranking",
                "matches": int(ranking_overlap.sum()),
                "accuracy": float(
                    (
                        ranking_prediction.astype(int)
                        == features.loc[ranking_overlap, "player_A_win"].astype(int)
                    ).mean()
                ),
                "log_loss": np.nan,
                "brier": np.nan,
                "roc_auc": np.nan,
            },
            {
                "model": "Logistic Phase 4",
                **probability_metrics(
                    features.loc[ranking_overlap, "player_A_win"],
                    final_probability.loc[ranking_overlap],
                ),
            },
        ]
    )
    atp_overlap.to_csv(report_dir / "phase4_atp_overlap.csv", index=False)

    elo_probability = ratings["final_calibrated_expected_A"]
    variants = {
        "Elo Phase 3 calibrated": elo_probability,
        "Logistic Phase 4": final_probability,
    }
    segment_rows: list[dict[str, Any]] = []
    for surface in ("Hard", "Clay", "Grass"):
        mask = test & (features["surface"] == surface)
        segment_rows.extend(_segment_rows(features, mask, variants, "surface", surface))
    years = pd.to_datetime(features["date"]).dt.year
    for year in sorted(years.loc[test].unique()):
        mask = test & (years == year)
        segment_rows.extend(_segment_rows(features, mask, variants, "year", str(year)))
    segments = pd.DataFrame(segment_rows)
    segments.to_csv(report_dir / "phase4_test_segments.csv", index=False)

    calibration = calibration_table(
        features["player_A_win"],
        {
            "Logistic raw": test_raw_values,
            "Logistic final": test_final_values,
            "Elo calibrated": elo_probability.loc[test].to_numpy(),
        },
        test_index,
    )
    calibration.to_csv(report_dir / "phase4_calibration.csv", index=False)
    coefficients = standardized_coefficients(final_model, final_columns)
    coefficients.to_csv(report_dir / "phase4_coefficients.csv", index=False)

    selected = {
        "point_prior_strength": point_strength,
        "surface_prior_strength": surface_strength,
        "model": final_name,
        "logistic_c": selected_models[final_name][1],
        "features": final_columns,
        "platt_used": use_calibration,
        "platt_intercept": float(calibrator.intercept_[0]),
        "platt_slope": float(calibrator.coef_[0, 0]),
    }
    (report_dir / "phase4_selected.json").write_text(
        json.dumps(selected, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# Phase 4 — statistiques historiques et régression logistique

## A. Audit des statistiques

Le bloc service complet couvre {audit["complete_service_bundle_matches"]:,} matchs
({audit["complete_service_bundle_coverage"]:.2%}); la durée valide couvre
{audit["duration_matches"]:,} matchs ({audit["duration_coverage"]:.2%}). Le train 2000–2018 est
conservé : les premières années sont déjà proches de 87–90 %, et le manque principal vient de
la Davis Cup. Voir `stats_availability_audit.md`.

## B. Features créées

Le dataset contient {len(features.columns):,} colonnes et {len(features):,} matchs. Les ratios
service/retour utilisent des numérateurs et opportunités cumulés, avec horizons carrière, 365j
et 90j, priors temporels, variantes surface, forme et charge 7/14/30j. Définitions complètes :
`docs/features_v2.md`.

Shrinkage sélectionné sur validation : points C={point_strength:g}, surface C={surface_strength:g}.

## C–D. Signal incrémental et ablation (validation 2019–2022)

{_markdown_table(validation_comparison, ["model", "features", "c_value", "accuracy", "log_loss", "brier", "roc_auc"])}

Le meilleur bloc est `{final_name}`. Les variables corrélées rendent les coefficients individuels
conditionnels aux autres variables; l'ablation est donc la preuve principale d'apport.

## E. Comparaison finale — test 2023–2026

{_markdown_table(comparison, ["model", "matches", "accuracy", "log_loss", "brier", "roc_auc"])}

Comparaison appariée sur les matchs avec deux rangs ATP disponibles et distincts :

{_markdown_table(atp_overlap, ["model", "matches", "accuracy", "log_loss", "brier", "roc_auc"])}

## F–G. Surfaces et années

{_markdown_table(segments, ["segment_type", "segment", "model", "matches", "accuracy", "log_loss", "brier", "roc_auc"])}

## H. Calibration

Calibration Platt retenue : {use_calibration}. Elle a été ajustée sur 2019–2022 uniquement,
après entraînement du modèle sur 2000–2018. Buckets complets : `phase4_calibration.csv`.

{_markdown_table(calibration_validation, ["model", "matches", "accuracy", "log_loss", "brier", "roc_auc"])}

## I. Coefficients standardisés

{_markdown_table(coefficients.head(20), ["feature", "standardized_coefficient"])}

Ils décrivent des associations conditionnelles, pas des effets causaux. Les ratios service et
retour sont fortement liés; leur importance doit être lue avec l'ablation.

## J. Protections anti-leakage

- snapshot de toutes les features avant le batch de round, puis mise à jour;
- stats, score et durée du match courant visibles seulement au batch suivant;
- priors circuit calculés uniquement avec l'historique antérieur;
- surface et fatigue strictement antérieures;
- imputer, scaler et régression ajustés sur 2000–2018 seulement;
- shrinkage, blocs, régularisation et calibration choisis sur 2019–2022;
- aucune fonction de sélection ne reçoit le masque test; l'évaluation test se trouve après le gel complet.

La date exacte des matchs n'étant pas disponible, la charge intra-tournoi suit l'ordre des
rounds et ne prétend pas reconstruire des jours calendaires inexistants.
"""
    (report_dir / "phase4_report.md").write_text(report, encoding="utf-8")
    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "selected": selected,
        "test_comparison": [
            {
                key: (None if isinstance(value, float) and math.isnan(value) else value)
                for key, value in row.items()
            }
            for row in comparison.to_dict("records")
        ],
        "atp_overlap": [
            {
                key: (None if isinstance(value, float) and math.isnan(value) else value)
                for key, value in row.items()
            }
            for row in atp_overlap.to_dict("records")
        ],
    }
    (report_dir / "phase4_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {"selected": selected, "comparison": comparison, "segments": segments}


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Build Phase 4 features and logistic models")
    command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    command.add_argument("--ratings", type=Path, default=PROCESSED_DIR / "match_ratings_v2.parquet")
    command.add_argument(
        "--timeline", type=Path, default=PROCESSED_DIR / "analytics" / "match_timeline.parquet"
    )
    command.add_argument("--atp-dir", type=Path, default=PROCESSED_DIR / "atp")
    command.add_argument("--report-dir", type=Path, default=REPORTS_DIR)
    command.add_argument("--output", type=Path, default=PROCESSED_DIR / "match_features_v2.parquet")
    command.add_argument(
        "--model-output",
        type=Path,
        default=PROCESSED_DIR / "models" / "logistic_phase4.joblib",
    )
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results = run_phase4(
        config=Phase4Config.from_toml(args.config),
        ratings=pd.read_parquet(args.ratings),
        timeline=pd.read_parquet(args.timeline),
        matches=pd.read_parquet(args.atp_dir / "matches.parquet"),
        stats=pd.read_parquet(args.atp_dir / "match_stats.parquet"),
        editions=pd.read_parquet(args.atp_dir / "tournament_editions.parquet"),
        report_dir=args.report_dir,
        output_path=args.output,
        model_path=args.model_output,
    )
    print(results["comparison"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
