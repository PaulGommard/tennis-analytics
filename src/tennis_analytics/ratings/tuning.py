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

from tennis_analytics.paths import PROCESSED_DIR, PROJECT_ROOT, REPORTS_DIR
from tennis_analytics.ratings.chronology import (
    apply_round_chronology,
    chronology_audit,
    write_chronology_audit,
)
from tennis_analytics.ratings.elo import EloConfig
from tennis_analytics.ratings.engine import (
    InactivityConfig,
    blended_surface_probability,
    run_elo_engine,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "elo_phase3.toml"


@dataclass(frozen=True)
class Phase3Config:
    k_values: tuple[float, ...]
    scale_values: tuple[float, ...]
    surface_c_values: tuple[float, ...]
    inactivity_threshold_days: tuple[int, ...]
    inactivity_strengths: tuple[float, ...]
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp

    @classmethod
    def from_toml(cls, path: Path = DEFAULT_CONFIG) -> Phase3Config:
        with path.open("rb") as stream:
            values = tomllib.load(stream)
        search = values["search"]
        evaluation = values["evaluation"]
        return cls(
            k_values=tuple(float(value) for value in search["k_values"]),
            scale_values=tuple(float(value) for value in search["scale_values"]),
            surface_c_values=tuple(float(value) for value in search["surface_c_values"]),
            inactivity_threshold_days=tuple(
                int(value) for value in search["inactivity_threshold_days"]
            ),
            inactivity_strengths=tuple(float(value) for value in search["inactivity_strengths"]),
            validation_start=pd.Timestamp(evaluation["validation_start"]),
            validation_end=pd.Timestamp(evaluation["validation_end"]),
            test_start=pd.Timestamp(evaluation["test_start"]),
        )


def probability_metrics(target: pd.Series, probability: pd.Series) -> dict[str, float | int]:
    mask = probability.notna()
    y = target.loc[mask].astype(float)
    p = probability.loc[mask].astype(float).clip(1e-15, 1 - 1e-15)
    favorite = p != 0.5
    return {
        "matches": int(mask.sum()),
        "accuracy_n": int(favorite.sum()),
        "accuracy": (
            float(((p.loc[favorite] > 0.5).astype(int) == y.loc[favorite]).mean())
            if favorite.any()
            else np.nan
        ),
        "log_loss": (
            -float((y * np.log(p) + (1 - y) * np.log(1 - p)).mean()) if len(p) else np.nan
        ),
        "brier": float(((p - y) ** 2).mean()) if len(p) else np.nan,
    }


def assert_validation_only(dates: pd.Series, config: Phase3Config) -> None:
    parsed = pd.to_datetime(dates)
    if not parsed.between(config.validation_start, config.validation_end).all():
        raise ValueError("Hyperparameter selection received rows outside 2019–2022 validation")


def select_best(trials: pd.DataFrame) -> pd.Series:
    """Select only by validation log loss, then Brier, then accuracy."""
    required = {"validation_log_loss", "validation_brier", "validation_accuracy"}
    if not required.issubset(trials.columns):
        raise ValueError(f"Selection table lacks: {sorted(required.difference(trials.columns))}")
    ordered = trials.sort_values(
        ["validation_log_loss", "validation_brier", "validation_accuracy"],
        ascending=[True, True, False],
        kind="stable",
    )
    return ordered.iloc[0]


def _validation_mask(frame: pd.DataFrame, config: Phase3Config) -> pd.Series:
    dates = pd.to_datetime(frame["date"])
    return dates.between(config.validation_start, config.validation_end)


def _test_mask(frame: pd.DataFrame, config: Phase3Config) -> pd.Series:
    return pd.to_datetime(frame["date"]) >= config.test_start


def _metrics_for_period(
    frame: pd.DataFrame,
    probability: pd.Series,
    mask: pd.Series,
) -> dict[str, float | int]:
    return probability_metrics(frame.loc[mask, "player_A_win"], probability.loc[mask])


def tune_global(
    chronology: pd.DataFrame,
    search: Phase3Config,
    surfaces: tuple[str, ...],
) -> tuple[pd.DataFrame, EloConfig]:
    validation = _validation_mask(chronology, search)
    assert_validation_only(chronology.loc[validation, "date"], search)
    rows: list[dict[str, float | int]] = []
    for k_factor in search.k_values:
        for scale in search.scale_values:
            config = EloConfig(k_factor=k_factor, scale=scale, surfaces=surfaces)
            ratings = run_elo_engine(chronology, config)
            metrics = _metrics_for_period(
                ratings, ratings["elo_expected_A"], _validation_mask(ratings, search)
            )
            rows.append(
                {
                    "k_factor": k_factor,
                    "scale": scale,
                    **{f"validation_{key}": value for key, value in metrics.items()},
                }
            )
    trials = pd.DataFrame(rows)
    best = select_best(trials)
    return trials, EloConfig(
        k_factor=float(best["k_factor"]),
        scale=float(best["scale"]),
        surfaces=surfaces,
    )


def tune_surface_shrinkage(
    ratings: pd.DataFrame,
    search: Phase3Config,
    scale: float,
) -> tuple[pd.DataFrame, float]:
    validation = _validation_mask(ratings, search)
    assert_validation_only(ratings.loc[validation, "date"], search)
    rows = []
    for c_value in search.surface_c_values:
        probability = blended_surface_probability(ratings, c_value=c_value, scale=scale)
        metrics = _metrics_for_period(ratings, probability, validation)
        rows.append(
            {
                "c_value": c_value,
                **{f"validation_{key}": value for key, value in metrics.items()},
            }
        )
    trials = pd.DataFrame(rows)
    best = select_best(trials)
    return trials, float(best["c_value"])


def _variant_probability(
    ratings: pd.DataFrame,
    variant: str,
    *,
    c_value: float,
    scale: float,
) -> pd.Series:
    if variant == "global":
        return ratings["elo_expected_A"]
    if variant == "surface_independent":
        return ratings["surface_elo_expected_A"]
    if variant == "surface_shrinkage":
        return blended_surface_probability(ratings, c_value=c_value, scale=scale)
    raise ValueError(f"Unknown rating variant: {variant}")


def choose_raw_variant(
    ratings: pd.DataFrame,
    search: Phase3Config,
    *,
    c_value: float,
    scale: float,
) -> tuple[pd.DataFrame, str]:
    validation = _validation_mask(ratings, search)
    rows = []
    for variant in ("global", "surface_independent", "surface_shrinkage"):
        probability = _variant_probability(ratings, variant, c_value=c_value, scale=scale)
        metrics = _metrics_for_period(ratings, probability, validation)
        rows.append(
            {
                "variant": variant,
                **{f"validation_{key}": value for key, value in metrics.items()},
            }
        )
    comparison = pd.DataFrame(rows)
    return comparison, str(select_best(comparison)["variant"])


def tune_inactivity(
    chronology: pd.DataFrame,
    elo_config: EloConfig,
    search: Phase3Config,
    *,
    raw_variant: str,
    c_value: float,
) -> tuple[pd.DataFrame, InactivityConfig | None, InactivityConfig | None]:
    candidates: list[InactivityConfig | None] = [None]
    candidates.extend(
        InactivityConfig(threshold, strength)
        for threshold in search.inactivity_threshold_days
        for strength in search.inactivity_strengths
    )
    rows = []
    for candidate in candidates:
        ratings = run_elo_engine(chronology, elo_config, candidate)
        probability = _variant_probability(
            ratings,
            raw_variant,
            c_value=c_value,
            scale=elo_config.scale,
        )
        validation = _validation_mask(ratings, search)
        metrics = _metrics_for_period(ratings, probability, validation)
        rows.append(
            {
                "threshold_days": candidate.threshold_days if candidate else 0,
                "strength": candidate.strength if candidate else 0.0,
                "is_no_decay": candidate is None,
                **{f"validation_{key}": value for key, value in metrics.items()},
            }
        )
    trials = pd.DataFrame(rows)
    selected = select_best(trials)
    selected_config = (
        None
        if bool(selected["is_no_decay"])
        else InactivityConfig(int(selected["threshold_days"]), float(selected["strength"]))
    )
    nonzero = trials.loc[~trials["is_no_decay"]]
    best_nonzero = select_best(nonzero)
    best_nonzero_config = InactivityConfig(
        int(best_nonzero["threshold_days"]), float(best_nonzero["strength"])
    )
    return trials, selected_config, best_nonzero_config


def _sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    result = np.empty_like(values, dtype=float)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def fit_platt(
    probability: pd.Series,
    target: pd.Series,
    dates: pd.Series,
    config: Phase3Config,
) -> tuple[float, float]:
    assert_validation_only(dates, config)
    mask = probability.notna()
    p = probability.loc[mask].astype(float).clip(1e-6, 1 - 1e-6).to_numpy()
    y = target.loc[mask].astype(float).to_numpy()
    logit = np.log(p / (1 - p))
    design = np.column_stack([np.ones(len(logit)), logit])
    coefficients = np.array([0.0, 1.0])
    for _ in range(100):
        fitted = _sigmoid(design @ coefficients)
        weights = np.clip(fitted * (1 - fitted), 1e-9, None)
        gradient = design.T @ (y - fitted)
        information = design.T @ (weights[:, None] * design)
        step = np.linalg.solve(information + np.eye(2) * 1e-9, gradient)
        coefficients += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(coefficients[0]), float(coefficients[1])


def apply_platt(probability: pd.Series, intercept: float, slope: float) -> pd.Series:
    result = pd.Series(np.nan, index=probability.index, dtype=float)
    mask = probability.notna()
    p = probability.loc[mask].astype(float).clip(1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    result.loc[mask] = _sigmoid((intercept + slope * logit).to_numpy())
    return result


def calibration_table(
    frame: pd.DataFrame,
    variants: dict[str, pd.Series],
    mask: pd.Series,
) -> pd.DataFrame:
    edges = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.0000001]
    labels = ["50-55%", "55-60%", "60-65%", "65-70%", "70-75%", "75-80%", "80-90%", "90-100%"]
    rows = []
    for name, probability in variants.items():
        subset = pd.DataFrame(
            {"probability": probability.loc[mask], "target": frame.loc[mask, "player_A_win"]}
        ).dropna()
        subset = subset.loc[subset["probability"] != 0.5]
        subset["favorite_probability"] = np.maximum(
            subset["probability"], 1 - subset["probability"]
        )
        subset["favorite_won"] = np.where(
            subset["probability"] > 0.5, subset["target"], 1 - subset["target"]
        )
        subset["bin"] = pd.cut(
            subset["favorite_probability"], edges, labels=labels, right=False, include_lowest=True
        )
        grouped = (
            subset.groupby("bin", observed=False)
            .agg(
                matches=("favorite_won", "size"),
                mean_probability=("favorite_probability", "mean"),
                actual_win_rate=("favorite_won", "mean"),
            )
            .reset_index()
        )
        grouped.insert(0, "model", name)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def _comparison_rows(
    frame: pd.DataFrame,
    models: dict[str, pd.Series],
    mask: pd.Series,
    *,
    segment_type: str = "overall",
    segment: str = "All",
) -> list[dict[str, Any]]:
    rows = []
    for name, probability in models.items():
        rows.append(
            {
                "segment_type": segment_type,
                "segment": segment,
                "model": name,
                **probability_metrics(frame.loc[mask, "player_A_win"], probability.loc[mask]),
            }
        )
    return rows


def ranking_metrics(frame: pd.DataFrame, mask: pd.Series) -> dict[str, Any]:
    eligible = (
        mask
        & frame["rank_A"].notna()
        & frame["rank_B"].notna()
        & (frame["rank_A"] != frame["rank_B"])
    )
    prediction = frame.loc[eligible, "rank_A"] < frame.loc[eligible, "rank_B"]
    return {
        "segment_type": "overall",
        "segment": "All",
        "model": "ATP Ranking",
        "matches": int(eligible.sum()),
        "accuracy_n": int(eligible.sum()),
        "accuracy": float(
            (prediction.astype(int) == frame.loc[eligible, "player_A_win"].astype(int)).mean()
        ),
        "log_loss": np.nan,
        "brier": np.nan,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _markdown_table(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    labels: dict[str, str] | None = None,
) -> str:
    labels = labels or {}
    header = [labels.get(column, column) for column in columns]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for record in frame[columns].to_dict("records"):
        values = []
        for column in columns:
            value = record[column]
            if pd.isna(value):
                rendered = "—"
            elif column in {
                "accuracy",
                "log_loss",
                "brier",
                "validation_accuracy",
                "validation_log_loss",
                "validation_brier",
                "mean_probability",
                "actual_win_rate",
                "gap",
            }:
                rendered = f"{float(value):.4f}"
            elif isinstance(value, (float, np.floating)):
                rendered = f"{float(value):g}"
            elif isinstance(value, (int, np.integer)):
                rendered = f"{int(value):,}"
            else:
                rendered = str(value)
            values.append(rendered.replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_phase3_report(
    report_dir: Path,
    *,
    audit: dict[str, Any],
    selected: dict[str, Any],
    comparison: pd.DataFrame,
    atp_overlap: pd.DataFrame,
    segmented: pd.DataFrame,
    chronology_impact: pd.DataFrame,
    raw_comparison: pd.DataFrame,
    inactivity_trials: pd.DataFrame,
    calibration: pd.DataFrame,
) -> None:
    chronology_overall = chronology_impact.loc[chronology_impact["segment_type"] == "overall"]
    surface_test = segmented.loc[
        (segmented["segment_type"] == "surface")
        & segmented["model"].isin(
            [
                "Elo global optimized",
                "Surface Elo independent optimized",
                "Surface Elo shrinkage",
                "Elo final raw",
                "Elo final calibrated",
            ]
        )
    ]
    inactivity_ordered = inactivity_trials.sort_values(
        ["validation_log_loss", "validation_brier", "validation_accuracy"],
        ascending=[True, True, False],
    ).head(5)
    calibration_display = calibration.copy()
    calibration_display["gap"] = (
        calibration_display["actual_win_rate"] - calibration_display["mean_probability"]
    )
    inactivity = selected["inactivity"]
    inactivity_text = (
        "aucune (la variante sans décroissance gagne sur validation)"
        if inactivity is None
        else f"seuil {inactivity['threshold_days']} jours, force {inactivity['strength']:g}"
    )
    text = f"""# Phase 3 — optimisation et calibration Elo

## Protocole figé

- Historique continu depuis 2000, sans réinitialisation.
- Sélection des hyperparamètres exclusivement sur 2019–2022.
- Test final exclusivement à partir de 2023, consulté après gel des choix.
- Critère principal : log loss ; départage par Brier score puis accuracy.
- Aucun modèle de ML avancé : Elo global, Elo surface, shrinkage, inactivité simple et Platt.

## Audit de chronologie

Les {audit["editions"]:,} éditions ({audit["matches_in_single_date_editions"]:,} matchs) ont leurs
matchs regroupés sur une seule date source. La nouvelle chronologie exploite donc uniquement
l’ordre structurel des tours, avec mise à jour après chaque tour complet. Davis Cup, labels
inconnus et dates présentant un joueur dans plusieurs éditions restent conservateurs. Un seul
couple joueur/date inter-éditions a été détecté.

Ordre : Q1 → Q2 → Q3 → Q4 → RR → ER → R128 → R64 → R32 → R16 → QF → SF → FINAL.
F et BR partagent le même batch FINAL. Aucun match Q1–Q4 n’est présent dans ce corpus.

### Impact isolé de la chronologie sur le test (K=32, scale=400)

{_markdown_table(chronology_overall, ["model", "matches", "accuracy", "log_loss", "brier"])}

Les détails par surface, niveau et année sont dans `elo_chronology_impact.csv`; l’audit complet
est dans `round_chronology_audit.md` et `round_chronology_audit.json`.

## Sélection sur validation 2019–2022

- Elo global : K={selected["k_factor"]:g}, scale={selected["scale"]:g}.
- Shrinkage surface : C={selected["surface_c"]:g}, avec `w = n / (n + C)` par joueur.
- Variante brute retenue : `{selected["raw_variant"]}`.
- Inactivité retenue : {inactivity_text}.
- Calibration de Platt : intercept={selected["platt_intercept"]:.6f}, pente={selected["platt_slope"]:.6f}.

### Comparaison des variantes brutes sur validation

{_markdown_table(raw_comparison, ["variant", "validation_matches", "validation_accuracy", "validation_log_loss", "validation_brier"])}

### Cinq meilleures variantes d’inactivité sur validation

{_markdown_table(inactivity_ordered, ["threshold_days", "strength", "is_no_decay", "validation_accuracy", "validation_log_loss", "validation_brier"])}

Les grilles exhaustives mais volontairement bornées sont dans `elo_global_tuning.csv`,
`elo_surface_shrinkage_tuning.csv` et `elo_inactivity_tuning.csv`.

## Résultat final sur le test 2023+

{_markdown_table(comparison, ["model", "matches", "accuracy", "log_loss", "brier"])}

### Comparaison appariée avec le classement ATP

Chaque paire conserve exactement le même dénominateur : rangs ATP disponibles et distincts,
puis exclusion des éventuelles probabilités Elo exactement égales à 0,5 pour la paire brute.

{_markdown_table(atp_overlap, ["model", "matches", "accuracy", "log_loss", "brier"])}

### Résultats par surface

{_markdown_table(surface_test, ["segment", "model", "matches", "accuracy", "log_loss", "brier"])}

Les résultats complets par surface et niveau sont dans `phase3_test_segments.csv`.

## Calibration sur le test

{_markdown_table(calibration_display, ["model", "bin", "matches", "mean_probability", "actual_win_rate", "gap"])}

La calibration est apprise sur la seule validation puis appliquée au test. Les probabilités
calibrées historiques antérieures au test ne sont pas publiées dans le parquet afin d’éviter
toute interprétation walk-forward incorrecte.

## Artefact retenu

`data/processed/match_ratings_v2.parquet` contient les ratings pré-match, les compteurs de
surface antérieurs, les probabilités brutes et la probabilité Platt uniquement sur 2023+.
`elo_annual_walk_forward.csv` donne les diagnostics annuels des variantes de rating fixes ;
il exclut volontairement Platt, appris ultérieurement sur 2019–2022.

## Limites

- La date source reste la date de début du tournoi : l’ordre des tours est structurel, pas daté.
- Tous les matchs d’un même tour restent gelés dans un batch.
- Les formats ambigus sont volontairement sous-informatifs plutôt que reconstruits artificiellement.
- Le niveau ATP `A` ne permet pas de séparer fiablement ATP 250 et ATP 500.
- Une bonne log loss ne constitue pas à elle seule une preuve de rentabilité de paris.
"""
    (report_dir / "phase3_report.md").write_text(text, encoding="utf-8")


def run_phase3(
    timeline: pd.DataFrame,
    phase2_features: pd.DataFrame,
    *,
    search: Phase3Config,
    config_path: Path,
    report_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    audit = chronology_audit(timeline)
    write_chronology_audit(audit, report_dir)
    chronology = apply_round_chronology(timeline, config_path=config_path)
    surfaces = ("Hard", "Clay", "Grass", "Carpet")

    LOGGER.info("Measuring chronology-only impact")
    base_config = EloConfig(k_factor=32.0, scale=400.0, surfaces=surfaces)
    round_base = run_elo_engine(chronology, base_config)
    old = phase2_features.set_index("match_id")
    old_global = round_base["match_id"].map(old["elo_expected_A"])
    old_surface = round_base["match_id"].map(old["surface_elo_expected_A"])

    LOGGER.info(
        "Tuning %d global Elo configurations on validation",
        len(search.k_values) * len(search.scale_values),
    )
    global_trials, best_config = tune_global(chronology, search, surfaces)
    global_trials.to_csv(report_dir / "elo_global_tuning.csv", index=False)
    optimized = run_elo_engine(chronology, best_config)

    LOGGER.info("Tuning surface shrinkage")
    surface_trials, best_c = tune_surface_shrinkage(optimized, search, best_config.scale)
    surface_trials.to_csv(report_dir / "elo_surface_shrinkage_tuning.csv", index=False)
    raw_comparison, raw_variant = choose_raw_variant(
        optimized, search, c_value=best_c, scale=best_config.scale
    )
    raw_comparison.to_csv(report_dir / "elo_raw_variant_validation.csv", index=False)

    LOGGER.info("Testing controlled inactivity variants on validation")
    inactivity_trials, selected_inactivity, best_nonzero_inactivity = tune_inactivity(
        chronology,
        best_config,
        search,
        raw_variant=raw_variant,
        c_value=best_c,
    )
    inactivity_trials.to_csv(report_dir / "elo_inactivity_tuning.csv", index=False)
    final_ratings = run_elo_engine(chronology, best_config, selected_inactivity)
    best_nonzero_ratings = run_elo_engine(chronology, best_config, best_nonzero_inactivity)
    final_raw = _variant_probability(
        final_ratings,
        raw_variant,
        c_value=best_c,
        scale=best_config.scale,
    )
    best_nonzero_probability = _variant_probability(
        best_nonzero_ratings,
        raw_variant,
        c_value=best_c,
        scale=best_config.scale,
    )

    validation = _validation_mask(final_ratings, search)
    test = _test_mask(final_ratings, search)
    intercept, slope = fit_platt(
        final_raw.loc[validation],
        final_ratings.loc[validation, "player_A_win"],
        final_ratings.loc[validation, "date"],
        search,
    )
    calibrated = apply_platt(final_raw, intercept, slope)

    optimized_shrink = blended_surface_probability(
        optimized, c_value=best_c, scale=best_config.scale
    )
    models = {
        "Elo V1 date batch": old_global,
        "Elo round K32/400": round_base["elo_expected_A"],
        "Elo global optimized": optimized["elo_expected_A"],
        "Surface Elo V1 date batch": old_surface,
        "Surface Elo independent optimized": optimized["surface_elo_expected_A"],
        "Surface Elo shrinkage": optimized_shrink,
        "Elo + inactivity best tested": best_nonzero_probability,
        "Elo final raw": final_raw,
        "Elo final calibrated": calibrated,
    }

    # The final test is touched only here, after every selection is frozen.
    aligned_phase2 = (
        phase2_features.set_index("match_id").loc[final_ratings["match_id"]].reset_index()
    )
    comparison_rows = [ranking_metrics(aligned_phase2, test)]
    comparison_rows.extend(_comparison_rows(final_ratings, models, test))
    comparison = pd.DataFrame(comparison_rows)
    atp_eligible = (
        test
        & aligned_phase2["rank_A"].notna()
        & aligned_phase2["rank_B"].notna()
        & (aligned_phase2["rank_A"] != aligned_phase2["rank_B"])
    )
    raw_overlap = atp_eligible & (final_raw != 0.5)
    calibrated_overlap = atp_eligible & (calibrated != 0.5)
    atp_raw = ranking_metrics(aligned_phase2, raw_overlap)
    atp_raw["model"] = "ATP Ranking (raw overlap)"
    atp_calibrated = ranking_metrics(aligned_phase2, calibrated_overlap)
    atp_calibrated["model"] = "ATP Ranking (calibrated overlap)"
    atp_overlap_rows = [atp_raw]
    atp_overlap_rows.extend(
        _comparison_rows(
            final_ratings,
            {"Elo final raw": final_raw},
            raw_overlap,
        )
    )
    atp_overlap_rows.append(atp_calibrated)
    atp_overlap_rows.extend(
        _comparison_rows(
            final_ratings,
            {"Elo final calibrated": calibrated},
            calibrated_overlap,
        )
    )
    atp_overlap = pd.DataFrame(atp_overlap_rows)

    segmented_rows: list[dict[str, Any]] = []
    for surface in ("Hard", "Clay", "Grass"):
        segment_mask = test & (final_ratings["surface"] == surface)
        segmented_rows.extend(
            _comparison_rows(
                final_ratings,
                models,
                segment_mask,
                segment_type="surface",
                segment=surface,
            )
        )
    for level in sorted(final_ratings["level"].dropna().unique()):
        segment_mask = test & (final_ratings["level"] == level)
        segmented_rows.extend(
            _comparison_rows(
                final_ratings,
                models,
                segment_mask,
                segment_type="level",
                segment=str(level),
            )
        )
    segmented = pd.DataFrame(segmented_rows)
    comparison.to_csv(report_dir / "phase3_test_comparison.csv", index=False)
    atp_overlap.to_csv(report_dir / "phase3_atp_overlap.csv", index=False)
    segmented.to_csv(report_dir / "phase3_test_segments.csv", index=False)

    chronology_rows: list[dict[str, Any]] = []
    chronology_models = {
        "Elo V1 date batch": old_global,
        "Elo round K32/400": round_base["elo_expected_A"],
    }
    chronology_rows.extend(_comparison_rows(round_base, chronology_models, test))
    for column, segment_type in (("surface", "surface"), ("level", "level")):
        for value in sorted(round_base[column].dropna().unique()):
            segment_mask = test & (round_base[column] == value)
            chronology_rows.extend(
                _comparison_rows(
                    round_base,
                    chronology_models,
                    segment_mask,
                    segment_type=segment_type,
                    segment=str(value),
                )
            )
    years = pd.to_datetime(round_base["date"]).dt.year
    for year in sorted(years.loc[test].unique()):
        year_mask = test & (years == year)
        chronology_rows.extend(
            _comparison_rows(
                round_base,
                chronology_models,
                year_mask,
                segment_type="year",
                segment=str(year),
            )
        )
    chronology_impact = pd.DataFrame(chronology_rows)
    chronology_impact.to_csv(report_dir / "elo_chronology_impact.csv", index=False)

    annual_rows = []
    principal = {
        "Elo V1": old_global,
        "Elo chronology": round_base["elo_expected_A"],
        "Elo optimized": optimized["elo_expected_A"],
        "Surface shrinkage": optimized_shrink,
        "Final raw": final_raw,
    }
    for year in sorted(years.unique()):
        year_mask = years == year
        annual_rows.extend(
            _comparison_rows(
                final_ratings,
                principal,
                year_mask,
                segment_type="year",
                segment=str(year),
            )
        )
    annual = pd.DataFrame(annual_rows)
    annual.to_csv(report_dir / "elo_annual_walk_forward.csv", index=False)

    calibration = calibration_table(
        final_ratings,
        {"Final raw": final_raw, "Final calibrated": calibrated},
        test,
    )
    calibration.to_csv(report_dir / "elo_phase3_calibration.csv", index=False)

    output = final_ratings.copy()
    output["surface_blended_expected_A"] = _variant_probability(
        final_ratings,
        "surface_shrinkage",
        c_value=best_c,
        scale=best_config.scale,
    )
    output["final_raw_expected_A"] = final_raw
    output["final_calibrated_expected_A"] = calibrated.where(test)
    ranks = phase2_features.set_index("match_id")[["rank_A", "rank_B", "rank_diff"]]
    output = output.join(ranks, on="match_id", validate="one_to_one")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".parquet.tmp")
    output.to_parquet(temporary, index=False)
    temporary.replace(output_path)

    selected = {
        "k_factor": best_config.k_factor,
        "scale": best_config.scale,
        "surface_c": best_c,
        "raw_variant": raw_variant,
        "inactivity": (
            None
            if selected_inactivity is None
            else {
                "threshold_days": selected_inactivity.threshold_days,
                "strength": selected_inactivity.strength,
            }
        ),
        "platt_intercept": intercept,
        "platt_slope": slope,
    }
    (report_dir / "elo_phase3_selected.json").write_text(
        json.dumps(selected, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "path": str(output_path.relative_to(PROJECT_ROOT)),
        "rows": len(output),
        "columns": list(output.columns),
        "date_min": output["date"].min().date().isoformat(),
        "date_max": output["date"].max().date().isoformat(),
        "unique_match_ids": int(output["match_id"].nunique()),
        "selected": selected,
        "calibration_scope": "final_calibrated_expected_A is populated only for 2023+ test rows",
    }
    output_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "selected": selected,
        "test_comparison": [
            {key: _json_safe(value) for key, value in row.items()}
            for row in comparison.to_dict("records")
        ],
        "atp_overlap": [
            {key: _json_safe(value) for key, value in row.items()}
            for row in atp_overlap.to_dict("records")
        ],
    }
    (report_dir / "phase3_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_phase3_report(
        report_dir,
        audit=audit,
        selected=selected,
        comparison=comparison,
        atp_overlap=atp_overlap,
        segmented=segmented,
        chronology_impact=chronology_impact,
        raw_comparison=raw_comparison,
        inactivity_trials=inactivity_trials,
        calibration=calibration,
    )
    return {
        "audit": audit,
        "selected": selected,
        "comparison": comparison,
        "atp_overlap": atp_overlap,
        "segmented": segmented,
        "chronology_impact": chronology_impact,
        "annual": annual,
        "calibration": calibration,
        "global_trials": global_trials,
        "surface_trials": surface_trials,
        "inactivity_trials": inactivity_trials,
        "raw_comparison": raw_comparison,
        "validation_mask": validation,
        "test_mask": test,
    }


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Tune and calibrate Phase 3 Elo ratings")
    command.add_argument(
        "--timeline", type=Path, default=PROCESSED_DIR / "analytics" / "match_timeline.parquet"
    )
    command.add_argument(
        "--phase2-features", type=Path, default=PROCESSED_DIR / "match_features_v1.parquet"
    )
    command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    command.add_argument("--report-dir", type=Path, default=REPORTS_DIR)
    command.add_argument("--output", type=Path, default=PROCESSED_DIR / "match_ratings_v2.parquet")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results = run_phase3(
        pd.read_parquet(args.timeline),
        pd.read_parquet(args.phase2_features),
        search=Phase3Config.from_toml(args.config),
        config_path=args.config,
        report_dir=args.report_dir,
        output_path=args.output,
    )
    print(results["comparison"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
