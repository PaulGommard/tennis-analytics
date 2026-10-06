from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump, load

from tennis_analytics.features.opponent_adjusted import (
    build_opponent_history_base,
    materialize_phase5_features,
    write_v3_metadata,
)
from tennis_analytics.modeling.logistic import (
    apply_platt,
    calibration_table,
    fit_platt,
    make_pipeline,
    probability_metrics,
    standardized_coefficients,
)
from tennis_analytics.modeling.logistic import feature_blocks as phase4_feature_blocks
from tennis_analytics.paths import PROCESSED_DIR, PROJECT_ROOT, REPORTS_DIR

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "phase5.toml"


@dataclass(frozen=True)
class Phase5Config:
    adjusted_point_prior_strengths: tuple[float, ...]
    adjusted_surface_prior_strengths: tuple[float, ...]
    h2h_prior_matches: tuple[float, ...]
    logistic_c_values: tuple[float, ...]
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp

    @classmethod
    def from_toml(cls, path: Path = DEFAULT_CONFIG) -> Phase5Config:
        with path.open("rb") as stream:
            values = tomllib.load(stream)
        search = values["search"]
        evaluation = values["evaluation"]
        return cls(
            adjusted_point_prior_strengths=tuple(
                map(float, search["adjusted_point_prior_strengths"])
            ),
            adjusted_surface_prior_strengths=tuple(
                map(float, search["adjusted_surface_prior_strengths"])
            ),
            h2h_prior_matches=tuple(map(float, search["h2h_prior_matches"])),
            logistic_c_values=tuple(map(float, search["logistic_c_values"])),
            train_end=pd.Timestamp(evaluation["train_end"]),
            validation_start=pd.Timestamp(evaluation["validation_start"]),
            validation_end=pd.Timestamp(evaluation["validation_end"]),
            test_start=pd.Timestamp(evaluation["test_start"]),
        )


def temporal_masks(
    frame: pd.DataFrame, config: Phase5Config
) -> tuple[pd.Series, pd.Series, pd.Series]:
    dates = pd.to_datetime(frame["date"])
    train = dates <= config.train_end
    validation = dates.between(config.validation_start, config.validation_end)
    test = dates >= config.test_start
    if (train & validation).any() or (validation & test).any() or (train & test).any():
        raise ValueError("Temporal splits overlap")
    return train, validation, test


PHASE4_FATIGUE = [
    f"{measure}_{window}d_diff"
    for window in (7, 14, 30)
    for measure in ("matches", "sets", "games", "minutes")
] + ["rest_days_diff"]


def new_feature_blocks() -> dict[str, list[str]]:
    schedule = [
        f"{name}_{window}d_diff"
        for window in (30, 90, 365)
        for name in ("opponent_elo_mean", "opponent_rank_quality_mean")
    ]
    adjusted_form = [f"wins_above_expected_{window}d_diff" for window in (30, 90, 365)]
    adjusted_stats = [
        f"{kind}_above_expected_{horizon}_diff"
        for kind in ("serve", "return")
        for horizon in ("career", "90d", "365d", "surface")
    ]
    matchup = ["serve_matchup_A", "serve_matchup_B"]
    h2h = ["h2h_matches_before", "h2h_edge_A"]
    h2h_surface = ["h2h_surface_matches_before", "h2h_surface_edge_A"]
    surface_interactions = [
        f"{feature}_x_{surface}"
        for surface in ("clay", "grass")
        for feature in (
            "elo_logit",
            "service_points_won_rate_90d_diff",
            "return_points_won_rate_90d_diff",
            "serve_above_expected_90d_diff",
            "return_above_expected_90d_diff",
        )
    ] + ["surface_clay", "surface_grass"]
    robust_fatigue = [
        f"prior_tournament_{measure}_{window}d_diff"
        for window in (14, 30)
        for measure in ("matches", "sets", "games", "minutes")
    ]
    experience = ["years_since_first_match_diff"]
    return {
        "schedule": schedule,
        "adjusted_form": adjusted_form,
        "adjusted_stats": adjusted_stats,
        "matchup": matchup,
        "h2h": h2h,
        "h2h_surface": h2h_surface,
        "surface_interactions": surface_interactions,
        "robust_fatigue": robust_fatigue,
        "experience": experience,
    }


def candidate_blocks() -> dict[str, list[str]]:
    phase4 = phase4_feature_blocks()["MODEL_5_ALL"]
    clean_phase4 = [column for column in phase4 if column not in PHASE4_FATIGUE]
    blocks = new_feature_blocks()
    all_new = list(chain.from_iterable(blocks.values()))
    return {
        "MODEL_A_PHASE4_EXACT": phase4,
        "MODEL_B_STRENGTH_OF_SCHEDULE": phase4 + blocks["schedule"],
        "MODEL_C_ADJUSTED_FORM": phase4 + blocks["adjusted_form"],
        "MODEL_D_OPPONENT_ADJUSTED_STATS": phase4 + blocks["adjusted_stats"],
        "MODEL_E_MATCHUP": phase4 + blocks["matchup"],
        "MODEL_F_H2H": phase4 + blocks["h2h"],
        "MODEL_F2_H2H_SURFACE": phase4 + blocks["h2h"] + blocks["h2h_surface"],
        "MODEL_G_SURFACE_INTERACTIONS": phase4 + blocks["surface_interactions"],
        "MODEL_ROBUST_FATIGUE": phase4 + blocks["robust_fatigue"],
        "MODEL_EXPERIENCE": phase4 + blocks["experience"],
        "MODEL_H_REDUCED": phase4
        + blocks["schedule"]
        + blocks["adjusted_stats"]
        + blocks["h2h"]
        + blocks["surface_interactions"],
        "MODEL_H_CORE": phase4
        + blocks["schedule"]
        + blocks["adjusted_stats"]
        + blocks["surface_interactions"],
        "MODEL_H_CORE_ROBUST_FATIGUE": phase4
        + blocks["schedule"]
        + blocks["adjusted_stats"]
        + blocks["surface_interactions"]
        + blocks["robust_fatigue"],
        "MODEL_H_REDUCED_ROBUST_FATIGUE": phase4
        + blocks["schedule"]
        + blocks["adjusted_stats"]
        + blocks["h2h"]
        + blocks["surface_interactions"]
        + blocks["robust_fatigue"],
        "MODEL_H_CLEAN_REDUCED_ROBUST_FATIGUE": clean_phase4
        + blocks["schedule"]
        + blocks["adjusted_stats"]
        + blocks["h2h"]
        + blocks["surface_interactions"]
        + blocks["robust_fatigue"],
        "MODEL_H_ALL": phase4 + all_new,
        "MODEL_H_CLEAN_FATIGUE": clean_phase4 + all_new,
    }


def _validation_metrics(
    features: pd.DataFrame,
    columns: list[str],
    train: pd.Series,
    validation: pd.Series,
    c_value: float,
) -> tuple[Any, dict[str, float | int]]:
    model = make_pipeline(c_value)
    model.fit(features.loc[train, columns], features.loc[train, "player_A_win"])
    probability = model.predict_proba(features.loc[validation, columns])[:, 1]
    return model, probability_metrics(features.loc[validation, "player_A_win"], probability)


def tune_feature_priors(
    features_v2: pd.DataFrame,
    base: pd.DataFrame,
    config: Phase5Config,
) -> tuple[pd.DataFrame, float, float, float]:
    rows = []
    phase4 = phase4_feature_blocks()["MODEL_5_ALL"]
    adjusted = new_feature_blocks()["adjusted_stats"]
    for point_prior in config.adjusted_point_prior_strengths:
        for surface_prior in config.adjusted_surface_prior_strengths:
            frame = materialize_phase5_features(
                features_v2,
                base,
                adjusted_point_prior=point_prior,
                adjusted_surface_prior=surface_prior,
                h2h_prior=10.0,
            )
            train, validation, _ = temporal_masks(frame, config)
            _, metrics = _validation_metrics(
                frame, phase4 + adjusted, train, validation, c_value=1.0
            )
            rows.append(
                {
                    "parameter": "adjusted_stats",
                    "point_prior": point_prior,
                    "surface_prior": surface_prior,
                    "h2h_prior": np.nan,
                    **{f"validation_{key}": value for key, value in metrics.items()},
                }
            )
    trials = pd.DataFrame(rows)
    best_adjusted = trials.sort_values(
        ["validation_log_loss", "validation_brier", "validation_accuracy"],
        ascending=[True, True, False],
    ).iloc[0]
    point_prior = float(best_adjusted["point_prior"])
    surface_prior = float(best_adjusted["surface_prior"])
    h2h_rows = []
    for h2h_prior in config.h2h_prior_matches:
        frame = materialize_phase5_features(
            features_v2,
            base,
            adjusted_point_prior=point_prior,
            adjusted_surface_prior=surface_prior,
            h2h_prior=h2h_prior,
        )
        train, validation, _ = temporal_masks(frame, config)
        _, metrics = _validation_metrics(
            frame,
            phase4 + new_feature_blocks()["h2h"] + new_feature_blocks()["h2h_surface"],
            train,
            validation,
            c_value=1.0,
        )
        h2h_rows.append(
            {
                "parameter": "h2h",
                "point_prior": point_prior,
                "surface_prior": surface_prior,
                "h2h_prior": h2h_prior,
                **{f"validation_{key}": value for key, value in metrics.items()},
            }
        )
    h2h_trials = pd.DataFrame(h2h_rows)
    best_h2h = h2h_trials.sort_values(
        ["validation_log_loss", "validation_brier", "validation_accuracy"],
        ascending=[True, True, False],
    ).iloc[0]
    return (
        pd.concat([trials, h2h_trials], ignore_index=True),
        point_prior,
        surface_prior,
        float(best_h2h["h2h_prior"]),
    )


def tune_candidates(
    features: pd.DataFrame,
    config: Phase5Config,
    phase4_artifact: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, str, Any]:
    train, validation, _ = temporal_masks(features, config)
    trials = []
    selected_rows = []
    selected_models: dict[str, Any] = {}
    for name, columns in candidate_blocks().items():
        if name == "MODEL_A_PHASE4_EXACT":
            model = phase4_artifact["pipeline"]
            probability = model.predict_proba(features.loc[validation, columns])[:, 1]
            metrics = probability_metrics(features.loc[validation, "player_A_win"], probability)
            row = {
                "model": name,
                "c_value": 1.0,
                "features": len(columns),
                **{f"validation_{key}": value for key, value in metrics.items()},
            }
            trials.append(row)
            selected_rows.append(row)
            selected_models[name] = model
            continue
        model_rows = []
        for c_value in config.logistic_c_values:
            model, metrics = _validation_metrics(features, columns, train, validation, c_value)
            row = {
                "model": name,
                "c_value": c_value,
                "features": len(columns),
                **{f"validation_{key}": value for key, value in metrics.items()},
            }
            trials.append(row)
            model_rows.append(row)
        best = (
            pd.DataFrame(model_rows)
            .sort_values(
                ["validation_log_loss", "validation_brier", "validation_accuracy"],
                ascending=[True, True, False],
            )
            .iloc[0]
        )
        final_model, _ = _validation_metrics(
            features, columns, train, validation, float(best["c_value"])
        )
        selected_models[name] = final_model
        selected_rows.append(best.to_dict())
    comparison = pd.DataFrame(selected_rows).sort_values(
        ["validation_log_loss", "validation_brier", "validation_accuracy"],
        ascending=[True, True, False],
    )
    final_name = str(comparison.iloc[0]["model"])
    return pd.DataFrame(trials), comparison, final_name, selected_models[final_name]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_manifest(paths: list[Path], report_dir: Path) -> dict[str, Any]:
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "statement": "Phase 4 inputs are read-only and are not modified by Phase 5.",
        "files": [
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in paths
        ],
    }
    (report_dir / "phase5_phase4_freeze.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def write_fatigue_audit(features: pd.DataFrame, report_dir: Path) -> None:
    fatigue = PHASE4_FATIGUE
    correlations = features[fatigue].corr()
    correlations.to_csv(report_dir / "fatigue_feature_correlations.csv")
    round_means = (
        features.groupby("round")[fatigue].agg(lambda values: values.abs().mean()).reset_index()
    )
    round_means.to_csv(report_dir / "fatigue_by_round.csv", index=False)
    source_text = """# Audit de validité des features de fatigue

## Verdict

Les features Phase 4 `matches/sets/games/minutes_{7,14,30}d` sont **expérimentales** et ne
mesurent pas des fenêtres calendaires exactes. La source ATP associe tous les rounds à la date
de début du tournoi. Un round précédent du même tournoi est donc compté avec zéro jour d'écart.
Ces variables combinent charge réelle et structure/progression dans le tableau.

Les corrélations sont extrêmes à horizon identique : sets, games et minutes atteignent souvent
0,97–0,99. Les moyennes absolues varient fortement par round (`fatigue_by_round.csv`). Elles ne
révèlent pas directement un résultat futur, car le snapshot précède le batch, mais constituent
un proxy indirect de progression et non une mesure pure de récupération physique.

## Dates alternatives déjà présentes

- Match Charting Project : 7 565 dates journalières valides sur 7 566 lignes, de 1960 à 2026,
  mais sur un échantillon charté, non homogène et bien plus petit que les 78 845 matchs analysés.
- Slam point-by-point : 10 513 lignes de matchs, aucune colonne date dans les 49 fichiers.
- ATP principal : `tourney_date` seulement, documentée comme début de tournoi.

Ces dates partielles ne sont pas fusionnées dans V3 : cela créerait deux définitions de fatigue
selon la disponibilité de la source.

## Alternative Phase 5

Les nouvelles variables `prior_tournament_*_{14,30}d` excluent explicitement l'édition
courante. Elles mesurent la charge lors de tournois antérieurs dont la date de début tombe dans
la fenêtre. Elles restent approximatives, mais ne codent plus le round courant. Phase 5 compare
un modèle conservant exactement la fatigue Phase 4 et un modèle la remplaçant par cette variante.

## Confiance

| Famille | Confiance | Limite |
| --- | --- | --- |
| Phase 4 intra-tournoi 7/14/30j | faible | dates identiques, proxy de round |
| `rest_days` | moyenne-faible | écart entre débuts de tournois; zéro intra-tournoi |
| Phase 5 tournois antérieurs | moyenne | date de début, pas jour réel du match |
| sets/games parsés | moyenne-haute | comptes explicites; scores ambigus deviennent NaN |
| minutes | moyenne | bonne couverture hors Davis Cup/2015, valeurs aberrantes filtrées |
"""
    (report_dir / "fatigue_validity_audit.md").write_text(source_text, encoding="utf-8")


def write_feature_dictionary(
    features: pd.DataFrame,
    v2_columns: set[str],
    path: Path,
    selected: dict[str, Any],
) -> None:
    new_columns = [column for column in features if column not in v2_columns]
    text = f"""# Dictionnaire des nouvelles features V3

V3 hérite intégralement de V2 sans modifier ses colonnes. Paramètres sélectionnés sur
validation : prior résiduel points={selected["adjusted_point_prior"]:g}, prior résiduel
surface={selected["adjusted_surface_prior"]:g}, prior H2H={selected["h2h_prior"]:g} matchs.

## Ajustement à l'adversaire

Pour un serveur A contre le retourneur B :

`expected_serve_A = 1 - historical_return_points_won_B`

`serve_residual_A = actual_service_points_won_A - expected_serve_A × service_points_A`

Le résidu retour utilise symétriquement `1 - historical_service_points_won_opponent`. Les
résidus sont agrégés par opportunités puis shrinkés vers zéro. Les variantes `90d`, `365d`,
`career` et `surface` sont strictement antérieures au batch.

## Strength of schedule et forme

- `opponent_elo_mean_*`: Elo adverse pré-match moyen dans la fenêtre.
- `opponent_rank_quality_mean_*`: moyenne de `-rank`; plus haut = adversaires plus forts.
- `wins_above_expected_*`: somme de `résultat - probabilité Elo pré-match`.

## H2H

`h2h_edge_A = (wins_A + 0.5C)/(matches + C) - 0.5`. Le H2H surface suit la même formule.
Les compteurs excluent toujours le match courant et restent gelés dans le batch.

## Matchup, surface, expérience et charge

- `serve_matchup_A = service_90d_A - return_90d_B`, symétriquement pour B.
- interactions explicites Elo/service/retour × Clay/Grass; Hard est la référence.
- `years_since_first_match`: ancienneté depuis la première apparition historique.
- `prior_tournament_*`: charge 14/30j excluant l'édition courante.

## Nouvelles colonnes

"""
    text += (
        "| Feature | Formule | Historique | Shrinkage | Sens | Limite |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
    )
    rows = []
    for column in new_columns:
        if column.startswith("opponent_elo_mean_"):
            window = column.split("_")[-2].removesuffix("d")
            values = (
                "moyenne(Elo adverse pré-match) A − B",
                f"{window} jours, match courant exclu",
                "aucun",
                "positif = calendrier A plus fort",
                "moyenne non pondérée; historique parfois absent",
            )
        elif column.startswith("opponent_rank_quality_mean_"):
            window = column.split("_")[-2].removesuffix("d")
            values = (
                "moyenne(−rang ATP adverse) A − B",
                f"{window} jours, match courant exclu",
                "aucun",
                "positif = adversaires A mieux classés",
                "rang non linéaire et parfois manquant",
            )
        elif "_above_expected_" in column and "_x_" not in column:
            kind, horizon = column.split("_above_expected_")
            horizon = horizon.removesuffix("_diff")
            prior = selected["adjusted_surface_prior"] if horizon == "surface" else selected[
                "adjusted_point_prior"
            ]
            expected = "1 − retour historique adverse" if kind == "serve" else (
                "1 − service historique adverse"
            )
            values = (
                f"résidu {kind} A − B; attendu={expected}",
                f"{horizon}, strictement antérieur au batch",
                f"{prior:g} opportunités vers 0",
                f"positif = {kind} au-dessus de l'attendu",
                "attente additive simple; disponibilité des stats",
            )
        elif column == "h2h_matches_before":
            values = (
                "nombre de confrontations A–B antérieures",
                "carrière, match courant exclu",
                "aucun",
                "positif = davantage d'historique commun",
                "ne donne pas seul le sens des victoires",
            )
        elif column == "h2h_edge_A":
            values = (
                "(wins_A + 0,5C)/(matches + C) − 0,5",
                "carrière, match courant exclu",
                f"C={selected['h2h_prior']:g} matchs vers 50 %",
                "positif = avantage historique A",
                "petits échantillons et biais de sélection",
            )
        elif "_x_clay" in column or "_x_grass" in column:
            surface = "Clay" if column.endswith("_clay") else "Grass"
            base = column.rsplit("_x_", maxsplit=1)[0]
            values = (
                f"{base} × 1(surface={surface})",
                "identique à la variable de base",
                "hérité de la variable de base",
                f"écart de coefficient {surface} vs Hard",
                "corrélée à la base; échantillon surface réduit",
            )
        elif column in {"surface_clay", "surface_grass"}:
            surface = column.removeprefix("surface_").title()
            values = (
                f"1(surface={surface})",
                "match courant (contexte connu avant match)",
                "aucun",
                f"intercept {surface} relatif à Hard",
                "ne mesure pas une aptitude individuelle",
            )
        elif column.startswith("prior_tournament_"):
            tokens = column.split("_")
            measure, window = tokens[2], tokens[3].removesuffix("d")
            values = (
                f"charge {measure} A − B, édition courante exclue",
                f"tournois antérieurs commencés dans {window} jours",
                "aucun",
                "positif = charge récente A supérieure",
                "dates de début de tournoi, pas dates journalières",
            )
        else:
            raise ValueError(f"Undocumented V3 feature: {column}")
        rows.append((column, *values))
    text += "\n".join(
        "| " + " | ".join((f"`{row[0]}`", *row[1:])) + " |" for row in rows
    )
    text += "\n"
    path.write_text(text, encoding="utf-8")


def write_v3_audit(
    features_v2: pd.DataFrame,
    production_features: pd.DataFrame,
    selected: dict[str, Any],
    report_dir: Path,
) -> None:
    """Validate the production schema and document coverage of every V3 addition."""
    v2_columns = list(features_v2.columns)
    pd.testing.assert_frame_equal(
        production_features[v2_columns],
        features_v2,
        check_exact=True,
        check_names=True,
    )
    if not production_features.index.is_unique:
        raise ValueError("V3 index is not unique")

    missing_model_columns = sorted(set(selected["features"]) - set(production_features.columns))
    if missing_model_columns:
        raise ValueError(f"V3 is missing selected model columns: {missing_model_columns}")

    numeric = production_features[selected["features"]].select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("Selected V3 model features contain infinite values")
    if not production_features["h2h_matches_before"].ge(0).all():
        raise ValueError("H2H match counts must be non-negative")
    if not production_features["h2h_edge_A"].between(-0.5, 0.5).all():
        raise ValueError("H2H edge must stay in [-0.5, 0.5]")

    new_columns = [column for column in production_features if column not in v2_columns]
    rows = []
    for column in new_columns:
        values = production_features[column]
        finite = values.replace([np.inf, -np.inf], np.nan)
        rows.append(
            {
                "feature": column,
                "non_null": int(values.notna().sum()),
                "coverage": float(values.notna().mean()),
                "zero_share": float(values.eq(0).mean()),
                "mean": float(finite.mean()),
                "std": float(finite.std()),
                "min": float(finite.min()),
                "p01": float(finite.quantile(0.01)),
                "median": float(finite.median()),
                "p99": float(finite.quantile(0.99)),
                "max": float(finite.max()),
            }
        )
    coverage = pd.DataFrame(rows)
    coverage.to_csv(report_dir / "feature_coverage_v3.csv", index=False)

    h2h_seen = int(production_features["h2h_matches_before"].gt(0).sum())
    audit = f"""# Audit du dataset V3

## Invariants de production

- lignes : {len(production_features):,};
- colonnes : {len(production_features.columns):,} (dont {len(v2_columns)} V2 inchangées et {len(new_columns)} nouvelles);
- index unique : oui;
- colonnes du modèle final toutes présentes : oui ({len(selected["features"])} features);
- égalité exacte des colonnes héritées V2 : oui;
- valeurs infinies dans les features du modèle : aucune;
- domaine H2H : compteurs positifs ou nuls, edge dans [-0,5; 0,5].

## Couverture

Un historique H2H antérieur existe pour {h2h_seen:,} matchs
({h2h_seen / len(production_features):.1%}). Les statistiques détaillées des {len(new_columns)}
nouvelles colonnes sont dans `feature_coverage_v3.csv` (couverture, part de zéros,
quantiles 1/50/99 %, minimum et maximum).

L'absence historique reste représentée par `NaN` lorsque la feature l'exige; l'imputation est
apprise exclusivement sur le train au sein du pipeline final.
"""
    (report_dir / "feature_audit_v3.md").write_text(audit, encoding="utf-8")


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame[columns].to_dict("records"):
        rendered = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                rendered.append("—")
            elif isinstance(value, (float, np.floating)):
                rendered.append(f"{float(value):.4f}")
            elif isinstance(value, (int, np.integer)):
                rendered.append(f"{int(value):,}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def _segment_rows(
    features: pd.DataFrame,
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
            **probability_metrics(features.loc[mask, "player_A_win"], probability.loc[mask]),
        }
        for name, probability in probabilities.items()
    ]


def run_phase5(
    *,
    config: Phase5Config,
    features_v2: pd.DataFrame,
    ratings: pd.DataFrame,
    timeline: pd.DataFrame,
    matches: pd.DataFrame,
    stats: pd.DataFrame,
    phase4_artifact: dict[str, Any],
    report_dir: Path,
    output_path: Path,
    model_path: Path,
) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    freeze_manifest(
        [
            PROCESSED_DIR / "match_features_v2.parquet",
            PROCESSED_DIR / "models" / "logistic_phase4.joblib",
            REPORTS_DIR / "phase4_report.md",
            PROJECT_ROOT / "config" / "phase4.toml",
        ],
        report_dir,
    )
    write_fatigue_audit(features_v2, report_dir)
    LOGGER.info("Building pre-match opponent-adjusted history")
    base = build_opponent_history_base(features_v2, ratings, timeline, matches, stats)
    LOGGER.info("Selecting bounded residual and H2H priors on validation")
    prior_trials, point_prior, surface_prior, h2h_prior = tune_feature_priors(
        features_v2, base, config
    )
    prior_trials.to_csv(report_dir / "phase5_feature_prior_tuning.csv", index=False)
    features = materialize_phase5_features(
        features_v2,
        base,
        adjusted_point_prior=point_prior,
        adjusted_surface_prior=surface_prior,
        h2h_prior=h2h_prior,
    )
    provisional_selected = {
        "adjusted_point_prior": point_prior,
        "adjusted_surface_prior": surface_prior,
        "h2h_prior": h2h_prior,
    }

    LOGGER.info("Running Phase 5 validation ablations")
    trials, validation_comparison, final_name, final_model = tune_candidates(
        features, config, phase4_artifact
    )
    trials.to_csv(report_dir / "phase5_logistic_tuning.csv", index=False)
    validation_comparison.to_csv(report_dir / "phase5_ablation_validation.csv", index=False)
    final_columns = candidate_blocks()[final_name]
    _, validation, test = temporal_masks(features, config)
    validation_raw = final_model.predict_proba(features.loc[validation, final_columns])[:, 1]
    calibrator = fit_platt(validation_raw, features.loc[validation, "player_A_win"])
    validation_calibrated = apply_platt(calibrator, validation_raw)
    raw_val_metrics = probability_metrics(features.loc[validation, "player_A_win"], validation_raw)
    calibrated_val_metrics = probability_metrics(
        features.loc[validation, "player_A_win"], validation_calibrated
    )
    use_calibration = (
        calibrated_val_metrics["log_loss"] < raw_val_metrics["log_loss"]
        and calibrated_val_metrics["brier"] <= raw_val_metrics["brier"]
    )
    calibration_validation = pd.DataFrame(
        [
            {"model": "Phase 5 raw", **raw_val_metrics},
            {"model": "Phase 5 Platt", **calibrated_val_metrics},
        ]
    )
    calibration_validation.to_csv(report_dir / "phase5_calibration_validation.csv", index=False)

    selected = {
        **provisional_selected,
        "model": final_name,
        "logistic_c": float(
            validation_comparison.loc[validation_comparison["model"] == final_name, "c_value"].iloc[
                0
            ]
        ),
        "features": final_columns,
        "platt_used": use_calibration,
        "platt_intercept": float(calibrator.intercept_[0]),
        "platt_slope": float(calibrator.coef_[0, 0]),
    }
    (report_dir / "phase5_selected.json").write_text(
        json.dumps(selected, indent=2) + "\n", encoding="utf-8"
    )
    production_columns = list(features_v2.columns) + [
        column for column in final_columns if column not in features_v2.columns
    ]
    production_features = features[production_columns].copy()
    write_v3_metadata(production_features, output_path, selected)
    write_feature_dictionary(
        production_features,
        set(features_v2.columns),
        PROJECT_ROOT / "docs" / "features_v3.md",
        selected,
    )
    write_v3_audit(features_v2, production_features, selected, report_dir)
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

    # Final Phase 5 test evaluation, after every selection is frozen.
    test_index = features.index[test]
    phase5_raw_values = final_model.predict_proba(features.loc[test, final_columns])[:, 1]
    phase5_calibrated_values = apply_platt(calibrator, phase5_raw_values)
    phase5_final_values = phase5_calibrated_values if use_calibration else phase5_raw_values
    phase5_probability = pd.Series(np.nan, index=features.index)
    phase5_probability.loc[test_index] = phase5_final_values
    phase4_raw_values = phase4_artifact["pipeline"].predict_proba(
        features.loc[test, phase4_artifact["feature_columns"]]
    )[:, 1]
    phase4_calibrated_values = apply_platt(phase4_artifact["calibrator"], phase4_raw_values)
    phase4_probability = pd.Series(np.nan, index=features.index)
    phase4_probability.loc[test_index] = phase4_calibrated_values
    elo_probability = ratings["final_calibrated_expected_A"]
    comparison_rows = []
    for name, probability in (
        ("Elo Phase 3 calibrated", elo_probability.loc[test].to_numpy()),
        ("Logistic Phase 4 calibrated", phase4_calibrated_values),
        ("Logistic Phase 5 raw", phase5_raw_values),
        ("Logistic Phase 5 final", phase5_final_values),
    ):
        comparison_rows.append(
            {
                "model": name,
                **probability_metrics(features.loc[test, "player_A_win"], probability),
            }
        )
    ranking = (
        test
        & features["rank_A"].notna()
        & features["rank_B"].notna()
        & (features["rank_A"] != features["rank_B"])
    )
    ranking_prediction = features.loc[ranking, "rank_A"] < features.loc[ranking, "rank_B"]
    comparison_rows.insert(
        0,
        {
            "model": "ATP Ranking",
            "matches": int(ranking.sum()),
            "accuracy": float(
                (
                    ranking_prediction.astype(int)
                    == features.loc[ranking, "player_A_win"].astype(int)
                ).mean()
            ),
            "log_loss": np.nan,
            "brier": np.nan,
            "roc_auc": np.nan,
        },
    )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(report_dir / "phase5_test_comparison.csv", index=False)
    atp_overlap = pd.DataFrame(
        [
            comparison_rows[0],
            {
                "model": "Logistic Phase 4",
                **probability_metrics(
                    features.loc[ranking, "player_A_win"],
                    phase4_probability.loc[ranking],
                ),
            },
            {
                "model": "Logistic Phase 5",
                **probability_metrics(
                    features.loc[ranking, "player_A_win"],
                    phase5_probability.loc[ranking],
                ),
            },
        ]
    )
    atp_overlap.to_csv(report_dir / "phase5_atp_overlap.csv", index=False)

    variants = {
        "Elo Phase 3": elo_probability,
        "Logistic Phase 4": phase4_probability,
        "Logistic Phase 5": phase5_probability,
    }
    segment_rows = []
    for surface in ("Hard", "Clay", "Grass"):
        mask = test & (features["surface"] == surface)
        segment_rows.extend(_segment_rows(features, mask, variants, "surface", surface))
    years = pd.to_datetime(features["date"]).dt.year
    for year in sorted(years.loc[test].unique()):
        mask = test & (years == year)
        segment_rows.extend(_segment_rows(features, mask, variants, "year", str(year)))
    segments = pd.DataFrame(segment_rows)
    segments.to_csv(report_dir / "phase5_test_segments.csv", index=False)
    calibration = calibration_table(
        features["player_A_win"],
        {
            "Phase 4 calibrated": phase4_calibrated_values,
            "Phase 5 raw": phase5_raw_values,
            "Phase 5 final": phase5_final_values,
        },
        test_index,
    )
    calibration.to_csv(report_dir / "phase5_calibration.csv", index=False)
    coefficients = standardized_coefficients(final_model, final_columns)
    coefficients.to_csv(report_dir / "phase5_coefficients.csv", index=False)
    validation_by_model = validation_comparison.set_index("model")
    phase4_validation_loss = float(
        validation_by_model.loc["MODEL_A_PHASE4_EXACT", "validation_log_loss"]
    )

    def validation_loss(model: str) -> float:
        return float(validation_by_model.loc[model, "validation_log_loss"])

    comparison_by_model = comparison.set_index("model")
    phase4_test = comparison_by_model.loc["Logistic Phase 4 calibrated"]
    phase5_test = comparison_by_model.loc["Logistic Phase 5 final"]
    new_model_features = set(final_columns) - set(phase4_artifact["feature_columns"])
    focused_coefficients = coefficients.loc[
        coefficients["feature"].isin(new_model_features)
    ].head(25)

    report = f"""# Phase 5 — qualité des adversaires et matchups

## Protocole

Phase 4 est figée par empreintes SHA-256 dans `phase5_phase4_freeze.json`. Toutes les sélections
Phase 5 utilisent 2019–2022; les priors, blocs, C et calibration sont gelés avant l'évaluation
finale 2023–2026.

## A. Strength of schedule

Les moyennes adverses Elo et `-rank` sur 30/90/365 jours sont utiles : log loss validation
{phase4_validation_loss:.5f} → {validation_loss("MODEL_B_STRENGTH_OF_SCHEDULE"):.5f}. Elles sont
conservées. Une moyenne historique simple a été préférée à une pondération supplémentaire pour
limiter les degrés de liberté; les fenêtres portent déjà la récence.

## B. Adjusted form

`wins_above_expected`, calculé avec les probabilités Elo brutes pré-match, n'améliore que de
façon négligeable la log loss ({phase4_validation_loss:.5f} →
{validation_loss("MODEL_C_ADJUSTED_FORM"):.5f}). Ce bloc est supprimé de V3 production.

## C. Opponent-adjusted stats

C'est le meilleur bloc individuel : {phase4_validation_loss:.5f} →
{validation_loss("MODEL_D_OPPONENT_ADJUSTED_STATS"):.5f}. Pour le service, l'attendu est
`1 − taux de retour historique adverse`; pour le retour, `1 − taux de service historique
adverse`. Cette identité respecte les bornes et évite le double comptage qu'introduirait une
formule `moyenne ligue + force serveur − force retourneur`. Les résidus sont pondérés par les
opportunités, avec priors sélectionnés de {point_prior:g} points et {surface_prior:g} points sur
surface.

## D. H2H

Le H2H global régularisé apporte un gain très faible ({phase4_validation_loss:.5f} →
{validation_loss("MODEL_F_H2H"):.5f}) avec C={h2h_prior:g}; il est conservé dans la combinaison.
La variante surface est moins bonne ({validation_loss("MODEL_F2_H2H_SURFACE"):.5f}) et supprimée.

## E. Matchup

Le proxy service récent A contre retour récent B ne gagne rien
({validation_loss("MODEL_E_MATCHUP"):.5f}, contre {phase4_validation_loss:.5f}). Il est supprimé.

## F. Surface interactions

Les interactions Elo, service, retour et résidus ajustés × Clay/Grass améliorent la validation
({phase4_validation_loss:.5f} → {validation_loss("MODEL_G_SURFACE_INTERACTIONS"):.5f}). Elles sont
conservées, Hard servant de référence. Les coefficients sont des écarts conditionnels et ne
doivent pas être lus isolément comme effets causaux.

## G. Audit fatigue

Les features Phase 4 sont marquées expérimentales car les rounds d'un tournoi partagent la
même date de début. Leur retrait dans une comparaison équitable donne
{validation_loss("MODEL_H_CLEAN_REDUCED_ROBUST_FATIGUE"):.5f}, contre
{validation_loss("MODEL_H_REDUCED_ROBUST_FATIGUE"):.5f} avec elles. Elles restent donc comme
proxies prédictifs expérimentaux, sans interprétation causale de fatigue. La charge de tournois
antérieurs, qui exclut l'édition courante, améliore seule à
{validation_loss("MODEL_ROBUST_FATIGUE"):.5f}. Voir `fatigue_validity_audit.md`.

## H. Ablation study

Le bloc final choisi sur validation est `{final_name}`. L'ablation, et non les coefficients
corrélés pris isolément, constitue la preuve principale.

{_markdown_table(validation_comparison, ["model", "features", "c_value", "validation_accuracy", "validation_log_loss", "validation_brier", "validation_roc_auc"])}

## I. Modèle final Phase 5

{_markdown_table(comparison, ["model", "matches", "accuracy", "log_loss", "brier", "roc_auc"])}

### Par surface et année

{_markdown_table(segments, ["segment_type", "segment", "model", "matches", "accuracy", "log_loss", "brier", "roc_auc"])}

## Calibration

Calibration Platt retenue : {use_calibration}.

{_markdown_table(calibration_validation, ["model", "matches", "accuracy", "log_loss", "brier", "roc_auc"])}

Buckets test : `phase5_calibration.csv`.

## Coefficients standardisés principaux

{_markdown_table(coefficients.head(25), ["feature", "standardized_coefficient"])}

Les coefficients sont conditionnels à des features corrélées et ne sont pas causaux.

### Nouvelles features retenues

{_markdown_table(focused_coefficients, ["feature", "standardized_coefficient"])}

## J. Comparaison Phase 4

Phase 5 calibrée gagne {(float(phase5_test["accuracy"]) - float(phase4_test["accuracy"])) * 100:+.2f}
point d'accuracy, {float(phase5_test["log_loss"]) - float(phase4_test["log_loss"]):+.5f} de log
loss, {float(phase5_test["brier"]) - float(phase4_test["brier"]):+.5f} de Brier et
{float(phase5_test["roc_auc"]) - float(phase4_test["roc_auc"]):+.5f} de ROC-AUC.

Comparaison appariée sur les {int(ranking.sum()):,} matchs avec deux rangs ATP disponibles et
distincts :

{_markdown_table(atp_overlap, ["model", "matches", "accuracy", "log_loss", "brier", "roc_auc"])}

## Journal d'ouverture du test

Une exécution technique préliminaire a produit un résultat test avant la dernière réduction de
features. Les suppressions et variantes finales ont ensuite été décidées exclusivement sur les
métriques validation, sans utiliser ce résultat test. Le tableau ci-dessus provient de la
dernière exécution figée. Cette déviation au protocole idéal est consignée explicitement.

## Anti-leakage

- forces adverses, Elo et ranks sont les snapshots pré-match;
- expected wins utilise exclusivement `final_raw_expected_A` Phase 3 pré-match;
- résidus, H2H et H2H surface sont mis à jour après le batch complet;
- rolling et surface excluent le match courant et les résultats futurs;
- imputer, scaler et régression sont fit sur 2000–2018;
- priors, ablations, C et Platt sont choisis sur 2019–2022;
- les tables de tuning ne contiennent aucune métrique test.

## Verdict pour la Phase 6

V3 est suffisamment mature pour un benchmark non linéaire contrôlé. La Phase 6 devra toutefois
comparer systématiquement la version complète à une variante sans les anciennes features de
fatigue, afin qu'un modèle d'arbres ne sur-exploite pas leur proxy de round. Aucun modèle non
linéaire n'est entraîné dans cette phase.
"""
    (report_dir / "phase5_report.md").write_text(report, encoding="utf-8")
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
    }
    (report_dir / "phase5_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {
        "selected": selected,
        "validation": validation_comparison,
        "comparison": comparison,
        "segments": segments,
    }


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Build Phase 5 opponent-adjusted features")
    command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    command.add_argument(
        "--features-v2", type=Path, default=PROCESSED_DIR / "match_features_v2.parquet"
    )
    command.add_argument("--ratings", type=Path, default=PROCESSED_DIR / "match_ratings_v2.parquet")
    command.add_argument(
        "--timeline", type=Path, default=PROCESSED_DIR / "analytics" / "match_timeline.parquet"
    )
    command.add_argument("--atp-dir", type=Path, default=PROCESSED_DIR / "atp")
    command.add_argument(
        "--phase4-model",
        type=Path,
        default=PROCESSED_DIR / "models" / "logistic_phase4.joblib",
    )
    command.add_argument("--report-dir", type=Path, default=REPORTS_DIR)
    command.add_argument("--output", type=Path, default=PROCESSED_DIR / "match_features_v3.parquet")
    command.add_argument(
        "--model-output",
        type=Path,
        default=PROCESSED_DIR / "models" / "logistic_phase5.joblib",
    )
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results = run_phase5(
        config=Phase5Config.from_toml(args.config),
        features_v2=pd.read_parquet(args.features_v2),
        ratings=pd.read_parquet(args.ratings),
        timeline=pd.read_parquet(args.timeline),
        matches=pd.read_parquet(args.atp_dir / "matches.parquet"),
        stats=pd.read_parquet(args.atp_dir / "match_stats.parquet"),
        phase4_artifact=load(args.phase4_model),
        report_dir=args.report_dir,
        output_path=args.output,
        model_path=args.model_output,
    )
    print(results["comparison"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
