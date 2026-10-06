from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from tennis_analytics.paths import PROCESSED_DIR, REPORTS_DIR
from tennis_analytics.tournament.replay import AUDIT_PATH, MODEL_DIR, OUTPUT_PATH, ROUND_ORDER

REPORT_PATH = REPORTS_DIR / "tournament_adapter_v1_report.md"


def _format(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "—"
        return f"{value:.6f}"
    return str(value)


def _table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_format(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def _comparison_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame[["model", "matches", "accuracy", "log_loss", "brier", "auc"]]


def _segment_delta(segments: pd.DataFrame, segment_type: str) -> pd.DataFrame:
    part = segments.loc[segments["segment_type"] == segment_type]
    base = part.loc[part["model"] == "PREMATCH_V1"].set_index("segment")
    adapter = part.loc[part["model"] == "TOURNAMENT_ADAPTER"].set_index("segment")
    result = pd.DataFrame(
        {
            "segment": base.index,
            "matches": base["matches"].astype(int),
            "base_log_loss": base["log_loss"],
            "adapter_log_loss": adapter.reindex(base.index)["log_loss"],
            "gain_log_loss": base["log_loss"] - adapter.reindex(base.index)["log_loss"],
            "base_brier": base["brier"],
            "adapter_brier": adapter.reindex(base.index)["brier"],
        }
    ).reset_index(drop=True)
    if segment_type == "round":
        result["order"] = result["segment"].map(ROUND_ORDER).fillna(999)
        result = result.sort_values("order").drop(columns="order")
    return result


def _historical_replay(predictions: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    candidates = predictions.loc[
        (predictions["level_group"] == "Grand Slam")
        & (pd.to_datetime(predictions["date"]).dt.year <= 2025)
    ]
    complete_ids = [
        edition_id
        for edition_id, edition in candidates.groupby("tournament_edition_id")
        if "F" in set(edition["round"])
    ]
    candidates = candidates.loc[candidates["tournament_edition_id"].isin(complete_ids)]
    chosen_id = str(
        candidates.sort_values("date").iloc[-1]["tournament_edition_id"]
    )
    edition = candidates.loc[candidates["tournament_edition_id"].astype(str) == chosen_id].copy()
    rows = []
    for round_name, group in edition.groupby("round", observed=True):
        rows.append(
            {
                "round": round_name,
                "matches": len(group),
                "observed_before": int(group["matches_observed_before_round"].iloc[0]),
                "base_log_loss": log_loss(
                    group["player_A_win"], group["p_base"], labels=[0, 1]
                ),
                "adapter_log_loss": log_loss(
                    group["player_A_win"], group["p_adjusted"], labels=[0, 1]
                ),
                "mean_abs_correction_pp": 100 * group["delta_probability"].abs().mean(),
            }
        )
    replay = pd.DataFrame(rows)
    replay["order"] = replay["round"].map(ROUND_ORDER).fillna(999)
    replay = replay.sort_values("order").drop(columns="order")
    replay.to_csv(REPORTS_DIR / "tournament_adapter_historical_replay_example.csv", index=False)
    name = str(edition.iloc[0]["tournament_name"])
    year = int(pd.to_datetime(edition.iloc[0]["date"]).year)
    return f"{name} {year} ({chosen_id})", replay


def write_tournament_report() -> Path:
    audit = json.loads(AUDIT_PATH.read_text())
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text())
    context = pd.read_parquet(OUTPUT_PATH)
    predictions = pd.read_parquet(
        PROCESSED_DIR / "tournament_adapter_v1_predictions_test.parquet"
    )
    benchmark = _comparison_table(REPORTS_DIR / "tournament_adapter_test_comparison.csv")
    ablation = pd.read_csv(REPORTS_DIR / "tournament_adapter_ablation_validation.csv")
    shrinkage = pd.read_csv(REPORTS_DIR / "tournament_adapter_shrinkage_validation.csv")
    regularization = pd.read_csv(
        REPORTS_DIR / "tournament_adapter_regularization_validation.csv"
    )
    cap = pd.read_csv(REPORTS_DIR / "tournament_adapter_cap_validation.csv")
    calibration = pd.read_csv(
        REPORTS_DIR / "tournament_adapter_calibration_validation.csv"
    )
    segments = pd.read_csv(REPORTS_DIR / "tournament_adapter_test_segments.csv")
    corrections = pd.read_csv(REPORTS_DIR / "tournament_adapter_correction_buckets.csv")
    coefficients = pd.read_csv(REPORTS_DIR / "tournament_adapter_coefficients.csv")
    replay_name, replay = _historical_replay(predictions)

    ablation_view = ablation[
        ["model", "features", "matches", "accuracy", "log_loss", "brier"]
    ].copy()
    base_validation = float(
        ablation.loc[ablation["model"] == "T0_PREMATCH_V1", "log_loss"].iloc[0]
    )
    ablation_view["gain_vs_T0"] = base_validation - ablation_view["log_loss"]
    round_delta = _segment_delta(segments, "round")
    observed_delta = _segment_delta(segments, "observed_matches")
    surface_delta = _segment_delta(segments, "surface")
    level_delta = _segment_delta(segments, "level")
    best_of_delta = _segment_delta(segments, "best_of")

    top_coefficients = coefficients.head(15).copy()
    clustered = metadata["clustered_test_gain"]
    mean_abs_delta = 100 * predictions["delta_probability"].abs().mean()
    max_abs_delta = 100 * predictions["delta_probability"].abs().max()
    selected_shrinkage = shrinkage.iloc[0]
    selected_regularization = regularization.iloc[0]
    recommendation = (
        "CANDIDAT PRODUCTION"
        if metadata["production_recommended"]
        else "NE PAS ACTIVER EN PRODUCTION — conserver PRE-MATCH V1"
    )

    report = f"""# Tournament Context Adapter V1 — rapport final

## Conclusion exécutive

**Recommandation : {recommendation}.**

L'adapter conservateur avec cap à 5 points améliore très légèrement la log loss agrégée
sur le test 2023–2026 (`0.609876 → 0.609692`), mais dégrade le Brier
(`0.211585 → 0.211623`) et surtout ne généralise pas de façon stable entre éditions : le
gain moyen par tournoi vaut `{clustered['mean_edition_log_loss_gain']:.6f}`, avec IC 95 %
`[{clustered['ci95_low']:.6f}, {clustered['ci95_high']:.6f}]`. Cet intervalle est entièrement
négatif. La V1 répond donc à la question scientifique : un signal existe dans certains
segments, mais il n'est pas assez stable pour dépasser PRE-MATCH V1 en production.

## A. Dataset

- `{audit['rows']:,}` matchs, `{audit['editions']:,}` éditions, {audit['years'][0]}–{audit['years'][1]} ;
- `{context['tournament_name_key'].nunique():,}` clés de tournois historiques normalisées ;
- couverture minutes `{audit['minutes_coverage']:.1%}`, stats détaillées `{audit['stat_coverage']:.1%}` ;
- `0` violation du gel avant round ;
- test final entièrement hors échantillon : 2023–2026, `{len(predictions):,}` matchs.

La chronologie est structurelle et tous les matchs d'un round utilisent le même snapshot.
PRE-MATCH V1 est figé ; son empreinte et celle des sources sont dans
`models/tournament_adapter_v1/prematch_v1_freeze.json`.

## B. Player tournament state

L'état contient charge, scores, durée avec couverture, taux bruts et résidus par rapport à
l'attendu. Le shrinkage joueur choisi en validation est `C={int(selected_shrinkage['player_prior'])}`.
Les plus gros coefficients standardisés sont descriptifs, pas causaux :

{_table(top_coefficients)}

## C. Tournament global state

Le bloc global seul (`T3`) n'améliore pas T0 en validation. Les conditions globales sont
orientées uniquement via des interactions de style ou de confiance, afin de ne jamais
favoriser arbitrairement A. Le shrinkage tournoi sélectionné est
`C={int(selected_shrinkage['tournament_prior'])}`. Le proxy de vitesse n'est pas une mesure
physique de court speed.

## D. Performance vs expected

`T2_PERFORMANCE_VS_EXPECTED` (`0.599387`) est meilleur que les taux bruts seuls
`T2B_RAW_TOURNAMENT_STATS` (`0.599535`) en log loss validation. Les combiner n'améliore pas
le bloc résiduel seul. Le signal « au-delà de l'attendu » est donc préférable, mais son gain
reste exploratoire.

## E. Fatigue

Le workload seul ne bat pas T0. Ajouter les minutes apporte un gain validation minuscule
dans le modèle combiné (`T6B 0.599103` contre `T6 0.599125`), avec seulement 42 % de
couverture. La variante avec minutes est conservée pour la recherche, jamais interprétée
comme une mesure complète de fatigue.

## F. Ablation walk-forward 2019–2022

Chaque année est prédite par un modèle entraîné uniquement sur les années précédentes ; une
édition entière reste dans un seul split.

{_table(ablation_view)}

La régularisation sélectionnée est `{selected_regularization['regularization']:.1f}`. Le
meilleur modèle validation est combiné avec minutes, mais son avantage ne se confirme pas
de façon stable sur le test final.

## G. Performance par round — test 2023–2026

Un gain positif signifie une log loss plus faible que PRE-MATCH V1.

{_table(round_delta)}

QF et finale sont favorables, SF défavorable. Cette alternance et les petits effectifs des
rounds avancés ne permettent pas de valider l'hypothèse d'un gain croissant régulier.

### Selon le nombre de matchs déjà observés

{_table(observed_delta)}

Le bucket `0–8` se dégrade ; les buckets plus informés s'améliorent, mais cela ne suffit pas
à rendre le gain stable par tournoi.

## H. Surface, niveau et format

### Surface

{_table(surface_delta)}

### Niveau

{_table(level_delta)}

### BO3 / BO5

{_table(best_of_delta)}

Le gazon et les Grands Chelems sont favorables sur ce test, la terre battue est défavorable.
Ces découpages sont des diagnostics, pas de nouvelles sélections de modèle.

## I. Corrections et erreurs

- correction absolue moyenne : `{mean_abs_delta:.3f}` point de probabilité ;
- correction absolue maximale : `{max_abs_delta:.3f}` points ;
- cap sélectionné sur validation : `{metadata['selected_cap']}`.

Résultats des caps en validation :

{_table(cap)}

{_table(corrections)}

Les 200 plus fortes corrections, avec résultat réel, sont conservées dans
`reports/tournament_adapter_largest_corrections.csv`.

## J. Calibration

{_table(calibration)}

La calibration température symétrique n'est pas retenue (`uses_temperature_calibration =
{metadata['uses_temperature_calibration']}`). Le contrôle PRE-MATCH recalibré n'est pas
retenu non plus ; l'amélioration/dégradation observée vient donc bien de l'adapter et du cap,
pas d'un simple intercept de recalibration.

### Comparaison finale

{_table(benchmark)}

L'offset respecte exactement `logit(p_final) = logit(p_base) + correction_context`. La
régression logistique complète, autorisée à réapprendre le socle, n'est pas meilleure.

## K. Replay et simulation

Exemple de chemin historique complet : **{replay_name}**.

{_table(replay)}

Ce tableau rejoue le chemin réellement observé round par round. L'API
`simulate_tournament` permet en plus des Monte Carlo reproductibles : mode pré-tournoi avec
PRE-MATCH V1 seul, ou mode live avec snapshot réel gelé. Elle ne fabrique jamais les stats
des tours futurs. Un ancien bracket contrefactuel complet demanderait des features
point-in-time pour des confrontations jamais jouées ; V1 refuse de les inventer.

## L. Recommandation finale

**Conserver PRE-MATCH V1 seul en production.** Garder Tournament Adapter V1 comme artefact
de recherche reproductible et comme API expérimentale désactivée. Les critères de succès ne
sont pas tous satisfaits : log loss globale légèrement meilleure, mais Brier moins bon et IC
clusterisé par édition négatif. Une V2 ne devrait être tentée qu'avec davantage de couverture
statistique, une validation pré-enregistrée et des interactions de conditions mieux motivées.

La prochaine phase logique, sans activer cet adapter, est **PRODUCT / WEB PLATFORM V1** en
s'appuyant sur les moteurs déjà validés. Aucun travail web n'est commencé ici.
"""
    REPORT_PATH.write_text(report)
    return REPORT_PATH
