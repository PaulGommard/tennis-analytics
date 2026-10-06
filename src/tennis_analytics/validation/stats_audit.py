from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from tennis_analytics.features.score import parse_score_workload

STAT_COLUMNS = (
    "aces",
    "double_faults",
    "service_points",
    "first_serve_in",
    "first_serve_won",
    "second_serve_won",
    "service_games",
    "break_points_saved",
    "break_points_faced",
)


def build_stats_audit(
    matches: pd.DataFrame,
    stats: pd.DataFrame,
    editions: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    context = matches[["match_id", "tournament_edition_id", "date", "minutes", "score"]].merge(
        editions[["tournament_edition_id", "surface", "level"]],
        on="tournament_edition_id",
        how="left",
        validate="many_to_one",
    )
    context["year"] = pd.to_datetime(context["date"]).dt.year
    counts = stats.groupby("match_id")[list(STAT_COLUMNS)].count().eq(2)
    context = context.join(counts.add_prefix("complete_"), on="match_id")
    context["complete_service_bundle"] = context[
        [f"complete_{column}" for column in STAT_COLUMNS]
    ].all(axis=1)
    context["duration_available"] = (context["minutes"].gt(0) & context["minutes"].le(360)).fillna(
        False
    )
    context["score_status"] = context["score"].map(lambda value: parse_score_workload(value).status)

    def coverage(columns: list[str]) -> pd.DataFrame:
        return (
            context.groupby(columns, dropna=False)
            .agg(
                matches=("match_id", "size"),
                service_coverage=("complete_service_bundle", "mean"),
                duration_coverage=("duration_available", "mean"),
            )
            .reset_index()
        )

    by_year = coverage(["year"])
    by_level = coverage(["level"])
    by_surface = coverage(["surface"])
    second_attempts = stats["service_points"] - stats["first_serve_in"]
    anomalies = {
        "negative_values": int((stats[list(STAT_COLUMNS)] < 0).any(axis=1).sum()),
        "first_serve_in_above_service_points": int(
            (stats["first_serve_in"] > stats["service_points"]).sum()
        ),
        "first_serve_won_above_first_serve_in": int(
            (stats["first_serve_won"] > stats["first_serve_in"]).sum()
        ),
        "second_serve_won_above_attempts": int((stats["second_serve_won"] > second_attempts).sum()),
        "break_points_saved_above_faced": int(
            (stats["break_points_saved"] > stats["break_points_faced"]).sum()
        ),
        "breaks_conceded_above_service_games": int(
            (
                stats["break_points_faced"] - stats["break_points_saved"] > stats["service_games"]
            ).sum()
        ),
        "nonpositive_duration": int((matches["minutes"].notna() & matches["minutes"].le(0)).sum()),
        "duration_above_360_minutes": int(matches["minutes"].gt(360).sum()),
    }
    field_coverage = {
        column: {
            "complete_matches": int(context[f"complete_{column}"].sum()),
            "coverage": float(context[f"complete_{column}"].mean()),
        }
        for column in STAT_COLUMNS
    }
    audit = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "matches": len(matches),
        "stat_rows": len(stats),
        "field_coverage": field_coverage,
        "complete_service_bundle_matches": int(context["complete_service_bundle"].sum()),
        "complete_service_bundle_coverage": float(context["complete_service_bundle"].mean()),
        "duration_matches": int(context["duration_available"].sum()),
        "duration_coverage": float(context["duration_available"].mean()),
        "score_status": {
            str(key): int(value) for key, value in context["score_status"].value_counts().items()
        },
        "anomalies": anomalies,
        "training_cutoff_decision": (
            "Keep 2000-2018: service coverage is already 86.9%-90.0% in 2000-2014; "
            "missingness is concentrated in Davis Cup rather than the early era."
        ),
    }
    return audit, by_year, by_level, by_surface


def write_stats_audit(
    report_dir: Path,
    audit: dict[str, Any],
    by_year: pd.DataFrame,
    by_level: pd.DataFrame,
    by_surface: pd.DataFrame,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "stats_availability_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    by_year.to_csv(report_dir / "stats_coverage_by_year.csv", index=False)
    by_level.to_csv(report_dir / "stats_coverage_by_level.csv", index=False)
    by_surface.to_csv(report_dir / "stats_coverage_by_surface.csv", index=False)
    fields = "\n".join(
        f"- `{name}`: {values['complete_matches']:,} matchs complets ({values['coverage']:.2%})"
        for name, values in audit["field_coverage"].items()
    )
    anomalies = "\n".join(f"- {name}: {value:,}" for name, value in audit["anomalies"].items())
    text = f"""# Audit des statistiques ATP — Phase 4

## Schéma disponible

Une ligne `match_stats` existe par joueur et contient des comptes bruts, jamais des taux :

{fields}

Le bloc service complet est disponible pour {audit["complete_service_bundle_matches"]:,} matchs
({audit["complete_service_bundle_coverage"]:.2%}). Une durée strictement positive existe pour
{audit["duration_matches"]:,} matchs ({audit["duration_coverage"]:.2%}).

## Définitions fiables

- points de second service tentés = `service_points - first_serve_in`;
- points de service gagnés = `first_serve_won + second_serve_won`;
- points de retour gagnés = points de service adverses moins points de service gagnés adverses;
- breaks concédés = `break_points_faced - break_points_saved`;
- jeux de service tenus = `service_games - breaks_conceded`.

Les taux sont agrégés par somme des numérateurs divisée par somme des opportunités, jamais par
moyenne de pourcentages de matchs.

## Couverture temporelle

Les fichiers CSV joints donnent le détail par année, niveau et surface. La couverture service
est déjà comprise entre 86,9 % et 90,0 % sur 2000–2014, puis dépasse généralement 94 % depuis
2016. La rupture principale est institutionnelle : Davis Cup (`D`) n'a qu'environ 18 % de
couverture, tandis que A/G/M dépassent 98 %. La durée tombe à 46,3 % en 2015.

Décision : conserver le train 2000–2018 et apprendre l'imputation sur ce train. Un cutoff récent
ne corrigerait pas la faible couverture Davis Cup et supprimerait un historique utile.

## Anomalies filtrées localement

{anomalies}

Une anomalie invalide uniquement le ratio concerné, pas toutes les statistiques du match. Les
durées nulles/négatives et supérieures à 360 minutes sont exclues des charges en minutes.

## Limite de calendrier

La date est le début du tournoi. Les rounds fournissent un ordre structurel sûr, mais pas les
jours réels. La fatigue est donc mise à jour après chaque batch de round; les formats ambigus
restent en un seul batch. Aucun `consecutive_play_days` n'est inventé.
"""
    (report_dir / "stats_availability_audit.md").write_text(text, encoding="utf-8")
