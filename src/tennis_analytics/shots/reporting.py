from __future__ import annotations

import json
from collections import Counter
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tennis_analytics.paths import REPORTS_DIR
from tennis_analytics.shots.build import AUDIT_JSON_PATH, MATCHES_PATH, POINTS_PATH


def _percent(value: float) -> str:
    return f"{100 * value:.2f} %"


def _read(name: str) -> pd.DataFrame:
    path = REPORTS_DIR / name
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def _canonical_sets() -> dict[str, Any]:
    matches: set[str] = set()
    players: set[str] = set()
    tournaments: set[str] = set()
    surfaces: Counter[str] = Counter()
    years: Counter[int] = Counter()
    parquet = pq.ParquetFile(POINTS_PATH)
    for batch in parquet.iter_batches(
        columns=[
            "match_id",
            "player1_id",
            "player2_id",
            "tournament",
            "surface",
            "year",
        ],
        batch_size=150_000,
    ):
        frame = pa.Table.from_batches([batch]).to_pandas()
        matches.update(frame["match_id"].unique())
        players.update(frame["player1_id"].unique())
        players.update(frame["player2_id"].unique())
        tournaments.update(frame["tournament"].unique())
        surfaces.update(frame["surface"].value_counts().to_dict())
        years.update(frame["year"].value_counts().to_dict())
    return {
        "matches": matches,
        "players": players,
        "tournaments": tournaments,
        "surfaces": surfaces,
        "years": years,
    }


def write_audit_reports() -> dict[str, Any]:
    audit = json.loads(AUDIT_JSON_PATH.read_text())
    canonical = _canonical_sets()
    metadata = pd.read_csv(MATCHES_PATH, dtype=str)
    parsed_date = pd.to_datetime(metadata["Date"], format="%Y%m%d", errors="coerce")
    valid_surface = metadata["Surface"].isin(["Hard", "Clay", "Grass", "Carpet"])
    players = pd.concat([metadata["Player 1"], metadata["Player 2"]]).dropna()
    player_coverage = players.value_counts().head(20)
    direction_coverage = audit["direction_known_events"] / audit["direction_eligible_events"]
    depth_coverage = audit["return_depth_known"] / audit["return_depth_eligible"]
    parse_rate = audit["valid_points"] / audit["raw_points"]

    era = Counter()
    for year, points in audit["year_points"].items():
        value = int(year)
        label = "avant_1990" if value < 1990 else f"{value // 10 * 10}s"
        era[label] += points
    lines = [
        "# Audit des données shot-level MCP",
        "",
        "## Périmètre local",
        "",
        "- Genre disponible: hommes uniquement. Les fichiers WTA `charting-w-*` sont absents du snapshot local; couverture WTA = 0 match.",
        f"- Métadonnées: {audit['metadata_rows']:,} identifiants uniques ({audit['metadata_duplicate_ids']} doublon résolu par qualité des champs).",
        f"- Matchs avec au moins un point canonique: {len(canonical['matches']):,}.",
        f"- Points source: {audit['raw_points']:,}; points shot-level retenus: {audit['valid_points']:,} ({_percent(parse_rate)}).",
        f"- Frappes/événements retenus: {audit['valid_events']:,}.",
        f"- Joueurs MCP distincts: {len(canonical['players']):,}; tournois: {len(canonical['tournaments']):,}.",
        f"- Période canonique: {min(canonical['years'])}–{max(canonical['years'])}.",
        "",
        "## Couverture par surface (points)",
        "",
        "| Surface | Points | Part |",
        "|---|---:|---:|",
    ]
    for surface, count in canonical["surfaces"].most_common():
        lines.append(f"| {surface} | {count:,} | {_percent(count / audit['valid_points'])} |")
    lines.extend(["", "## Couverture par époque", "", "| Époque | Points |", "|---|---:|"])
    for label, count in sorted(era.items()):
        lines.append(f"| {label} | {count:,} |")
    lines.extend(
        [
            "",
            "## Joueurs les plus couverts dans les métadonnées",
            "",
            "| Joueur | Matchs |",
            "|---|---:|",
        ]
    )
    for name, count in player_coverage.items():
        lines.append(f"| {name} | {count:,} |")
    lines.extend(
        [
            "",
            "## Fiabilité des champs",
            "",
            "| Champ / contrôle | Couverture | Commentaire |",
            "|---|---:|---|",
            f"| Date métadonnée parseable | {_percent(parsed_date.notna().mean())} | format attendu YYYYMMDD |",
            f"| Surface reconnue | {_percent(valid_surface.mean())} | deux lignes source sont mal alignées/complémentaires |",
            f"| Main joueur 1 | {_percent(metadata['Pl 1 hand'].notna().mean())} | valeur source |",
            f"| Main joueur 2 | {_percent(metadata['Pl 2 hand'].notna().mean())} | valeur source |",
            f"| Direction des coups non-service | {_percent(direction_coverage)} | `0` et absence restent inconnus |",
            f"| Profondeur du retour | {_percent(depth_coverage)} | optionnelle selon le protocole |",
            f"| Cohérence vainqueur reconstruit | {_percent(1 - audit['winner_mismatches'] / audit['raw_points'])} | le cas divergent est exclu |",
            "",
            "## Biais et limites",
            "",
            "Le MCP est crowdsourcé et non exhaustif: la couverture favorise certains joueurs, matchs télévisés et périodes récentes. Les fréquences ne représentent donc pas automatiquement le circuit ATP. Les comparaisons de modèles sont confinées au même sous-ensemble MCP. La licence source est CC BY-NC-SA 4.0.",
            "",
        ]
    )
    (REPORTS_DIR / "mcp_shot_data_audit.md").write_text("\n".join(lines))

    parser_lines = [
        "# Audit du parseur MCP",
        "",
        f"Taux de parsing canonique: **{_percent(parse_rate)}** ({audit['valid_points']:,}/{audit['raw_points']:,}).",
        f"Événements écrits: **{audit['valid_events']:,}**. Points exclus: **{audit['excluded_points']:,}**.",
        "",
        "## Motifs d'exclusion",
        "",
        "| Motif | Points | Part des sources |",
        "|---|---:|---:|",
    ]
    for reason, count in sorted(audit["reason_counts"].items(), key=lambda item: -item[1]):
        parser_lines.append(f"| `{reason}` | {count:,} | {_percent(count / audit['raw_points'])} |")
    parser_lines.extend(
        [
            "",
            "Les codes `P/Q/R/S` expliquent l'essentiel de `special_point`: ils portent une issue de score mais aucune séquence observable. Les inconnues officielles (`0`, `q`, `e`) sont acceptées et marquées; les caractères non résolus restent des exclusions.",
            "",
            "## Contrôles",
            "",
            f"- Désaccords entre le vainqueur reconstruit et `PtWinner`: {audit['winner_mismatches']} (exclus).",
            f"- Métadonnées de match absentes: {audit['missing_metadata']}.",
            f"- Événements contenant une inconnue explicite: {audit['unknown_code_events']:,}.",
            "- Services manquants, fins absentes, multiples fins et caractères après terminal sont rejetés.",
            "- Les tests couvrent exemples officiels, second service, profondeur, positions, double faute, codes spéciaux et absence d'information future aux checkpoints.",
            "",
        ]
    )
    (REPORTS_DIR / "mcp_parser_audit.md").write_text("\n".join(parser_lines))
    return {
        "parse_rate": parse_rate,
        "matches": len(canonical["matches"]),
        "players": len(canonical["players"]),
        "tournaments": len(canonical["tournaments"]),
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 10) -> list[str]:
    if frame.empty:
        return ["Aucune ligne disponible."]
    selected = frame[columns].head(limit)
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in selected.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_phase9_report(summary: dict[str, Any]) -> dict[str, Any]:
    audit = json.loads(AUDIT_JSON_PATH.read_text())
    model = _read("shot_model_comparison.csv")
    effects = _read("shot_model_clustered_effects.csv")
    next_shot = _read("next_shot_comparison.csv")
    service = _read("shot_service_patterns.csv")
    court_service = _read("shot_serve_court_side.csv")
    serve_plus = _read("shot_serve_plus1_patterns.csv")
    return_plus = _read("shot_return_plus1_patterns.csv")
    rally = _read("shot_rally_length.csv")
    surfaces = _read("shot_surface_comparison.csv")
    pressure = _read("shot_pressure_comparison.csv")
    matchups = _read("shot_style_matchups.csv")
    test_model = model.query("split == 'test'").sort_values(["checkpoint", "model"])
    test_effect = effects.query("split == 'test'")
    strongest = test_effect.query("comparison == 'shot_minus_recalibrated_prior'").sort_values(
        "mean_match_log_loss_gain", ascending=False
    )
    evidence_shot = bool(
        not strongest.empty
        and strongest.iloc[0]["mean_match_log_loss_gain"] > 0
        and strongest.iloc[0]["ci95_low"] > 0
    )
    test_next = next_shot.query("split == 'test'")
    markov = test_next.loc[test_next["model"] == "markov_1", "log_loss"]
    unigram = test_next.loc[test_next["model"] == "unigram", "log_loss"]
    evidence_sequence = bool(not markov.empty and not unigram.empty and markov.iloc[0] < unigram.iloc[0])
    phase10 = evidence_shot and evidence_sequence

    serve_global = (
        service.groupby(["serve_number", "serve_direction"], as_index=False)
        .agg(attempts=("attempts", "sum"), wins=("wins", "sum"), aces=("aces", "sum"))
        if not service.empty
        else pd.DataFrame()
    )
    if not serve_global.empty:
        serve_global["win_rate"] = serve_global["wins"] / serve_global["attempts"]
        serve_global["ace_rate"] = serve_global["aces"] / serve_global["attempts"]

    lines = [
        "# Phase 9 — Shot-level analytics V1",
        "",
        "## Conclusion exécutive",
        "",
        (
            "Le shot-level apporte un signal prédictif incrémental robuste au-delà du prior recalibré et le modèle de transition bat l'unigramme. Une Phase 10 séquentielle est donc justifiée, sans avoir été lancée."
            if phase10
            else "La preuve nécessaire pour lancer un modèle profond de séquence n'est pas encore réunie: le gain shot-level robuste et le signal next-shot doivent être simultanément confirmés. Aucun Transformer n'a été lancé."
        ),
        "",
        "Les résultats restent associatifs, conditionnels au sous-ensemble MCP et ne doivent pas être lus causalement.",
        "",
        "## A. Dataset shot-level",
        "",
        f"- {summary['audit']['matches']:,} matchs, {audit['valid_points']:,} points, {audit['valid_events']:,} événements, {summary['audit']['players']:,} joueurs.",
        f"- Parsing: {_percent(summary['audit']['parse_rate'])}; hommes seulement dans le snapshot local.",
        "",
        "## B. Qualité du parser",
        "",
        f"{audit['excluded_points']:,} points exclus avec motifs auditables; {audit['winner_mismatches']} désaccord de vainqueur exclu. Voir `mcp_parser_audit.md`.",
        "",
        "## C. Service",
        "",
        *_markdown_table(serve_global.sort_values("attempts", ascending=False) if not serve_global.empty else serve_global, ["serve_number", "serve_direction", "attempts", "win_rate", "ace_rate"]),
        "",
        "Le côté est reconstruit par la parité du score avant le point (y compris tie-breaks):",
        "",
        *_markdown_table(court_service.sort_values("attempts", ascending=False), ["court_side", "surface", "serve_number", "serve_direction", "attempts", "win_rate_posterior", "ace_rate"]),
        "",
        "## D. Serve + 1",
        "",
        "Patterns avec support minimum 200; taux lissés Beta(1,1). Les meilleurs taux ne sont pas des effets causaux.",
        "",
        *_markdown_table(serve_plus.sort_values("attempts", ascending=False), ["serve", "serve_plus_1", "attempts", "win_rate_posterior", "lift_vs_global"]),
        "",
        "## E. Return + 1",
        "",
        *_markdown_table(return_plus.sort_values("attempts", ascending=False), ["return", "return_plus_1", "attempts", "win_rate_posterior", "lift_vs_global"]),
        "",
        "## F. Rally",
        "",
        *_markdown_table(rally, ["surface", "rally_length", "attempts", "win_rate_posterior"]),
        "",
        "La longueur finale sert uniquement aux statistiques descriptives. Les modèles ne voient que la longueur déjà observée.",
        "",
        "## G–H. Fingerprints et clustering",
        "",
        f"{summary['descriptive']['players_with_fingerprint']} joueurs passent les seuils; K-Means retient k={summary['descriptive']['selected_clusters']}. La similarité cosinus exclut ranking et niveau externe. Interprétation détaillée dans `player_style_clusters.md`.",
        "",
        "Interactions descriptives entre style du serveur et style du relanceur (sans interprétation causale):",
        "",
        *_markdown_table(matchups, ["server_style_cluster", "receiver_style_cluster", "surface", "attempts", "win_rate_posterior"]),
        "",
        "## I. Prédiction du point",
        "",
        "Même sous-ensemble MCP, split temporel par match; un contrôle `PRIOR_RECALIBRATED` sépare l'effet des coups d'une simple recalibration du prior.",
        "",
        *_markdown_table(test_model, ["checkpoint", "model", "points", "matches", "log_loss", "brier", "auc"] , limit=20),
        "",
        "Effets de log loss agrégés par match (positif = amélioration):",
        "",
        *_markdown_table(test_effect, ["checkpoint", "comparison", "mean_match_log_loss_gain", "ci95_low", "ci95_high", "share_matches_improved"], limit=20),
        "",
        "Les checkpoints après N frappes excluent les points déjà terminés: les chiffres mesurent l'information additionnelle parmi les échanges encore vivants, pas une métrique artificiellement parfaite sur aces/winners déjà observés.",
        "",
        "## J. Next-shot prediction",
        "",
        *_markdown_table(next_shot, ["split", "model", "transitions", "log_loss", "top1_accuracy"], limit=10),
        "",
        "Le modèle joueur utilise un shrinkage de 100 transitions vers le Markov global et un minimum de 200 observations joueur.",
        "",
        "## K. Surface",
        "",
        *_markdown_table(surfaces, ["surface", "points", "server_win_rate", "average_rally_length", "serve_wide_rate", "serve_body_rate", "serve_t_rate"]),
        "",
        "## L. Pression",
        "",
        f"Analyse limitée aux {summary['descriptive']['pressure_points_with_exact_flags']:,} points raccordés aux flags historiques de POINT MODEL V2.",
        "",
        *_markdown_table(pressure, ["pressure", "points", "server_win_rate", "average_rally_length", "serve_wide_rate", "serve_plus_1_attack_rate"]),
        "",
        "Les écarts décrivent des changements de choix observés; ils ne démontrent ni clutch persistant ni causalité.",
        "",
        "## Décision Phase 10",
        "",
        f"Décision: **{'GO prudent' if phase10 else 'NO-GO à ce stade'}**. Critères: gain shot-level robuste au-delà de la recalibration = {evidence_shot}; Markov meilleur que l'unigramme = {evidence_sequence}. La Phase 10 n'est pas commencée.",
        "",
    ]
    (REPORTS_DIR / "phase9_report.md").write_text("\n".join(lines))
    result = {
        "phase10_recommended": phase10,
        "shot_signal_robust": evidence_shot,
        "next_shot_signal": evidence_sequence,
    }
    (REPORTS_DIR / "phase9_results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
