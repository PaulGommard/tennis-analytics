from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from tennis_analytics.paths import EXTERNAL_DIR, REPORTS_DIR

SLAM_ROOT = EXTERNAL_DIR / "slam_pointbypoint"
MCP_ROOT = EXTERNAL_DIR / "match_charting_project"
SURFACE_BY_SLAM = {
    "ausopen": "Hard",
    "frenchopen": "Clay",
    "wimbledon": "Grass",
    "usopen": "Hard",
}


def _coverage(series: pd.Series, valid: tuple[Any, ...] | None = None) -> float:
    if len(series) == 0:
        return 0.0
    mask = series.notna()
    if valid is not None:
        mask &= series.isin(valid)
    return float(mask.mean())


def audit_slam() -> dict[str, Any]:
    point_files = sorted(SLAM_ROOT.rglob("*-points.csv"))
    match_files = sorted(SLAM_ROOT.rglob("*-matches.csv"))
    match_frames = [pd.read_csv(path, low_memory=False) for path in match_files]
    matches = pd.concat(match_frames, ignore_index=True)
    known_matches = set(matches["match_id"].astype(str))
    rows = 0
    point_matches: set[str] = set()
    years: Counter[int] = Counter()
    slams: Counter[str] = Counter()
    invalid_winner = 0
    invalid_server = 0
    sentinel = 0
    score_coverage_count = 0
    server_count = 0
    winner_count = 0
    ace_count = 0
    double_fault_count = 0
    duplicate_keys = 0
    tiebreak_score_rows = 0
    schema_variants: Counter[tuple[str, ...]] = Counter()
    for path in point_files:
        frame = pd.read_csv(path, low_memory=False)
        size = len(frame)
        rows += size
        schema_variants[tuple(frame.columns)] += 1
        match_ids = frame["match_id"].astype(str)
        point_matches.update(match_ids.unique())
        year = int(path.name[:4])
        slam = path.name.removeprefix(f"{year}-").removesuffix("-points.csv")
        years[year] += size
        slams[slam] += size
        winners = pd.to_numeric(frame["PointWinner"], errors="coerce")
        servers = pd.to_numeric(frame["PointServer"], errors="coerce")
        valid_point = pd.to_numeric(frame["PointNumber"], errors="coerce").fillna(0) > 0
        sentinel += int((~valid_point).sum())
        invalid_winner += int((valid_point & ~winners.isin([1, 2])).sum())
        invalid_server += int((valid_point & ~servers.isin([1, 2])).sum())
        winner_count += int((valid_point & winners.isin([1, 2])).sum())
        server_count += int((valid_point & servers.isin([1, 2])).sum())
        score_coverage_count += int(
            (valid_point & frame["P1Score"].notna() & frame["P2Score"].notna()).sum()
        )
        numeric_a = pd.to_numeric(frame["P1Score"], errors="coerce")
        numeric_b = pd.to_numeric(frame["P2Score"], errors="coerce")
        tiebreak_score_rows += int(
            (
                (numeric_a.between(1, 99) & ~numeric_a.isin([15, 30, 40]))
                | (numeric_b.between(1, 99) & ~numeric_b.isin([15, 30, 40]))
            ).sum()
        )
        ace_columns = [column for column in ("P1Ace", "P2Ace") if column in frame]
        df_columns = [column for column in ("P1DoubleFault", "P2DoubleFault") if column in frame]
        ace_count += int(frame[ace_columns].notna().all(axis=1).sum()) if ace_columns else 0
        double_fault_count += int(frame[df_columns].notna().all(axis=1).sum()) if df_columns else 0
        duplicate_keys += int(frame.duplicated(["match_id", "PointNumber"]).sum())
    effective = max(rows - sentinel, 1)
    match_num = pd.to_numeric(matches["match_num"], errors="coerce")
    division = match_num.floordiv(1000).map({1: "men", 2: "women"}).fillna("unknown")
    surfaces = matches["slam"].map(SURFACE_BY_SLAM).value_counts().to_dict()
    return {
        "source": "slam_pointbypoint",
        "license": "CC BY-NC-SA 4.0",
        "point_files": len(point_files),
        "match_files": len(match_files),
        "rows": rows,
        "metadata_matches": len(matches),
        "point_matches": len(point_matches),
        "orphan_point_matches": len(point_matches - known_matches),
        "years": [int(matches["year"].min()), int(matches["year"].max())],
        "year_row_counts": dict(sorted(years.items())),
        "slam_row_counts": dict(sorted(slams.items())),
        "surface_match_counts": surfaces,
        "division_match_counts_inferred_from_match_num": division.value_counts().to_dict(),
        "players_named": int(
            pd.concat([matches["player1"], matches["player2"]]).dropna().nunique()
        ),
        "schema_variants": len(schema_variants),
        "sentinel_or_non_point_rows": sentinel,
        "valid_point_winner_rate": winner_count / effective,
        "valid_server_rate": server_count / effective,
        "score_pair_rate": score_coverage_count / effective,
        "ace_flag_pair_rate_all_rows": ace_count / max(rows, 1),
        "double_fault_flag_pair_rate_all_rows": double_fault_count / max(rows, 1),
        "tiebreak_numeric_score_rows": tiebreak_score_rows,
        "invalid_winner_rows": invalid_winner,
        "invalid_server_rows": invalid_server,
        "duplicate_match_point_number_rows": duplicate_keys,
        "daily_date_available": False,
        "retirement_status_rate": _coverage(matches["status"]),
        "status_counts": matches["status"].fillna("missing").value_counts().to_dict(),
        "stable_player_id_pair_rate": float(
            (matches["player1id"].notna() & matches["player2id"].notna()).mean()
        ),
    }


def audit_mcp() -> dict[str, Any]:
    files = sorted(MCP_ROOT.glob("charting-m-points-*.csv"))
    metadata = pd.read_csv(MCP_ROOT / "charting-m-matches.csv", low_memory=False)
    known_matches = set(metadata["match_id"].astype(str))
    rows = 0
    match_ids: set[str] = set()
    invalid_winner = 0
    invalid_server = 0
    duplicate_keys = 0
    score_count = 0
    first_count = 0
    second_count = 0
    tiebreak_score_rows = 0
    for path in files:
        frame = pd.read_csv(path, low_memory=False)
        rows += len(frame)
        ids = frame["match_id"].astype(str)
        match_ids.update(ids.unique())
        invalid_winner += int(
            (~pd.to_numeric(frame["PtWinner"], errors="coerce").isin([1, 2])).sum()
        )
        invalid_server += int((~pd.to_numeric(frame["Svr"], errors="coerce").isin([1, 2])).sum())
        duplicate_keys += int(frame.duplicated(["match_id", "Pt"]).sum())
        score_count += int(frame["Pts"].notna().sum())
        first_count += int(frame["1st"].notna().sum())
        second_count += int(frame["2nd"].notna().sum())
        scores = frame["Pts"].fillna("").astype(str)
        parts = scores.str.extract(r"^(\d+)-(\d+)$").apply(pd.to_numeric, errors="coerce")
        regular = {0, 15, 30, 40}
        tiebreak_score_rows += int(
            (parts[0].notna() & (~parts[0].isin(regular) | ~parts[1].isin(regular))).sum()
        )
    numeric_dates = pd.to_numeric(metadata["Date"], errors="coerce").astype("Int64")
    dates = pd.to_datetime(numeric_dates.astype(str), format="%Y%m%d", errors="coerce")
    return {
        "source": "match_charting_project_men",
        "license": "CC BY-NC-SA 4.0",
        "point_files": len(files),
        "rows": rows,
        "metadata_matches": len(metadata),
        "point_matches": len(match_ids),
        "orphan_point_matches": len(match_ids - known_matches),
        "years": [int(dates.dt.year.min()), int(dates.dt.year.max())],
        "surface_match_counts": metadata["Surface"].fillna("missing").value_counts().to_dict(),
        "players_named": int(
            pd.concat([metadata["Player 1"], metadata["Player 2"]]).dropna().nunique()
        ),
        "valid_point_winner_rate": 1 - invalid_winner / max(rows, 1),
        "valid_server_rate": 1 - invalid_server / max(rows, 1),
        "score_rate": score_count / max(rows, 1),
        "first_serve_shot_string_rate": first_count / max(rows, 1),
        "second_serve_shot_string_rate": second_count / max(rows, 1),
        "tiebreak_numeric_score_rows": tiebreak_score_rows,
        "invalid_winner_rows": invalid_winner,
        "invalid_server_rows": invalid_server,
        "duplicate_match_point_rows": duplicate_keys,
        "malformed_metadata_rows": int(numeric_dates.isna().sum()),
        "best_of_counts": metadata["Best of"].fillna("missing").value_counts().to_dict(),
        "final_tiebreak_rule_counts": metadata["Final TB?"]
        .fillna("missing")
        .value_counts()
        .to_dict(),
        "daily_date_available": True,
        "stable_player_ids_available": False,
        "retirement_field_available": False,
        "explicit_ace_double_fault_flags": False,
    }


def run_audit(report_dir: Path = REPORTS_DIR) -> dict[str, Any]:
    audit = {"slam_pointbypoint": audit_slam(), "match_charting_project": audit_mcp()}
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "phase7a_pbp_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    slam, mcp = audit["slam_pointbypoint"], audit["match_charting_project"]
    text = f"""# Phase 7A — audit point par point

## Inventaire constaté

| Source | Fichiers points | Lignes | Matchs points | Période | Date quotidienne |
|---|---:|---:|---:|---|---|
| Grand Slam PBP | {slam["point_files"]} | {slam["rows"]:,} | {slam["point_matches"]:,} | {slam["years"][0]}–{slam["years"][1]} | non |
| Match Charting Project hommes | {mcp["point_files"]} | {mcp["rows"]:,} | {mcp["point_matches"]:,} | {mcp["years"][0]}–{mcp["years"][1]} | oui |

Les deux sources sont sous **CC BY-NC-SA 4.0** : attribution obligatoire,
usage commercial interdit, partage dans les mêmes conditions.

## Couverture réelle

- **Grand Slam PBP** : {slam["players_named"]:,} noms de joueurs bruts. La convention de
  `match_num` permet d'inférer {slam["division_match_counts_inferred_from_match_num"].get("men", 0):,}
  matchs hommes, {slam["division_match_counts_inferred_from_match_num"].get("women", 0):,} femmes
  et laisse {slam["division_match_counts_inferred_from_match_num"].get("unknown", 0):,} non classés.
  Surfaces inférées du tournoi : {slam["surface_match_counts"].get("Hard", 0):,} Hard,
  {slam["surface_match_counts"].get("Clay", 0):,} Clay et
  {slam["surface_match_counts"].get("Grass", 0):,} Grass. Le format BO3/BO5 n'est pas un champ
  source fiable de ces fichiers et ne doit pas être inventé.
- **MCP** : collection masculine uniquement, {mcp["players_named"]:,} noms bruts ;
  {mcp["surface_match_counts"].get("Hard", 0):,} Hard,
  {mcp["surface_match_counts"].get("Clay", 0):,} Clay et
  {mcp["surface_match_counts"].get("Grass", 0):,} Grass. Les métadonnées déclarent
  {mcp["best_of_counts"].get("3", 0):,} BO3 et {mcp["best_of_counts"].get("5", 0):,} BO5,
  avec {mcp["best_of_counts"].get("missing", 0):,} valeurs absentes et deux valeurs anormales.

## Formats et encodage

- Grand Slam : une ligne par point/sentinelle, `PointServer` et `PointWinner` dans {{1,2}},
  `SetNo`, `GameNo`, `P1GamesWon/P2GamesWon`, `P1Score/P2Score`. Ace et double faute sont des
  flags par joueur. Les scores numériques de tie-break apparaissent sur
  {slam["tiebreak_numeric_score_rows"]:,} lignes ; la règle de tie-break du match n'est pas
  fournie uniformément.
- MCP : `Pt` ordonne les points ; `Set1/Set2`, `Gm1/Gm2`, `Pts` et `Gm#` portent le score ;
  `Svr`/`PtWinner` utilisent {{1,2}}. `TbSet` indique qu'un set doit finir au tie-break, pas que
  le point courant est nécessairement joué dans celui-ci. `1st`, `2nd` et `Notes` portent
  l'encodage coup par coup. Les fichiers présents n'exposent pas les colonnes dérivées
  `isAce`, `isDouble`, `TB?` mentionnées dans le dictionnaire amont : elles devront être
  parsées et vérifiées, jamais supposées.
- Abandons : Grand Slam fournit 20 `Retired`, 628 `Complete`, 3 `In Progress`, mais le statut
  manque pour 9 862 matchs. MCP n'a pas de champ abandon dédié.

## Qualité et limites

- Grand Slam : {slam["schema_variants"]} variantes de schéma, {slam["sentinel_or_non_point_rows"]:,}
  lignes sentinelles/non-points, taux serveur valide {slam["valid_server_rate"]:.2%},
  taux gagnant valide {slam["valid_point_winner_rate"]:.2%}, {slam["orphan_point_matches"]:,}
  matchs points sans métadonnée, {slam["duplicate_match_point_number_rows"]:,} clés
  `(match_id, PointNumber)` dupliquées.
- Grand Slam : aucune date quotidienne fiable; l'année et le tournoi sont disponibles.
  Les identifiants joueurs ne sont renseignés en paire que dans
  {slam["stable_player_id_pair_rate"]:.2%} des métadonnées. Les divisions homme/femme
  sont inférées du millier de `match_num` et doivent être confirmées avant normalisation.
- MCP : taux serveur valide {mcp["valid_server_rate"]:.2%}, taux gagnant valide
  {mcp["valid_point_winner_rate"]:.2%}, {mcp["duplicate_match_point_rows"]:,} clés
  `(match_id, Pt)` dupliquées et {mcp["malformed_metadata_rows"]} ligne de métadonnées décalée.
  Les noms sont présents mais pas d'identifiants ATP stables.
- MCP : les aces, doubles fautes et types de coups sont encodés dans `1st`/`2nd`, pas
  fournis comme flags canoniques. Un parseur versionné sera nécessaire.
- Les abandons ne sont pas identifiables de façon homogène. Aucun entraînement live ne
  doit commencer avant résolution des doublons, mapping joueurs, ordre des points et règles
  de tie-break par époque/tournoi.

## Décision Phase 7A

Le volume est suffisant pour préparer un baseline live, mais la jointure avec la base ATP
et la normalisation doivent rester une étape dédiée et auditée. Ce rapport ne prétend pas
que la couverture est exhaustive.
"""
    (report_dir / "point_by_point_data_audit.md").write_text(text, encoding="utf-8")
    return audit
