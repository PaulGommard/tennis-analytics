from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from tennis_analytics.paths import REPORTS_DIR
from tennis_analytics.shots.build import EVENTS_PATH, POINTS_PATH
from tennis_analytics.shots.modeling import PREDICTIONS_PATH


def _posterior(count: int, total: int) -> tuple[float, float, float]:
    """Beta(1,1) posterior mean and normal-approximate 95% credible interval."""
    alpha = count + 1
    beta = total - count + 1
    mean = alpha / (alpha + beta)
    sd = math.sqrt(alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1)))
    return mean, max(0.0, mean - 1.96 * sd), min(1.0, mean + 1.96 * sd)


def _aggregate_rows(
    counter: dict[tuple[Any, ...], Counter[str]], names: list[str], minimum: int = 1
) -> list[dict[str, Any]]:
    rows = []
    for key, values in counter.items():
        attempts = values["attempts"]
        if attempts < minimum:
            continue
        mean, low, high = _posterior(values["wins"], attempts)
        rows.append(
            {
                **dict(zip(names, key, strict=True)),
                **dict(values),
                "win_rate_posterior": mean,
                "win_rate_ci95_low": low,
                "win_rate_ci95_high": high,
            }
        )
    return rows


def _with_lift(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    global_rate = frame["wins"].sum() / frame["attempts"].sum()
    frame["global_win_rate"] = global_rate
    frame["lift_vs_global"] = frame["win_rate_posterior"] - global_rate
    return frame


def _court_side(score: str) -> str:
    parts = str(score).upper().split("-")
    if len(parts) != 2:
        return "unknown"
    regular = {"0": 0, "15": 1, "30": 2, "40": 3, "A": 4, "AD": 4, "ADV": 4}
    if parts[0] in regular and parts[1] in regular:
        total = regular[parts[0]] + regular[parts[1]]
    else:
        try:
            total = int(parts[0]) + int(parts[1])
        except ValueError:
            return "unknown"
    return "deuce" if total % 2 == 0 else "ad"


def run_descriptive_analysis() -> dict[str, Any]:
    player: defaultdict[str, Counter[str]] = defaultdict(Counter)
    player_name: dict[str, str] = {}
    player_matches: defaultdict[str, set[str]] = defaultdict(set)
    service: defaultdict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    events = pq.ParquetFile(EVENTS_PATH)
    for batch in events.iter_batches(batch_size=150_000):
        frame = pa.Table.from_batches([batch]).to_pandas()
        for row in frame.itertuples(index=False):
            player_name[row.hitter_id] = row.hitter_name
            stats = player[row.hitter_id]
            stats["events"] += 1
            if row.is_serve:
                stats["serves"] += 1
                stats[f"serve_{row.serve_direction}"] += 1
                key = (row.hitter_id, row.hitter_name, row.surface, row.serve_number, row.serve_direction)
                service[key]["attempts"] += 1
                service[key]["wins"] += int(row.won_point)
                service[key]["aces"] += int(row.outcome == "ace")
                service[key]["unreturned"] += int(row.outcome == "unreturned")
                service[key]["faults"] += int(row.outcome == "fault")
                continue
            stats["shots"] += 1
            stats[row.wing] += 1
            stats["net"] += int(row.at_net or row.is_approach)
            stats["winners"] += int(row.outcome == "winner")
            stats["unforced_errors"] += int(row.outcome == "unforced_error")
            stats["forced_errors"] += int(row.outcome == "forced_error")
            if row.tactical_role == "serve_plus_1":
                stats["serve_plus_1"] += 1
                stats["serve_plus_1_aggressive"] += int(
                    row.outcome == "winner" or row.at_net or row.is_approach
                )

    service_rows = _aggregate_rows(
        service,
        ["player_id", "player_name", "surface", "serve_number", "serve_direction"],
        minimum=20,
    )
    service_frame = pd.DataFrame(service_rows)
    if not service_frame.empty:
        service_frame["ace_rate"] = service_frame["aces"] / service_frame["attempts"]
        service_frame["unreturned_rate"] = service_frame["unreturned"] / service_frame["attempts"]
        service_frame["fault_rate"] = service_frame["faults"] / service_frame["attempts"]
    service_frame.to_csv(REPORTS_DIR / "shot_service_patterns.csv", index=False)

    serve_plus: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    return_plus: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    ngrams: defaultdict[tuple[int, str], Counter[str]] = defaultdict(Counter)
    ngrams_surface: defaultdict[tuple[str, int, str], Counter[str]] = defaultdict(Counter)
    ngrams_player: defaultdict[tuple[str, int, str], Counter[str]] = defaultdict(Counter)
    transitions: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    rally: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    surfaces: defaultdict[str, Counter[str]] = defaultdict(Counter)
    pressure: defaultdict[str, Counter[str]] = defaultdict(Counter)
    court_service: defaultdict[tuple[str, int, str, str], Counter[str]] = defaultdict(Counter)

    flags: dict[str, tuple[int, int, int, int]] = {}
    if PREDICTIONS_PATH.exists():
        flag_frame = pq.read_table(
            PREDICTIONS_PATH,
            columns=["point_id", "is_break_point", "is_set_point", "is_match_point", "is_tiebreak"],
        ).to_pandas()
        flags = dict(
            zip(
                flag_frame["point_id"],
                zip(
                    flag_frame["is_break_point"],
                    flag_frame["is_set_point"],
                    flag_frame["is_match_point"],
                    flag_frame["is_tiebreak"],
                    strict=True,
                ),
                strict=True,
            )
        )

    points = pq.ParquetFile(POINTS_PATH)
    for batch in points.iter_batches(batch_size=100_000):
        frame = pa.Table.from_batches([batch]).to_pandas()
        for row in frame.itertuples(index=False):
            player_matches[row.player1_id].add(row.match_id)
            player_matches[row.player2_id].add(row.match_id)
            player[row.player1_id]["rally_points"] += 1
            player[row.player2_id]["rally_points"] += 1
            player[row.player1_id]["rally_length_sum"] += row.final_rally_length
            player[row.player2_id]["rally_length_sum"] += row.final_rally_length
            tokens = str(row.sequence_tokens).split()
            court_side = _court_side(row.score_raw)
            for event in json.loads(row.sequence_json):
                if not event["is_serve"]:
                    continue
                key = (court_side, int(event["serve_number"]), event["serve_direction"], row.surface)
                court_service[key]["attempts"] += 1
                court_service[key]["wins"] += row.server_won_point
                court_service[key]["aces"] += int(event["outcome"] == "ace")
            surface_stats = surfaces[row.surface]
            surface_stats["points"] += 1
            surface_stats["server_wins"] += row.server_won_point
            surface_stats["rally_length_sum"] += row.final_rally_length
            surface_stats[f"serve_{tokens[0].split(':')[-1] if tokens else 'missing'}"] += 1
            category = (
                "0_4"
                if row.final_rally_length <= 4
                else "5_9"
                if row.final_rally_length <= 9
                else "10_plus"
            )
            rally[(row.surface, category)]["attempts"] += 1
            rally[(row.surface, category)]["wins"] += row.server_won_point
            if len(tokens) >= 3:
                key = (tokens[0], tokens[2])
                serve_plus[key]["attempts"] += 1
                serve_plus[key]["wins"] += row.server_won_point
            if len(tokens) >= 4:
                key = (tokens[1], tokens[3])
                return_plus[key]["attempts"] += 1
                return_plus[key]["wins"] += 1 - row.server_won_point
            for index in range(len(tokens) - 1):
                key = (tokens[index], tokens[index + 1])
                transitions[key]["attempts"] += 1
                owner_won = row.server_won_point if index % 2 == 0 else 1 - row.server_won_point
                transitions[key]["wins"] += owner_won
            for size in (2, 3):
                for index in range(len(tokens) - size + 1):
                    pattern = " > ".join(tokens[index : index + size])
                    key = (size, pattern)
                    ngrams[key]["attempts"] += 1
                    owner_won = row.server_won_point if index % 2 == 0 else 1 - row.server_won_point
                    ngrams[key]["wins"] += owner_won
                    surface_key = (row.surface, size, pattern)
                    ngrams_surface[surface_key]["attempts"] += 1
                    ngrams_surface[surface_key]["wins"] += owner_won
                    if size == 2:
                        owner = row.server_id if index % 2 == 0 else row.receiver_id
                        player_key = (owner, size, pattern)
                        ngrams_player[player_key]["attempts"] += 1
                        ngrams_player[player_key]["wins"] += owner_won
            point_flags = flags.get(row.point_id)
            if point_flags is not None:
                is_break, is_set, is_match, is_tb = point_flags
                label = (
                    "match_point"
                    if is_match
                    else "set_point"
                    if is_set
                    else "break_point"
                    if is_break
                    else "tiebreak"
                    if is_tb
                    else "normal"
                )
                values = pressure[label]
                values["points"] += 1
                values["server_wins"] += row.server_won_point
                values["rally_length_sum"] += row.final_rally_length
                if tokens:
                    values[f"serve_{tokens[0].split(':')[-1]}"] += 1
                if len(tokens) >= 3:
                    code = tokens[2].split(":", 1)[0]
                    values["serve_plus_1_observed"] += 1
                    values["serve_plus_1_attack"] += int(code in {"u", "v", "z", "o", "p", "j", "k"})

    _with_lift(
        _aggregate_rows(serve_plus, ["serve", "serve_plus_1"], minimum=200)
    ).sort_values("attempts", ascending=False).to_csv(
        REPORTS_DIR / "shot_serve_plus1_patterns.csv", index=False
    )
    _with_lift(
        _aggregate_rows(return_plus, ["return", "return_plus_1"], minimum=200)
    ).sort_values("attempts", ascending=False).to_csv(
        REPORTS_DIR / "shot_return_plus1_patterns.csv", index=False
    )
    _with_lift(
        _aggregate_rows(ngrams, ["n", "pattern"], minimum=500)
    ).sort_values("attempts", ascending=False).to_csv(REPORTS_DIR / "shot_ngrams.csv", index=False)
    _with_lift(
        _aggregate_rows(ngrams_surface, ["surface", "n", "pattern"], minimum=200)
    ).sort_values("attempts", ascending=False).to_csv(
        REPORTS_DIR / "shot_ngrams_surface.csv", index=False
    )
    player_patterns = _with_lift(
        _aggregate_rows(ngrams_player, ["player_id", "n", "pattern"], minimum=100)
    )
    if not player_patterns.empty:
        player_patterns.insert(
            1, "player_name", player_patterns["player_id"].map(player_name).fillna("Unknown")
        )
    player_patterns.sort_values("attempts", ascending=False).to_csv(
        REPORTS_DIR / "shot_ngrams_player.csv", index=False
    )
    _with_lift(
        _aggregate_rows(transitions, ["from_token", "to_token"], minimum=200)
    ).sort_values("attempts", ascending=False).to_csv(
        REPORTS_DIR / "shot_transition_matrix.csv", index=False
    )
    pd.DataFrame(
        _aggregate_rows(rally, ["surface", "rally_length"], minimum=1)
    ).to_csv(REPORTS_DIR / "shot_rally_length.csv", index=False)

    surface_rows = []
    for surface, values in surfaces.items():
        points_n = values["points"]
        surface_rows.append(
            {
                "surface": surface,
                "points": points_n,
                "server_win_rate": values["server_wins"] / points_n,
                "average_rally_length": values["rally_length_sum"] / points_n,
                "serve_wide_rate": values["serve_wide"] / points_n,
                "serve_body_rate": values["serve_body"] / points_n,
                "serve_t_rate": values["serve_t"] / points_n,
            }
        )
    pd.DataFrame(surface_rows).sort_values("points", ascending=False).to_csv(
        REPORTS_DIR / "shot_surface_comparison.csv", index=False
    )
    pressure_rows = []
    for label, values in pressure.items():
        points_n = values["points"]
        plus_n = values["serve_plus_1_observed"]
        pressure_rows.append(
            {
                "pressure": label,
                "points": points_n,
                "server_win_rate": values["server_wins"] / points_n,
                "average_rally_length": values["rally_length_sum"] / points_n,
                "serve_wide_rate": values["serve_wide"] / points_n,
                "serve_body_rate": values["serve_body"] / points_n,
                "serve_t_rate": values["serve_t"] / points_n,
                "serve_plus_1_attack_rate": (
                    values["serve_plus_1_attack"] / plus_n if plus_n else np.nan
                ),
            }
        )
    pd.DataFrame(pressure_rows).sort_values("points", ascending=False).to_csv(
        REPORTS_DIR / "shot_pressure_comparison.csv", index=False
    )
    court_rows = _aggregate_rows(
        court_service,
        ["court_side", "serve_number", "serve_direction", "surface"],
        minimum=100,
    )
    court_frame = pd.DataFrame(court_rows)
    court_frame["ace_rate"] = court_frame["aces"] / court_frame["attempts"]
    court_frame.to_csv(REPORTS_DIR / "shot_serve_court_side.csv", index=False)

    fingerprint_rows = []
    for player_id, values in player.items():
        shots = values["shots"]
        matches = len(player_matches[player_id])
        if shots < 500 or matches < 10:
            continue
        wing_total = values["forehand"] + values["backhand"]
        plus = values["serve_plus_1"]
        serve_total = values["serves"]
        fingerprint_rows.append(
            {
                "player_id": player_id,
                "player_name": player_name.get(player_id, "Unknown"),
                "matches": matches,
                "shots": shots,
                "forehand_frequency": values["forehand"] / wing_total if wing_total else np.nan,
                "backhand_frequency": values["backhand"] / wing_total if wing_total else np.nan,
                "net_approach_rate": values["net"] / shots,
                "winner_rate": values["winners"] / shots,
                "unforced_error_rate": values["unforced_errors"] / shots,
                "forced_error_rate": values["forced_errors"] / shots,
                "serve_plus_1_aggression": values["serve_plus_1_aggressive"] / plus if plus else np.nan,
                "average_rally_length": values["rally_length_sum"] / values["rally_points"],
                "serve_wide_rate": values["serve_wide"] / serve_total if serve_total else np.nan,
                "serve_body_rate": values["serve_body"] / serve_total if serve_total else np.nan,
                "serve_t_rate": values["serve_t"] / serve_total if serve_total else np.nan,
            }
        )
    fingerprints = pd.DataFrame(fingerprint_rows).dropna()
    fingerprints.to_csv(REPORTS_DIR / "player_style_fingerprints.csv", index=False)
    clustering = _cluster_and_compare(fingerprints)
    cluster_by_player = clustering.pop("cluster_by_player", {})
    _style_matchups(cluster_by_player)
    return {
        "players_with_fingerprint": len(fingerprints),
        "selected_clusters": clustering["selected_clusters"],
        "service_groups": len(service_frame),
        "pressure_points_with_exact_flags": sum(values["points"] for values in pressure.values()),
    }


def _cluster_and_compare(fingerprints: pd.DataFrame) -> dict[str, Any]:
    features = [
        "forehand_frequency",
        "backhand_frequency",
        "net_approach_rate",
        "winner_rate",
        "unforced_error_rate",
        "forced_error_rate",
        "serve_plus_1_aggression",
        "average_rally_length",
        "serve_wide_rate",
        "serve_body_rate",
        "serve_t_rate",
    ]
    if len(fingerprints) < 10:
        pd.DataFrame().to_csv(REPORTS_DIR / "player_style_clusters.csv", index=False)
        pd.DataFrame().to_csv(REPORTS_DIR / "player_style_similarity.csv", index=False)
        return {"selected_clusters": 0, "cluster_by_player": {}}
    raw = fingerprints[features].to_numpy(dtype=float)
    scaled = StandardScaler().fit_transform(raw)
    scores = []
    models: dict[int, KMeans] = {}
    for clusters in range(2, min(8, len(fingerprints) - 1) + 1):
        model = KMeans(n_clusters=clusters, random_state=42, n_init=20).fit(scaled)
        models[clusters] = model
        scores.append((clusters, silhouette_score(scaled, model.labels_)))
    selected, silhouette = max(scores, key=lambda item: item[1])
    labels = models[selected].labels_
    coords = PCA(n_components=2, random_state=42).fit_transform(scaled)
    output = fingerprints[["player_id", "player_name", "matches", "shots"]].copy()
    output["cluster"] = labels
    output["pca_1"] = coords[:, 0]
    output["pca_2"] = coords[:, 1]
    output.to_csv(REPORTS_DIR / "player_style_clusters.csv", index=False)

    similarity = cosine_similarity(raw)
    similarity_rows = []
    for index, row in fingerprints.reset_index(drop=True).iterrows():
        nearest = np.argsort(-similarity[index])
        rank = 0
        for other in nearest:
            if other == index:
                continue
            rank += 1
            similarity_rows.append(
                {
                    "player_id": row.player_id,
                    "player_name": row.player_name,
                    "rank": rank,
                    "similar_player_id": fingerprints.iloc[other].player_id,
                    "similar_player_name": fingerprints.iloc[other].player_name,
                    "cosine_similarity": similarity[index, other],
                }
            )
            if rank == 5:
                break
    pd.DataFrame(similarity_rows).to_csv(REPORTS_DIR / "player_style_similarity.csv", index=False)

    standardized = pd.DataFrame(scaled, columns=features)
    standardized["cluster"] = labels
    cluster_means = standardized.groupby("cluster")[features].mean()
    lines = [
        "# Clustering tactique des joueurs — Phase 9",
        "",
        f"Population: {len(fingerprints)} joueurs (minimum 10 matchs et 500 frappes).",
        f"K-Means sélectionné: k={selected}, silhouette={silhouette:.4f}. HDBSCAN n'est pas une dépendance du projet et n'a pas été forcé.",
        "",
        "Les descriptions ci-dessous sont générées après clustering à partir des écarts standardisés; ce ne sont pas des labels imposés.",
        "",
    ]
    for cluster, values in cluster_means.iterrows():
        extremes = values.reindex(values.abs().sort_values(ascending=False).index[:3])
        description = ", ".join(
            f"{'haut' if value > 0 else 'bas'} {name} ({value:+.2f}σ)"
            for name, value in extremes.items()
        )
        members = output.loc[output["cluster"] == cluster, "player_name"].head(10).tolist()
        lines.extend(
            [
                f"## Cluster {cluster}",
                "",
                f"Profil: {description}.",
                f"Exemples: {', '.join(members)}.",
                "",
            ]
        )
    (REPORTS_DIR / "player_style_clusters.md").write_text("\n".join(lines))
    return {
        "selected_clusters": selected,
        "silhouette": silhouette,
        "cluster_by_player": dict(zip(output["player_id"], output["cluster"], strict=True)),
    }


def _style_matchups(cluster_by_player: dict[str, int]) -> None:
    matchup: defaultdict[tuple[int, int, str], Counter[str]] = defaultdict(Counter)
    parquet = pq.ParquetFile(POINTS_PATH)
    for batch in parquet.iter_batches(
        columns=["surface", "server_id", "receiver_id", "server_won_point"],
        batch_size=150_000,
    ):
        frame = pa.Table.from_batches([batch]).to_pandas()
        for row in frame.itertuples(index=False):
            server_cluster = cluster_by_player.get(row.server_id)
            receiver_cluster = cluster_by_player.get(row.receiver_id)
            if server_cluster is None or receiver_cluster is None:
                continue
            key = (server_cluster, receiver_cluster, row.surface)
            matchup[key]["attempts"] += 1
            matchup[key]["wins"] += row.server_won_point
    rows = _aggregate_rows(
        matchup,
        ["server_style_cluster", "receiver_style_cluster", "surface"],
        minimum=500,
    )
    pd.DataFrame(rows).sort_values("attempts", ascending=False).to_csv(
        REPORTS_DIR / "shot_style_matchups.csv", index=False
    )
