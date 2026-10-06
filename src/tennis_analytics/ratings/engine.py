from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tennis_analytics.ratings.elo import EloConfig


@dataclass(frozen=True)
class InactivityConfig:
    threshold_days: int
    strength: float


def apply_date_chronology(timeline: pd.DataFrame) -> pd.DataFrame:
    data = timeline.loc[timeline["is_elo_eligible"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data = data.sort_values(["date", "match_id"], kind="stable", ignore_index=True)
    data["chronology_batch_id"] = pd.factorize(data["date"], sort=False)[0]
    data["chronology_order"] = data["chronology_batch_id"]
    data["chronology_stage"] = "DATE"
    data["chronology_strategy"] = "date_conservative"
    return data


def run_elo_engine(
    matches: pd.DataFrame,
    config: EloConfig,
    inactivity: InactivityConfig | None = None,
) -> pd.DataFrame:
    """Compute pre-match global/surface Elo using pre-assigned frozen batches."""
    required = {
        "match_id",
        "date",
        "chronology_batch_id",
        "player_A_id",
        "player_B_id",
        "player_A_win",
        "surface",
    }
    missing = required.difference(matches.columns)
    if missing:
        raise ValueError(f"Missing Elo engine columns: {sorted(missing)}")
    data = matches.sort_values(["chronology_order", "match_id"], kind="stable").reset_index(
        drop=True
    )
    count = len(data)
    all_players = pd.concat([data["player_A_id"], data["player_B_id"]], ignore_index=True)
    player_labels, player_codes = pd.factorize(all_players, sort=True)
    # pandas returns (codes, uniques), despite the deliberately explicit names.
    codes = player_labels
    players = player_codes
    a_codes = codes[:count]
    b_codes = codes[count:]
    player_count = len(players)
    surface_lookup = {surface: index for index, surface in enumerate(config.surfaces)}
    surface_codes = data["surface"].map(surface_lookup).fillna(-1).astype(int).to_numpy()
    outcomes = data["player_A_win"].astype(float).to_numpy()
    date_days = pd.to_datetime(data["date"]).to_numpy(dtype="datetime64[D]").astype("int64")
    batch_values = data["chronology_batch_id"].to_numpy()
    starts = np.r_[0, np.flatnonzero(batch_values[1:] != batch_values[:-1]) + 1]
    ends = np.r_[starts[1:], count]

    ratings = np.full(player_count, config.initial_rating, dtype=float)
    surface_ratings = np.full(
        (player_count, len(config.surfaces)), config.initial_rating, dtype=float
    )
    career_matches = np.zeros(player_count, dtype=np.int64)
    surface_matches = np.zeros((player_count, len(config.surfaces)), dtype=np.int64)
    last_day = np.full(player_count, np.iinfo(np.int64).min, dtype=np.int64)

    global_a = np.empty(count, dtype=float)
    global_b = np.empty(count, dtype=float)
    surface_a = np.full(count, np.nan, dtype=float)
    surface_b = np.full(count, np.nan, dtype=float)
    surface_count_a = np.zeros(count, dtype=np.int64)
    surface_count_b = np.zeros(count, dtype=np.int64)
    career_a = np.zeros(count, dtype=np.int64)
    career_b = np.zeros(count, dtype=np.int64)
    days_a = np.full(count, np.nan, dtype=float)
    days_b = np.full(count, np.nan, dtype=float)
    expected_global = np.empty(count, dtype=float)
    expected_surface = np.full(count, np.nan, dtype=float)

    for start, end in zip(starts, ends, strict=True):
        index = slice(start, end)
        a = a_codes[index]
        b = b_codes[index]
        y = outcomes[index]
        day = date_days[start]
        participants = np.unique(np.concatenate([a, b]))

        if inactivity is not None:
            seen = last_day[participants] != np.iinfo(np.int64).min
            inactive = np.zeros(len(participants), dtype=bool)
            inactive[seen] = day - last_day[participants[seen]] > inactivity.threshold_days
            decayed_players = participants[inactive]
            if len(decayed_players):
                keep = 1.0 - inactivity.strength
                ratings[decayed_players] = (
                    config.initial_rating
                    + (ratings[decayed_players] - config.initial_rating) * keep
                )
                surface_ratings[decayed_players, :] = (
                    config.initial_rating
                    + (surface_ratings[decayed_players, :] - config.initial_rating) * keep
                )

        ra = ratings[a]
        rb = ratings[b]
        probability = 1.0 / (1.0 + np.power(10.0, (rb - ra) / config.scale))
        global_a[index] = ra
        global_b[index] = rb
        expected_global[index] = probability
        career_a[index] = career_matches[a]
        career_b[index] = career_matches[b]
        seen_a = last_day[a] != np.iinfo(np.int64).min
        seen_b = last_day[b] != np.iinfo(np.int64).min
        local_days_a = np.full(len(a), np.nan)
        local_days_b = np.full(len(b), np.nan)
        local_days_a[seen_a] = day - last_day[a[seen_a]]
        local_days_b[seen_b] = day - last_day[b[seen_b]]
        days_a[index] = local_days_a
        days_b[index] = local_days_b

        valid_surface = surface_codes[index] >= 0
        if valid_surface.any():
            positions = np.flatnonzero(valid_surface)
            surfaces = surface_codes[index][valid_surface]
            valid_a = a[valid_surface]
            valid_b = b[valid_surface]
            rsa = surface_ratings[valid_a, surfaces]
            rsb = surface_ratings[valid_b, surfaces]
            surface_a[start + positions] = rsa
            surface_b[start + positions] = rsb
            surface_count_a[start + positions] = surface_matches[valid_a, surfaces]
            surface_count_b[start + positions] = surface_matches[valid_b, surfaces]
            expected_surface[start + positions] = 1.0 / (
                1.0 + np.power(10.0, (rsb - rsa) / config.scale)
            )

        delta = config.k_factor * (y - probability)
        np.add.at(ratings, a, delta)
        np.add.at(ratings, b, -delta)
        np.add.at(career_matches, a, 1)
        np.add.at(career_matches, b, 1)

        if valid_surface.any():
            surface_probability = expected_surface[start:end][valid_surface]
            surface_delta = config.k_factor * (y[valid_surface] - surface_probability)
            surfaces = surface_codes[index][valid_surface]
            valid_a = a[valid_surface]
            valid_b = b[valid_surface]
            np.add.at(surface_ratings, (valid_a, surfaces), surface_delta)
            np.add.at(surface_ratings, (valid_b, surfaces), -surface_delta)
            np.add.at(surface_matches, (valid_a, surfaces), 1)
            np.add.at(surface_matches, (valid_b, surfaces), 1)
        last_day[participants] = day

    result = data[
        [
            "match_id",
            "date",
            "surface",
            "level",
            "round",
            "player_A_id",
            "player_B_id",
            "player_A_win",
            "chronology_batch_id",
            "chronology_stage",
            "chronology_strategy",
        ]
    ].copy()
    result["elo_A"] = global_a
    result["elo_B"] = global_b
    result["elo_diff"] = global_a - global_b
    result["elo_expected_A"] = expected_global
    result["surface_elo_A"] = surface_a
    result["surface_elo_B"] = surface_b
    result["surface_elo_diff"] = surface_a - surface_b
    result["surface_elo_expected_A"] = expected_surface
    result["surface_matches_A"] = surface_count_a
    result["surface_matches_B"] = surface_count_b
    result["career_matches_A"] = career_a
    result["career_matches_B"] = career_b
    result["days_since_last_match_A"] = days_a
    result["days_since_last_match_B"] = days_b
    return result


def blended_surface_probability(
    ratings: pd.DataFrame,
    *,
    c_value: float,
    scale: float,
) -> pd.Series:
    weight_a = ratings["surface_matches_A"] / (ratings["surface_matches_A"] + c_value)
    weight_b = ratings["surface_matches_B"] / (ratings["surface_matches_B"] + c_value)
    blended_a = weight_a * ratings["surface_elo_A"] + (1 - weight_a) * ratings["elo_A"]
    blended_b = weight_b * ratings["surface_elo_B"] + (1 - weight_b) * ratings["elo_B"]
    probability = 1.0 / (1.0 + np.power(10.0, (blended_b - blended_a) / scale))
    global_probability = 1.0 / (1.0 + np.power(10.0, (ratings["elo_B"] - ratings["elo_A"]) / scale))
    # Unknown surfaces have no surface evidence and therefore reduce exactly
    # to the global rating instead of becoming unevaluable.
    return probability.where(
        ratings["surface_elo_A"].notna() & ratings["surface_elo_B"].notna(),
        global_probability,
    )
