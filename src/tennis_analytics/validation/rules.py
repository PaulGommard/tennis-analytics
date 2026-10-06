from __future__ import annotations

import pandas as pd

from tennis_analytics.preprocessing.parsing import valid_score


def duplicate_primary_keys(frame: pd.DataFrame, columns: list[str]) -> int:
    return int(frame.duplicated(columns, keep=False).sum())


def winner_not_in_participants(matches: pd.DataFrame, participants: pd.DataFrame) -> int:
    membership = participants.groupby("match_id")["player_id"].agg(set)
    aligned = matches.set_index("match_id")
    return sum(
        winner not in membership.get(match_id, set())
        for match_id, winner in aligned["winner_player_id"].items()
    )


def self_matches(participants: pd.DataFrame) -> int:
    distinct = participants.groupby("match_id")["player_id"].nunique(dropna=True)
    return int((distinct < 2).sum())


def negative_statistics(stats: pd.DataFrame) -> int:
    numeric = stats.drop(columns=["match_id", "player_id"], errors="ignore")
    return int((numeric.apply(pd.to_numeric, errors="coerce") < 0).sum().sum())


def invalid_rankings(participants: pd.DataFrame) -> int:
    rank = pd.to_numeric(participants["rank"], errors="coerce")
    return int((rank.notna() & (rank < 1)).sum())


def invalid_scores(matches: pd.DataFrame) -> int:
    return int((~matches["score"].map(valid_score)).sum())
