import pandas as pd

from tennis_analytics.preprocessing.normalize import normalize
from tennis_analytics.validation.rules import (
    invalid_rankings,
    negative_statistics,
    self_matches,
    winner_not_in_participants,
)


def raw_match() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "tourney_id": "2024-001",
                "tourney_name": "Example Open",
                "surface": "hard",
                "draw_size": 32,
                "tourney_level": "A",
                "tourney_date": 20240101,
                "match_num": 1,
                "winner_id": 200,
                "winner_name": "Winner Person",
                "winner_hand": "R",
                "winner_ht": 190,
                "winner_ioc": "FRA",
                "winner_age": 25.1,
                "winner_rank": 10,
                "winner_rank_points": 3000,
                "loser_id": 100,
                "loser_name": "Other Person",
                "loser_hand": "L",
                "loser_ht": 180,
                "loser_ioc": "ESP",
                "loser_age": 27.2,
                "loser_rank": 20,
                "loser_rank_points": 1800,
                "score": "6-4 6-4",
                "best_of": 3,
                "round": "R32",
                "minutes": 90,
                "w_ace": 8,
                "w_df": 2,
                "w_svpt": 60,
                "w_1stIn": 40,
                "w_1stWon": 30,
                "w_2ndWon": 12,
                "w_SvGms": 10,
                "w_bpSaved": 3,
                "w_bpFaced": 4,
                "l_ace": 3,
                "l_df": 4,
                "l_svpt": 65,
                "l_1stIn": 38,
                "l_1stWon": 25,
                "l_2ndWon": 10,
                "l_SvGms": 10,
                "l_bpSaved": 5,
                "l_bpFaced": 8,
                "_source_file": "atp_matches_2024.csv",
                "_source_row": 2,
            }
        ]
    )


def test_slots_do_not_encode_winner():
    tables = normalize(raw_match(), pd.DataFrame())
    participants = tables["match_participants"]
    winner = participants.loc[participants["is_winner"]].iloc[0]
    loser = participants.loc[~participants["is_winner"]].iloc[0]
    assert winner["slot"] == "B"
    assert loser["slot"] == "A"
    assert tables["matches"].iloc[0]["winner_player_id"] == "atp:200"


def test_core_integrity_rules():
    tables = normalize(raw_match(), pd.DataFrame())
    assert tables["players"]["player_id"].is_unique
    assert tables["matches"]["match_id"].is_unique
    assert winner_not_in_participants(tables["matches"], tables["match_participants"]) == 0
    assert self_matches(tables["match_participants"]) == 0
    assert negative_statistics(tables["match_stats"]) == 0
    assert invalid_rankings(tables["match_participants"]) == 0
    assert set(tables["match_stats"]["service_points"]) == {60, 65}


def test_missing_source_id_is_visible_and_not_fuzzy_matched():
    raw = raw_match()
    raw.loc[0, "winner_id"] = None
    tables = normalize(raw, pd.DataFrame())
    synthetic = tables["match_participants"].query("id_source == 'synthetic'")
    assert len(synthetic) == 1
    assert synthetic.iloc[0]["player_id"].startswith("synthetic:")
