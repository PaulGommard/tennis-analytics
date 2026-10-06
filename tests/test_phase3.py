from pathlib import Path

import pandas as pd
import pytest

from tennis_analytics.ratings.chronology import apply_round_chronology
from tennis_analytics.ratings.elo import EloConfig
from tennis_analytics.ratings.engine import InactivityConfig, run_elo_engine
from tennis_analytics.ratings.tuning import (
    Phase3Config,
    assert_validation_only,
    fit_platt,
    select_best,
)


def match(
    match_id: str,
    date: str,
    edition: str,
    round_name: str,
    player_a: str,
    player_b: str,
    a_wins: int,
    *,
    level: str = "A",
    surface: str = "Hard",
) -> dict[str, object]:
    return {
        "match_id": match_id,
        "date": pd.Timestamp(date),
        "tournament": edition,
        "tournament_edition_id": edition,
        "level": level,
        "surface": surface,
        "round": round_name,
        "player_A_id": player_a,
        "player_B_id": player_b,
        "player_A_win": a_wins,
        "is_elo_eligible": True,
    }


@pytest.fixture
def chronology_config(tmp_path: Path) -> Path:
    path = tmp_path / "elo.toml"
    path.write_text(
        """[chronology]
ordered_rounds = ["RR", "R32", "R16", "QF", "SF", "F"]
conservative_levels = ["D"]
""",
        encoding="utf-8",
    )
    return path


def knockout_timeline() -> pd.DataFrame:
    return pd.DataFrame(
        [
            match("m1", "2020-01-01", "open-2020", "R32", "A", "B", 1),
            match("m2", "2020-01-01", "open-2020", "R32", "C", "D", 1),
            match("m3", "2020-01-01", "open-2020", "R16", "A", "C", 1),
        ]
    )


def search_config() -> Phase3Config:
    return Phase3Config(
        k_values=(32.0,),
        scale_values=(400.0,),
        surface_c_values=(20.0,),
        inactivity_threshold_days=(90,),
        inactivity_strengths=(0.1,),
        validation_start=pd.Timestamp("2019-01-01"),
        validation_end=pd.Timestamp("2022-12-31"),
        test_start=pd.Timestamp("2023-01-01"),
    )


def test_same_round_is_frozen_and_next_round_sees_updates(chronology_config: Path):
    chronology = apply_round_chronology(knockout_timeline(), config_path=chronology_config)
    rows = chronology.set_index("match_id")
    assert rows.loc["m1", "chronology_batch_id"] == rows.loc["m2", "chronology_batch_id"]
    assert rows.loc["m3", "chronology_batch_id"] != rows.loc["m1", "chronology_batch_id"]

    ratings = run_elo_engine(chronology, EloConfig()).set_index("match_id")
    assert ratings.loc["m1", "elo_A"] == 1500
    assert ratings.loc["m2", "elo_A"] == 1500
    assert ratings.loc["m3", "elo_A"] == 1516
    assert ratings.loc["m3", "surface_matches_A"] == 1


def test_future_round_result_cannot_change_prior_round_features(chronology_config: Path):
    original = knockout_timeline()
    changed = original.copy()
    changed.loc[changed["match_id"] == "m3", "player_A_win"] = 0
    first = run_elo_engine(
        apply_round_chronology(original, config_path=chronology_config), EloConfig()
    )
    second = run_elo_engine(
        apply_round_chronology(changed, config_path=chronology_config), EloConfig()
    )
    feature_columns = [
        column for column in first.columns if column not in {"match_id", "player_A_win"}
    ]
    pd.testing.assert_frame_equal(
        first.loc[first["round"] == "R32", feature_columns].reset_index(drop=True),
        second.loc[second["round"] == "R32", feature_columns].reset_index(drop=True),
    )


def test_davis_cup_and_cross_edition_overlap_stay_conservative(chronology_config: Path):
    davis = pd.DataFrame(
        [
            match("d1", "2020-02-01", "tie", "RR", "A", "B", 1, level="D"),
            match("d2", "2020-02-01", "tie", "F", "C", "D", 1, level="D"),
        ]
    )
    davis_rows = apply_round_chronology(davis, config_path=chronology_config)
    assert davis_rows["chronology_batch_id"].nunique() == 1
    assert set(davis_rows["chronology_strategy"]) == {"tournament_conservative"}

    overlap = pd.DataFrame(
        [
            match("o1", "2020-03-01", "event-1", "R32", "A", "B", 1),
            match("o2", "2020-03-01", "event-2", "R16", "A", "C", 1),
        ]
    )
    overlap_rows = apply_round_chronology(overlap, config_path=chronology_config)
    assert overlap_rows["chronology_batch_id"].nunique() == 1
    assert set(overlap_rows["chronology_strategy"]) == {"date_overlap_conservative"}


def test_inactivity_regression_is_applied_before_return_match():
    timeline = pd.DataFrame(
        [
            match("m1", "2020-01-01", "e1", "R32", "A", "B", 1),
            match("m2", "2020-04-11", "e2", "R32", "A", "C", 1),
        ]
    )
    timeline["chronology_order"] = [0, 1]
    timeline["chronology_batch_id"] = [0, 1]
    timeline["chronology_stage"] = "R32"
    timeline["chronology_strategy"] = "round"
    ratings = run_elo_engine(
        timeline,
        EloConfig(),
        InactivityConfig(threshold_days=90, strength=0.5),
    ).set_index("match_id")
    assert ratings.loc["m2", "elo_A"] == 1508
    assert ratings.loc["m2", "surface_elo_A"] == 1508


def test_hyperparameter_selection_uses_validation_metrics_only():
    trials = pd.DataFrame(
        [
            {
                "name": "validation-winner",
                "validation_log_loss": 0.60,
                "validation_brier": 0.21,
                "validation_accuracy": 0.64,
                "test_log_loss": 0.99,
            },
            {
                "name": "test-winner",
                "validation_log_loss": 0.61,
                "validation_brier": 0.20,
                "validation_accuracy": 0.65,
                "test_log_loss": 0.50,
            },
        ]
    )
    assert select_best(trials)["name"] == "validation-winner"


def test_validation_guards_reject_test_rows():
    config = search_config()
    dates = pd.Series(pd.to_datetime(["2022-12-31", "2023-01-01"]))
    with pytest.raises(ValueError, match="outside"):
        assert_validation_only(dates, config)
    with pytest.raises(ValueError, match="outside"):
        fit_platt(
            pd.Series([0.4, 0.6]),
            pd.Series([0, 1]),
            dates,
            config,
        )
