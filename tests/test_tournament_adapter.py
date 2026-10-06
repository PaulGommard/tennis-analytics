from __future__ import annotations

import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from tennis_analytics.modeling.logistic import apply_platt
from tennis_analytics.paths import PROCESSED_DIR, PROJECT_ROOT
from tennis_analytics.tournament.engine import (
    predict_tournament_match,
    simulate_tournament,
)
from tennis_analytics.tournament.replay import AUDIT_PATH, MODEL_DIR, OUTPUT_PATH


@pytest.fixture(scope="module")
def context() -> pd.DataFrame:
    return pd.read_parquet(OUTPUT_PATH)


def test_round_state_is_frozen_and_contains_only_past_rounds(
    context: pd.DataFrame,
) -> None:
    grouped = context.groupby(
        ["tournament_edition_id", "round_order"], sort=False, observed=True
    )
    assert grouped["round_state_version"].nunique().eq(1).all()
    assert grouped["matches_observed_before_round"].nunique().eq(1).all()
    for (_, round_order), frame in list(grouped)[:500]:
        edition = context.loc[
            context["tournament_edition_id"] == frame.iloc[0]["tournament_edition_id"]
        ]
        expected = int((edition["round_order"] < round_order).sum())
        assert int(frame.iloc[0]["matches_observed_before_round"]) == expected
        assert int(frame.iloc[0]["tournament_matches_observed"]) == expected


def test_current_match_and_same_round_are_absent_from_player_state(
    context: pd.DataFrame,
) -> None:
    sampled_editions = context["tournament_edition_id"].drop_duplicates().iloc[::73].head(50)
    sample = context.loc[context["tournament_edition_id"].isin(sampled_editions)]
    for edition_id, edition in sample.groupby("tournament_edition_id", observed=True):
        del edition_id
        for row in edition.itertuples(index=False):
            past = edition.loc[edition["round_order"] < row.round_order]
            expected_a = int(
                (past["player_A_id"].astype(str) == str(row.player_A_id)).sum()
                + (past["player_B_id"].astype(str) == str(row.player_A_id)).sum()
            )
            expected_b = int(
                (past["player_A_id"].astype(str) == str(row.player_B_id)).sum()
                + (past["player_B_id"].astype(str) == str(row.player_B_id)).sum()
            )
            assert row.A_matches_played == expected_a
            assert row.B_matches_played == expected_b


def test_p_base_is_exact_frozen_prematch_probability(context: pd.DataFrame) -> None:
    audit = json.loads(AUDIT_PATH.read_text())
    digest = hashlib.sha256(
        context[["match_id", "p_base"]].to_csv(index=False).encode()
    ).hexdigest()
    assert digest == audit["base_probability_sha256"]

    prematch = joblib.load(PROJECT_ROOT / "models" / "prematch_v1" / "model.joblib")
    features = pd.read_parquet(PROCESSED_DIR / "match_features_v3.parquet").head(256)
    raw = prematch["pipeline"].predict_proba(features[prematch["feature_columns"]])[:, 1]
    exact = (
        apply_platt(prematch["calibrator"], raw)
        if prematch.get("calibrator") is not None
        else raw
    )
    persisted = context.set_index("match_id").loc[features["match_id"], "p_base"].to_numpy()
    np.testing.assert_allclose(persisted, exact, rtol=0, atol=1e-15)


def test_adapter_is_serializable_and_zero_context_preserves_base() -> None:
    artifact = joblib.load(MODEL_DIR / "model.joblib")
    assert artifact["p_base_contract"] == "exact frozen PRE-MATCH V1 probability"
    prediction = predict_tournament_match(0.63, {})
    assert prediction.p_base == 0.63
    assert prediction.p_adjusted == 0.63


def test_offset_is_symmetric_when_context_is_reversed(context: pd.DataFrame) -> None:
    artifact = joblib.load(MODEL_DIR / "model.joblib")
    row = context.loc[context["matches_observed_before_round"] > 20].iloc[0]
    original = {feature: float(row[feature]) for feature in artifact["feature_columns"]}
    reversed_context = {feature: -value for feature, value in original.items()}
    forward = predict_tournament_match(float(row["p_base"]), original).p_adjusted
    reverse = predict_tournament_match(1 - float(row["p_base"]), reversed_context).p_adjusted
    assert forward + reverse == pytest.approx(1.0, abs=1e-12)


def test_simulation_is_reproducible_and_stage_sums_are_coherent() -> None:
    pairs = [("A", "B"), ("C", "D"), ("E", "F"), ("G", "H")]

    def probability(player_a: str, player_b: str) -> float:
        del player_a, player_b
        return 0.57

    first = simulate_tournament(pairs, probability, n_simulations=2_000, seed=123)
    second = simulate_tournament(pairs, probability, n_simulations=2_000, seed=123)
    assert first == second
    assert first["stage_probability_sums"] == pytest.approx(
        {"QF": 8.0, "SF": 4.0, "F": 2.0, "Champion": 1.0}
    )


def test_pre_tournament_simulation_does_not_require_adapter_context() -> None:
    result = simulate_tournament(
        [("A", "B")],
        lambda player_a, player_b: 0.7,
        n_simulations=1_000,
        seed=5,
        mode="pre_tournament",
    )
    assert result["future_context_policy"] == "PRE-MATCH V1 only"
    assert result["players"]["A"]["P(Champion)"] == pytest.approx(0.7, abs=0.04)
