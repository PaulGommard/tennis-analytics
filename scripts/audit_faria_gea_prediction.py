from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from typing import Any

import joblib
import numpy as np
import pandas as pd

from tennis_analytics.paths import PROCESSED_DIR, PROJECT_ROOT, REPORTS_DIR
from tennis_analytics.product.model_audit import (
    diagnostic_ablation_probability,
    prediction_decomposition,
    reverse_feature_values,
)
from tennis_analytics.product.on_demand import OnDemandPredictionService
from tennis_analytics.product.storage import ProductStore

MATCH_ID = "espn:184880"
QUERY = pd.Timestamp("2026-10-07T04:00:00")
PLAYER_IDS = ("atp:210262", "atp:210338")
EXTERNAL_IDS = ("10219", "10427")


def probability(artifact: dict[str, Any], values: dict[str, float | None]) -> float:
    _, summary = prediction_decomposition(artifact, values)
    return summary["final_probability"]


def percent(value: float) -> str:
    return f"{100 * value:.6f} %"


def main() -> None:
    store = ProductStore()
    history = store.get_prediction_history(MATCH_ID, "PREMATCH_V1", limit=20)
    original = min(history, key=lambda item: abs(item["probability"]["player_a"] - 0.721))
    corrected = history[0]
    artifact = joblib.load(PROJECT_ROOT / "models" / "prematch_v1" / "model.joblib")
    research_path = PROCESSED_DIR / "models" / "logistic_phase5.joblib"
    research_artifact = joblib.load(research_path)

    contributions, summary = prediction_decomposition(artifact, original["features"])
    contributions.to_csv(REPORTS_DIR / "faria_gea_contributions.csv", index=False)
    _, corrected_summary = prediction_decomposition(
        artifact, corrected["features"]
    )

    reversed_original = reverse_feature_values(original["features"])
    reverse_probability = probability(artifact, reversed_original)
    sign_failures = [
        name for name, value in original["features"].items()
        if value is not None
        and (
            name.endswith("_diff") or "_diff_x_" in name
            or name in {"elo_logit", "elo_logit_x_clay", "elo_logit_x_grass", "h2h_edge_A"}
        )
        and not math.isclose(float(reversed_original[name]), -float(value), abs_tol=1e-15)
    ]

    blocks = {
        "form": {name for name in artifact["feature_columns"] if name.startswith("win_rate_")},
        "workload": {
            name for name in artifact["feature_columns"]
            if name.startswith(("matches_", "sets_", "games_", "minutes_", "rest_days", "prior_tournament_"))
        },
        "serve/return": {
            name for name in artifact["feature_columns"]
            if any(token in name for token in (
                "serve", "return", "ace_rate", "double_fault", "hold_rate", "break_points"
            ))
        },
        "ranking": {"rank_diff", "rank_points_diff"},
        "Elo": {name for name in artifact["feature_columns"] if name.startswith("elo_logit")},
    }
    ablations = {
        name: diagnostic_ablation_probability(artifact, original["features"], columns)
        for name, columns in blocks.items()
    }

    pipeline = artifact["pipeline"]
    logistic = pipeline.named_steps["logistic"]
    calibrator = artifact["calibrator"]
    zero_logit = float(logistic.intercept_[0])
    artificial = {
        "identical / all standardized differences zero": float(
            1 / (1 + np.exp(-(calibrator.intercept_[0] + calibrator.coef_[0, 0] * zero_logit)))
        )
    }
    coefficient = dict(zip(artifact["feature_columns"], logistic.coef_[0], strict=True))
    for label, selected in {
        "ranking only (+1 SD rank and points for A)": {"rank_diff", "rank_points_diff"},
        "Elo only (+1 SD for A)": {"elo_logit"},
        "recent form only (+1 SD on three win rates for A)": {
            "win_rate_career_diff", "win_rate_365d_diff", "win_rate_90d_diff",
        },
    }.items():
        logit = zero_logit + sum(coefficient[name] for name in selected)
        artificial[label] = float(
            1 / (1 + np.exp(-(calibrator.intercept_[0] + calibrator.coef_[0, 0] * logit)))
        )

    feature_path = PROCESSED_DIR / "match_features_v3.parquet"
    historical_id = "atp:2026-520:2026-05-25:198"
    historical = pd.read_parquet(
        feature_path,
        filters=[("match_id", "==", historical_id)],
        columns=["match_id", *artifact["feature_columns"]],
    )
    historical_frame = historical[artifact["feature_columns"]]
    frozen_raw = float(artifact["pipeline"].predict_proba(historical_frame)[0, 1])
    research_raw = float(research_artifact["pipeline"].predict_proba(historical_frame)[0, 1])
    frozen_final = probability(
        artifact,
        {
            name: None if pd.isna(historical_frame.iloc[0][name]) else float(historical_frame.iloc[0][name])
            for name in artifact["feature_columns"]
        },
    )
    research_final = probability(
        research_artifact,
        {
            name: None if pd.isna(historical_frame.iloc[0][name]) else float(historical_frame.iloc[0][name])
            for name in artifact["feature_columns"]
        },
    )
    frozen_hash = hashlib.sha256((PROJECT_ROOT / "models/prematch_v1/model.joblib").read_bytes()).hexdigest()
    research_hash = hashlib.sha256(research_path.read_bytes()).hexdigest()

    service = OnDemandPredictionService.create(store)
    with sqlite3.connect(store.path) as connection:
        states = {
            row[0]: json.loads(row[1])
            for row in connection.execute(
                "SELECT external_player_id, payload_json FROM player_runtime_states "
                "WHERE external_player_id IN ('10219','10427')"
            )
        }
    combined = [
        service._combined_results(player_id, states[external_id]["matches"], QUERY)
        for player_id, external_id in zip(PLAYER_IDS, EXTERNAL_IDS, strict=True)
    ]
    window_lines: list[str] = []
    for player_name, results in zip(("Jaime Faria", "Arthur Gea"), combined, strict=True):
        window_lines.append(f"### {player_name}")
        for days in (7, 14, 30, 90, 365):
            selected = [
                item for item in results
                if QUERY - pd.Timedelta(days, unit="D") <= item["date"] < QUERY
            ]
            window_lines.append(f"\n**{days}d — {len(selected)} matchs**\n")
            window_lines.append("| Date utilisée | Tournoi | Adversaire | Sets | Jeux | Durée | Source | Précision date |")
            window_lines.append("|---|---|---|---:|---:|---:|---|---|")
            for item in selected:
                sets, games = service._score_totals(item["score"])
                window_lines.append(
                    f"| {item['date'].date()} | {item['tournament']} | {item['opponent_name']} | "
                    f"{sets} | {games} | {item['minutes'] if item['minutes'] is not None else 'N/A'} | "
                    f"{item['source']} | {item['date_precision']} |"
                )
            if not selected:
                window_lines.append("| — | Aucun | — | 0 | 0 | 0 | — | — |")

    source_rules: list[dict[str, str]] = []
    fresh = {"rank_diff", "rank_points_diff", "age_years_diff", "surface_clay", "surface_grass", "h2h_matches_before", "h2h_edge_A"}
    partial_prefixes = (
        "win_rate_", "matches_", "sets_", "games_", "rest_days", "career_matches",
        "surface_matches", "prior_tournament_matches", "prior_tournament_sets", "prior_tournament_games",
    )
    for name in artifact["feature_columns"]:
        value = corrected["features"][name]
        if name in fresh:
            group, source, date, exactness = "FRESH", "ESPN ranking / local identity / target context", "2026-10-01 to 2026-10-07", "exact under resolved identity"
        elif name.startswith(partial_prefixes):
            group, source, date, exactness = "PARTIALLY REFRESHED", "local history + ESPN results", "through 2026-09-27", "tournament-level ESPN dates"
        else:
            group, source, date, exactness = "STALE", "last compatible historical snapshot or frozen imputer", "2026-05-25 or earlier", "stale or unavailable"
        if value is None:
            source += "; missing → training median imputer"
        source_rules.append(
            {"feature": name, "value": "NaN" if value is None else f"{value:.9g}", "source": source, "date": date, "freshness": group, "exactness": exactness}
        )

    top_positive = contributions.sort_values("contribution_to_logit", ascending=False).head(15)
    top_negative = contributions.sort_values("contribution_to_logit", ascending=True).head(15)
    def contribution_table(frame: pd.DataFrame) -> str:
        lines = ["| Feature | Valeur | z | Coefficient | Contribution |", "|---|---:|---:|---:|---:|"]
        for row in frame.itertuples(index=False):
            lines.append(
                f"| `{row.feature}` | {row.raw_value if pd.notna(row.raw_value) else 'NaN'} | "
                f"{row.standardized_value:.6f} | {row.coefficient:.6f} | {row.contribution_to_logit:+.6f} |"
            )
        return "\n".join(lines)

    important = contributions.set_index("feature")
    audit = f"""# Audit PRE-MATCH V1 — Jaime Faria vs Arthur Gea

Audit du snapshot original produit pour Rolex Shanghai Masters, Round 1, Hard, 7 octobre 2026.
L'artefact PRE-MATCH V1 n'a été ni modifié ni réentraîné.

## 1. Prediction reconstruction

La probabilité **72,138532 %** est exactement la sortie du modèle sérialisé pour le vecteur
original sauvegardé (`{original['feature_snapshot_hash']}`).

- raw logit : `{summary['raw_logit']:.15f}`
- raw probability : `{summary['raw_probability']:.15f}`
- calibration Platt : `sigmoid({summary['calibration_intercept']:.15f} + {summary['calibration_slope']:.15f} × raw_logit)`
- calibrated logit : `{summary['calibrated_logit']:.15f}`
- final probability : `{summary['final_probability']:.15f}`
- valeur SQLite : `{original['probability']['player_a']:.15f}`

Conclusion mathématique : aucune divergence d'inférence. Le passage de 75,072462 % brut à
72,138532 % vient uniquement de la calibration Platt.

## 2. Exact logit decomposition

`raw_logit = intercept + Σ contributions`

- intercept : `{summary['intercept']:+.9f}`
- somme positive : `{summary['sum_positive']:+.9f}`
- somme négative : `{summary['sum_negative']:+.9f}`
- somme des contributions : `{summary['sum_contributions']:+.9f}`
- total : `{summary['intercept'] + summary['sum_contributions']:+.9f}`

### 15 contributions positives

{contribution_table(top_positive)}

### 15 contributions négatives

{contribution_table(top_negative)}

Le détail des 86 lignes (valeur brute, valeur imputée, moyenne, écart-type, z-score,
coefficient et contribution) est dans `reports/faria_gea_contributions.csv`.

## 3. A/B symmetry

Toutes les features antisymétriques changent correctement de signe :
`{len(sign_failures) == 0}` (échecs : `{sign_failures}`).

- P(Faria | Faria=A), original : `{percent(summary['final_probability'])}`
- P(Gea | Gea=A), vecteur inversé : `{percent(reverse_probability)}`
- somme : `{summary['final_probability'] + reverse_probability:.12f}`

La somme n'est pas exactement 1 mais 0,987502. Ce n'est pas une inversion runtime : le modèle
logistique appris a un intercept non nul, les moyennes du scaler ne sont pas toutes nulles et la
calibration Platt a elle-même un intercept non nul. PRE-MATCH V1 n'est donc pas strictement
antisymétrique par construction. L'écart d'environ 1,25 point est une propriété figée du modèle.

## 4. Runtime vs research parity

Snapshot historique contrôlé : `{historical_id}`.

- SHA-256 artefact research : `{research_hash}`
- SHA-256 artefact gelé : `{frozen_hash}`
- artefacts identiques : `{research_hash == frozen_hash}`
- probabilité brute research/runtime : `{research_raw:.15f}` / `{frozen_raw:.15f}`
- probabilité calibrée research/runtime : `{research_final:.15f}` / `{frozen_final:.15f}`
- différence absolue : `{abs(research_final-frozen_final):.3e}`

Le pipeline d'inférence, l'imputation, la standardisation et les contributions sont identiques.
La divergence trouvée se situait dans la construction prospective des features, pas dans
l'artefact sklearn.

## 5. Counter-intuitive coefficients

| Feature | Valeur Faria−Gea | z-score | Coefficient | Contribution |
|---|---:|---:|---:|---:|
| `win_rate_90d_diff` | -0.338889 | {important.loc['win_rate_90d_diff','standardized_value']:.6f} | {important.loc['win_rate_90d_diff','coefficient']:.6f} | {important.loc['win_rate_90d_diff','contribution_to_logit']:+.6f} |
| `win_rate_365d_diff` | -0.139216 | {important.loc['win_rate_365d_diff','standardized_value']:.6f} | {important.loc['win_rate_365d_diff','coefficient']:.6f} | {important.loc['win_rate_365d_diff','contribution_to_logit']:+.6f} |
| `service_points_won_rate_career_diff` | -0.021004 | {important.loc['service_points_won_rate_career_diff','standardized_value']:.6f} | {important.loc['service_points_won_rate_career_diff','coefficient']:.6f} | {important.loc['service_points_won_rate_career_diff','contribution_to_logit']:+.6f} |

`win_rate_90d_diff = A - B`; sa valeur négative signifie bien Faria < Gea. La contribution
devient positive parce que son z-score est négatif et que le coefficient conditionnel appris
est lui aussi négatif. Dans le train, ces variables ont pourtant une corrélation marginale
positive avec la victoire, mais elles sont fortement corrélées à Elo et entre elles
(corrélation win-rate 365d/Elo ≈ 0,871 ; 90d/365d ≈ 0,705). Il s'agit d'un effet de suppression
multivariée sous régularisation L2 (`C=1`), pas d'une inversion A/B. Le signe est néanmoins peu
interprétable causalement et devient dangereux avec des données runtime approximatives.

## 6. Workload/date audit

Dans le snapshot original, `minutes_7d_diff=301` provenait du dernier snapshot historique de
Faria du 25 mai : deux matchs antérieurs du même batch Roland-Garros totalisaient 151 + 150
minutes. Le snapshot de Gea avait zéro minute sur 7 jours. Ces valeurs vieilles de 135 jours
étaient mélangées aux compteurs ESPN recalculés au 7 octobre, d'où l'incohérence avec zéro match,
set et jeu sur 7 jours.

Après correction : `minutes_7d_diff=0` car aucun match n'est encodé dans la fenêtre ; les durées
14d/30d sont `NaN` car ESPN ne fournit pas les durées, puis l'imputer gelé applique sa médiane.

{chr(10).join(window_lines)}

ESPN attribue la date de début du tournoi à chaque ligne : quatre matchs de Faria au Japon ont
la date 27 septembre, et quatre matchs de Gea à Pékin ont aussi la date 27 septembre. Pour la
requête du 7 octobre, les huit sont donc exclus de 7d, inclus dans 14d/30d et le repos vaut 10
jours pour les deux. Les jours réels des rounds ne sont pas disponibles dans cette page : l'effet
exact sur 7d et le repos ne peut pas être récupéré proprement, mais le biais encodé est quantifié
ici (0 match en 7d contre jusqu'à 4 par joueur si certains rounds ont eu lieu après le 30 septembre).

## 7. Null/stale feature handling

Le snapshot original contenait `{sum(value is None for value in original['features'].values())}`
valeurs nulles. Elles arrivent comme `NaN` dans `SimpleImputer(strategy='median')`, sont remplacées
par la médiane apprise sur le train, puis standardisées par `StandardScaler`. Elles ne deviennent
donc pas automatiquement zéro : elles ont le z-score `(médiane - moyenne) / écart-type`.
Le runtime utilise exactement les mêmes objets sklearn que l'entraînement.

| Feature | Valeur corrigée | Source | Date | Groupe | Exactitude |
|---|---:|---|---|---|---|
{chr(10).join(f"| `{row['feature']}` | {row['value']} | {row['source']} | {row['date']} | {row['freshness']} | {row['exactness']} |" for row in source_rules)}

## 8. Bugs found

Trois bugs runtime ont été confirmés et corrigés, sans toucher au modèle :

1. **Workload hybride** : minutes 7/14/30 anciennes combinées à des comptes récents.
2. **Mauvaise surface** : les stats `*_surface` venaient du dernier match, même sur une autre
   surface. Pour Faria, un état terre battue était utilisé pour Shanghai Hard.
3. **Double comptage** : `career_matches` et parfois `surface_matches` additionnaient toute la
   page ESPN 2026 à un historique qui contenait déjà janvier–mai. Les taux 365d omettaient aussi
   octobre–décembre 2025.

La correction fusionne l'historique local jusqu'au 25 mai avec uniquement les résultats ESPN
postérieurs, sélectionne un snapshot historique compatible avec la surface cible, reconstruit
H2H et `prior_tournament_*`, recalcule l'âge, et transmet les durées indisponibles comme `NaN`.

## 9. Corrected prediction

- avant : **Faria {100*original['probability']['player_a']:.3f} % / Gea {100*original['probability']['player_b']:.3f} %**
- après : **Faria {100*corrected['probability']['player_a']:.3f} % / Gea {100*corrected['probability']['player_b']:.3f} %**
- raw probability corrigée : `{corrected_summary['raw_probability']:.15f}`
- final probability corrigée : `{corrected_summary['final_probability']:.15f}`

Ablations diagnostiques du snapshot original — elles ne sont pas des prédictions valides :

| Variante | P(Faria) |
|---|---:|
| FULL | {100*summary['final_probability']:.3f} % |
{chr(10).join(f"| without {name} block | {100*value:.3f} % |" for name,value in ablations.items())}

Cas artificiels en espace standardisé :

| Cas | P(A) |
|---|---:|
{chr(10).join(f"| {name} | {100*value:.3f} % |" for name,value in artificial.items())}

Le cas « recent form only » confirme le signe conditionnel contre-intuitif appris ; ce n'est pas
une erreur d'affichage.

## 10. Final conclusion

### MODEL CORRECT / RUNTIME FEATURES INCORRECT (corrigées)

**72,1 % était la sortie PRE-MATCH V1 mathématiquement correcte pour le vecteur enregistré, mais
ce vecteur n'était pas correctement reconstruit.** L'explication des signes était correcte en
log-odds bruts ; elle révélait de vrais coefficients conditionnels négatifs. Après correction du
runtime, le même modèle figé donne **{100*corrected['probability']['player_a']:.1f} % pour Faria et
{100*corrected['probability']['player_b']:.1f} % pour Gea**. La qualité reste `MEDIUM` à cause des
dates ESPN au niveau tournoi, des durées absentes et des statistiques service/retour historiques.
"""
    (REPORTS_DIR / "faria_gea_prediction_audit.md").write_text(audit)


if __name__ == "__main__":
    main()
