# Phase 6 — méthodologie du benchmark non linéaire

## Périmètre

Phase 6 utilise exclusivement les 86 features retenues en Phase 5. Elle compare
XGBoost, LightGBM et CatBoost à la régression logistique sans créer de données
point-by-point, live ou postérieures au match.

La période 2023–2026 a déjà été consultée lors des phases précédentes. Elle est
donc appelée **historical benchmark** et ne participe à aucune décision.

## Validation temporelle

Les folds valident successivement 2018, 2019, 2020, 2021 et 2022. Pour chaque
fold, l'entraînement s'arrête au 31 décembre de l'année précédente. Il n'existe
ni shuffle ni mélange entre passé et futur.

Chaque configuration est classée selon la moyenne des cinq folds : log loss,
puis Brier, ROC-AUC et accuracy. L'early stopping utilise seulement l'année de
validation du fold. Le nombre final d'arbres est la médiane des meilleures
itérations des cinq folds.

## Variantes de features

- `full` : les 86 features Logistic Phase 5;
- `no_fatigue_proxy` : 73 features après retrait de `matches`, `sets`, `games`
  et `minutes` sur 7/14/30 jours, ainsi que `rest_days`.

La charge `prior_tournament_*` reste présente : elle exclut explicitement
l'édition courante et ne code pas directement la progression dans son tableau.

## Calibration

Les prédictions out-of-fold 2018–2022 sont conservées. Pour évaluer une méthode
sur une année entre 2019 et 2022, son calibrateur est ajusté uniquement sur les
prédictions out-of-fold des années antérieures. Raw, Platt et isotonic sont
comparés avec la même priorité de métriques. Le calibrateur final est ensuite
ajusté sur l'ensemble des prédictions out-of-fold, jamais sur 2023–2026.

## Modèles finaux et benchmark

Après gel de la configuration, de la variante et de la calibration, chaque
modèle est réentraîné sur les matchs antérieurs au 1er janvier 2023. Les
artefacts sérialisent l'imputer médian, l'estimateur, le calibrateur, les
features et les métadonnées d'entraînement.

Le benchmark est ensuite ouvert une seule fois pour produire les comparaisons
globales, par surface, année et niveau source. Ces résultats ne peuvent pas
modifier le modèle sélectionné.

## Interprétation

Les importances natives et par permutation sont calculées sur 2022 avec un
modèle diagnostique entraîné jusqu'en 2021. SHAP porte uniquement sur le meilleur
arbre calibré gelé. Les valeurs SHAP décrivent les prédictions du modèle en
unités internes/log-odds; elles ne démontrent aucune causalité.
