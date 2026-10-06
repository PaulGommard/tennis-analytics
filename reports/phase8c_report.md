# Phase 8C — stabilisation de la probabilité live du match

## Résumé

LIVE V3 améliore la probabilité de match sans abandonner le signal dynamique de V2. La variante
retenue est un mélange linéaire cohérent des probabilités de point au service :

```text
w(n) = n / (n + 200)
p_service_V3 = (1 - w) × p_service_V1 + w × p_service_V2
```

Le moteur scoring/Markov exact reste inchangé. À zéro point, V3 égale exactement V1 et le prior
PRE-MATCH. Après 20, 50, 100 et 200 points, le poids V2 vaut respectivement 9,1 %, 20 %, 33,3 %
et 50 %. Cette montée lente est précisément ce qui empêche l'overreaction.

Sur le test 2023–2026, la log-loss P(match) passe de 0,458376 pour LIVE V1 et 0,464330 pour
LIVE V2 à **0,455247 pour LIVE V3**. Le Brier passe de 0,152214 à **0,151185** et le ROC-AUC
de 0,860902 à **0,862860**. LIVE V3 satisfait donc le critère de promotion.

## Protocole

V1 et V2 sont figés par empreintes SHA-256 dans `models/live_v3/phase8_freeze.json` et ne sont
jamais réentraînés ou réécrits. Les constantes 25/50/75/100/150/200 et les mélanges linéaire et
logit-tempéré sont présélectionnés sur 2019–2022, puis les quatre meilleurs candidats passent
dans le Markov exact sur 12 877 checkpoints de validation.

Pour la calibration, les modèles sont ajustés sur 2019–2020, sélectionnés sur 2021–2022, puis
réajustés sur toute la validation. Le test 2023–2026 n'est ouvert qu'après fixation de V3.
P(point) et P(game) sont mesurées sur 413 965 états test ; P(set) et P(match) sur les mêmes
46 433 checkpoints déterministes que Phase 8.

## A. Origine de l'overconfidence

LIVE V2 injecte toute la correction dynamique à chaque horizon futur. Les petites variations de
service/retour, raisonnables pour le prochain point, sont ainsi composées sur de nombreux jeux et
sets. Le problème est une **propagation répétée trop forte**, pas une mauvaise information de
point.

Le diagnostic par confiance le montre clairement :

| Confiance match | LIVE V1 prédit / observé | LIVE V2 prédit / observé | LIVE V3 prédit / observé |
|---|---:|---:|---:|
| 60–70 % | 65,0 / 64,2 % | 65,0 / 61,0 % | 64,9 / 62,8 % |
| 70–80 % | 75,0 / 74,8 % | 75,1 / 69,1 % | 75,0 / 73,5 % |
| 80–90 % | 85,1 / 86,0 % | 85,2 / 80,1 % | 85,1 / 84,5 % |
| 90–95 % | 92,6 / 94,2 % | 92,6 / 88,4 % | 92,6 / 92,5 % |
| 95–100 % | 97,8 / 99,0 % | 98,3 / 96,8 % | 98,0 / 98,5 % |

V2 est surtout trop confiant entre 60 et 95 %. V3 ramène presque partout le taux observé vers
la probabilité annoncée.

## B. Meilleure fonction de blending

La présélection et le Markov exact choisissent le mélange **linéaire C=200**. Sur les quatre
finalistes, la log-loss match validation exacte est :

| Méthode | C | Log-loss match validation |
|---|---:|---:|
| Linéaire | **200** | **0,499350** |
| Logit tempéré | 200 | 0,499364 |
| Linéaire | 150 | 0,499680 |
| Linéaire | 100 | 0,500600 |

Le logit tempéré est pratiquement équivalent à C=200, mais le mélange linéaire gagne légèrement
et reste plus facile à expliquer. Les valeurs plus petites de C, donc plus agressives, se
dégradent régulièrement.

## C. Calibration spécifique aux horizons

Platt, isotonic et temperature scaling ont été testés séparément pour jeu, set et match.
Sur la sélection temporelle 2021–2022, **aucune calibration ne bat la sortie Markov brute** :

| Horizon | Aucune | Température | Platt | Isotonic |
|---|---:|---:|---:|---:|
| Jeu | **0,438720** | 0,438967 | 0,438986 | 0,451347 |
| Set | **0,551747** | 0,552151 | 0,552125 | 0,556420 |
| Match | **0,486260** | 0,486877 | 0,487674 | 0,490899 |

LIVE V3 n'applique donc aucune correction postérieure. La cohérence point → jeu → set → match
reste complète.

## D. Impact point, jeu, set et match

| Horizon | LIVE V1 | LIVE V2 | LIVE V3 | Meilleur |
|---|---:|---:|---:|---|
| Point | 0,645612 | **0,645081** | 0,645290 | V2 |
| Jeu | 0,425470 | **0,422358** | 0,423735 | V2 |
| Set | 0,498791 | **0,491959** | 0,493745 | V2 |
| Match | 0,458376 | 0,464330 | **0,455247** | V3 |

La séparation d'horizon est justifiée par les données : V2 complet reste optimal pour point,
jeu et set ; V3 stabilisé doit alimenter le match. V3 ne sacrifie toutefois pas fortement les
horizons courts : il reste meilleur que V1 sur les trois.

## E. Profondeur du match

| Checkpoint | LIVE V1 | LIVE V2 | LIVE V3 |
|---|---:|---:|---:|
| Avant match | 0,555570 | 0,555570 | 0,555570 |
| Après 20 points | 0,520616 | 0,521094 | **0,518532** |
| Après 50 points | 0,469627 | 0,475471 | **0,465028** |
| Après 100 points | 0,389365 | 0,398797 | **0,386316** |
| Après 1 set | 0,416662 | 0,434857 | **0,409792** |
| Après 2 sets | **0,466441** | 0,500328 | 0,468374 |

V3 apprend progressivement et gagne nettement après 20–100 points et après le premier set.
Après deux sets, V1 reste légèrement meilleur sur les 1 373 matchs concernés ; cette tranche
mélange surtout les matchs décisifs et les BO5. Le blending lent limite néanmoins fortement la
dégradation de V2.

V3 améliore également V1 dans le set 1 (0,528976 → 0,527021), le set 2
(0,398713 → 0,393764) et les sets 3+ (0,450532 → 0,448172), ainsi qu'au début, milieu et à la
fin des sets.

## F. Cas où V2 overreact

| Écart absolu V2–V1 | États | Matchs | Part des états où V2 aide | Gain moyen de log-loss V2 |
|---|---:|---:|---:|---:|
| ≥ 10 points | 10 393 | 1 733 | 65,8 % | **-0,0184** |
| ≥ 20 points | 1 697 | 629 | 58,6 % | **-0,0701** |
| ≥ 30 points | 210 | 133 | 52,4 % | **-0,1889** |

V2 va souvent dans la bonne direction, mais ses rares erreurs très confiantes coûtent beaucoup
plus que ses corrections réussies ne rapportent. C'est exactement le profil d'une bonne source
d'information mal amplifiée.

Les composantes service et retour apportent presque le même signal point : log-loss test
0,645962 et 0,645960 ; les combiner donne 0,645954. Aucune n'est seule responsable. Supprimer
les fenêtres last-3/last-5 donne 0,645871 contre 0,645859 pour V2 complet : la recency courte
n'est pas la cause principale de l'overconfidence et apporte un très petit gain point.

## G. Architecture finale recommandée

```text
P(next point)  → POINT MODEL V2
P(game)        → LIVE V2
P(set)         → LIVE V2
P(match)       → LIVE V3 : blend V1/V2, w=n/(n+200), Markov exact
```

LIVE V1 reste la baseline officielle reproductible. LIVE V3 est promu comme nouvelle variante
officielle pour P(match), car il améliore validation et test sans calibration opaque. L'API doit
exposer le poids dynamique et les probabilités V1/V2 sous-jacentes pour audit.

La Phase 8C s'arrête ici. Aucun code shot-level ou Phase 9 n'a été commencé. La prochaine phase
pourra étudier les directions de service/retour, coup droit/revers, géométrie des frappes, rally
length, winner/error, serve+1 et return+1 du Match Charting Project.
