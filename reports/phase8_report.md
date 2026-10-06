# Phase 8 — adaptation intra-match et momentum

## Conclusion exécutive

La performance observée pendant le match contient bien un signal prédictif supplémentaire,
mais il est faible au niveau du prochain point et doit être fortement régularisé. Le meilleur
prior représente **50 points équivalents**. POINT MODEL V2 améliore la log-loss test de
0,646444 à **0,645859** et le Brier de 0,227142 à **0,226880**.

L'effet n'apparaît pratiquement pas dans les 20 premiers points, puis grandit avec le volume
d'observations. Le signal principal vient de l'adaptation shrinkée des forces service/retour ;
les séquences récentes, jeux et set précédent ajoutent de petits gains. En revanche, aucune
capacité « clutch » persistante convaincante n'est détectée.

Injecter V2 sans recalibration dans le Markov améliore P(game) et P(set), mais dégrade la
log-loss et la calibration de P(match). La recommandation est donc : **promouvoir V2 comme
modèle de prochain point expérimental, conserver LIVE V1 comme moteur match officiel, et ne
pas remplacer toute la chaîne V1 par V2 en l'état**.

## Protocole et anti-leakage

Phase 7B est figée par huit empreintes SHA-256 dans
`models/point_v2/phase7b_freeze.json`. Aucun artefact V1 n'a été modifié. Le split reste
strictement identique : train jusqu'en 2018, validation 2019–2022, test 2023–2026, avec tous
les points d'un match dans le même split.

`point_features_v2.parquet` conserve 2 724 048 points et 15 353 matchs. Chaque statistique
intra-match est enregistrée avant le point courant. Les mises à jour se font ensuite. Les tests
vérifient notamment les fenêtres T−1, l'activation retardée des breaks/sets, le démarrage du
shrinkage au prior et l'absence de contamination entre matchs du même jour pour le clutch.

Les champs ace/double faute sont conservés lorsqu'ils sont réellement disponibles, mais ne
sont pas utilisés dans V2 car leur couverture diffère entre Grand Chelem et MCP. La performance
de première balle n'est pas reconstructible de façon homogène dans le canonique V1.

## A. POINT MODEL V2

Le modèle sélectionné sur validation est `P5_ALL`, une régression logistique combinant V1,
statistiques cumulées, serve/return dynamiques, fenêtres récentes, momentum jeu/set et clutch.

| Modèle | Log-loss validation | Log-loss test | Brier test | ROC-AUC test |
|---|---:|---:|---:|---:|
| POINT MODEL V1 | 0,648333 | 0,646444 | 0,227142 | 0,558046 |
| Cumulatif match | 0,648050 | 0,645990 | 0,226939 | 0,560751 |
| Serve/return dynamique | 0,648001 | 0,645954 | 0,226923 | 0,560926 |
| Fenêtres de points | 0,648216 | 0,646249 | 0,227055 | 0,559174 |
| Jeux/set/pression | 0,648244 | 0,646293 | 0,227074 | 0,558954 |
| **V2 complet** | **0,647902** | **0,645859** | **0,226880** | **0,561445** |

Le gain V2 test est 0,000586 de log-loss point-pondérée. L'analyse appariée par match donne
0,000639, IC 95 % [0,000520 ; 0,000758], avec 50,4 % des matchs améliorés. L'accuracy reste
quasiment inchangée, comme attendu pour une cible dominée par le serveur.

## B. Serve/return dynamique

Les poids de prior 10, 20, 50 et 100 ont été testés exclusivement sur validation. Les log-loss
respectives sont 0,648058, 0,648024, **0,648001** et 0,648002 : 50 est retenu, même si 50 et
100 sont pratiquement équivalents.

Le modèle dynamique isolé gagne 0,000331 en validation et 0,000491 en test contre V1. Dans le
modèle complet, `dynamic_serve_w50`, `dynamic_return_w50` et
`performance_today_vs_expected` portent un signal réel. L'adaptation est donc mesurable, mais
le prior de 50 points montre qu'elle doit rester prudente pendant une grande partie du match.

## C. Momentum des points

Le bloc `last_3/5/10` gagne 0,000117 de log-loss en validation et 0,000196 en test. L'IC 95 %
apparié validation est [0,000048 ; 0,000187]. Le signal n'est donc pas nul, mais sa taille est
très faible. Les dix derniers points de service/retour sont plus utiles que le simple fait
d'avoir gagné les trois derniers points.

Conclusion scientifique : il existe une faible persistance locale compatible avec la forme du
jour, mais pas un « hot hand » autonome assez fort pour justifier une règle ad hoc ou une forte
modification des probabilités.

## D. Momentum des jeux

Le bloc jeux seul améliore la log-loss de 0,000047 en validation et 0,000063 en test. Son IC
95 % validation [0,000009 ; 0,000078] est positif, mais l'effet pratique est minuscule.
`break_recently`, `was_broken_recently` et les séries de jeux capturent donc au mieux une petite
information résiduelle après contrôle du score et des forces.

## E. Set précédent

Le bloc set précédent gagne 0,000046 en validation et 0,000079 en test. L'effet apparié moyen
est positif, mais très faible. Le score du set précédent est plus informatif que son seul
vainqueur ; cela ressemble davantage à une mesure bruitée de la performance du jour qu'à un
momentum psychologique démontré.

## F. Clutch persistant

Les profils break point, tie-break et deciding set sont construits à partir des résidus face à
une attente service/retour strictement pré-match, avec un prior de 200 observations. Le bloc clutch
dégrade la log-loss de 0,000002 en validation. Son IC 95 % apparié
[-0,000003 ; 0,000004] contient zéro. Les coefficients standardisés restent minuscules :
break -0,0024, tie-break -0,00001, deciding set -0,0012.

Dans ces données et avec ce protocole, il n'existe donc **pas de preuve robuste d'une capacité
clutch individuelle persistante** au-delà du niveau normal, du serveur, de la surface et du
score.

## G. Progression dans le match

| Segment | V1 log-loss | V2 log-loss | Gain V2 |
|---|---:|---:|---:|
| Points 1–20 | 0,645863 | 0,645868 | -0,000005 |
| Points 21–50 | 0,645985 | 0,645751 | +0,000234 |
| Points 51–100 | 0,645331 | 0,644654 | +0,000677 |
| Points 100+ | 0,647497 | 0,646677 | +0,000820 |

L'hypothèse centrale est confirmée : avant environ 20 points, les observations du jour ne
doivent presque rien changer. Leur valeur augmente ensuite régulièrement.

Le gain point est présent dans les deux sources : 0,000574 de log-loss sur MCP et 0,000625 sur
Grand Chelem. Il est également présent sur terre, gazon et dur. Il ne semble donc pas être créé
par un encodage source unique.

## H. Probabilités live

P(game) est évaluée sur les 413 965 états. P(set) et P(match), plus coûteuses, utilisent un
échantillon déterministe pré-déclaré de 46 433 états : tous les dix points et chaque début de
set. Le moteur Markov exact de Phase 7B reste inchangé ; seules ses probabilités de point au
service évoluent.

| Cible | V1 log-loss | V2 log-loss | V1 Brier | V2 Brier |
|---|---:|---:|---:|---:|
| Prochain point | 0,645612 | **0,645081** | 0,226742 | **0,226505** |
| Jeu courant | 0,425470 | **0,422358** | 0,136345 | **0,135284** |
| Set courant | 0,498791 | **0,491959** | 0,167991 | **0,165630** |
| Match | **0,458376** | 0,464330 | **0,152214** | 0,153631 |

L'adaptation aide donc point, jeu et set, mais son accumulation rend P(match) trop extrême.
Cette dégradation apparaît dans MCP comme dans Grand Chelem : ce n'est pas un simple biais de
source. Il faut calibrer la propagation dynamique à l'horizon match avant promotion.

## I. Calibration et retournements

L'erreur de calibration pondérée par buckets (ECE) du prochain point passe de 0,00400 à
0,00218, et celle du set de 0,02222 à 0,00918. Pour le match, elle se dégrade de 0,00941 à
0,03475 : V2 surestime les extrêmes. Par exemple, les états match à 80–90 % prédit donnent
79,8 % observé, et ceux à 10–20 % donnent 19,6 % observé.

Parmi 475 matchs où PRE-MATCH favorisait A mais V2 passe sous 50 % avant la fin du premier set,
B gagne finalement 50,7 % du temps. Le seuil faible est peu discriminant. Lorsque V2 descend
sous 40 %, B gagne 60,6 % des 213 cas ; sous 30 %, 63,5 % des 74 cas. Ces retournements forts
sont informatifs, mais l'ensemble doit être recalibré.

## J. Recommandation

1. Conserver POINT MODEL V1 et LIVE V1 comme baseline officielle immuable.
2. Accepter POINT MODEL V2 comme candidat expérimental supérieur pour P(next point).
3. Utiliser V2 pour P(game) et P(set) dans les analyses, avec suivi de calibration.
4. Ne pas remplacer P(match) V1 par P(match) V2 avant une calibration temporelle dédiée du
   déplacement dynamique, idéalement choisie sur validation au niveau match.
5. Ne pas investir davantage dans des règles de hot hand ou clutch manuelles : leurs effets
   sont trop faibles.

La Phase 8 s'arrête ici. Aucun Transformer, LSTM, deep learning ou modèle shot-level n'a été
commencé. Une éventuelle Phase 9 peut étudier les séquences tactiques MCP (direction de service,
retour, coup droit/revers, rally length, winner/error), avec un nouveau protocole figé.

## Artefacts reproductibles

- `data/processed/point_features_v2.parquet`
- `data/processed/live_v2_probabilities_test.parquet`
- `models/point_v2/model.joblib`, métadonnées, schéma et empreintes V1
- `reports/point_v2_comparison.csv`, ablations, coefficients, calibration et segments
- `reports/live_v2_validation.csv`, calibration, retournements et anomalies
- `scripts/build_point_features_v2.py`, `train_point_model_v2.py`, `evaluate_live_v2.py`
