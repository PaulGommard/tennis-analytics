# Phase 7B — audit de reconstruction du score

Chaque match est rejoué depuis 0–0 avec `TennisScore`. L'état canonique est capturé **avant**
le point, puis le gagnant du point est appliqué. MCP est comparé à son score source avant point ;
Grand Slam à son score source après point. Une divergence exclut le match entier du dataset ML.

| Source | Matchs bruts | Score cohérent | Terminés éligibles | Points bruts |
|---|---:|---:|---:|---:|
| mcp | 7,532 | 7,429 | 7,315 | 1,283,895 |
| slam | 10,508 | 8,300 | 8,038 | 1,911,538 |

Total : 18,040 matchs, 15,729 séquences cohérentes, 15,353 matchs terminés et
éligibles au modèle point. Couverture PRE-MATCH V1 parmi les matchs éligibles :
62.08%.

Les exclusions distinguent doublons/gaps, métadonnées invalides, score/serveur divergent,
match incomplet ou abandon. Les formats Grand Chelem historiques sont versionnés : avantage,
Wimbledon 12–12, Australian Open 10 points et règle unifiée 2022. Les super tie-breaks et
formats non standard non représentables sont exclus par divergence, jamais corrigés en silence.

Le détail match par match est conservé dans `data/processed/point_matches_v1.parquet`.
