# Audit des données shot-level MCP

## Périmètre local

- Genre disponible: hommes uniquement. Les fichiers WTA `charting-w-*` sont absents du snapshot local; couverture WTA = 0 match.
- Métadonnées: 7,565 identifiants uniques (1 doublon résolu par qualité des champs).
- Matchs avec au moins un point canonique: 7,532.
- Points source: 1,284,276; points shot-level retenus: 1,264,752 (98.48 %).
- Frappes/événements retenus: 6,396,818.
- Joueurs MCP distincts: 1,002; tournois: 398.
- Période canonique: 1960–2026.

## Couverture par surface (points)

| Surface | Points | Part |
|---|---:|---:|
| Hard | 791,819 | 62.61 % |
| Clay | 306,686 | 24.25 % |
| Grass | 166,059 | 13.13 % |
| Unknown | 188 | 0.01 % |

## Couverture par époque

| Époque | Points |
|---|---:|
| 1990s | 140,893 |
| 2000s | 174,462 |
| 2010s | 353,820 |
| 2020s | 544,133 |
| avant_1990 | 51,444 |

## Joueurs les plus couverts dans les métadonnées

| Joueur | Matchs |
|---|---:|
| Roger Federer | 723 |
| Novak Djokovic | 553 |
| Rafael Nadal | 425 |
| Hubert Hurkacz | 320 |
| Jannik Sinner | 298 |
| Daniil Medvedev | 286 |
| Andy Murray | 257 |
| Pete Sampras | 237 |
| Carlos Alcaraz | 222 |
| Andre Agassi | 215 |
| Stefanos Tsitsipas | 200 |
| Alexander Zverev | 196 |
| Stefan Edberg | 193 |
| Dominic Thiem | 183 |
| Andrey Rublev | 176 |
| Grigor Dimitrov | 174 |
| Casper Ruud | 171 |
| Boris Becker | 155 |
| Juan Martin Del Potro | 153 |
| Alex De Minaur | 140 |

## Fiabilité des champs

| Champ / contrôle | Couverture | Commentaire |
|---|---:|---|
| Date métadonnée parseable | 99.99 % | format attendu YYYYMMDD |
| Surface reconnue | 99.97 % | deux lignes source sont mal alignées/complémentaires |
| Main joueur 1 | 99.56 % | valeur source |
| Main joueur 2 | 99.56 % | valeur source |
| Direction des coups non-service | 98.17 % | `0` et absence restent inconnus |
| Profondeur du retour | 71.09 % | optionnelle selon le protocole |
| Cohérence vainqueur reconstruit | 100.00 % | le cas divergent est exclu |

## Biais et limites

Le MCP est crowdsourcé et non exhaustif: la couverture favorise certains joueurs, matchs télévisés et périodes récentes. Les fréquences ne représentent donc pas automatiquement le circuit ATP. Les comparaisons de modèles sont confinées au même sous-ensemble MCP. La licence source est CC BY-NC-SA 4.0.
