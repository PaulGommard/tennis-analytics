# Features SHOT V1

## Unités canoniques

`shot_events_v1.parquet` contient une ligne par événement, y compris une première faute et le second service. `active_shot_index` est nul pour la première faute; il commence à 1 au service qui met effectivement le point en jeu. `shot_points_v1.parquet` contient une ligne par point retenu, la séquence textuelle tokenisée et une représentation JSON structurée.

Les identifiants joueurs MCP sont des hachages stables de noms normalisés. Ils servent aux agrégats de style, jamais à simuler une résolution ATP certaine.

## Champs tactiques

- Service: numéro, direction, issue, service-volée.
- Frappe: type, aile, zone cible brute, relation crosscourt/milieu/long de ligne, profondeur.
- Position: approche, filet, fond, net cord, stop volley, shank.
- Rôle: service, retour, serve+1, return+1, rallye.
- Terminal: winner, faute provoquée/directe, ace, non-retourné, double faute.

`final_rally_length`, l'issue terminale et le vainqueur appartiennent à la table point pour l'analyse a posteriori. Ils ne sont jamais injectés dans les features prédictives d'un préfixe.

## Checkpoints prédictifs

Les checkpoints sont 0 (avant service), 1 (après service), 2 (après retour), 3 et 5 frappes actives. Une ligne n'existe que si le point est encore non terminal après le préfixe observé. Les seules informations de séquence sont celles du préfixe: service, retour lorsqu'il est déjà observé, dernière frappe observée, longueur observée et prochain rôle.

POINT MODEL V2 fournit le prior du serveur sur les lignes MCP qui existent dans son snapshot. Les comparaisons sont faites sur les mêmes points, avec split temporel strict par match: entraînement jusqu'au 31 décembre 2018, validation 2019–2022, test à partir de 2023.

## Prévention des fuites

- jamais de `final_rally_length`, vainqueur, issue terminale ou frappe future dans un checkpoint;
- aucun match partagé entre partitions temporelles;
- points terminés au checkpoint exclus de la tâche «issue future»;
- baseline et candidat évalués exactement sur les mêmes lignes MCP;
- effets et intervalles calculés au niveau match pour limiter la pseudo-réplication des points.

## Fingerprints

Les taux de style sont calculés avec un minimum de 10 matchs et 500 frappes: fréquence coup droit/revers, filet/approche, winners, fautes, agressivité serve+1, longueur moyenne, directions de service. Aucun ranking ni résultat de carrière n'entre dans la similarité ou le clustering.

`serve_plus_1_aggression` signifie ici winner, approche ou frappe au filet sur le troisième coup actif. C'est un proxy descriptif, pas une vérité causale.
