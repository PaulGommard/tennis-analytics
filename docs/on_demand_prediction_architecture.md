# Architecture de prédiction ATP à la demande

## Principe

Le chemin prospectif est séparé du replay historique. Il ne contient ni cron, ni polling, ni
entraînement. Une lecture de `/upcoming` demande seulement le calendrier. Le refresh des deux
joueurs et l'inférence PREMATCH V1 ne commencent qu'après un clic explicite.

```text
Navigateur /upcoming
  └─ GET /api/upcoming?days=3
       └─ EspnTennisProvider → cache HTTP 30 min → prochains simples ATP

Bouton « Actualiser & calculer »
  └─ POST /api/upcoming/{espn:id}/refresh-prediction
       ├─ résolution ESPN id → player_id local (mapping, nom exact, alias contrôlé)
       ├─ 1 page résultats ESPN par joueur
       ├─ état runtime SQLite daté et traçable
       ├─ snapshot historique + variables actualisables
       ├─ validation du vecteur ordonné de 86 features
       └─ artefact figé PREMATCH_V1 → ledger SQLite immuable
```

## Composants

- `OnDemandTennisDataProvider` définit le contrat minimal et garde le fournisseur remplaçable.
- `EspnTennisProvider` applique timeout, une seule relance, User-Agent explicite, cache disque,
  repli sur cache périmé et plafond de six appels par opération.
- `OnDemandPredictionService` interdit le fuzzy matching silencieux. Les noms sont résolus par
  mapping persistant, égalité normalisée ou petite table d'alias contrôlés.
- `ProductStore` conserve les mappings externes, matchs upcoming, états joueurs et prédictions.
- `FastPreMatchService` expose la construction du vecteur, sans modifier le modèle ni ses poids.

## Cohérence et absence de fuite

Seuls les résultats strictement antérieurs à l'heure du match cible sont utilisés. Les deux
joueurs sont rafraîchis dans la même requête utilisateur. Les valeurs indisponibles ne sont pas
inventées : les statistiques de service/retour restent celles du dernier snapshot historique,
avec un avertissement et une date de fraîcheur. Une surface non résolue ou une identité non
résolue provoque un refus explicite.

## Qualité et coût

ESPN seul plafonne la qualité à `MEDIUM`, car ses pages publiques ne livrent pas les durées ni
les statistiques détaillées de service/retour nécessaires à une actualisation exacte des 86
variables. Un premier calcul entièrement froid consomme au plus cinq requêtes externes :
calendrier, index et snapshot du classement, puis deux pages joueur. Si calendrier et classement
sont déjà en cache, il n'en consomme que deux ; le cache peut ramener ce nombre à zéro.

## Extension fournisseur

Un futur provider peut implémenter le même contrat. Le fallback se fait provider par provider :
une réponse complète garde une provenance unique. Le système ne fusionne jamais silencieusement
des champs contradictoires ; toute future fusion devra définir une priorité par champ et exposer
les conflits dans la réponse API.
