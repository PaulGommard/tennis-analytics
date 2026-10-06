# Smoke test — prédiction ATP à la demande

Dernière révision : 7 octobre 2026 (Europe/Paris).

## Statut

Le chemin produit a été validé de bout en bout : calendrier public, ouverture d'une fiche match,
refresh explicite, construction des 86 features, inférence PRE-MATCH V1 et stockage local du
snapshot. Ouvrir `/upcoming` ne déclenche toujours aucune prédiction automatiquement.

Les cinq valeurs publiées par le smoke test initial du 6 octobre ne constituent cependant plus
des résultats valides. L'audit détaillé du match Jaime Faria–Arthur Gea a découvert trois erreurs
dans la construction prospective des features : workload hybride, snapshot de mauvaise surface
et double comptage de résultats ESPN. PRE-MATCH V1 lui-même était correct et n'a pas été modifié.

## Cas audité

| Match | Ancien snapshot | Snapshot runtime corrigé | Qualité |
| --- | ---:| ---:| --- |
| Jaime Faria – Arthur Gea | Faria 72,139 % | **Faria 59,511 %** | MEDIUM |

Les quatre autres probabilités du premier smoke test sont retirées de ce rapport plutôt que
présentées comme corrigées sans audit individuel. Elles peuvent être recalculées à la demande par
le runtime corrigé lorsqu'elles sont encore disponibles chez le fournisseur.

## Contrôles permanents

- exactement 86 colonnes, dans l'ordre attendu par l'artefact ;
- probabilités dans `[0, 1]` ;
- orientation A/B contrôlée sur toutes les features antisymétriques ;
- parité stricte entre l'artefact research et l'artefact PRE-MATCH V1 figé ;
- historique local fusionné uniquement avec les résultats ESPN postérieurs au cutoff local ;
- statistiques de surface prises sur un snapshot compatible avec la surface cible ;
- durées récentes inconnues transmises comme `NaN` à l'imputer figé ;
- fournisseur, fraîcheur, qualité et hash du snapshot exposés ;
- identité ambiguë refusée sans fuzzy matching silencieux.

## Interprétation

Ce smoke test valide le fonctionnement logiciel, pas la justesse prospective. La qualité reste
`MEDIUM` : rangs et résultats sont récents, mais les dates ESPN sont parfois au niveau tournoi,
les durées sont absentes et plusieurs statistiques service/retour restent historiques. Voir
[`faria_gea_prediction_audit.md`](faria_gea_prediction_audit.md) pour la reconstruction complète,
les contributions et les limites.
