# Audit du dataset V3

## Invariants de production

- lignes : 78,845;
- colonnes : 251 (dont 215 V2 inchangées et 36 nouvelles);
- index unique : oui;
- colonnes du modèle final toutes présentes : oui (86 features);
- égalité exacte des colonnes héritées V2 : oui;
- valeurs infinies dans les features du modèle : aucune;
- domaine H2H : compteurs positifs ou nuls, edge dans [-0,5; 0,5].

## Couverture

Un historique H2H antérieur existe pour 34,896 matchs
(44.3%). Les statistiques détaillées des 36
nouvelles colonnes sont dans `feature_coverage_v3.csv` (couverture, part de zéros,
quantiles 1/50/99 %, minimum et maximum).

L'absence historique reste représentée par `NaN` lorsque la feature l'exige; l'imputation est
apprise exclusivement sur le train au sein du pipeline final.
