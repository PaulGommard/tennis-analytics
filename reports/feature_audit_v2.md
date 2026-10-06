# Audit des features V2

- Matchs : 78,845
- Colonnes totales : 215
- Features numériques hors cible : 207
- Paires de différences avec |corrélation| ≥ 0,80 : 54

Les distributions et taux de valeurs manquantes complets sont dans
`feature_coverage_v2.csv`. Les couvertures annuelles des statistiques service,
retour, durée et ranking sont dans `feature_coverage_v2_by_year.csv`.

Les taux sont bornés par construction lorsque les comptes source sont cohérents.
Les valeurs incohérentes sont transformées en données manquantes pour le ratio
concerné. L'imputation médiane est apprise uniquement sur 2000–2018.

Les corrélations fortes sont conservées dans le dataset pour l'ablation mais
leurs coefficients ne doivent pas être interprétés isolément.
