# UI Bitcoin Scalper

## Structure générale

L'interface utilisateur du bot Bitcoin Scalper est conçue pour offrir une expérience "trader pro" : lisibilité, feedback immédiat, sécurité, personnalisation.

- **MainWindow** : fenêtre principale, orchestre tous les panneaux (compte, positions, signaux, risque, logs, graphique).
- **DashboardSimple** : panneau synthétique (statut, PnL, capital, alertes, STOP).
- **AccountInfoPanel** : solde, PnL, statut bot, dernier prix.
- **RiskPanel** : drawdown, PnL journalier, pic capital, capital actuel.
- **SignalPanel** : signal de trading (buy/sell/hold).
- **Table des positions** : enrichie par un delegate custom (icônes, couleurs, tooltips, surlignage critique).
- **Bandeau d'alerte globale** : messages critiques affichés en overlay en haut de la fenêtre.

## Principes UX/UI

- **Palette sombre pro** (QSS, personnalisable).
- **Feedback immédiat** : couleurs, badges, alertes, logs colorés.
- **Sécurité** : aucune donnée sensible affichée, gestion des erreurs, alertes critiques.
- **Accessibilité** : police lisible, contrastes, surlignage des situations à risque.
- **Modularité** : chaque panneau est un widget indépendant, facilement extensible.

## Personnalisation (QSS)

- Tous les styles sont centralisés dans `ui/styles/dark_theme.qss`.
- Les états (succès, erreur, warning, info) sont gérés par des classes CSS.
- Possibilité d'ajouter un thème clair (structure prête).
- Les badges, surlignages et couleurs sont modifiables sans toucher au code Python.

## Table des positions enrichie

- **Icônes achat/vente** dans la colonne Sens.
- **Coloration automatique** des lignes selon le PnL (vert/rouge).
- **Surlignage critique** si perte > 5% ou drawdown > 5% (rouge clair + badge triangle rouge).
- **Tooltips détaillés** sur chaque cellule (toutes les infos de la position).

## Alertes et feedback

- **Bandeau d'alerte globale** : utilisable par tout composant pour signaler une situation critique (drawdown, perte, erreur système, etc.).
- **Logs colorés** dans la console intégrée.
- **Modales de confirmation** pour les actions sensibles (à venir).

## Extension future

- **Dashboard avancé** : intégration du dashboard simple comme page d'accueil ou panneau rétractable.
- **Thème clair** : ajout d'un QSS light.
- **Personnalisation utilisateur** : taille police, préférences, etc.
- **Notifications système** : intégration desktop ou mobile.

## Tests et qualité

- Couverture >95% sur les modules UI critiques.
- Tests d'intégration pour le bandeau d'alerte, la table des positions, les panneaux principaux.
- Documentation générée automatiquement (Sphinx/MkDocs). 