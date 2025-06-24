UI (Interface graphique)
========================

.. automodule:: ui.main_window
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: ui.account_info_panel
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: ui.dashboard_simple
    :members:
    :undoc-members:
    :show-inheritance:

Nouveautés UI (2024)
-------------------

- Thème sombre professionnel (voir ui/styles/dark_theme.qss)
- Widget AccountInfoPanel : affiche solde, PnL, statut bot (avec icône), dernier prix, statut dynamique
- Affichage du statut bot et du dernier prix mis à jour dès réception de données
- Message explicite sur le graphique si aucune donnée OHLCV reçue ou colonnes manquantes
- Nouvelle organisation :
    - Infos compte/statut à gauche (haut)
    - Positions à gauche (bas)
    - Graphique au centre
    - Contrôles/signaux/risque à droite (onglets)
    - Logs en bas (affichage couleur, niveau, timestamp)
    - Barre d'état pour messages système
- Icônes SVG pour statuts et actions (dossier resources/ à la racine)

Nouveau module : DashboardSimple
-------------------------------

- Mode débutant/synthétique
- Statut bot (ON/OFF), PnL temps réel, capital, nombre de positions
- Graphique capital simple
- Bouton STOP toujours visible
- Bandeau d'alerte critique
- Design moderne, dark mode, responsive

.. automodule:: ui.main_window
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: ui.account_info_panel
    :members:
    :undoc-members:
    :show-inheritance: 