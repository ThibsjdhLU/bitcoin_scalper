Prétraitement des données Bitcoin Scalper
=========================================

.. automodule:: scripts.prepare_features
    :members:
    :undoc-members:
    :show-inheritance:

Pipeline de préparation
----------------------

Le module prépare les données pour l'apprentissage automatique et le backtesting, en garantissant l'absence de look-ahead bias et la robustesse temporelle.

Fonctions principales
---------------------

- ``prepare_dataset`` : pipeline complet, du brut à l'export CSV sécurisé.
- ``generate_signal`` : génération du label de trading robuste et équilibré.
- ``check_temporal_integrity`` : validation de l'absence de fuite d'information.

Exemple d'utilisation
---------------------

.. code-block:: python

    from scripts.prepare_features import prepare_dataset
    prepare_dataset('input.csv', 'output.csv')

Sécurité temporelle
-------------------

.. warning::
   Toutes les features et signaux sont calculés sur données décalées (shift(1)).
   Toute modification doit préserver cette propriété pour garantir la validité des backtests et du ML. 