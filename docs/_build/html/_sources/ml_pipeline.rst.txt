ml\_pipeline module
===================

.. automodule:: ml_pipeline
   :members:
   :show-inheritance:
   :undoc-members:

.. note::

   **Sécurité ML** : Lors de la prédiction, la classe ``MLPipeline`` vérifie que les noms et l'ordre des features fournis correspondent exactement à ceux utilisés lors de l'entraînement (fichier ``features_list.pkl``). Un warning est loggué si des colonnes sont manquantes ou en trop, et les features sont automatiquement réordonnées pour garantir la cohérence. Cela protège contre les erreurs silencieuses et les dérives de données.
