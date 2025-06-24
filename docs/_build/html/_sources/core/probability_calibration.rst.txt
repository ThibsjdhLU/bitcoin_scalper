probability_calibration module
============================

.. automodule:: bitcoin_scalper.core.probability_calibration
   :members:
   :undoc-members:
   :show-inheritance:

Exemple d'utilisation
---------------------

.. code-block:: python

   from bitcoin_scalper.core.probability_calibration import ProbabilityCalibrator
   calibrator = ProbabilityCalibrator(method="sigmoid")
   calibrator.fit(model, X, y)
   proba = calibrator.predict_proba(X)

Sécurité
--------
- Aucune donnée sensible n'est stockée en clair.
- Les calibrateurs peuvent être sauvegardés/chargés via joblib (fichiers chiffrés recommandés en production). 