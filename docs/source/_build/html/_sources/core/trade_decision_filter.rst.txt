trade_decision_filter module
===========================

.. automodule:: bitcoin_scalper.core.trade_decision_filter
   :members:
   :undoc-members:
   :show-inheritance:

Exemple d'utilisation
---------------------

.. code-block:: python

   from bitcoin_scalper.core.trade_decision_filter import TradeDecisionFilter
   filter = TradeDecisionFilter(dynamic=True)
   accept, reason = filter.filter(0.52)
   print(accept, reason)

Sécurité
--------
- Aucun secret n'est stocké ou loggé.
- Toutes les décisions sont journalisées pour auditabilité. 