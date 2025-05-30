Feature Engineering
===================

.. automodule:: bitcoin_scalper.core.feature_engineering
   :members:
   :undoc-members:
   :show-inheritance:

Features générées
-----------------

- rsi
- macd, macd_signal, macd_diff
- ema_21, ema_50, sma_20
- bb_high, bb_low, bb_width
- atr, supertrend, vwap
- tickvol (copie de volume si absent)
- close_sma_3 (SMA 3 sur close)
- atr_sma_20 (SMA 20 sur ATR)

Exemple d'utilisation
---------------------

.. code-block:: python

   from bitcoin_scalper.core.feature_engineering import FeatureEngineering
   import pandas as pd
   df = pd.DataFrame({
       'close': [100, 101, 102],
       'high': [101, 102, 103],
       'low': [99, 100, 101],
       'volume': [10, 12, 11]
   })
   fe = FeatureEngineering()
   df_feat = fe.add_indicators(df)
   print(df_feat.columns)

Sécurité temporelle
------------------

- Tous les indicateurs sont décalés d'une bougie (shift(1)) pour éviter tout look-ahead bias.
- VWAP est calculé de façon cumulative et décalée.
- Aucun indicateur ne regarde le futur : sécurité stricte pour le ML et le backtest. 