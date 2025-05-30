ml_train module
===============

.. automodule:: bitcoin_scalper.core.ml_train
    :members:
    :undoc-members:
    :show-inheritance:

Utilisation CLI
---------------

.. code-block:: bash

    python3 -m bitcoin_scalper.core.ml_train --input_csv data/features/BTCUSD_M1_features_trend_following.csv --model_out model_rf.pkl --scaler_out scaler.pkl

Exemple d'appel Python
----------------------

.. code-block:: python

    from bitcoin_scalper.core import ml_train
    clf, scaler = ml_train.train_ml_model(
        input_csv="data/features/BTCUSD_M1_features_trend_following.csv",
        model_out="model_rf.pkl",
        scaler_out="scaler.pkl",
        use_smote=True
    )

.. autofunction:: bitcoin_scalper.core.ml_train.analyse_label_balance 