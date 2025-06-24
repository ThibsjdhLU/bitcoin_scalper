import numpy as np
import pandas as pd
from bitcoin_scalper.core.modeling import train_qvalue_model

def test_train_qvalue_model_catboost():
    X = pd.DataFrame(np.random.randn(100, 5))
    Y = pd.DataFrame(np.random.randn(100, 3), columns=['q_buy', 'q_sell', 'q_hold'])
    Xtr, Xv = X.iloc[:80], X.iloc[80:]
    Ytr, Yv = Y.iloc[:80], Y.iloc[80:]
    model = train_qvalue_model(Xtr, Ytr, Xv, Yv, algo='catboost')
    preds = model.predict(Xv)
    assert preds.shape == (20, 3)
    assert np.all(np.isfinite(preds)) 