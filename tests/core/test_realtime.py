import pytest
import pandas as pd
import numpy as np
import os
from bitcoin_scalper.core.realtime import RealTimeExecutor

class DummyModel:
    def predict(self, X):
        # Signal alterné : 1, 0, -1, 0, 1, ...
        return np.array([1 if i%4==0 else -1 if i%4==2 else 0 for i in range(len(X))])

def make_realtime_data(n=20):
    idx = pd.date_range('2024-06-01', periods=n, freq='H', tz='UTC')
    df = pd.DataFrame({
        'feat0': np.random.randn(n),
        'feat1': np.random.randn(n),
        'close': np.linspace(100, 120, n)
    }, index=idx)
    return df

def test_realtime_executor_simulation(tmp_path):
    df = make_realtime_data(12)
    data_iter = iter([df.iloc[[i]] for i in range(len(df))])
    def data_source():
        try:
            return next(data_iter)
        except StopIteration:
            return pd.DataFrame()
    model = DummyModel()
    out_dir = tmp_path / "realtime_report"
    executor = RealTimeExecutor(
        model=model,
        data_source=data_source,
        signal_col="signal",
        price_col="close",
        initial_capital=1000.0,
        fee=0.001,
        slippage=0.0005,
        mode="simulation",
        sleep_time=0.0,
        out_dir=str(out_dir)
    )
    executor.run(max_steps=12)
    # Vérifie que les fichiers de reporting sont générés
    assert os.path.exists(f"{out_dir}/trades.csv")
    assert os.path.exists(f"{out_dir}/equity_curve.csv")
    assert os.path.exists(f"{out_dir}/equity_curve.png") 