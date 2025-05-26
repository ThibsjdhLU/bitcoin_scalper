import pandas as pd
import numpy as np
from app.core.backtesting import Backtester

def make_df():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "close": np.linspace(100, 110, 10),
        "signal": [1, 0, 0, -1, 0, 1, 0, 0, -1, 0]
    }, index=idx)
    return df

def test_backtester_run():
    df = make_df()
    bt = Backtester(df)
    out_df, trades, kpis = bt.run()
    assert "equity_curve" in out_df.columns
    assert len(trades) > 0
    assert "sharpe" in kpis
    assert "max_drawdown" in kpis
    assert "profit_factor" in kpis
    assert "win_rate" in kpis
    assert kpis["nb_trades"] == len(trades)

def test_backtester_no_trades():
    df = make_df()
    df["signal"] = 0
    bt = Backtester(df)
    out_df, trades, kpis = bt.run()
    assert len(trades) == 0
    assert kpis["win_rate"] == 0
    assert kpis["final_return"] == 0

def test_backtester_flat():
    df = make_df()
    df["signal"] = [0]*10
    bt = Backtester(df)
    out_df, trades, kpis = bt.run()
    assert np.all(out_df["returns"] == 0)
    assert np.all(out_df["equity_curve"] == 1) 