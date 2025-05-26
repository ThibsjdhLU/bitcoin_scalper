import pandas as pd
import numpy as np
from app.core.strategies import MeanReversionStrategy, MomentumStrategy, BreakoutStrategy, ArbitrageDummyStrategy, AdaptivePositionSizing, DynamicStop

def make_df():
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame({
        "close": np.linspace(100, 110, 30) + np.random.normal(0, 1, 30),
        "btc_close": np.linspace(100, 110, 30),
        "eth_close": np.linspace(50, 60, 30)
    }, index=idx)
    return df

def test_mean_reversion():
    df = make_df()
    strat = MeanReversionStrategy(z_thresh=1.5)
    sig = strat.generate_signal(df)
    assert len(sig) == len(df)
    assert np.all(np.isin(sig, [-1, 0, 1]))

def test_momentum():
    df = make_df()
    strat = MomentumStrategy(window=5)
    sig = strat.generate_signal(df)
    assert len(sig) == len(df)
    assert np.all(np.isin(sig, [-1, 0, 1]))

def test_breakout():
    df = make_df()
    strat = BreakoutStrategy(window=10)
    sig = strat.generate_signal(df)
    assert len(sig) == len(df)
    assert np.all(np.isin(sig, [-1, 0, 1]))

def test_arbitrage():
    df = make_df()
    strat = ArbitrageDummyStrategy()
    sig = strat.generate_signal(df, ref_col="btc_close", hedge_col="eth_close")
    assert len(sig) == len(df)
    assert np.all(np.isin(sig, [-1, 0, 1]))

def test_adaptive_position_sizing():
    strat = AdaptivePositionSizing(risk_per_trade=0.02)
    size = strat.compute_size(capital=10000, stop_loss=100, atr=2)
    assert size > 0

def test_dynamic_stop():
    strat = DynamicStop(atr_mult=2.5)
    stop_long = strat.compute_stop(entry_price=100, atr=2, direction=1)
    stop_short = strat.compute_stop(entry_price=100, atr=2, direction=-1)
    stop_flat = strat.compute_stop(entry_price=100, atr=2, direction=0)
    assert stop_long < 100
    assert stop_short > 100
    assert stop_flat == 100 