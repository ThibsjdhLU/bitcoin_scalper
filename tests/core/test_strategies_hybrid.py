import pytest
import numpy as np
import pandas as pd
from app.core.strategies_hybrid import (
    MeanReversionStrategy, MomentumStrategy, BreakoutStrategy, ArbitrageStrategy, MultiTimeframeStrategy,
    OnlineParameterAdaptation, KellySizer, VaRSizer, TrailingStop, ExecutionAlgo, HybridStrategyEngine, BaseStrategy
)

@pytest.fixture
def price_data():
    np.random.seed(42)
    df = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99
    })
    return df

@pytest.fixture
def arbitrage_data():
    np.random.seed(42)
    df = pd.DataFrame({
        'asset1': np.random.randn(100).cumsum() + 100,
        'asset2': np.random.randn(100).cumsum() + 100
    })
    return df

@pytest.fixture
def multi_tf_data(price_data):
    return {'1min': price_data, '5min': price_data.copy()}

def test_mean_reversion(price_data):
    strat = MeanReversionStrategy(window=10, threshold=1.5)
    sig = strat.predict(price_data)
    assert sig.shape == (100,)
    assert np.all(np.isin(sig, [-1, 0, 1]))

def test_momentum(price_data):
    strat = MomentumStrategy(window=5)
    sig = strat.predict(price_data)
    assert sig.shape == (100,)
    assert np.all(np.isin(sig, [-1, 0, 1]))

def test_breakout(price_data):
    strat = BreakoutStrategy(window=10)
    sig = strat.predict(price_data)
    assert sig.shape == (100,)
    assert np.all(np.isin(sig, [-1, 0, 1]))

def test_arbitrage(arbitrage_data):
    strat = ArbitrageStrategy(spread_threshold=0.5)
    sig = strat.predict(arbitrage_data)
    assert sig.shape == (100,)
    assert np.all(np.isin(sig, [-1, 0, 1]))

def test_multi_timeframe(multi_tf_data):
    strat = MultiTimeframeStrategy({'1min': MeanReversionStrategy(), '5min': MomentumStrategy()})
    sig = strat.predict(multi_tf_data)
    assert sig.shape == (100,)
    assert np.all(np.isin(sig, [-1, 0, 1]))

def test_online_adaptation(price_data):
    base = MeanReversionStrategy(window=10)
    adapt = OnlineParameterAdaptation(base)
    y = np.random.randint(0, 2, 100)
    old_window = base.window
    adapt.update(price_data, pd.Series(y))
    assert base.window >= 5

def test_kelly_sizer():
    sizer = KellySizer()
    size = sizer.size(0.6, 2.0)
    assert 0.01 <= size <= 1.0
    size2 = sizer.size(0.5, 0.0)
    assert size2 == 0.01

def test_var_sizer():
    sizer = VaRSizer()
    pnl = pd.Series(np.random.randn(100))
    size = sizer.size(pnl)
    assert 0.01 <= size <= 0.1

def test_trailing_stop():
    stop = TrailingStop(distance=0.02)
    s_long = stop.get_stop(100, 105, 1)
    s_short = stop.get_stop(100, 95, -1)
    s_flat = stop.get_stop(100, 100, 0)
    assert s_long <= 105 and s_long >= 100
    assert s_short >= 95 and s_short <= 100
    assert s_flat == 100

def test_execution_algo():
    iceberg = ExecutionAlgo.iceberg(10, 3)
    assert sum(iceberg) == pytest.approx(10)
    twap = ExecutionAlgo.twap(10, 4)
    assert sum(twap) == pytest.approx(10)
    vwap = ExecutionAlgo.vwap([100,101,102], [1,2,3], 12)
    assert sum(vwap) == pytest.approx(12)

def test_hybrid_engine(price_data):
    engine = HybridStrategyEngine([
        MeanReversionStrategy(window=5),
        MomentumStrategy(window=3)
    ])
    engine.fit(price_data)
    sig = engine.predict(price_data)
    assert sig.shape == (100,)
    engine.update(price_data)
    size = engine.position_size(0.6, 2.0)
    assert 0.01 <= size <= 1.0
    stop = engine.get_stop(100, 105, 1)
    assert stop <= 105 and stop >= 100
    orders = engine.execute_order(10, mode="iceberg", max_qty=3)
    assert sum(orders) == pytest.approx(10)
    orders2 = engine.execute_order(10, mode="twap", n_slices=5)
    assert sum(orders2) == pytest.approx(10)
    orders3 = engine.execute_order(12, mode="vwap", prices=[100,101,102], vols=[1,2,3])
    assert sum(orders3) == pytest.approx(12)
    with pytest.raises(ValueError):
        engine.execute_order(10, mode="unknown") 