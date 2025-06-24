import numpy as np
import pytest
from bitcoin_scalper.core.trade_decision_filter import TradeDecisionFilter

def test_static_filter_accept():
    f = TradeDecisionFilter(lower_bound=0.4, upper_bound=0.6, dynamic=False)
    accepted, reason = f.filter(0.7)
    assert accepted
    assert "Accepté" in reason

def test_static_filter_refuse():
    f = TradeDecisionFilter(lower_bound=0.4, upper_bound=0.6, dynamic=False)
    accepted, reason = f.filter(0.5)
    assert not accepted
    assert "Refusé" in reason

def test_dynamic_bounds():
    f = TradeDecisionFilter(dynamic=True, window_size=10)
    # Injecte des probas croissantes
    for p in np.linspace(0.1, 0.9, 20):
        f.filter(float(p))
    # Après 10 valeurs, les bornes doivent s'adapter
    assert f.lower_bound < f.upper_bound

def test_logging(caplog):
    f = TradeDecisionFilter(lower_bound=0.4, upper_bound=0.6, dynamic=False)
    with caplog.at_level("INFO"):
        f.filter(0.5)
    assert any("refusé" in m.lower() or "accepté" in m.lower() for m in caplog.messages)

def test_adaptive_scheduler_basic():
    from bitcoin_scalper.core.adaptive_scheduler import AdaptiveStrategyScheduler
    class DummyRiskManager:
        def can_open_position(self, symbol, volume):
            if volume > 0.5:
                return {"allowed": False, "reason": "Taille trop grande"}
            return {"allowed": True, "reason": "OK"}
    scheduler = AdaptiveStrategyScheduler(risk_manager=DummyRiskManager(), sizing_method="confidence", min_size=0.1, max_size=0.5)
    # Signal neutre
    res = scheduler.schedule_trade("BTCUSD", 0, 0.9)
    assert res is None
    # Signal accepté, sizing dynamique
    res2 = scheduler.schedule_trade("BTCUSD", 1, 0.4)
    assert res2["action"] == "buy"
    assert 0.1 <= res2["volume"] <= 0.5
    # Signal refusé par risk manager
    res3 = scheduler.schedule_trade("BTCUSD", 1, 0.9)
    assert res3 is not None  # max_size=0.5 donc accepté
    # Test refus si volume > 0.5
    scheduler2 = AdaptiveStrategyScheduler(risk_manager=DummyRiskManager(), sizing_method="confidence", min_size=0.6, max_size=1.0)
    res4 = scheduler2.schedule_trade("BTCUSD", 1, 0.9)
    assert res4 is None 