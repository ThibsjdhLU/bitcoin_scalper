import unittest
import numpy as np
from bitcoin_scalper.core.adaptive_scheduler import AdaptiveStrategyScheduler
from bitcoin_scalper.core.risk_management import RiskManager
from bitcoin_scalper.core.trade_decision_filter import TradeDecisionFilter

class MockClient:
    def _request(self, method, endpoint):
        # Mock account info for RiskManager
        if endpoint == "/account":
            return {"balance": 10000.0, "equity": 10000.0}
        # Mock symbol info
        if "symbol" in endpoint:
            return {"tick_value": 1.0}
        return {}

class TestIntegrationScheduler(unittest.TestCase):
    def setUp(self):
        self.rm = RiskManager(client=MockClient(), risk_per_trade=0.01) # 1% risk -> 100$
        self.filter = TradeDecisionFilter(max_entropy=0.8)
        self.scheduler = AdaptiveStrategyScheduler(
            risk_manager=self.rm,
            filter=self.filter,
            sizing_method="risk_adjusted"
        )

    def test_entropy_rejection(self):
        # Case 1: High Confidence, Low Entropy -> Accept
        probs = [0.9, 0.1]
        decision = self.scheduler.schedule_trade(
            symbol="BTCUSD",
            signal=1,
            proba=0.9,
            probs=probs,
            stop_loss=100.0
        )
        self.assertIsNotNone(decision)
        self.assertEqual(decision['action'], 'buy')
        self.assertEqual(decision['reason'], 'Accepté')

        # Case 2: Low Confidence (but above threshold), High Entropy -> Reject
        # Example: 3 classes [0.4, 0.3, 0.3]. Max proba 0.4 (might be below threshold depending on config)
        # Let's say filter lower_bound is 0.35. Proba 0.4 passed threshold.
        # But Entropy: - (0.4log0.4 + 0.3log0.3 + 0.3log0.3) approx 1.5 > 0.8
        probs_confused = [0.4, 0.3, 0.3]
        decision = self.scheduler.schedule_trade(
            symbol="BTCUSD",
            signal=1,
            proba=0.4, # Assume 0.4 is > filter lower bound (default 0.45, so let's adjust filter)
            probs=probs_confused,
            stop_loss=100.0
        )
        # Default lower_bound is 0.45, so 0.4 would be rejected by proba anyway.
        # Let's update filter bound to test entropy specifically.
        self.scheduler.filter.lower_bound = 0.3

        decision = self.scheduler.schedule_trade(
            symbol="BTCUSD",
            signal=1,
            proba=0.4,
            probs=probs_confused,
            stop_loss=100.0
        )

        self.assertIsNone(decision)
        # Ideally check reason, but schedule_trade returns None.
        # Logs would show "Refusé : Modèle confus"

    def test_dynamic_sizing(self):
        # 1% Risk on 10000 = 100$
        # Stop Loss distance = 50 points. Tick value = 1.
        # Position Size should be 100 / (50 * 1) = 2.0 lots

        decision = self.scheduler.schedule_trade(
            symbol="BTCUSD",
            signal=1,
            proba=0.9,
            probs=[0.9, 0.1],
            stop_loss=50.0 # 50 points distance
        )

        self.assertIsNotNone(decision)
        self.assertAlmostEqual(decision['volume'], 1.0)

        # Test Cap (max_position_size default is 1.0 in RiskManager)
        # Wait, RiskManager init default max_position_size=1.0.
        # So it should be capped at 1.0.
        self.assertEqual(decision['volume'], 1.0)

        # Update cap to test calculation
        self.scheduler.risk_manager.max_position_size = 5.0
        self.scheduler.max_size = 5.0
        decision = self.scheduler.schedule_trade(
            symbol="BTCUSD",
            signal=1,
            proba=0.9,
            probs=[0.9, 0.1],
            stop_loss=50.0
        )
        self.assertAlmostEqual(decision['volume'], 2.0)

if __name__ == '__main__':
    unittest.main()
