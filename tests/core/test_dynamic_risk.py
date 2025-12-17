import unittest
import numpy as np
from bitcoin_scalper.core.risk_management import RiskManager, MonteCarloSimulator
from bitcoin_scalper.core.trade_decision_filter import TradeDecisionFilter
from bitcoin_scalper.connectors.mt5_rest_client import MT5RestClient

class MockClient:
    def _request(self, method, endpoint):
        return {}

class TestDynamicRisk(unittest.TestCase):
    def setUp(self):
        self.rm = RiskManager(client=MockClient())
        self.filter = TradeDecisionFilter()

    def test_calculate_entropy(self):
        # Binary High Entropy (0.5, 0.5)
        probs = np.array([0.5, 0.5])
        ent = self.filter.calculate_entropy(probs)
        self.assertAlmostEqual(ent, 1.0, places=2)

        # Binary Low Entropy (0.9, 0.1)
        probs = np.array([0.9, 0.1])
        ent = self.filter.calculate_entropy(probs)
        # - (0.9 log2 0.9 + 0.1 log2 0.1) approx 0.469
        self.assertLess(ent, 0.5)

        # 3 classes (0.33, 0.33, 0.33) -> log2(3) approx 1.585
        probs = np.array([1/3, 1/3, 1/3])
        ent = self.filter.calculate_entropy(probs)
        self.assertAlmostEqual(ent, 1.585, places=2)

    def test_dynamic_stops(self):
        entry = 30000
        atr = 100

        # High Confidence -> Loose SL (k=2)
        res = self.rm.calculate_dynamic_stops(entry, atr, 'buy', 0.9)
        self.assertEqual(res['sl'], 29800) # 30000 - 2*100

        # Low Confidence -> Tight SL (k=1.5)
        res = self.rm.calculate_dynamic_stops(entry, atr, 'buy', 0.6)
        self.assertEqual(res['sl'], 29850) # 30000 - 1.5*100

        # Sell side
        res = self.rm.calculate_dynamic_stops(entry, atr, 'sell', 0.9)
        self.assertEqual(res['sl'], 30200) # 30000 + 2*100

    def test_monte_carlo(self):
        # Fake PnL history
        pnl = [100, -50, 200, -100, 50, 50, -20, 100]
        mc = MonteCarloSimulator(pnl_history=pnl, initial_capital=1000)
        res = mc.run_simulation(n_simulations=50, n_trades=10)

        self.assertIn("risk_of_ruin", res)
        self.assertIn("max_drawdown_95", res)
        self.assertIn("median_final_capital", res)

        # With positive expectancy, median capital should be > initial (mostly)
        # but n_trades is small, variance high. Just check types.
        self.assertIsInstance(res['risk_of_ruin'], float)

if __name__ == '__main__':
    unittest.main()
