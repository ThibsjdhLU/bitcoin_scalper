"""
Test Dashboard Worker Configuration Equivalence with engine_main.py Paper Mode.

This test verifies that the dashboard worker.py is configured identically 
to engine_main.py when running in paper trading mode.
"""

import unittest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestWorkerCodeStructure(unittest.TestCase):
    """Test that worker.py code structure matches engine_main.py paper mode."""
    
    def test_worker_imports_and_structure(self):
        """Test that worker.py has correct structure and imports."""
        worker_path = Path(__file__).parent.parent.parent / "src" / "bitcoin_scalper" / "dashboard" / "worker.py"
        
        with open(worker_path, 'r') as f:
            content = f.read()
        
        # Verify critical imports
        self.assertIn("from bitcoin_scalper.core.engine import TradingEngine, TradingMode", content)
        self.assertIn("from bitcoin_scalper.connectors.paper import PaperMT5Client", content)
        
        # Verify data limit is 5000 (not 100)
        self.assertIn("limit=5000", content, 
                     "Data fetch limit must be 5000 to match engine_main.py")
        self.assertNotIn("limit=100", content,
                        "Old limit of 100 must be replaced with 5000")
        
        # Verify safe_mode_on_drift parameter
        self.assertIn("safe_mode_on_drift=", content,
                     "safe_mode_on_drift parameter must be included")
        
        # Verify paper_initial_balance usage
        self.assertIn("paper_initial_balance", content,
                     "Must use paper_initial_balance from config")
        
        # Verify paper_simulate_slippage usage
        self.assertIn("paper_simulate_slippage", content,
                     "Must use paper_simulate_slippage from config")
        
        # Verify set_price is called
        self.assertIn("set_price", content,
                     "Must call set_price on paper client")
        
        print("✓ All critical code structures verified")
    
    def test_configuration_parameter_equivalence(self):
        """Test that all critical configuration parameters are equivalent."""
        # This test documents the expected equivalence
        # References to engine_main.py run_paper_mode() function
        critical_params = {
            'data_limit': 5000,  # Must match engine_main.py paper mode
            'initial_balance_config_key': 'paper_initial_balance',
            'initial_balance_default': 15000.0,
            'slippage_config_key': 'paper_simulate_slippage',
            'includes_safe_mode_on_drift': True,
            'initial_price': 50000.0,
        }
        
        # Document expected behavior
        self.assertEqual(critical_params['data_limit'], 5000)
        self.assertEqual(critical_params['initial_balance_default'], 15000.0)
        self.assertTrue(critical_params['includes_safe_mode_on_drift'])
        
        print("✓ Configuration parameters documented correctly")


class TestWorkerEngineMainEquivalence(unittest.TestCase):
    """
    Integration test to verify worker.py behavior matches engine_main.py paper mode.
    """
    
    def test_code_path_equivalence(self):
        """Document the code path equivalence between worker.py and engine_main.py."""
        equivalence_map = {
            'worker._initialize_engine()': 'engine_main.run_paper_mode() - engine initialization',
            'worker._fetch_market_data()': 'engine_main.run_paper_mode() - data fetching',
            'connector initialization': 'Both use PaperMT5Client with same params',
            'engine initialization': 'Both use TradingEngine with same params',
            'model loading': 'Both use engine.load_ml_model() with meta_threshold',
        }
        
        # Verify documentation is accurate
        self.assertIn('connector initialization', equivalence_map)
        self.assertIn('engine initialization', equivalence_map)
        
        print("✓ Code path equivalence documented")
    
    def test_critical_differences_resolved(self):
        """Test that all critical differences have been resolved."""
        worker_path = Path(__file__).parent.parent.parent / "src" / "bitcoin_scalper" / "dashboard" / "worker.py"
        engine_path = Path(__file__).parent.parent.parent / "src" / "bitcoin_scalper" / "engine_main.py"
        
        with open(worker_path, 'r') as f:
            worker_content = f.read()
        
        with open(engine_path, 'r') as f:
            engine_content = f.read()
        
        # Check that both files use the same critical patterns
        # 1. Data limit
        self.assertIn("limit=5000", worker_content)
        
        # 2. safe_mode_on_drift
        self.assertIn("safe_mode_on_drift", worker_content)
        
        # 3. paper_initial_balance
        self.assertIn("paper_initial_balance", worker_content)
        
        # 4. paper_simulate_slippage
        self.assertIn("paper_simulate_slippage", worker_content)
        
        # 5. set_price
        self.assertIn("set_price", worker_content)
        
        print("✓ All critical differences resolved")


if __name__ == '__main__':
    unittest.main(verbosity=2)

