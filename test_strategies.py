import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.strategies.ema_strategy import EMAStrategy
from src.strategies.rsi_strategy import RSIStrategy
from src.strategies.macd_strategy import MACDStrategy
from src.strategies.bollinger_strategy import BollingerStrategy
from src.strategies.combined_strategy import CombinedStrategy

class TestEMAStrategy(unittest.TestCase):
    def setUp(self):
        """Configuration initiale pour chaque test."""
        self.strategy = EMAStrategy()
        
        # Données de test
        self.test_data = pd.DataFrame({
            'close': [50000.0, 51000.0, 52000.0, 53000.0, 54000.0],
            'volume': [100.0, 200.0, 300.0, 400.0, 500.0]
        })

    def test_calculate_signals(self):
        """Test le calcul des signaux EMA."""
        signals = self.strategy.calculate_signals(self.test_data)
        
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('signal', signals.columns)
        self.assertIn('ema_fast', signals.columns)
        self.assertIn('ema_slow', signals.columns)

    def test_generate_signal(self):
        """Test la génération de signaux."""
        # Test signal d'achat
        signals = pd.DataFrame({
            'ema_fast': [50000.0, 51000.0],
            'ema_slow': [49000.0, 50000.0]
        })
        signal = self.strategy.generate_signal(signals)
        self.assertEqual(signal, 'BUY')
        
        # Test signal de vente
        signals = pd.DataFrame({
            'ema_fast': [49000.0, 48000.0],
            'ema_slow': [50000.0, 49000.0]
        })
        signal = self.strategy.generate_signal(signals)
        self.assertEqual(signal, 'SELL')
        
        # Test signal neutre
        signals = pd.DataFrame({
            'ema_fast': [50000.0, 50000.0],
            'ema_slow': [50000.0, 50000.0]
        })
        signal = self.strategy.generate_signal(signals)
        self.assertEqual(signal, 'HOLD')

class TestRSIStrategy(unittest.TestCase):
    def setUp(self):
        """Configuration initiale pour chaque test."""
        self.strategy = RSIStrategy()
        
        # Données de test
        self.test_data = pd.DataFrame({
            'close': [50000.0, 51000.0, 52000.0, 53000.0, 54000.0],
            'volume': [100.0, 200.0, 300.0, 400.0, 500.0]
        })

    def test_calculate_signals(self):
        """Test le calcul des signaux RSI."""
        signals = self.strategy.calculate_signals(self.test_data)
        
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('signal', signals.columns)
        self.assertIn('rsi', signals.columns)

    def test_generate_signal(self):
        """Test la génération de signaux."""
        # Test signal d'achat (RSI < 30)
        signals = pd.DataFrame({
            'rsi': [25.0, 20.0]
        })
        signal = self.strategy.generate_signal(signals)
        self.assertEqual(signal, 'BUY')
        
        # Test signal de vente (RSI > 70)
        signals = pd.DataFrame({
            'rsi': [75.0, 80.0]
        })
        signal = self.strategy.generate_signal(signals)
        self.assertEqual(signal, 'SELL')
        
        # Test signal neutre
        signals = pd.DataFrame({
            'rsi': [50.0, 50.0]
        })
        signal = self.strategy.generate_signal(signals)
        self.assertEqual(signal, 'HOLD')

class TestMACDStrategy(unittest.TestCase):
    def setUp(self):
        """Configuration initiale pour chaque test."""
        self.strategy = MACDStrategy()
        
        # Données de test
        self.test_data = pd.DataFrame({
            'close': [50000.0, 51000.0, 52000.0, 53000.0, 54000.0],
            'volume': [100.0, 200.0, 300.0, 400.0, 500.0]
        })

    def test_calculate_signals(self):
        """Test le calcul des signaux MACD."""
        signals = self.strategy.calculate_signals(self.test_data)
        
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('signal', signals.columns)
        self.assertIn('macd', signals.columns)
        self.assertIn('signal_line', signals.columns)

    def test_generate_signal(self):
        """Test la génération de signaux."""
        # Test signal d'achat (MACD croise au-dessus de la ligne de signal)
        signals = pd.DataFrame({
            'macd': [1.0, 2.0],
            'signal_line': [2.0, 1.0]
        })
        signal = self.strategy.generate_signal(signals)
        self.assertEqual(signal, 'BUY')
        
        # Test signal de vente (MACD croise en-dessous de la ligne de signal)
        signals = pd.DataFrame({
            'macd': [2.0, 1.0],
            'signal_line': [1.0, 2.0]
        })
        signal = self.strategy.generate_signal(signals)
        self.assertEqual(signal, 'SELL')
        
        # Test signal neutre
        signals = pd.DataFrame({
            'macd': [1.0, 1.0],
            'signal_line': [1.0, 1.0]
        })
        signal = self.strategy.generate_signal(signals)
        self.assertEqual(signal, 'HOLD')

class TestBollingerStrategy(unittest.TestCase):
    def setUp(self):
        """Configuration initiale pour chaque test."""
        self.strategy = BollingerStrategy()
        
        # Données de test
        self.test_data = pd.DataFrame({
            'close': [50000.0, 51000.0, 52000.0, 53000.0, 54000.0],
            'volume': [100.0, 200.0, 300.0, 400.0, 500.0]
        })

    def test_calculate_signals(self):
        """Test le calcul des signaux Bollinger."""
        signals = self.strategy.calculate_signals(self.test_data)
        
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('signal', signals.columns)
        self.assertIn('upper_band', signals.columns)
        self.assertIn('lower_band', signals.columns)
        self.assertIn('middle_band', signals.columns)

    def test_generate_signal(self):
        """Test la génération de signaux."""
        # Test signal d'achat (prix sous la bande inférieure)
        signals = pd.DataFrame({
            'close': [49000.0],
            'lower_band': [50000.0]
        })
        signal = self.strategy.generate_signal(signals)
        self.assertEqual(signal, 'BUY')
        
        # Test signal de vente (prix au-dessus de la bande supérieure)
        signals = pd.DataFrame({
            'close': [51000.0],
            'upper_band': [50000.0]
        })
        signal = self.strategy.generate_signal(signals)
        self.assertEqual(signal, 'SELL')
        
        # Test signal neutre
        signals = pd.DataFrame({
            'close': [50000.0],
            'middle_band': [50000.0]
        })
        signal = self.strategy.generate_signal(signals)
        self.assertEqual(signal, 'HOLD')

class TestCombinedStrategy(unittest.TestCase):
    def setUp(self):
        """Configuration initiale pour chaque test."""
        self.strategy = CombinedStrategy()
        
        # Données de test
        self.test_data = pd.DataFrame({
            'close': [50000.0, 51000.0, 52000.0, 53000.0, 54000.0],
            'volume': [100.0, 200.0, 300.0, 400.0, 500.0]
        })

    def test_calculate_signals(self):
        """Test le calcul des signaux combinés."""
        signals = self.strategy.calculate_signals(self.test_data)
        
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('signal', signals.columns)
        self.assertIn('ema_signal', signals.columns)
        self.assertIn('rsi_signal', signals.columns)
        self.assertIn('macd_signal', signals.columns)
        self.assertIn('bollinger_signal', signals.columns)

    def test_generate_signal(self):
        """Test la génération de signaux."""
        # Test signal d'achat (majorité des stratégies en BUY)
        signals = pd.DataFrame({
            'ema_signal': ['BUY'],
            'rsi_signal': ['BUY'],
            'macd_signal': ['BUY'],
            'bollinger_signal': ['HOLD']
        })
        signal = self.strategy.generate_signal(signals)
        self.assertEqual(signal, 'BUY')
        
        # Test signal de vente (majorité des stratégies en SELL)
        signals = pd.DataFrame({
            'ema_signal': ['SELL'],
            'rsi_signal': ['SELL'],
            'macd_signal': ['SELL'],
            'bollinger_signal': ['HOLD']
        })
        signal = self.strategy.generate_signal(signals)
        self.assertEqual(signal, 'SELL')
        
        # Test signal neutre (égalité ou majorité en HOLD)
        signals = pd.DataFrame({
            'ema_signal': ['HOLD'],
            'rsi_signal': ['HOLD'],
            'macd_signal': ['BUY'],
            'bollinger_signal': ['SELL']
        })
        signal = self.strategy.generate_signal(signals)
        self.assertEqual(signal, 'HOLD')

if __name__ == '__main__':
    unittest.main() 