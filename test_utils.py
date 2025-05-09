import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

from src.utils.data_processor import DataProcessor
from src.utils.risk_manager import RiskManager
from src.utils.logger import setup_logger
from utils.config_loader import ConfigLoader

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        """Configuration initiale pour chaque test."""
        self.processor = DataProcessor()
        
        # Données de test
        self.test_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=5, freq='1H'),
            'open': [50000.0, 51000.0, 52000.0, 53000.0, 54000.0],
            'high': [51000.0, 52000.0, 53000.0, 54000.0, 55000.0],
            'low': [49000.0, 50000.0, 51000.0, 52000.0, 53000.0],
            'close': [50500.0, 51500.0, 52500.0, 53500.0, 54500.0],
            'volume': [100.0, 200.0, 300.0, 400.0, 500.0]
        })

    def test_calculate_indicators(self):
        """Test le calcul des indicateurs techniques."""
        indicators = self.processor.calculate_indicators(self.test_data)
        
        self.assertIsInstance(indicators, pd.DataFrame)
        self.assertIn('sma_20', indicators.columns)
        self.assertIn('ema_20', indicators.columns)
        self.assertIn('rsi_14', indicators.columns)
        self.assertIn('macd', indicators.columns)
        self.assertIn('macd_signal', indicators.columns)
        self.assertIn('bollinger_upper', indicators.columns)
        self.assertIn('bollinger_lower', indicators.columns)

    def test_normalize_data(self):
        """Test la normalisation des données."""
        normalized_data = self.processor.normalize_data(self.test_data)
        
        self.assertIsInstance(normalized_data, pd.DataFrame)
        self.assertTrue(normalized_data['close'].max() <= 1.0)
        self.assertTrue(normalized_data['close'].min() >= 0.0)

    def test_detect_outliers(self):
        """Test la détection des valeurs aberrantes."""
        outliers = self.processor.detect_outliers(self.test_data['close'])
        
        self.assertIsInstance(outliers, pd.Series)
        self.assertIsInstance(outliers.dtype, bool)

    def test_calculate_volatility(self):
        """Test le calcul de la volatilité."""
        volatility = self.processor.calculate_volatility(self.test_data['close'])
        
        self.assertIsInstance(volatility, float)
        self.assertTrue(volatility >= 0.0)

class TestRiskManager(unittest.TestCase):
    def setUp(self):
        """Configuration initiale pour chaque test."""
        self.risk_manager = RiskManager()
        
        # Configuration de test
        self.config = {
            'initial_capital': 10000.0,
            'risk_per_trade': 1.0,
            'max_trades': 3,
            'stop_loss': 2.0,
            'take_profit': 4.0
        }

    def test_calculate_position_size(self):
        """Test le calcul de la taille de position."""
        price = 50000.0
        stop_loss = 49000.0
        
        position_size = self.risk_manager.calculate_position_size(
            self.config['initial_capital'],
            self.config['risk_per_trade'],
            price,
            stop_loss
        )
        
        self.assertIsInstance(position_size, float)
        self.assertTrue(position_size > 0.0)

    def test_calculate_stop_loss(self):
        """Test le calcul du stop loss."""
        entry_price = 50000.0
        direction = 'BUY'
        
        stop_loss = self.risk_manager.calculate_stop_loss(
            entry_price,
            direction,
            self.config['stop_loss']
        )
        
        self.assertIsInstance(stop_loss, float)
        self.assertTrue(stop_loss < entry_price)

    def test_calculate_take_profit(self):
        """Test le calcul du take profit."""
        entry_price = 50000.0
        direction = 'BUY'
        
        take_profit = self.risk_manager.calculate_take_profit(
            entry_price,
            direction,
            self.config['take_profit']
        )
        
        self.assertIsInstance(take_profit, float)
        self.assertTrue(take_profit > entry_price)

    def test_check_risk_limits(self):
        """Test la vérification des limites de risque."""
        current_trades = 2
        new_position_size = 0.5
        
        is_allowed = self.risk_manager.check_risk_limits(
            current_trades,
            new_position_size,
            self.config
        )
        
        self.assertIsInstance(is_allowed, bool)

class TestLogger(unittest.TestCase):
    def setUp(self):
        """Configuration initiale pour chaque test."""
        self.log_dir = Path("test_logs")
        self.log_dir.mkdir(exist_ok=True)
        
        self.logger = setup_logger("test_logger", self.log_dir / "test.log")

    def test_logger_creation(self):
        """Test la création du logger."""
        self.assertIsNotNone(self.logger)
        self.assertEqual(self.logger.name, "test_logger")

    def test_logging_levels(self):
        """Test les différents niveaux de logging."""
        with patch('logging.Logger.info') as mock_info:
            self.logger.info("Test info message")
            mock_info.assert_called_once()
        
        with patch('logging.Logger.warning') as mock_warning:
            self.logger.warning("Test warning message")
            mock_warning.assert_called_once()
        
        with patch('logging.Logger.error') as mock_error:
            self.logger.error("Test error message")
            mock_error.assert_called_once()

    def tearDown(self):
        """Nettoyage après chaque test."""
        # Supprimer les fichiers de log de test
        for file in self.log_dir.glob("*.log"):
            file.unlink()
        self.log_dir.rmdir()

class TestConfigLoader(unittest.TestCase):
    def setUp(self):
        """Configuration initiale pour chaque test."""
        self.config_loader = ConfigLoader()
        
        # Configuration de test
        self.test_config = {
            'trading': {
                'initial_capital': 10000.0,
                'risk_per_trade': 1.0,
                'max_trades': 3
            },
            'risk': {
                'stop_loss': 2.0,
                'take_profit': 4.0
            }
        }

    def test_load_config(self):
        """Test le chargement de la configuration."""
        with patch('json.load') as mock_load:
            mock_load.return_value = self.test_config
            config = self.config_loader.load_config()
            
            self.assertEqual(config, self.test_config)
            mock_load.assert_called_once()

    def test_save_config(self):
        """Test la sauvegarde de la configuration."""
        with patch('json.dump') as mock_dump:
            self.config_loader.save_config(self.test_config)
            mock_dump.assert_called_once()

    def test_validate_config(self):
        """Test la validation de la configuration."""
        is_valid = self.config_loader.validate_config(self.test_config)
        self.assertTrue(is_valid)
        
        # Test avec une configuration invalide
        invalid_config = self.test_config.copy()
        invalid_config['trading']['initial_capital'] = -1000.0
        is_valid = self.config_loader.validate_config(invalid_config)
        self.assertFalse(is_valid)

if __name__ == '__main__':
    unittest.main() 