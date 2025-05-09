import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import streamlit as st

from src.strategies.ema_strategy import EMAStrategy
from src.strategies.rsi_strategy import RSIStrategy
from src.strategies.macd_strategy import MACDStrategy
from src.strategies.bollinger_strategy import BollingerStrategy
from src.strategies.combined_strategy import CombinedStrategy
from src.utils.data_processor import DataProcessor
from src.services.mt5_service import MT5Service
from src.services.dashboard_service import DashboardService
from src.utils.risk_manager import RiskManager
from src.utils.config_loader import ConfigLoader
from utils.logger import setup_logger

class TestIntegration(unittest.TestCase):
    def setUp(self):
        """Configuration initiale pour chaque test."""
        # Créer les dossiers nécessaires
        self.log_dir = Path("test_logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialiser les services
        self.mt5_service = MT5Service()
        self.dashboard_service = DashboardService()
        self.dashboard_service.mt5_service = self.mt5_service
        
        # Initialiser les stratégies
        self.ema_strategy = EMAStrategy()
        self.rsi_strategy = RSIStrategy()
        self.macd_strategy = MACDStrategy()
        self.bollinger_strategy = BollingerStrategy()
        self.combined_strategy = CombinedStrategy()
        
        # Initialiser les utilitaires
        self.data_processor = DataProcessor()
        self.risk_manager = RiskManager()
        self.logger = setup_logger("test_integration", self.log_dir / "test_integration.log")
        self.config_loader = ConfigLoader()
        
        # Données de test
        self.test_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='1H'),
            'open': np.random.normal(50000, 1000, 100),
            'high': np.random.normal(51000, 1000, 100),
            'low': np.random.normal(49000, 1000, 100),
            'close': np.random.normal(50000, 1000, 100),
            'volume': np.random.normal(1000, 100, 100)
        })
        
        # Configuration de test
        self.test_config = {
            'trading': {
                'initial_capital': 10000.0,
                'risk_per_trade': 1.0,
                'max_trades': 3,
                'symbols': ['BTCUSDT', 'ETHUSDT']
            },
            'risk': {
                'stop_loss': 2.0,
                'take_profit': 4.0,
                'trailing_stop': True
            },
            'strategy': {
                'ema_fast': 12,
                'ema_slow': 26,
                'rsi_period': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'bollinger_period': 20,
                'bollinger_std': 2
            }
        }

    def test_full_trading_cycle(self):
        """Test un cycle complet de trading."""
        # 1. Charger la configuration
        with patch('json.load', return_value=self.test_config):
            config = self.config_loader.load_config()
            self.assertIsNotNone(config)
        
        # 2. Traiter les données
        processed_data = self.data_processor.calculate_indicators(self.test_data)
        self.assertIsNotNone(processed_data)
        
        # 3. Calculer les signaux pour chaque stratégie
        ema_signals = self.ema_strategy.calculate_signals(processed_data)
        rsi_signals = self.rsi_strategy.calculate_signals(processed_data)
        macd_signals = self.macd_strategy.calculate_signals(processed_data)
        bollinger_signals = self.bollinger_strategy.calculate_signals(processed_data)
        
        self.assertIsNotNone(ema_signals)
        self.assertIsNotNone(rsi_signals)
        self.assertIsNotNone(macd_signals)
        self.assertIsNotNone(bollinger_signals)
        
        # 4. Combiner les signaux
        combined_signals = self.combined_strategy.calculate_signals(processed_data)
        self.assertIsNotNone(combined_signals)
        
        # 5. Calculer la taille de position et les niveaux de prix
        current_price = processed_data['close'].iloc[-1]
        stop_loss = self.risk_manager.calculate_stop_loss(
            current_price,
            'BUY',
            config['risk']['stop_loss']
        )
        take_profit = self.risk_manager.calculate_take_profit(
            current_price,
            'BUY',
            config['risk']['take_profit']
        )
        
        position_size = self.risk_manager.calculate_position_size(
            config['trading']['initial_capital'],
            config['trading']['risk_per_trade'],
            current_price,
            stop_loss
        )
        
        self.assertIsNotNone(stop_loss)
        self.assertIsNotNone(take_profit)
        self.assertIsNotNone(position_size)
        
        # 6. Vérifier les limites de risque
        is_allowed = self.risk_manager.check_risk_limits(
            0,  # Aucun trade en cours
            position_size,
            config['trading']
        )
        self.assertTrue(is_allowed)

    def test_dashboard_integration(self):
        """Test l'intégration avec le dashboard."""
        # 1. Initialiser le RefreshManager
        refresh_manager = RefreshManager(self.dashboard_service)
        self.assertIsNotNone(refresh_manager)
        
        # 2. Simuler des données de marché
        with patch.object(self.mt5_service, 'get_market_data', return_value=self.test_data):
            # 3. Rafraîchir les données
            refresh_manager.start()
            time.sleep(1)  # Attendre un peu pour le rafraîchissement
            
            # 4. Vérifier les données mises à jour
            latest_data = refresh_manager.get_latest_data()
            self.assertIsNotNone(latest_data)
            
            # 5. Arrêter le rafraîchissement
            refresh_manager.stop()
            self.assertFalse(refresh_manager.running)

    def test_error_handling(self):
        """Test la gestion des erreurs."""
        # 1. Simuler une erreur de connexion MT5
        with patch.object(self.mt5_service, 'get_market_data', side_effect=Exception("Connection error")):
            with self.assertRaises(Exception):
                self.dashboard_service.get_raw_data()
        
        # 2. Simuler une erreur de calcul d'indicateurs
        with patch.object(self.data_processor, 'calculate_indicators', side_effect=Exception("Calculation error")):
            with self.assertRaises(Exception):
                self.data_processor.calculate_indicators(self.test_data)
        
        # 3. Simuler une erreur de configuration
        invalid_config = self.test_config.copy()
        invalid_config['trading']['initial_capital'] = -1000.0
        with patch('json.load', return_value=invalid_config):
            with self.assertRaises(ValueError):
                self.config_loader.validate_config(invalid_config)

    def test_performance_metrics(self):
        """Test les métriques de performance."""
        # 1. Simuler des trades
        trades = pd.DataFrame({
            'time': pd.date_range(start='2024-01-01', periods=10, freq='1D'),
            'type': ['BUY', 'SELL'] * 5,
            'price_open': np.random.normal(50000, 1000, 10),
            'price_close': np.random.normal(51000, 1000, 10),
            'profit': np.random.normal(100, 50, 10),
            'duration': np.random.normal(24, 6, 10)
        })
        
        # 2. Calculer les statistiques
        stats = self.dashboard_service.calculate_statistics()
        
        self.assertIn('win_rate', stats)
        self.assertIn('winning_trades', stats)
        self.assertIn('total_trades', stats)
        self.assertIn('avg_profit', stats)
        self.assertIn('max_drawdown', stats)

    def tearDown(self):
        """Nettoyage après chaque test."""
        # Supprimer les fichiers de log de test
        for file in self.log_dir.glob("*.log"):
            file.unlink()
        self.log_dir.rmdir()

if __name__ == '__main__':
    unittest.main() 