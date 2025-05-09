import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

from services.dashboard_service import DashboardService
from services.mt5_service import MT5Service

class TestDashboardService(unittest.TestCase):
    def setUp(self):
        """Configuration initiale pour chaque test."""
        self.mock_mt5_service = Mock()
        self.dashboard_service = DashboardService()
        self.dashboard_service.mt5_service = self.mock_mt5_service

    def test_get_raw_data(self):
        """Test la récupération des données brutes."""
        mock_data = {
            'prices': pd.Series([50000.0, 51000.0]),
            'volumes': pd.Series([100.0, 200.0])
        }
        self.mock_mt5_service.get_market_data.return_value = mock_data
        
        result = self.dashboard_service.get_raw_data()
        self.assertEqual(result, mock_data)
        self.mock_mt5_service.get_market_data.assert_called_once()

    def test_calculate_statistics(self):
        """Test le calcul des statistiques."""
        mock_trades = pd.DataFrame({
            'time': [datetime.now()],
            'type': ['BUY'],
            'price_open': [50000.0],
            'price_close': [51000.0],
            'profit': [100.0],
            'duration': [1.0]
        })
        
        self.mock_mt5_service.get_trades_history.return_value = mock_trades
        
        stats = self.dashboard_service.calculate_statistics()
        
        self.assertIn('win_rate', stats)
        self.assertIn('winning_trades', stats)
        self.assertIn('total_trades', stats)
        self.assertIn('avg_profit', stats)
        self.assertIn('max_drawdown', stats)

    def test_create_price_chart(self):
        """Test la création du graphique des prix."""
        mock_data = {
            'prices': pd.Series([50000.0, 51000.0]),
            'volumes': pd.Series([100.0, 200.0])
        }
        self.mock_mt5_service.get_market_data.return_value = mock_data
        
        chart = self.dashboard_service.create_price_chart()
        self.assertIsNotNone(chart)

    def test_get_available_symbols(self):
        """Test la récupération des symboles disponibles."""
        mock_symbols = ['BTCUSDT', 'ETHUSDT']
        self.mock_mt5_service.get_available_symbols.return_value = mock_symbols
        
        symbols = self.dashboard_service.get_available_symbols()
        self.assertEqual(symbols, mock_symbols)

    def test_save_trading_params(self):
        """Test la sauvegarde des paramètres de trading."""
        params = {
            'initial_capital': 1000.0,
            'risk_per_trade': 1.0,
            'strategy': ['EMA Crossover'],
            'take_profit': 2.0,
            'stop_loss': 1.0,
            'trailing_stop': True
        }
        
        with patch('json.dump') as mock_dump:
            self.dashboard_service.save_trading_params(params)
            mock_dump.assert_called_once()

    def test_handle_bot_action(self):
        """Test la gestion des actions du bot."""
        with patch('streamlit.session_state') as mock_session:
            mock_session.bot_status = 'Inactif'
            
            # Test démarrage
            self.dashboard_service.handle_bot_action('start')
            self.mock_mt5_service.start_bot.assert_called_once()
            
            # Test arrêt
            self.dashboard_service.handle_bot_action('stop')
            self.mock_mt5_service.stop_bot.assert_called_once()
            
            # Test réinitialisation
            self.dashboard_service.handle_bot_action('reset')
            self.mock_mt5_service.reset_bot.assert_called_once()

class TestMT5Service(unittest.TestCase):
    def setUp(self):
        """Configuration initiale pour chaque test."""
        self.mt5_service = MT5Service()

    def test_is_demo_mode(self):
        """Test la vérification du mode démo."""
        with patch('os.getenv', return_value='1'):
            self.assertTrue(self.mt5_service.is_demo_mode())
        
        with patch('os.getenv', return_value='0'):
            self.assertFalse(self.mt5_service.is_demo_mode())

    def test_set_demo_mode(self):
        """Test la configuration du mode démo."""
        with patch('os.environ.__setitem__') as mock_set:
            self.mt5_service.set_demo_mode(True)
            mock_set.assert_called_with('DEMO_MODE', '1')
            
            self.mt5_service.set_demo_mode(False)
            mock_set.assert_called_with('DEMO_MODE', '0')

    def test_get_market_data(self):
        """Test la récupération des données de marché."""
        with patch('MetaTrader5.copy_rates_from_pos') as mock_copy:
            mock_copy.return_value = np.array([
                (datetime.now().timestamp(), 50000.0, 51000.0, 49000.0, 50500.0, 100.0),
                (datetime.now().timestamp(), 51000.0, 52000.0, 50000.0, 51500.0, 200.0)
            ])
            
            data = self.mt5_service.get_market_data('BTCUSDT', '1h', 100)
            self.assertIn('prices', data)
            self.assertIn('volumes', data)

    def test_get_trades_history(self):
        """Test la récupération de l'historique des trades."""
        with patch('MetaTrader5.history_deals_get') as mock_history:
            mock_history.return_value = [
                Mock(
                    time=datetime.now().timestamp(),
                    type=0,  # BUY
                    price=50000.0,
                    profit=100.0,
                    volume=1.0
                )
            ]
            
            trades = self.mt5_service.get_trades_history()
            self.assertIsInstance(trades, pd.DataFrame)

    def test_get_available_symbols(self):
        """Test la récupération des symboles disponibles."""
        with patch('MetaTrader5.symbols_get') as mock_symbols:
            mock_symbols.return_value = [
                Mock(name='BTCUSDT'),
                Mock(name='ETHUSDT')
            ]
            
            symbols = self.mt5_service.get_available_symbols()
            self.assertEqual(symbols, ['BTCUSDT', 'ETHUSDT'])

if __name__ == '__main__':
    unittest.main() 