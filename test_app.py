import unittest
from unittest.mock import Mock, patch
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path
import json
import os
import threading
import queue

from app import (
    RefreshManager,
    apply_css,
    header,
    refresh_controls,
    symbol_selector,
    price_chart,
    statistics,
    trades_history,
    logs_console,
    config_panel,
    check_critical_alerts,
    main
)

class TestApp(unittest.TestCase):
    def setUp(self):
        """Configuration initiale pour chaque test."""
        # Mock Streamlit session state
        self.mock_session_state = {
            'bot_status': 'Inactif',
            'last_refresh': datetime.now(),
            'refresh_interval': 10,
            'data_loaded': False,
            'confirm_action': None,
            'selected_symbol': 'BTCUSDT',
            'available_symbols': ['BTCUSDT', 'ETHUSDT'],
            'trading_params': {
                'initial_capital': 1000.0,
                'risk_per_trade': 1.0,
                'strategy': ['EMA Crossover'],
                'take_profit': 2.0,
                'stop_loss': 1.0,
                'trailing_stop': True
            },
            'account_stats': {
                'balance': 1000.0,
                'profit': 0.0,
                'max_drawdown': 0.0
            },
            'trades_history': pd.DataFrame({
                'time': [datetime.now()],
                'type': ['BUY'],
                'price_open': [50000.0],
                'price_close': [51000.0],
                'profit': [100.0],
                'duration': [1.0]
            }),
            'indicators': {
                'show_sma': True,
                'sma_period': 20,
                'show_ema': True,
                'ema_period': 20,
                'show_bollinger': True,
                'bollinger_period': 20,
                'show_rsi': True,
                'rsi_period': 14,
                'show_macd': True,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9
            }
        }
        
        # Mock Streamlit functions
        self.mock_st = Mock()
        self.mock_st.session_state = self.mock_session_state
        
        # Mock DashboardService
        self.mock_dashboard_service = Mock()
        self.mock_dashboard_service.get_raw_data.return_value = {
            'prices': pd.Series([50000.0, 51000.0]),
            'volumes': pd.Series([100.0, 200.0])
        }
        self.mock_dashboard_service.calculate_statistics.return_value = {
            'win_rate': 60.0,
            'winning_trades': 3,
            'total_trades': 5,
            'avg_profit': 100.0,
            'max_drawdown': 5.0
        }
        self.mock_dashboard_service.create_price_chart.return_value = Mock()
        self.mock_dashboard_service.get_available_symbols.return_value = ['BTCUSDT', 'ETHUSDT']
        
        # Patch Streamlit
        self.st_patcher = patch('streamlit.st', self.mock_st)
        self.st_patcher.start()
        
        # Patch DashboardService
        self.dashboard_patcher = patch('app.DashboardService', return_value=self.mock_dashboard_service)
        self.dashboard_patcher.start()

    def tearDown(self):
        """Nettoyage après chaque test."""
        self.st_patcher.stop()
        self.dashboard_patcher.stop()

    def test_refresh_manager_initialization(self):
        """Test l'initialisation du RefreshManager."""
        refresh_manager = RefreshManager(self.mock_dashboard_service)
        self.assertFalse(refresh_manager.running)
        self.assertEqual(refresh_manager.refresh_interval, 10)
        self.assertIsInstance(refresh_manager.refresh_queue, queue.Queue)

    def test_refresh_manager_start_stop(self):
        """Test le démarrage et l'arrêt du RefreshManager."""
        refresh_manager = RefreshManager(self.mock_dashboard_service)
        refresh_manager.start()
        self.assertTrue(refresh_manager.running)
        self.assertIsInstance(refresh_manager.thread, threading.Thread)
        
        refresh_manager.stop()
        self.assertFalse(refresh_manager.running)

    def test_refresh_manager_get_latest_data(self):
        """Test la récupération des dernières données."""
        refresh_manager = RefreshManager(self.mock_dashboard_service)
        test_data = {'timestamp': datetime.now(), 'data': {'test': 'data'}}
        refresh_manager.refresh_queue.put(test_data)
        
        latest_data = refresh_manager.get_latest_data()
        self.assertEqual(latest_data, test_data)

    def test_apply_css(self):
        """Test l'application du CSS."""
        with patch('streamlit.markdown') as mock_markdown:
            apply_css()
            mock_markdown.assert_called_once()

    def test_header(self):
        """Test l'affichage de l'en-tête."""
        with patch('streamlit.columns') as mock_columns:
            mock_columns.return_value = [Mock(), Mock()]
            header()
            mock_columns.assert_called_once()

    def test_refresh_controls(self):
        """Test les contrôles de rafraîchissement."""
        with patch('streamlit.columns') as mock_columns:
            mock_columns.return_value = [Mock(), Mock(), Mock(), Mock()]
            refresh_controls()
            mock_columns.assert_called_once()

    def test_symbol_selector(self):
        """Test le sélecteur de symbole."""
        with patch('streamlit.selectbox') as mock_selectbox:
            mock_selectbox.return_value = 'BTCUSDT'
            symbol_selector()
            mock_selectbox.assert_called_once()

    def test_price_chart(self):
        """Test l'affichage du graphique des prix."""
        with patch('streamlit.expander') as mock_expander:
            mock_expander.return_value.__enter__.return_value = Mock()
            price_chart()
            mock_expander.assert_called_once()

    def test_statistics(self):
        """Test l'affichage des statistiques."""
        with patch('streamlit.columns') as mock_columns:
            mock_columns.return_value = [Mock(), Mock(), Mock(), Mock()]
            statistics()
            mock_columns.assert_called_once()

    def test_trades_history(self):
        """Test l'affichage de l'historique des trades."""
        with patch('streamlit.dataframe') as mock_dataframe:
            trades_history()
            mock_dataframe.assert_called_once()

    def test_logs_console(self):
        """Test l'affichage de la console des logs."""
        with patch('streamlit.columns') as mock_columns:
            mock_columns.return_value = [Mock(), Mock(), Mock()]
            logs_console()
            mock_columns.assert_called_once()

    def test_config_panel(self):
        """Test l'affichage du panneau de configuration."""
        with patch('streamlit.sidebar') as mock_sidebar:
            config_panel()
            mock_sidebar.assert_called_once()

    def test_check_critical_alerts(self):
        """Test la vérification des alertes critiques."""
        self.mock_session_state['account_stats']['max_drawdown'] = 20.0
        with patch('streamlit.markdown') as mock_markdown:
            check_critical_alerts()
            mock_markdown.assert_called_once()

    def test_main(self):
        """Test la fonction principale."""
        with patch('streamlit.set_page_config') as mock_config:
            main()
            mock_config.assert_called_once()

if __name__ == '__main__':
    unittest.main() 