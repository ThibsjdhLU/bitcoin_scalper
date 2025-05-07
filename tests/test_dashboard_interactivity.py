import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
import shutil
import os
import streamlit as st
from unittest.mock import patch, MagicMock

from dashboard.app import Dashboard

class TestDashboardInteractivity(unittest.TestCase):
    """Tests d'interactivité pour le dashboard."""
    
    def setUp(self):
        """Prépare l'environnement de test."""
        self.test_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.test_dir) / "data"
        self.logs_dir = Path(self.test_dir) / "logs"
        os.makedirs(self.data_dir)
        os.makedirs(self.logs_dir)
        
        # Créer des données de test
        self._create_test_data()
        
        # Initialiser le dashboard
        self.dashboard = Dashboard()
        self.dashboard.data_dir = self.data_dir
        self.dashboard.logs_dir = self.logs_dir
        self.dashboard.load_data()
        
    def _create_test_data(self):
        """Crée des données de test pour les tests d'interactivité."""
        # Données de trading avec différents symboles et stratégies
        trades_data = {
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='h'),
            'symbol': ['BTCUSD'] * 50 + ['ETHUSD'] * 50,
            'strategy': ['EMA'] * 25 + ['RSI'] * 25 + ['BB'] * 25 + ['MACD'] * 25,
            'side': ['BUY', 'SELL'] * 50,
            'volume': np.random.uniform(0.01, 1.0, 100),
            'price': np.random.uniform(40000, 50000, 100),
            'profit': np.random.uniform(-100, 100, 100)
        }
        trades_df = pd.DataFrame(trades_data)
        trades_df.to_csv(self.data_dir / "trades.csv", index=False)
        
        # Données de performance
        performance_data = {
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='h'),
            'balance': np.cumsum(np.random.uniform(-100, 100, 100)) + 10000,
            'equity': np.cumsum(np.random.uniform(-100, 100, 100)) + 10000
        }
        performance_df = pd.DataFrame(performance_data)
        performance_df.to_csv(self.data_dir / "performance.csv", index=False)
        
        # Logs d'erreurs variés
        error_types = ['Connection', 'Order', 'Data', 'System']
        for i in range(20):
            error_log = {
                'timestamp': (datetime.now() - timedelta(hours=i)).isoformat(),
                'error': f'{error_types[i % 4]} error',
                'details': f'Error details {i}'
            }
            with open(self.logs_dir / f"error_{i}.json", 'w') as f:
                json.dump(error_log, f)
                
    @patch('streamlit.selectbox')
    def test_symbol_filter(self, mock_selectbox):
        """Test le filtre par symbole."""
        # Simuler la sélection d'un symbole
        mock_selectbox.return_value = 'BTCUSD'
        
        # Appliquer le filtre
        filtered_data = self.dashboard.trades_df[self.dashboard.trades_df['symbol'] == 'BTCUSD']
        
        # Vérifier le filtrage
        self.assertEqual(len(filtered_data), 50)
        self.assertTrue(all(filtered_data['symbol'] == 'BTCUSD'))
        
    @patch('streamlit.multiselect')
    def test_strategy_filter(self, mock_multiselect):
        """Test le filtre par stratégie."""
        # Simuler la sélection de stratégies
        mock_multiselect.return_value = ['EMA', 'RSI']
        
        # Appliquer le filtre
        filtered_data = self.dashboard.trades_df[self.dashboard.trades_df['strategy'].isin(['EMA', 'RSI'])]
        
        # Vérifier le filtrage
        self.assertEqual(len(filtered_data), 50)
        self.assertTrue(all(filtered_data['strategy'].isin(['EMA', 'RSI'])))
        
    @patch('streamlit.date_input')
    def test_date_range_filter(self, mock_date_input):
        """Test le filtre par plage de dates."""
        # Simuler la sélection d'une plage de dates
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 2)
        mock_date_input.return_value = (start_date, end_date)
        
        # Appliquer le filtre
        filtered_data = self.dashboard.trades_df[
            (self.dashboard.trades_df['timestamp'] >= start_date) &
            (self.dashboard.trades_df['timestamp'] <= end_date)
        ]
        
        # Vérifier le filtrage
        self.assertTrue(len(filtered_data) > 0)
        self.assertTrue(all(filtered_data['timestamp'] >= start_date))
        self.assertTrue(all(filtered_data['timestamp'] <= end_date))
        
    @patch('streamlit.button')
    def test_refresh_button(self, mock_button):
        """Test le bouton de rafraîchissement."""
        # Simuler le clic sur le bouton
        mock_button.return_value = True
        
        # Ajouter de nouvelles données
        new_trade = pd.DataFrame({
            'timestamp': [datetime.now()],
            'symbol': ['BTCUSD'],
            'strategy': ['EMA'],
            'side': ['BUY'],
            'volume': [0.1],
            'price': [45000],
            'profit': [50]
        })
        new_trade.to_csv(self.data_dir / "trades.csv", mode='a', header=False, index=False)
        
        # Rafraîchir les données
        self.dashboard.load_data()
        
        # Vérifier que les nouvelles données sont chargées
        self.assertEqual(len(self.dashboard.trades_df), 101)
        
    @patch('streamlit.expander')
    def test_error_log_expander(self, mock_expander):
        """Test l'expansion des logs d'erreurs."""
        # Simuler l'expansion
        mock_expander.return_value = MagicMock()
        
        # Vérifier que les erreurs sont chargées
        self.assertIsInstance(self.dashboard.errors_df, pd.DataFrame)
        self.assertEqual(len(self.dashboard.errors_df), 20)
        
    def test_chart_interactivity(self):
        """Test l'interactivité des graphiques."""
        # Vérifier que les données pour les graphiques sont correctement formatées
        self.assertIn('timestamp', self.dashboard.performance_df.columns)
        self.assertIn('balance', self.dashboard.performance_df.columns)
        self.assertIn('equity', self.dashboard.performance_df.columns)
        
        # Vérifier que les données sont triées par timestamp
        self.assertTrue(self.dashboard.performance_df['timestamp'].is_monotonic_increasing)
        
    def tearDown(self):
        """Nettoie l'environnement de test."""
        shutil.rmtree(self.test_dir)
        
if __name__ == '__main__':
    unittest.main() 