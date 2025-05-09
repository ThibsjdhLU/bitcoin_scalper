"""
Tests unitaires pour le service MT5.
"""

import unittest
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime
from src.bitcoin_scalper.services.mt5_service import MT5Service

class TestMT5Service(unittest.TestCase):
    def setUp(self):
        """Configuration initiale pour chaque test."""
        self.service = MT5Service()
        
    @patch('services.mt5_service.AvatraderMT5')
    def test_connect_success(self, mock_mt5):
        """Test de la connexion réussie à MT5."""
        # Configuration du mock
        mock_mt5.return_value = Mock()
        
        # Test
        result = self.service.connect()
        
        # Vérifications
        self.assertTrue(result)
        mock_mt5.assert_called_once()
        
    @patch('services.mt5_service.AvatraderMT5')
    def test_connect_failure(self, mock_mt5):
        """Test de l'échec de connexion à MT5."""
        # Configuration du mock pour lever une exception
        mock_mt5.side_effect = Exception("Connection failed")
        
        # Test
        result = self.service.connect()
        
        # Vérifications
        self.assertFalse(result)
        
    @patch('services.mt5_service.MT5Service.connect')
    def test_get_positions(self, mock_connect):
        """Test de la récupération des positions."""
        # Configuration du mock
        mock_positions = [
            {
                'time': datetime.now(),
                'type': 'BUY',
                'price': 50000.0,
                'volume': 0.1,
                'profit': 100.0
            }
        ]
        self.service._connection = Mock()
        self.service._connection.get_positions.return_value = mock_positions
        
        # Test
        result = self.service.get_positions()
        
        # Vérifications
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        self.assertIn('duration', result.columns)
        
    @patch('services.mt5_service.MT5Service.connect')
    def test_get_account_info(self, mock_connect):
        """Test de la récupération des informations du compte."""
        # Configuration du mock
        mock_info = {
            'balance': 10000.0,
            'equity': 10100.0,
            'profit': 100.0
        }
        self.service._connection = Mock()
        self.service._connection.get_account_info.return_value = mock_info
        
        # Test
        result = self.service.get_account_info()
        
        # Vérifications
        self.assertEqual(result, mock_info)
        
    @patch('services.mt5_service.MT5Service.connect')
    def test_get_price_history(self, mock_connect):
        """Test de la récupération de l'historique des prix."""
        # Configuration du mock
        mock_data = pd.DataFrame({
            'open': [50000.0],
            'high': [51000.0],
            'low': [49000.0],
            'close': [50500.0],
            'volume': [100.0]
        })
        self.service._connection = Mock()
        self.service._connection.get_price_history.return_value = mock_data
        
        # Test
        result = self.service.get_price_history("BTCUSD")
        
        # Vérifications
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        
    @patch('services.mt5_service.MT5Service.connect')
    def test_get_available_symbols(self, mock_connect):
        """Test de la récupération des symboles disponibles."""
        # Configuration du mock
        mock_symbols = ["BTCUSD", "ETHUSD", "XRPUSD"]
        self.service._connection = Mock()
        self.service._connection.get_available_symbols.return_value = mock_symbols
        
        # Test
        result = self.service.get_available_symbols()
        
        # Vérifications
        self.assertEqual(result, mock_symbols)

if __name__ == '__main__':
    unittest.main() 