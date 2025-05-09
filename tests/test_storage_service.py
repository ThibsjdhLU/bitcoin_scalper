"""
Tests unitaires pour le service de stockage.
"""

import unittest
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import shutil
from src.bitcoin_scalper.services.storage_service import StorageService

class TestStorageService(unittest.TestCase):
    def setUp(self):
        """Configuration initiale pour chaque test."""
        self.test_data_dir = Path("test_data")
        self.test_logs_dir = Path("test_logs")
        self.service = StorageService()
        self.service.data_dir = self.test_data_dir
        self.service.logs_dir = self.test_logs_dir
        self.service._ensure_directories()
        
    def tearDown(self):
        """Nettoyage après chaque test."""
        if self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir)
        if self.test_logs_dir.exists():
            shutil.rmtree(self.test_logs_dir)
            
    def test_save_and_load_trades(self):
        """Test de la sauvegarde et du chargement des trades."""
        # Création de données de test
        trades_df = pd.DataFrame({
            'time': [datetime.now()],
            'type': ['BUY'],
            'price': [50000.0],
            'volume': [0.1],
            'profit': [100.0]
        })
        
        # Test de sauvegarde
        save_result = self.service.save_trades(trades_df)
        self.assertTrue(save_result)
        
        # Test de chargement
        loaded_df = self.service.load_trades()
        self.assertIsNotNone(loaded_df)
        self.assertEqual(len(loaded_df), 1)
        self.assertEqual(loaded_df.iloc[0]['type'], 'BUY')
        
    def test_save_and_load_log(self):
        """Test de la sauvegarde des logs."""
        # Test de sauvegarde
        message = "Test log message"
        save_result = self.service.save_log(message, "info")
        self.assertTrue(save_result)
        
        # Vérification du fichier
        log_files = list(self.test_logs_dir.glob("trading_*.log"))
        self.assertEqual(len(log_files), 1)
        
        # Vérification du contenu
        with open(log_files[0], 'r') as f:
            content = f.read()
            self.assertIn(message, content)
            
    def test_save_and_load_error(self):
        """Test de la sauvegarde et du chargement des erreurs."""
        # Création de données de test
        error_data = {
            'timestamp': datetime.now().isoformat(),
            'error_type': 'ConnectionError',
            'message': 'Test error message'
        }
        
        # Test de sauvegarde
        save_result = self.service.save_error(error_data)
        self.assertTrue(save_result)
        
        # Test de chargement
        loaded_errors = self.service.load_errors()
        self.assertEqual(len(loaded_errors), 1)
        self.assertEqual(loaded_errors[0]['error_type'], 'ConnectionError')
        
    def test_save_and_load_backtest_results(self):
        """Test de la sauvegarde et du chargement des résultats de backtest."""
        # Création de données de test
        backtest_results = {
            'timestamp': datetime.now().isoformat(),
            'strategy': 'MA Crossover',
            'metrics': {
                'total_trades': 100,
                'win_rate': 0.6,
                'profit_factor': 1.5
            }
        }
        
        # Test de sauvegarde
        save_result = self.service.save_backtest_results(backtest_results)
        self.assertTrue(save_result)
        
        # Test de chargement
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        loaded_results = self.service.load_backtest_results(timestamp)
        self.assertIsNotNone(loaded_results)
        self.assertEqual(loaded_results['strategy'], 'MA Crossover')
        
    def test_directory_creation(self):
        """Test de la création des répertoires."""
        # Suppression des répertoires
        if self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir)
        if self.test_logs_dir.exists():
            shutil.rmtree(self.test_logs_dir)
            
        # Création des répertoires
        self.service._ensure_directories()
        
        # Vérification
        self.assertTrue(self.test_data_dir.exists())
        self.assertTrue(self.test_logs_dir.exists())

if __name__ == '__main__':
    unittest.main() 