import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import cProfile
import pstats
import io
import os
from pathlib import Path
import psutil
import threading

from src.strategies.ema_strategy import EMAStrategy
from src.strategies.rsi_strategy import RSIStrategy
from src.strategies.macd_strategy import MACDStrategy
from src.strategies.bollinger_strategy import BollingerStrategy
from src.utils.data_processor import DataProcessor
from src.services.mt5_service import MT5Service
from src.services.dashboard_service import DashboardService
from src.app import main, RefreshManager
from utils.risk_manager import RiskManager

class TestPerformance(unittest.TestCase):
    def setUp(self):
        """Configuration initiale pour chaque test."""
        # Initialiser les services et composants
        self.mt5_service = MT5Service()
        self.dashboard_service = DashboardService()
        self.dashboard_service.mt5_service = self.mt5_service
        
        self.ema_strategy = EMAStrategy()
        self.rsi_strategy = RSIStrategy()
        self.macd_strategy = MACDStrategy()
        self.bollinger_strategy = BollingerStrategy()
        self.combined_strategy = CombinedStrategy()
        
        self.data_processor = DataProcessor()
        self.risk_manager = RiskManager()
        
        # Données de test (plus grandes pour les tests de performance)
        self.test_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=1000, freq='1H'),
            'open': np.random.normal(50000, 1000, 1000),
            'high': np.random.normal(51000, 1000, 1000),
            'low': np.random.normal(49000, 1000, 1000),
            'close': np.random.normal(50000, 1000, 1000),
            'volume': np.random.normal(1000, 100, 1000)
        })

    def test_data_processing_performance(self):
        """Test les performances du traitement des données."""
        # Profiler le traitement des données
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Traiter les données
        processed_data = self.data_processor.calculate_indicators(self.test_data)
        
        profiler.disable()
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats()
        
        # Vérifier que le traitement est rapide (moins de 1 seconde)
        self.assertLess(time.time() - profiler.start_time, 1.0)
        
        # Vérifier la mémoire utilisée
        memory_usage = processed_data.memory_usage(deep=True).sum()
        self.assertLess(memory_usage, 100 * 1024 * 1024)  # Moins de 100 MB

    def test_strategy_performance(self):
        """Test les performances des stratégies."""
        # Profiler les stratégies
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Calculer les signaux pour chaque stratégie
        ema_signals = self.ema_strategy.calculate_signals(self.test_data)
        rsi_signals = self.rsi_strategy.calculate_signals(self.test_data)
        macd_signals = self.macd_strategy.calculate_signals(self.test_data)
        bollinger_signals = self.bollinger_strategy.calculate_signals(self.test_data)
        combined_signals = self.combined_strategy.calculate_signals(self.test_data)
        
        profiler.disable()
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats()
        
        # Vérifier que le calcul des signaux est rapide (moins de 2 secondes)
        self.assertLess(time.time() - profiler.start_time, 2.0)

    def test_risk_management_performance(self):
        """Test les performances de la gestion des risques."""
        # Profiler la gestion des risques
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Simuler des calculs de risque
        for _ in range(1000):
            price = np.random.normal(50000, 1000)
            stop_loss = price * 0.98
            position_size = self.risk_manager.calculate_position_size(
                10000.0,  # capital initial
                1.0,      # risque par trade
                price,
                stop_loss
            )
        
        profiler.disable()
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats()
        
        # Vérifier que les calculs sont rapides (moins de 1 seconde)
        self.assertLess(time.time() - profiler.start_time, 1.0)

    def test_dashboard_refresh_performance(self):
        """Test les performances du rafraîchissement du dashboard."""
        # Profiler le rafraîchissement
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Simuler le rafraîchissement
        refresh_manager = RefreshManager(self.dashboard_service)
        refresh_manager.start()
        time.sleep(2)  # Attendre quelques rafraîchissements
        refresh_manager.stop()
        
        profiler.disable()
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats()
        
        # Vérifier que le rafraîchissement est rapide
        self.assertLess(time.time() - profiler.start_time, 3.0)

    def test_memory_usage(self):
        """Test l'utilisation de la mémoire."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Effectuer des opérations intensives
        for _ in range(10):
            processed_data = self.data_processor.calculate_indicators(self.test_data)
            ema_signals = self.ema_strategy.calculate_signals(processed_data)
            rsi_signals = self.rsi_strategy.calculate_signals(processed_data)
            macd_signals = self.macd_strategy.calculate_signals(processed_data)
            bollinger_signals = self.bollinger_strategy.calculate_signals(processed_data)
            combined_signals = self.combined_strategy.calculate_signals(processed_data)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Vérifier que l'augmentation de la mémoire est raisonnable (moins de 500 MB)
        self.assertLess(memory_increase, 500 * 1024 * 1024)

    def test_concurrent_operations(self):
        """Test les performances des opérations concurrentes."""
        def run_strategy(strategy, data):
            return strategy.calculate_signals(data)
        
        # Créer des threads pour chaque stratégie
        threads = []
        results = []
        
        start_time = time.time()
        
        for strategy in [self.ema_strategy, self.rsi_strategy, self.macd_strategy, self.bollinger_strategy]:
            thread = threading.Thread(
                target=lambda s=strategy: results.append(run_strategy(s, self.test_data))
            )
            threads.append(thread)
            thread.start()
        
        # Attendre que tous les threads soient terminés
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        # Vérifier que l'exécution parallèle est plus rapide que l'exécution séquentielle
        parallel_time = end_time - start_time
        
        # Exécution séquentielle
        start_time = time.time()
        for strategy in [self.ema_strategy, self.rsi_strategy, self.macd_strategy, self.bollinger_strategy]:
            strategy.calculate_signals(self.test_data)
        sequential_time = time.time() - start_time
        
        # L'exécution parallèle devrait être plus rapide
        self.assertLess(parallel_time, sequential_time)

    def test_data_io_performance(self):
        """Test les performances des opérations d'entrée/sortie."""
        # Créer un fichier temporaire
        temp_file = Path("temp_test_data.csv")
        
        # Mesurer le temps d'écriture
        start_time = time.time()
        self.test_data.to_csv(temp_file)
        write_time = time.time() - start_time
        
        # Mesurer le temps de lecture
        start_time = time.time()
        loaded_data = pd.read_csv(temp_file)
        read_time = time.time() - start_time
        
        # Nettoyer
        temp_file.unlink()
        
        # Vérifier que les opérations d'IO sont rapides
        self.assertLess(write_time, 1.0)
        self.assertLess(read_time, 1.0)

if __name__ == '__main__':
    unittest.main() 