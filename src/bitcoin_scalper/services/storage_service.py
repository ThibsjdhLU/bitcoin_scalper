"""
Service de gestion de la persistance des données.
"""

import logging
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
import os

logger = logging.getLogger(__name__)

class StorageService:
    def __init__(self):
        self.data_dir = Path("data")
        self.logs_dir = Path("logs")
        self._ensure_directories()
        
    def _ensure_directories(self):
        """Crée les répertoires nécessaires s'ils n'existent pas."""
        self.data_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
    def save_trades(self, trades_df: pd.DataFrame) -> bool:
        """Sauvegarde l'historique des trades."""
        try:
            file_path = self.data_dir / "trades.csv"
            trades_df.to_csv(file_path, index=False)
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des trades: {str(e)}")
            return False
            
    def load_trades(self) -> Optional[pd.DataFrame]:
        """Charge l'historique des trades."""
        try:
            file_path = self.data_dir / "trades.csv"
            if file_path.exists():
                return pd.read_csv(file_path)
            return None
        except Exception as e:
            logger.error(f"Erreur lors du chargement des trades: {str(e)}")
            return None
            
    def save_log(self, message: str, level: str = "info") -> bool:
        """Sauvegarde un message de log."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d")
            file_path = self.logs_dir / f"trading_{timestamp}.log"
            
            with open(file_path, "a") as f:
                f.write(f"[{datetime.now().strftime('%H:%M:%S')}] [{level.upper()}] {message}\n")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du log: {str(e)}")
            return False
            
    def save_error(self, error_data: Dict[str, Any]) -> bool:
        """Sauvegarde une erreur dans un fichier JSON."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = self.logs_dir / f"error_{timestamp}.json"
            
            with open(file_path, "w") as f:
                json.dump(error_data, f, indent=4)
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'erreur: {str(e)}")
            return False
            
    def load_errors(self) -> List[Dict[str, Any]]:
        """Charge toutes les erreurs sauvegardées."""
        errors = []
        try:
            for file_path in self.logs_dir.glob("error_*.json"):
                with open(file_path, "r") as f:
                    errors.append(json.load(f))
            return errors
        except Exception as e:
            logger.error(f"Erreur lors du chargement des erreurs: {str(e)}")
            return []
            
    def save_backtest_results(self, results: Dict[str, Any]) -> bool:
        """Sauvegarde les résultats d'un backtest."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = self.data_dir / f"backtest_{timestamp}.json"
            
            with open(file_path, "w") as f:
                json.dump(results, f, indent=4)
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des résultats de backtest: {str(e)}")
            return False
            
    def load_backtest_results(self, timestamp: str) -> Optional[Dict[str, Any]]:
        """Charge les résultats d'un backtest spécifique."""
        try:
            file_path = self.data_dir / f"backtest_{timestamp}.json"
            if file_path.exists():
                with open(file_path, "r") as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Erreur lors du chargement des résultats de backtest: {str(e)}")
            return None

    def save_logs(self, logs: List[str]) -> None:
        """Sauvegarde les logs dans un fichier."""
        try:
            log_file = os.path.join(self.logs_dir, f"trading_logs_{datetime.now().strftime('%Y%m%d')}.txt")
            with open(log_file, "a", encoding="utf-8") as f:
                for log in logs:
                    f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {log}\n")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des logs: {e}") 