import json
import os
from typing import Dict, Any, Optional

class ConfigManager:
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config: Dict[str, Any] = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Charge la configuration depuis le fichier"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print("Erreur lors du chargement de la configuration")
                return self._get_default_config()
        return self._get_default_config()

    def save_config(self) -> bool:
        """Sauvegarde la configuration dans le fichier"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
            return True
        except Exception as e:
            print(f"Erreur lors de la sauvegarde de la configuration: {str(e)}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """Récupère une valeur de configuration"""
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Définit une valeur de configuration"""
        self.config[key] = value
        self.save_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Retourne la configuration par défaut"""
        # Configuration par défaut
        self.default_config = {
            "symbol": "BTC/USD",
            "timeframe": "1m",
            "min_volume": 1000000,
            "min_volatility": 0.001,
            "signal_strength_threshold": 2,
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bb_period": 20,
            "bb_std": 2,
            "atr_period": 14,
            "atr_multiplier": 2
        }
        return {
            "platform": "mt5",
            "credentials": {
                "login": "101490774",  # Remplacez par votre numéro de compte MT5
                "password": "MatLB356&",  # Remplacez par votre mot de passe MT5
                "server": "Ava - Demo 1-MT5"  # Remplacez par votre serveur MT5
            },
            "trading": {
                "symbol": "BTC/USDT",
                "position_size": 0.01,
                "stop_loss": 2.0,
                "take_profit": 4.0,
                "max_risk": 1.0,
                "max_positions": 3
            },
            "indicators": {
                "rsi_period": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "bb_period": 20,
                "bb_std": 2
            }
        } 