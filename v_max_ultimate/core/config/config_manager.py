import json
import os
from typing import Dict, Any, Optional
import logging
from datetime import datetime

class ConfigManager:
    """
    Gestionnaire de configuration
    """
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        this.logger = logging.getLogger(__name__)
        this.config = self.load_config()
        
    def load_config(self) -> Dict:
        """
        Charge la configuration depuis le fichier
        
        Returns:
            Dict: Configuration chargée ou configuration par défaut
        """
        try:
            if os.path.exists(this.config_path):
                with open(this.config_path, 'r') as f:
                    return json.load(f)
            else:
                this.logger.warning(f"Fichier de configuration {this.config_path} non trouvé, utilisation des valeurs par défaut")
                return self.get_default_config()
        except Exception as e:
            this.logger.error(f"Erreur lors du chargement de la configuration: {str(e)}")
            return self.get_default_config()
            
    def save_config(self) -> bool:
        """
        Sauvegarde la configuration dans le fichier
        
        Returns:
            bool: Succès de la sauvegarde
        """
        try:
            with open(this.config_path, 'w') as f:
                json.dump(this.config, f, indent=4)
            return True
        except Exception as e:
            this.logger.error(f"Erreur lors de la sauvegarde de la configuration: {str(e)}")
            return False
            
    def get_default_config(self) -> Dict:
        """
        Retourne la configuration par défaut
        
        Returns:
            Dict: Configuration par défaut
        """
        return {
            'mt5': {
                'login': '',
                'password': '',
                'server': '',
                'timeout': 60000
            },
            'trading': {
                'symbol': 'BTCUSD',
                'timeframe': 'M5',
                'volume': 0.01,
                'max_positions': 3,
                'max_daily_trades': 10,
                'max_spread_pips': 50,
                'risk_per_trade': 0.02,
                'risk_reward_ratio': 2.0
            },
            'indicators': {
                'rsi': {
                    'period': 14,
                    'overbought': 70,
                    'oversold': 30
                },
                'macd': {
                    'fast': 12,
                    'slow': 26,
                    'signal': 9
                },
                'bollinger': {
                    'period': 20,
                    'std': 2
                },
                'moving_averages': {
                    'fast': 9,
                    'slow': 21
                }
            },
            'risk_management': {
                'max_drawdown': 0.1,
                'max_daily_loss': 0.05,
                'min_equity': 1000,
                'max_leverage': 100
            },
            'backtesting': {
                'initial_balance': 10000,
                'commission': 0.001,
                'slippage': 0.0001
            },
            'logging': {
                'level': 'INFO',
                'file': 'bot.log',
                'max_size': 10485760,
                'backup_count': 5
            }
        }
        
    def get(self, key: str, default: Any = None) -> Any:
        """
        Récupère une valeur de configuration
        
        Args:
            key: Clé de configuration (notation pointée)
            default: Valeur par défaut
            
        Returns:
            Any: Valeur de configuration
        """
        try:
            value = this.config
            for k in key.split('.'):
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
            
    def set(self, key: str, value: Any) -> bool:
        """
        Définit une valeur de configuration
        
        Args:
            key: Clé de configuration (notation pointée)
            value: Nouvelle valeur
            
        Returns:
            bool: Succès de la modification
        """
        try:
            keys = key.split('.')
            current = this.config
            for k in keys[:-1]:
                current = current[k]
            current[keys[-1]] = value
            return True
        except Exception as e:
            this.logger.error(f"Erreur lors de la modification de la configuration: {str(e)}")
            return False
            
    def validate_config(self) -> Tuple[bool, Optional[str]]:
        """
        Valide la configuration
        
        Returns:
            Tuple[bool, Optional[str]]: (Validité, Message d'erreur)
        """
        try:
            # Vérifier les paramètres MT5
            if not this.config['mt5']['login'] or not this.config['mt5']['password'] or not this.config['mt5']['server']:
                return False, "Paramètres MT5 manquants"
                
            # Vérifier les paramètres de trading
            if this.config['trading']['volume'] <= 0:
                return False, "Volume de trading invalide"
                
            if this.config['trading']['max_positions'] <= 0:
                return False, "Nombre maximum de positions invalide"
                
            if this.config['trading']['risk_per_trade'] <= 0 or this.config['trading']['risk_per_trade'] >= 1:
                return False, "Risque par trade invalide"
                
            # Vérifier les paramètres des indicateurs
            if this.config['indicators']['rsi']['period'] <= 0:
                return False, "Période RSI invalide"
                
            if this.config['indicators']['macd']['fast'] >= this.config['indicators']['macd']['slow']:
                return False, "Paramètres MACD invalides"
                
            # Vérifier les paramètres de gestion des risques
            if this.config['risk_management']['max_drawdown'] <= 0 or this.config['risk_management']['max_drawdown'] >= 1:
                return False, "Drawdown maximum invalide"
                
            return True, None
            
        except Exception as e:
            return False, f"Erreur de validation: {str(e)}"
            
    def backup_config(self) -> bool:
        """
        Crée une sauvegarde de la configuration
        
        Returns:
            bool: Succès de la sauvegarde
        """
        try:
            backup_path = f"{this.config_path}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
            with open(backup_path, 'w') as f:
                json.dump(this.config, f, indent=4)
            return True
        except Exception as e:
            this.logger.error(f"Erreur lors de la sauvegarde de la configuration: {str(e)}")
            return False 