import os
import json
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from base64 import b64decode, b64encode
from cryptography.hazmat.primitives import padding

# Standardized Training Constants
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1
DEFAULT_WARMUP_ROWS = 500
MIN_DATA_ROWS = 2000
DEFAULT_HORIZON = 30
DEFAULT_N_TRIALS_PRIMARY = 50
DEFAULT_N_TRIALS_META = 30
DEFAULT_PT_SL_MULTIPLIER = 1.5
DEFAULT_META_THRESHOLD = 0.55

class ConfigError(Exception):
    """Exception personnalisée pour la configuration."""
    pass

class SecureConfig:
    """
    Gère le chargement et le déchiffrement sécurisé de la configuration (clé API, login MT5, etc).
    Les secrets sont stockés dans un fichier JSON chiffré avec AES-256.
    """
    def __init__(self, encrypted_config_path: str, aes_key):
        self.encrypted_config_path = encrypted_config_path
        if isinstance(aes_key, str):
            if len(aes_key) == 64:
                self.aes_key = bytes.fromhex(aes_key)
            else:
                raise ConfigError("La clé AES doit être une chaîne hexadécimale de 64 caractères (256 bits)")
        elif isinstance(aes_key, bytes):
            self.aes_key = aes_key
        else:
            raise ConfigError("Format de clé AES non supporté")
        if len(self.aes_key) != 32:
            raise ConfigError("La clé AES doit faire 32 bytes (256 bits)")
        self._config = self._load_and_decrypt()

    def _load_and_decrypt(self) -> Dict:
        if not os.path.exists(self.encrypted_config_path):
            raise ConfigError(f"Fichier de configuration introuvable: {self.encrypted_config_path}")
        with open(self.encrypted_config_path, "rb") as f:
            data = f.read()
        try:
            # Format attendu: IV (16 bytes) + données chiffrées (base64)
            iv = data[:16]
            encrypted = b64decode(data[16:])
            cipher = Cipher(algorithms.AES(self.aes_key), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()
            decrypted = decryptor.update(encrypted) + decryptor.finalize()
            # Padding PKCS7: retirer les bytes de padding
            pad_len = decrypted[-1]
            decrypted = decrypted[:-pad_len]
            config = json.loads(decrypted.decode())
            return config
        except Exception as e:
            raise ConfigError(f"Erreur de déchiffrement: {e}")

    def get(self, key: str, default=None):
        return self._config.get(key, default)

    def as_dict(self) -> Dict:
        return self._config

    @staticmethod
    def encrypt_file(input_json_path: str, output_enc_path: str, aes_key: bytes):
        """
        Chiffre un fichier JSON en AES-256-CBC (IV aléatoire + base64) pour usage avec SecureConfig.
        """
        if len(aes_key) != 32:
            raise ValueError("La clé AES doit faire 32 bytes (256 bits)")
        with open(input_json_path, "r") as f:
            data = json.load(f)
        raw = json.dumps(data).encode()
        # Padding PKCS7
        padder = padding.PKCS7(128).padder()
        padded = padder.update(raw) + padder.finalize()
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        encrypted = encryptor.update(padded) + encryptor.finalize()
        with open(output_enc_path, "wb") as f:
            f.write(iv + b64encode(encrypted))

"""
Exemple d'utilisation:
config = SecureConfig("/chemin/vers/config.enc", os.environ["CONFIG_AES_KEY"])
mt5_login = config.get("mt5_login")

Recommandation sécurité :
- Ne jamais stocker la clé AES en clair dans le code ou les fichiers.
- Utiliser un gestionnaire de secrets externe (HashiCorp Vault, AWS Secrets Manager, Azure Key Vault) pour injecter la clé via variable d'environnement.
- Pour la production, interdire tout fallback sur un fichier de config non chiffré.
""" 

@dataclass
class TradingConfig:
    """
    Central configuration for the trading engine.
    
    This dataclass defines all parameters for trading:
    - Which model to use (XGBoost, CatBoost, PPO, DQN)
    - Risk parameters (drawdown limits, position sizing)
    - Timeframes and symbols
    - API credentials (loaded from environment variables)
    
    Can be loaded from:
    - YAML file
    - JSON file
    - Python dict
    - Environment variables
    
    Example YAML:
        trading:
          mode: ml  # or rl
          model_type: xgboost  # xgboost, catboost, ppo, dqn
          model_path: models/model
          symbol: BTCUSD
          timeframe: M1
        
        risk:
          max_drawdown: 0.05
          max_daily_loss: 0.05
          risk_per_trade: 0.01
          position_sizer: kelly  # or target_vol
          kelly_fraction: 0.25
          target_volatility: 0.15
        
        execution:
          order_type: market
          use_sl_tp: true
          sl_atr_mult: 2.0
          tp_atr_mult: 3.0
        
        drift:
          enabled: true
          safe_mode_on_drift: true
    """
    
    # Trading parameters
    mode: str = "ml"  # "ml" or "rl"
    model_type: str = "xgboost"  # xgboost, catboost, ppo, dqn
    model_path: Optional[str] = None
    symbol: str = "BTCUSD"
    timeframe: str = "M1"
    meta_threshold: float = 0.6
    
    # Risk parameters
    max_drawdown: float = 0.05
    max_daily_loss: float = 0.05
    risk_per_trade: float = 0.01
    max_position_size: float = 1.0
    position_sizer: str = "kelly"  # "kelly" or "target_vol"
    kelly_fraction: float = 0.25
    target_volatility: float = 0.15
    
    # Execution parameters
    order_type: str = "market"
    use_sl_tp: bool = True
    sl_atr_mult: float = 2.0
    tp_atr_mult: float = 3.0
    default_sl_pct: float = 0.01
    default_tp_pct: float = 0.02
    exec_algo: str = "market"  # market, iceberg, vwap, twap
    
    # Drift detection
    drift_enabled: bool = True
    safe_mode_on_drift: bool = True
    
    # Exchange configuration
    exchange: str = field(default_factory=lambda: os.getenv("EXCHANGE", "binance"))  # binance, mt5, paper
    
    # Binance API credentials (loaded from env vars)
    binance_api_key: str = field(default_factory=lambda: os.getenv("BINANCE_API_KEY", ""))
    binance_api_secret: str = field(default_factory=lambda: os.getenv("BINANCE_API_SECRET", ""))
    binance_testnet: bool = field(default_factory=lambda: os.getenv("BINANCE_TESTNET", "true").lower() == "true")
    
    # MT5 API credentials (loaded from env vars - legacy)
    mt5_rest_url: str = field(default_factory=lambda: os.getenv("MT5_REST_URL", "http://localhost:8000"))
    mt5_api_key: str = field(default_factory=lambda: os.getenv("MT5_REST_API_KEY", ""))
    
    # Database (TimescaleDB)
    tsdb_host: str = field(default_factory=lambda: os.getenv("TSDB_HOST", "localhost"))
    tsdb_port: int = field(default_factory=lambda: int(os.getenv("TSDB_PORT", "5432")))
    tsdb_name: str = field(default_factory=lambda: os.getenv("TSDB_NAME", "trading"))
    tsdb_user: str = field(default_factory=lambda: os.getenv("TSDB_USER", "postgres"))
    tsdb_password: str = field(default_factory=lambda: os.getenv("TSDB_PASSWORD", ""))
    tsdb_sslmode: str = field(default_factory=lambda: os.getenv("TSDB_SSLMODE", "prefer"))
    
    # Logging
    log_dir: Optional[str] = None
    log_level: str = "INFO"
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TradingConfig':
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML config file
        
        Returns:
            TradingConfig instance
        """
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_json(cls, json_path: str) -> 'TradingConfig':
        """
        Load configuration from JSON file.
        
        Args:
            json_path: Path to JSON config file
        
        Returns:
            TradingConfig instance
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingConfig':
        """
        Load configuration from dictionary.
        
        Args:
            data: Configuration dictionary
        
        Returns:
            TradingConfig instance
        """
        # Flatten nested dict structure
        flat = {}
        
        # Trading section
        if 'trading' in data:
            flat.update(data['trading'])
        
        # Risk section
        if 'risk' in data:
            for k, v in data['risk'].items():
                flat[k] = v
        
        # Execution section
        if 'execution' in data:
            for k, v in data['execution'].items():
                flat[k] = v
        
        # Drift section
        if 'drift' in data:
            flat['drift_enabled'] = data['drift'].get('enabled', True)
            flat['safe_mode_on_drift'] = data['drift'].get('safe_mode_on_drift', True)
        
        # API section
        if 'api' in data:
            for k, v in data['api'].items():
                flat[k] = v
        
        # Database section
        if 'database' in data:
            for k, v in data['database'].items():
                flat[k] = v
        
        # Logging section
        if 'logging' in data:
            for k, v in data['logging'].items():
                flat[k] = v
        
        # Also support flat structure
        for k, v in data.items():
            if k not in ['trading', 'risk', 'execution', 'drift', 'api', 'database', 'logging']:
                flat[k] = v
        
        # Filter to only fields defined in dataclass
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in flat.items() if k in valid_fields}
        
        return cls(**filtered)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'trading': {
                'mode': self.mode,
                'model_type': self.model_type,
                'model_path': self.model_path,
                'symbol': self.symbol,
                'timeframe': self.timeframe,
            },
            'risk': {
                'max_drawdown': self.max_drawdown,
                'max_daily_loss': self.max_daily_loss,
                'risk_per_trade': self.risk_per_trade,
                'max_position_size': self.max_position_size,
                'position_sizer': self.position_sizer,
                'kelly_fraction': self.kelly_fraction,
                'target_volatility': self.target_volatility,
            },
            'execution': {
                'order_type': self.order_type,
                'use_sl_tp': self.use_sl_tp,
                'sl_atr_mult': self.sl_atr_mult,
                'tp_atr_mult': self.tp_atr_mult,
                'default_sl_pct': self.default_sl_pct,
                'default_tp_pct': self.default_tp_pct,
                'exec_algo': self.exec_algo,
            },
            'drift': {
                'enabled': self.drift_enabled,
                'safe_mode_on_drift': self.safe_mode_on_drift,
            },
        }
    
    def to_yaml(self, output_path: str):
        """Save configuration to YAML file."""
        with open(output_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def to_json(self, output_path: str):
        """Save configuration to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
