import yaml
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from base64 import b64encode, b64decode
from typing import Any

class ConfigError(Exception):
    """Exception personnalisée pour la configuration."""
    pass

class SecureConfig:
    """
    Gestionnaire de configuration sécurisée avec chiffrement AES-256 pour les clés sensibles.
    """
    def __init__(self, config_path: str, encryption_key: bytes):
        self.config_path = config_path
        self.encryption_key = encryption_key
        self._config = self._load_config()

    def _load_config(self) -> dict:
        if not os.path.exists(self.config_path):
            raise ConfigError(f"Fichier de configuration introuvable : {self.config_path}")
        with open(self.config_path, 'r') as f:
            try:
                config = yaml.safe_load(f)
            except Exception as e:
                raise ConfigError(f"Erreur de lecture YAML : {e}")
        return config

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def get_encrypted(self, key: str) -> str:
        value = self._config.get(key)
        if value is None:
            raise ConfigError(f"Clé {key} absente de la configuration.")
        return self.decrypt(value)

    def set_encrypted(self, key: str, value: str):
        encrypted = self.encrypt(value)
        self._config[key] = encrypted
        self._save_config()

    def _save_config(self):
        with open(self.config_path, 'w') as f:
            yaml.safe_dump(self._config, f)

    def encrypt(self, plaintext: str) -> str:
        backend = default_backend()
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(self.encryption_key), modes.CBC(iv), backend=backend)
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(plaintext.encode()) + padder.finalize()
        encryptor = cipher.encryptor()
        ct = encryptor.update(padded_data) + encryptor.finalize()
        return b64encode(iv + ct).decode()

    def decrypt(self, ciphertext: str) -> str:
        backend = default_backend()
        data = b64decode(ciphertext)
        iv = data[:16]
        ct = data[16:]
        cipher = Cipher(algorithms.AES(self.encryption_key), modes.CBC(iv), backend=backend)
        decryptor = cipher.decryptor()
        padded_plaintext = decryptor.update(ct) + decryptor.finalize()
        unpadder = padding.PKCS7(128).unpadder()
        plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()
        return plaintext.decode() 