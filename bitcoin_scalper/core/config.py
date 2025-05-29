import os
import json
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from base64 import b64decode, b64encode
from typing import Dict
from cryptography.hazmat.primitives import padding

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
""" 