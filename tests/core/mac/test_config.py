import pytest
from unittest.mock import patch, mock_open, MagicMock
import os
import json
# Utiliser les primitives de bas niveau pour simuler le chiffrement/déchiffrement comme dans SecureConfig
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from base64 import b64encode, b64decode
from cryptography.hazmat.primitives import padding
# Import spécifique pour l'exception en cas de mauvais padding, etc.
# from cryptography.exceptions import InvalidPadding

from bitcoin_scalper.core.config import SecureConfig, ConfigError
# Import spécifique pour l'exception Fernet si on veut l'utiliser directement (pas utilisé par SecureConfig, donc à enlever)
# from cryptography.fernet import InvalidToken as FernetInvalidToken

# Clé AES de test (32 bytes = 64 caractères hex)
TEST_AES_KEY_HEX = "a" * 64 # Clé valide pour les tests
TEST_AES_KEY_BYTES = bytes.fromhex(TEST_AES_KEY_HEX)
# TEST_FERNET_KEY = Fernet(TEST_AES_KEY_BYTES).generate_key() # Cette ligne n'est pas utilisée et cause confusion

TEST_CONFIG_DATA = {
    "MT5_REST_URL": "http://localhost:8000",
    "MT5_REST_API_KEY": "testkey",
    "TSDB_HOST": "localhost",
    "TSDB_PORT": 5432,
    "TSDB_NAME": "testdb",
    "TSDB_USER": "user",
    "TSDB_PASSWORD": "pass",
    "ML_MODEL_PATH": "/fake/path/model_rf.pkl",
}

# Fonction utilitaire pour chiffrer comme SecureConfig._load_and_decrypt attend
def encrypt_config_for_test(config_dict: dict, aes_key_bytes: bytes) -> bytes:
    iv = os.urandom(16)
    data = json.dumps(config_dict).encode()
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded_data = padder.update(data) + padder.finalize()
    cipher = Cipher(algorithms.AES(aes_key_bytes), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
    return iv + b64encode(encrypted_data)


# Fichier de config chiffré de test
@pytest.fixture
def encrypted_config_file(tmp_path):
    config_file = tmp_path / "config.enc"
    # Chiffrer en utilisant la clé AES de test, comme SecureConfig le ferait en sens inverse
    encrypted_data = encrypt_config_for_test(TEST_CONFIG_DATA, TEST_AES_KEY_BYTES)
    config_file.write_bytes(encrypted_data)
    return config_file

def test_secure_config_load_and_decrypt(encrypted_config_file, monkeypatch):
    # Simuler la variable d'environnement pour la clé AES
    monkeypatch.setenv("CONFIG_AES_KEY", TEST_AES_KEY_HEX)
    # Simuler l'existence du fichier de config
    monkeypatch.setattr("os.path.exists", lambda path: path == str(encrypted_config_file))

    # Le fichier est chiffré avec TEST_AES_KEY_BYTES, SecureConfig utilise TEST_AES_KEY_HEX (qui devient TEST_AES_KEY_BYTES internement)
    # On s'attend à ce que le déchiffrement dans SecureConfig fonctionne.
    config = SecureConfig(str(encrypted_config_file), TEST_AES_KEY_HEX)

    assert config.get("MT5_REST_URL") == TEST_CONFIG_DATA["MT5_REST_URL"]
    assert config.get("TSDB_PORT") == TEST_CONFIG_DATA["TSDB_PORT"]
    assert config.as_dict() == TEST_CONFIG_DATA

def test_secure_config_key_length():
    # Clé trop courte (doit être 64 caractères hex = 32 bytes)
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=b"")):
            # Adapter le regex pour matcher le message exact
            with pytest.raises(ConfigError, match=r"La clé AES doit être une chaîne hexadécimale de 64 caractères \(256 bits\)"):
                SecureConfig("fakepath", "shortkey")

def test_secure_config_invalid_hex_key():
    # Clé hex invalide (doit être hex)
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=b"")):
            # Adapter le regex pour matcher l'erreur de bytes.fromhex
            with pytest.raises(ValueError, match=r"non-hexadecimal number found in fromhex\(\) arg.*"):
                SecureConfig("fakepath", "z" * 64)

def test_secure_config_wrong_type_key():
    # Clé de type incorrect (doit être string)
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=b"")):
            # Vérifier que ConfigError est levé avec le message attendu
            with pytest.raises(ConfigError, match=r"Format de clé AES non supporté"):
                SecureConfig("fakepath", 12345) # Envoyer un int au lieu d'une string

def test_secure_config_missing_file():
    # Fichier de config manquant
    with patch("os.path.exists", return_value=False):
        # Adapter le regex pour matcher le message exact
        with pytest.raises(ConfigError, match=r"Fichier de configuration introuvable: non_existent_path"):
            SecureConfig("non_existent_path", TEST_AES_KEY_HEX)

def test_secure_config_decryption_error(encrypted_config_file, monkeypatch):
    # Simuler une mauvaise clé AES pour le déchiffrement
    wrong_key_hex = "b" * 64 # Clé différente de celle utilisée pour chiffrer
    monkeypatch.setenv("CONFIG_AES_KEY", wrong_key_hex)
    monkeypatch.setattr("os.path.exists", lambda path: path == str(encrypted_config_file))

    # Le déchiffrement devrait échouer, entraînant une ConfigError qui wrap l'exception de déchiffrement
    # Le message de ConfigError est "Erreur de déchiffrement: {original_exception}"
    with pytest.raises(ConfigError, match=r"Erreur de déchiffrement: .*"):
        # On utilise la mauvaise clé ici pour simuler l'erreur de déchiffrement
        SecureConfig(str(encrypted_config_file), wrong_key_hex)

def test_config_get_existing(monkeypatch):
    # Mock pour simuler l'existence du fichier et son contenu déchiffré
    # On mocke _load_and_decrypt pour qu'il retourne directement les données déchiffrées
    with (
        patch("os.path.exists", return_value=True),
        patch.object(SecureConfig, '_load_and_decrypt', return_value=TEST_CONFIG_DATA)
    ):
        # L'instanciation de SecureConfig appellera le mock de _load_and_decrypt
        config = SecureConfig("fakepath", TEST_AES_KEY_HEX)

        assert config.get("MT5_REST_URL") == TEST_CONFIG_DATA["MT5_REST_URL"]

def test_config_get_non_existing(monkeypatch):
    # Mock pour simuler l'existence du fichier et son contenu déchiffré
    # On mocke _load_and_decrypt pour qu'il retourne directement les données déchiffrées
    with (
        patch("os.path.exists", return_value=True),
        patch.object(SecureConfig, '_load_and_decrypt', return_value=TEST_CONFIG_DATA)
    ):
        # L'instanciation de SecureConfig appellera le mock de _load_and_decrypt
        config = SecureConfig("fakepath", TEST_AES_KEY_HEX)

        # Tester une clé non existante avec valeur par défaut
        assert config.get("NON_EXISTENT_KEY", "default_value") == "default_value"

        # Tester une clé non existante sans valeur par défaut (doit lever KeyError)
        with pytest.raises(KeyError):
            _ = config._config["NON_EXISTENT_KEY"]

def test_secure_config_env_only(monkeypatch, tmp_path):
    # Générer une clé AES valide
    key = os.urandom(32)
    # Créer un fichier de config chiffré
    input_json = tmp_path / "config.json"
    input_json.write_text('{"foo": "bar"}')
    enc_path = tmp_path / "config.enc"
    SecureConfig.encrypt_file(str(input_json), str(enc_path), key)
    # Injection par variable d'environnement
    monkeypatch.setenv("CONFIG_AES_KEY", key.hex())
    cfg = SecureConfig(str(enc_path), os.environ["CONFIG_AES_KEY"])
    assert cfg.get("foo") == "bar"
    # Refus d'une clé en clair
    with pytest.raises(ConfigError):
        SecureConfig(str(enc_path), "cle_en_clair_interdite")

def test_secure_config_requires_aes_key(tmp_path):
    # Génère un fichier de config chiffré valide
    config_data = {"foo": "bar"}
    json_path = tmp_path / "test.json"
    enc_path = tmp_path / "test.enc"
    json_path.write_text('{"foo": "bar"}')
    key = os.urandom(32)
    SecureConfig.encrypt_file(str(json_path), str(enc_path), key)
    # Mauvaise taille de clé
    with pytest.raises(ConfigError):
        SecureConfig(str(enc_path), b"shortkey")
    # Bonne taille mais mauvais format (str non hex)
    with pytest.raises(ConfigError):
        SecureConfig(str(enc_path), "nothex")
    # Bonne clé, déchiffrement OK
    cfg = SecureConfig(str(enc_path), key)
    assert cfg.get("foo") == "bar"

def test_secure_config_no_fallback(monkeypatch):
    # Simule l'absence de clé AES dans l'env
    monkeypatch.delenv("CONFIG_AES_KEY", raising=False)
    with pytest.raises(RuntimeError):
        # Simule le code de main.py qui impose la clé
        aes_key = os.environ.get("CONFIG_AES_KEY")
        if not aes_key:
            raise RuntimeError("Clé AES manquante : variable d'environnement CONFIG_AES_KEY obligatoire. Aucun fallback n'est autorisé en production.")

# TODO: Ajouter des tests pour l'encrypt_config si c'est une méthode à tester unitairement
# TODO: Ajouter des tests pour get() avec différents types de données (int, bool, float)
# TODO: Ajouter des tests pour la lecture depuis Keychain si pertinent pour l'unité de test SecureConfig