import pytest
from app.core.config import SecureConfig, ConfigError
import tempfile
import os

@pytest.fixture
def sample_key():
    return b"0123456789abcdef0123456789abcdef"  # 32 bytes AES-256

@pytest.fixture
def sample_config_file(tmp_path):
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        f.write("api_key: test\n")
    return str(config_path)

def test_load_config_success(sample_config_file, sample_key):
    cfg = SecureConfig(sample_config_file, sample_key)
    assert cfg.get("api_key") == "test"

def test_load_config_missing():
    with pytest.raises(ConfigError):
        SecureConfig("/not/exist.yaml", b"0"*32)

def test_encrypt_decrypt(sample_config_file, sample_key):
    cfg = SecureConfig(sample_config_file, sample_key)
    secret = "supersecret"
    encrypted = cfg.encrypt(secret)
    assert isinstance(encrypted, str)
    decrypted = cfg.decrypt(encrypted)
    assert decrypted == secret

def test_set_encrypted_and_get_encrypted(tmp_path, sample_key):
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        f.write("")
    cfg = SecureConfig(str(config_path), sample_key)
    secret = "mysecret"
    cfg.set_encrypted("secret_key", secret)
    assert cfg.get_encrypted("secret_key") == secret 