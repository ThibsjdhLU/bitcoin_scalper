"""
Script de chiffrement de configuration pour SecureConfig (AES-256 CBC)
Usage :
    python app/core/encrypt_config.py config.json config.enc <clé AES 32 bytes en hex ou base64>
- config.json : fichier JSON contenant les secrets (ex: {"MT5_REST_URL": ..., "MT5_REST_API_KEY": ...})
- config.enc : fichier de sortie chiffré
- clé AES : 32 bytes (256 bits), en hexadécimal (64 caractères) ou base64 (44 caractères)
"""
import sys
import os
import json
from base64 import b64encode, b64decode
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
import secrets

def load_key(key_str):
    # Hex ou base64
    if len(key_str) == 64:
        return bytes.fromhex(key_str)
    elif len(key_str) == 44:
        return b64decode(key_str)
    else:
        raise ValueError("Clé AES attendue en hex (64) ou base64 (44)")

def main():
    if len(sys.argv) != 4:
        print("Usage: python app/core/encrypt_config.py config.json config.enc <clé AES 32 bytes en hex ou base64>")
        sys.exit(1)
    json_path, enc_path, key_str = sys.argv[1:4]
    key = load_key(key_str)
    if len(key) != 32:
        print("Clé AES doit faire 32 bytes (256 bits)")
        sys.exit(1)
    with open(json_path, "r") as f:
        data = json.load(f)
    plaintext = json.dumps(data).encode()
    # Padding PKCS7
    padder = padding.PKCS7(128).padder()
    padded = padder.update(plaintext) + padder.finalize()
    # IV aléatoire
    iv = secrets.token_bytes(16)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    encrypted = encryptor.update(padded) + encryptor.finalize()
    # Format : IV (16 bytes) + données chiffrées (base64)
    with open(enc_path, "wb") as f:
        f.write(iv + b64encode(encrypted))
    print(f"Config chiffrée écrite dans {enc_path}")

if __name__ == "__main__":
    main() 