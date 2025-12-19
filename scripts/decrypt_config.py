import sys
import json
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from base64 import b64decode
from cryptography.hazmat.primitives import padding

def decrypt_config(encrypted_config_path: str, aes_key: bytes) -> dict:
    """
    Déchiffre un fichier de configuration chiffré avec AES-256-CBC.
    """
    if len(aes_key) != 32:
        raise ValueError("La clé AES doit faire 32 bytes (256 bits)")

    with open(encrypted_config_path, "rb") as f:
        data = f.read()

    # Format attendu: IV (16 bytes) + données chiffrées (base64)
    iv = data[:16]
    encrypted = b64decode(data[16:])

    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()

    decrypted_padded = decryptor.update(encrypted) + decryptor.finalize()

    # Padding PKCS7: retirer les bytes de padding
    unpadder = padding.PKCS7(128).unpadder()
    decrypted = unpadder.update(decrypted_padded) + unpadder.finalize()

    config = json.loads(decrypted.decode('utf-8'))
    return config

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/decrypt_config.py <config.enc> <clé_AES_256_hex>")
        print("Example: python scripts/decrypt_config.py config/config.enc <key>")
        sys.exit(1)

    input_enc = sys.argv[1]
    aes_key_hex = sys.argv[2]
    try:
        aes_key = bytes.fromhex(aes_key_hex)
        config_data = decrypt_config(input_enc, aes_key)
        print("Contenu déchiffré:")
        print(json.dumps(config_data, indent=4))
    except Exception as e:
        print(f"Erreur lors du déchiffrement: {e}")
        sys.exit(1) 