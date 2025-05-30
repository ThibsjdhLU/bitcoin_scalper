"""
Script de migration : rechiffre config.enc avec une clé dérivée d'un mot de passe utilisateur (PBKDF2).
Usage : python scripts/migrate_config_to_password.py
- Nécessite config_clear.json et l'ancienne clé hexadécimale (copie temporaire).
- Demande le mot de passe utilisateur pour la nouvelle méthode.
- Écrase config.enc avec la nouvelle méthode (sauvegardez l'ancien avant !)
"""
import getpass
import hashlib
from bitcoin_scalper.core.config import SecureConfig
import os

OLD_KEY_HEX = "b7e2a1c3d4f5e6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1"
SALT = b"bitcoin_scalper_salt"
ITERATIONS = 200_000

if not os.path.exists("config_clear.json"):
    print("Erreur : config_clear.json introuvable.")
    exit(1)

# 1. Déchiffre l'ancien config.enc (optionnel, pour vérification)
try:
    old_key = bytes.fromhex(OLD_KEY_HEX)
    old_config = SecureConfig("config.enc", old_key)
    config_dict = old_config.as_dict()
    print("Ancienne config décodée avec succès.")
except Exception as e:
    print(f"Erreur lors du déchiffrement avec l'ancienne clé : {e}")
    print("On va utiliser config_clear.json pour rechiffrer.")
    import json
    with open("config_clear.json", "r") as f:
        config_dict = json.load(f)

# 2. Demande le mot de passe utilisateur
password = getpass.getpass("Nouveau mot de passe pour la config : ")
def derive_key_from_password(password: str, salt: bytes = SALT, iterations: int = ITERATIONS) -> bytes:
    return hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, iterations, dklen=32)
new_key = derive_key_from_password(password)

# 3. Rechiffre la config avec la nouvelle clé
import json
with open("config_clear.json", "r") as f:
    config_data = json.load(f)
SecureConfig.encrypt_file("config_clear.json", "config.enc", new_key)
print("Nouveau fichier config.enc généré avec la méthode mot de passe. Sauvegardez bien votre mot de passe !") 