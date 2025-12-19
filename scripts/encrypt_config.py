import sys
import os

# Ajouter le chemin src/ au PYTHONPATH pour importer bitcoin_scalper
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from bitcoin_scalper.core.config import SecureConfig

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python scripts/encrypt_config.py <config_clear.json> <config.enc> <clé_AES_256_hex>")
        print("Example: python scripts/encrypt_config.py config/config.json config/config.enc <key>")
        sys.exit(1)
    input_json = sys.argv[1]
    output_enc = sys.argv[2]
    aes_key_hex = sys.argv[3]
    aes_key = bytes.fromhex(aes_key_hex)
    SecureConfig.encrypt_file(input_json, output_enc, aes_key)
    print(f"Fichier {output_enc} généré avec succès.") 