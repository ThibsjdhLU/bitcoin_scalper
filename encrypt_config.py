import sys
from bitcoin_scalper.core.config import SecureConfig

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python encrypt_config.py <config_clear.json> <config.enc> <clé_AES_256_hex>")
        sys.exit(1)
    input_json = sys.argv[1]
    output_enc = sys.argv[2]
    aes_key_hex = sys.argv[3]
    aes_key = bytes.fromhex(aes_key_hex)
    SecureConfig.encrypt_file(input_json, output_enc, aes_key)
    print(f"Fichier {output_enc} généré avec succès.") 