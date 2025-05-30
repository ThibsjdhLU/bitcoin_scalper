import hashlib
import sys

SALT = b"bitcoin_scalper_salt"
ITERATIONS = 200_000

def derive_key_from_password(password: str, salt: bytes = SALT, iterations: int = ITERATIONS) -> bytes:
    """Dérive une clé AES-256 à partir d'un mot de passe utilisateur."""
    return hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt,
        iterations,
        dklen=32  # 256 bits
    )

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_password_key.py <password>")
        sys.exit(1)
    password = sys.argv[1]
    derived_key = derive_key_from_password(password)
    print(derived_key.hex()) 