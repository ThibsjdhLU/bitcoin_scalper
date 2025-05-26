import sys

if not (sys.version_info.major == 3 and sys.version_info.minor == 11):
    print(f"[ERREUR] Python 3.11.x requis, version détectée : {sys.version}", file=sys.stderr)
    sys.exit(1)
else:
    print(f"[OK] Python 3.11.x détecté : {sys.version}") 