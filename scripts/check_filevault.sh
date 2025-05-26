#!/bin/bash

status=$(fdesetup status)
if [[ "$status" == *"FileVault is On."* ]]; then
  echo "[OK] FileVault est activé."
  exit 0
else
  echo "[ERREUR] FileVault n'est PAS activé ! Activez-le pour la sécurité des données."
  exit 1
fi 