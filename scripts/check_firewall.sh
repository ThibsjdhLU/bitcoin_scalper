#!/bin/bash

status=$(/usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate)
if [[ "$status" == *"enabled"* ]]; then
  echo "[OK] Pare-feu macOS activé."
  exit 0
else
  echo "[ERREUR] Pare-feu macOS désactivé ! Activez-le pour la sécurité réseau."
  exit 1
fi 