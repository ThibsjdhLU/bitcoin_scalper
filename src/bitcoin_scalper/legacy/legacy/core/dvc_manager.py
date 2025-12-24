"""
Module DVC Manager : gestion du versioning data & artefacts via DVC (Python API + fallback CLI).
Compatible Python 3.11+, conçu pour intégration pipeline ML/data.
"""
import subprocess
from typing import List, Optional
import os

class DVCManager:
    """
    Classe utilitaire pour gérer DVC via Python (API ou CLI fallback).
    """
    def __init__(self, repo_path: Optional[str] = None):
        self.repo_path = repo_path or os.getcwd()

    def _run(self, args: List[str]) -> subprocess.CompletedProcess:
        """Exécute une commande DVC CLI et retourne le résultat."""
        return subprocess.run([
            "dvc", *args
        ], cwd=self.repo_path, capture_output=True, text=True, check=False)

    def init(self) -> bool:
        """Initialise DVC dans le repo courant."""
        result = self._run(["init"])
        return result.returncode == 0

    def add(self, path: str) -> bool:
        """Ajoute un fichier ou dossier à DVC tracking."""
        result = self._run(["add", path])
        return result.returncode == 0

    def commit(self, path: Optional[str] = None) -> bool:
        """Commit les changements DVC (optionnellement sur un chemin donné)."""
        args = ["commit"]
        if path:
            args.append(path)
        result = self._run(args)
        return result.returncode == 0

    def push(self) -> bool:
        """Push les artefacts DVC vers le remote."""
        result = self._run(["push"])
        return result.returncode == 0

    def pull(self) -> bool:
        """Pull les artefacts DVC depuis le remote."""
        result = self._run(["pull"])
        return result.returncode == 0

    def repro(self, target: Optional[str] = None) -> bool:
        """Reproduit le pipeline DVC (optionnellement sur une cible)."""
        args = ["repro"]
        if target:
            args.append(target)
        result = self._run(args)
        return result.returncode == 0

    def status(self) -> str:
        """Retourne le statut DVC (diff entre workspace et remote)."""
        result = self._run(["status"])
        return result.stdout

    def diff(self) -> str:
        """Retourne le diff DVC (données, artefacts, pipeline)."""
        result = self._run(["diff"])
        return result.stdout

    def gc(self, workspace: bool = False) -> bool:
        """Nettoie les artefacts inutiles (garbage collect)."""
        args = ["gc"]
        if workspace:
            args.append("--workspace")
        result = self._run(args)
        return result.returncode == 0

    def remote(self, action: str, name: str, url: Optional[str] = None) -> bool:
        """Gère les remotes DVC (add, remove, modify)."""
        args = ["remote", action, name]
        if url:
            args.append(url)
        result = self._run(args)
        return result.returncode == 0 