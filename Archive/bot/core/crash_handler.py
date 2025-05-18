"""
Gestionnaire de crash pour le bot de trading.
Gère la sauvegarde d'état et la reprise après crash.
"""
import json
import signal
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from loguru import logger


class CrashHandler:
    """
    Gère les crashes et la reprise du bot.

    Attributes:
        state_file (Path): Fichier de sauvegarde d'état
        last_save (datetime): Dernière sauvegarde
        save_interval (int): Intervalle de sauvegarde en secondes
    """

    def __init__(self, state_file: str = "data/bot_state.json"):
        """
        Initialise le gestionnaire de crash.

        Args:
            state_file: Chemin vers le fichier de sauvegarde d'état
        """
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.last_save = datetime.now()
        self.save_interval = 300  # 5 minutes

        # Enregistrer les handlers de signal
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

    def handle_signal(self, signum, frame):
        """
        Gère les signaux système.

        Args:
            signum: Numéro du signal
            frame: Frame actuelle
        """
        try:
            logger.info("Signal de terminaison reçu, arrêt en cours...")

            # Sauvegarder l'état
            self.save_state()

            # Déconnecter MT5 proprement
            try:
                import MetaTrader5 as mt5

                if mt5.shutdown():
                    logger.info("Déconnexion MT5 effectuée")
                else:
                    logger.error("Erreur lors de la déconnexion MT5")
            except Exception as e:
                logger.error(f"Erreur lors de la déconnexion MT5: {str(e)}")

            # Sauvegarder l'état une dernière fois
            self.save_state()

            # Ne pas forcer l'arrêt
            logger.info(
                "Arrêt demandé, mais ignoré pour permettre le contrôle via l'API"
            )

        except Exception as e:
            logger.error(f"Erreur lors de l'arrêt: {str(e)}")
            logger.info(
                "Arrêt demandé, mais ignoré pour permettre le contrôle via l'API"
            )

    def save_state(self, state: Optional[Dict] = None) -> None:
        """
        Sauvegarde l'état du bot.

        Args:
            state: État à sauvegarder
        """
        try:
            if state is None:
                state = {}

            # Ajouter des métadonnées
            state["timestamp"] = datetime.now().isoformat()
            state["version"] = "1.0.0"

            # Sauvegarder l'état
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=4)

            self.last_save = datetime.now()
            logger.info("État sauvegardé avec succès")

        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'état: {str(e)}")

    def load_state(self) -> Optional[Dict]:
        """
        Charge l'état sauvegardé.

        Returns:
            Optional[Dict]: État chargé ou None si erreur
        """
        try:
            if not self.state_file.exists():
                logger.warning("Aucun état sauvegardé trouvé")
                return None

            with open(self.state_file, "r") as f:
                state = json.load(f)

            logger.info("État chargé avec succès")
            return state

        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'état: {str(e)}")
            return None

    def handle_exception(self, exc_type, exc_value, exc_traceback) -> None:
        """
        Gère une exception non gérée.

        Args:
            exc_type: Type d'exception
            exc_value: Valeur de l'exception
            exc_traceback: Traceback
        """
        # Sauvegarder l'état avant de crasher
        self.save_state()

        # Logger l'erreur
        logger.error("Exception non gérée détectée:")
        logger.error(f"Type: {exc_type.__name__}")
        logger.error(f"Message: {str(exc_value)}")
        logger.error("Traceback:")
        logger.error("".join(traceback.format_tb(exc_traceback)))

    def should_save(self) -> bool:
        """
        Vérifie si une sauvegarde est nécessaire.

        Returns:
            bool: True si sauvegarde nécessaire
        """
        return (datetime.now() - self.last_save).total_seconds() >= self.save_interval

    def __enter__(self):
        """Context manager entry."""
        # Configurer le handler d'exception
        sys.excepthook = self.handle_exception
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            self.handle_exception(exc_type, exc_val, exc_tb)
        else:
            self.save_state()
