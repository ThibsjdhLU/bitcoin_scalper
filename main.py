"""
Point d'entrée principal du bot de trading crypto.
Orchestre l'initialisation et l'exécution des différents composants.
"""
import sys
from pathlib import Path

# Ajouter le répertoire racine au PYTHONPATH
root_dir = Path(__file__).parent
sys.path.append(str(root_dir))

from utils.logger import setup_logger, get_logger

def main():
    """
    Fonction principale du bot de trading.
    Initialise les composants et lance le cycle de trading.
    """
    # Configurer le logger
    setup_logger()
    logger = get_logger()
    
    logger.info("Démarrage du bot de trading crypto")
    
    try:
        # TODO: Initialiser les composants
        # - Connexion MT5
        # - Chargement des stratégies
        # - Initialisation du gestionnaire de risques
        
        logger.info("Initialisation terminée avec succès")
        
        # TODO: Lancer le cycle de trading
        # - Boucle principale
        # - Gestion des signaux
        # - Exécution des ordres
        
    except Exception as e:
        logger.error(f"Erreur critique: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 