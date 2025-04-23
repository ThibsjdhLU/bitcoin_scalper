#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Point d'entrée principal du bot de trading
"""

import argparse
import logging
from pathlib import Path

from .bot import TradingBot

def setup_argparse() -> argparse.ArgumentParser:
    """
    Configure le parseur d'arguments
    
    Returns:
        argparse.ArgumentParser: Parseur configuré
    """
    parser = argparse.ArgumentParser(
        description='Bot de trading automatisé',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '-c', '--config',
        type=str,
        default='config/.env',
        help='Chemin vers le fichier de configuration'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Active le mode verbeux'
    )
    
    return parser

def main():
    """Fonction principale"""
    # Configuration des arguments
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Configuration du logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger('main')
    
    try:
        # Vérification du fichier de configuration
        config_path = Path(args.config)
        if not config_path.exists():
            logger.warning(f"Le fichier de configuration {config_path} n'existe pas")
            logger.info("Un fichier de configuration par défaut sera créé")
        
        # Création et démarrage du bot
        bot = TradingBot(config_path=str(config_path))
        bot.start()
        
    except KeyboardInterrupt:
        logger.info("Arrêt demandé par l'utilisateur")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du bot: {e}")
        raise

if __name__ == '__main__':
    main() 