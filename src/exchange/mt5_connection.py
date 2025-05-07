import MetaTrader5 as mt5
import logging
from typing import Optional

class MT5Connection:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MT5Connection, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = logging.getLogger('TradingBot.MT5Connection')
            self._initialized = True
            self.connected = False
    
    def initialize(self, login: int, password: str, server: str) -> bool:
        """
        Initialise la connexion à MetaTrader 5.
        
        Args:
            login (int): Identifiant de connexion
            password (str): Mot de passe
            server (str): Serveur MT5
            
        Returns:
            bool: True si la connexion est établie
        """
        try:
            print("\n=== Tentative de connexion à MetaTrader 5 ===")
            
            # Vérification si MT5 est déjà initialisé
            if mt5.initialize():
                print("MetaTrader 5 est déjà initialisé, fermeture de la connexion existante...")
                mt5.shutdown()
            
            # Initialisation de MT5
            print("Initialisation de MetaTrader 5...")
            if not mt5.initialize():
                error = mt5.last_error()
                print(f"Erreur d'initialisation de MetaTrader 5: {error}")
                print("Vérifiez que MetaTrader 5 est bien installé et en cours d'exécution")
                return False
            
            print("MetaTrader 5 initialisé avec succès")
            print(f"Version de MetaTrader 5: {mt5.__version__}")
            
            # Vérification des paramètres de connexion
            print(f"\nParamètres de connexion:")
            print(f"- Login: {login}")
            print(f"- Server: {server}")
            
            # Tentative de connexion
            print("\nTentative d'authentification...")
            authorized = mt5.login(
                login=int(login),
                password=password,
                server=server
            )
            
            if not authorized:
                error = mt5.last_error()
                print(f"Échec de l'authentification: {error}")
                print("\nVérifiez que:")
                print("1. MetaTrader 5 est bien installé et en cours d'exécution")
                print("2. Les identifiants sont corrects")
                print("3. Le serveur est accessible")
                print("4. Vous avez une connexion Internet stable")
                mt5.shutdown()
                return False
            
            print("Authentification réussie!")
            self.connected = True
            print("=== Connexion établie avec succès ===\n")
            return True
            
        except Exception as e:
            print(f"Erreur inattendue lors de la connexion: {str(e)}")
            if mt5.initialize():
                mt5.shutdown()
            return False
    
    def shutdown(self):
        """Ferme la connexion à MT5."""
        if mt5.initialize():
            mt5.shutdown()
            self.connected = False
    
    def is_connected(self) -> bool:
        """Vérifie si la connexion est active."""
        return self.connected and mt5.initialize() 