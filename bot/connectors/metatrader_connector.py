import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import MetaTrader5 as mt5
import pandas as pd
from dotenv import load_dotenv

from .mt5_connection import MT5Connection


class AvatraderMT5:
    def __init__(self, login: int, password: str, server: str):
        """
        Initialise la connexion à AvaTrade.

        Args:
            login (int): Identifiant de connexion
            password (str): Mot de passe
            server (str): Serveur AvaTrade
        """
        self.login = login
        self.password = password
        self.server = server
        self.connected = False
        self.last_account_info_log = 0  # Pour suivre le dernier log d'info compte

        # Utilisation de la connexion singleton
        self.mt5_connection = MT5Connection()
        self.connected = self.mt5_connection.initialize(login, password, server)

    def connect(self) -> bool:
        """Établit la connexion avec MT5"""
        try:
            if not mt5.initialize():
                print(f"Erreur d'initialisation MT5: {mt5.last_error()}")
                return False

            if not mt5.login(
                login=self.login, password=self.password, server=self.server
            ):
                print(f"Erreur de connexion MT5: {mt5.last_error()}")
                return False

            self.connected = True
            return True

        except Exception as e:
            print(f"Erreur lors de la connexion: {str(e)}")
            return False

    def disconnect(self) -> None:
        """Déconnecte de MT5"""
        try:
            if self.connected:
                self.mt5_connection.shutdown()
                self.connected = False
        except Exception as e:
            print(f"Erreur lors de la déconnexion: {str(e)}")

    def update_account_info(self) -> None:
        """Met à jour les informations du compte"""
        if not self.connected:
            return

        try:
            account_info = mt5.account_info()
            if account_info is None:
                print("Erreur: Impossible de récupérer les informations du compte")
                return

            # Log des infos compte seulement toutes les 5 minutes
            current_time = time.time()
            if (
                current_time - self.last_account_info_log >= 300
            ):  # 300 secondes = 5 minutes
                print("\n=== Informations du compte ===")
                print(f"Balance: {account_info.balance:.2f} EUR")
                print(f"Equity: {account_info.equity:.2f} EUR")
                print(f"Profit: {account_info.profit:.2f} EUR")
                print(f"Marge: {account_info.margin:.2f} EUR")
                print(f"Marge Libre: {account_info.margin_free:.2f} EUR")
                print(f"Levier: 1:{account_info.leverage}")
                print("===========================\n")
                self.last_account_info_log = current_time

        except Exception as e:
            print(f"Erreur lors de la mise à jour des informations du compte: {str(e)}")

    def place_order(
        self, symbol: str, order_type: str, volume: float, price: float = None
    ) -> bool:
        """Place un ordre sur le marché"""
        if not self.connected:
            print("Erreur: Non connecté à MT5")
            return False

        try:
            # Vérification du symbole
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                print(f"Erreur: Symbole {symbol} non trouvé")
                return False

            # Préparation de la requête
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY
                if order_type == "BUY"
                else mt5.ORDER_TYPE_SELL,
                "price": price
                if price
                else symbol_info.ask
                if order_type == "BUY"
                else symbol_info.bid,
                "deviation": 20,
                "magic": 234000,
                "comment": "ordre python",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            # Envoi de l'ordre
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"Erreur lors du placement de l'ordre: {result.comment}")
                return False

            print(f"\n=== Nouvel ordre placé ===")
            print(f"Type: {order_type}")
            print(f"Symbole: {symbol}")
            print(f"Volume: {volume}")
            print(f"Prix: {request['price']}")
            print(f"Ticket: {result.order}")
            print("=====================\n")

            return True

        except Exception as e:
            print(f"Erreur lors du placement de l'ordre: {str(e)}")
            return False

    def get_positions(self) -> list:
        """Récupère les positions ouvertes"""
        if not self.connected:
            print("Erreur: Non connecté à MT5")
            return []

        try:
            positions = mt5.positions_get()
            if positions is None:
                return []

            # Log des positions si changement
            if len(positions) > 0:
                print("\n=== Positions ouvertes ===")
                for pos in positions:
                    print(f"Ticket: {pos.ticket}")
                    print(f"Symbole: {pos.symbol}")
                    print(f"Type: {'Achat' if pos.type == 0 else 'Vente'}")
                    print(f"Volume: {pos.volume}")
                    print(f"Prix d'ouverture: {pos.price_open}")
                    print(f"Prix actuel: {pos.price_current}")
                    print(f"Profit: {pos.profit}")
                    print("-------------------")
                print("=====================\n")

            return positions

        except Exception as e:
            print(f"Erreur lors de la récupération des positions: {str(e)}")
            return []

    def close_position(self, ticket: int) -> bool:
        """Ferme une position spécifique"""
        if not self.connected:
            print("Erreur: Non connecté à MT5")
            return False

        try:
            position = mt5.positions_get(ticket=ticket)
            if position is None or len(position) == 0:
                print(f"Erreur: Position {ticket} non trouvée")
                return False

            pos = position[0]
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
                "position": ticket,
                "price": mt5.symbol_info_tick(pos.symbol).bid
                if pos.type == 0
                else mt5.symbol_info_tick(pos.symbol).ask,
                "deviation": 20,
                "magic": 234000,
                "comment": "fermeture python",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"Erreur lors de la fermeture de la position: {result.comment}")
                return False

            print(f"\n=== Position fermée ===")
            print(f"Ticket: {ticket}")
            print(f"Symbole: {pos.symbol}")
            print(f"Volume: {pos.volume}")
            print(f"Profit: {pos.profit}")
            print("=====================\n")

            return True

        except Exception as e:
            print(f"Erreur lors de la fermeture de la position: {str(e)}")
            return False

    def get_open_positions(self) -> List[Dict]:
        """
        Récupère les positions ouvertes.

        Returns:
            List[Dict]: Liste des positions ouvertes
        """
        if not self.connected:
            if not self.connect():
                return []

        positions = mt5.positions_get()

        if positions is None:
            return []

        return [
            {
                "ticket": pos.ticket,
                "symbol": pos.symbol,
                "type": "LONG" if pos.type == mt5.POSITION_TYPE_BUY else "SHORT",
                "volume": pos.volume,
                "price": pos.price_open,
                "sl": pos.sl,
                "tp": pos.tp,
                "profit": pos.profit,
                "value": pos.volume * pos.price_open,
            }
            for pos in positions
        ]

    def create_market_buy_order(
        self,
        symbol: str,
        amount: float,
        stop_loss: float = None,
        take_profit: float = None,
    ) -> Dict:
        """
        Crée un ordre d'achat au marché.

        Args:
            symbol (str): Symbole de l'actif
            amount (float): Quantité à acheter
            stop_loss (float): Prix du stop loss
            take_profit (float): Prix du take profit

        Returns:
            Dict: Détails de l'ordre
        """
        if not self.connected:
            if not self.connect():
                return None

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": amount,
            "type": mt5.ORDER_TYPE_BUY,
            "price": mt5.symbol_info_tick(symbol).ask,
            "deviation": 20,
            "magic": 234000,
            "comment": "python script open",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        if stop_loss:
            request["sl"] = stop_loss
        if take_profit:
            request["tp"] = take_profit

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Erreur d'exécution de l'ordre: {result.comment}")
            return None

        return {
            "ticket": result.order,
            "volume": amount,
            "price": result.price,
            "comment": result.comment,
        }

    def create_market_sell_order(
        self,
        symbol: str,
        amount: float,
        stop_loss: float = None,
        take_profit: float = None,
    ) -> Dict:
        """
        Crée un ordre de vente au marché.

        Args:
            symbol (str): Symbole de l'actif
            amount (float): Quantité à vendre
            stop_loss (float): Prix du stop loss
            take_profit (float): Prix du take profit

        Returns:
            Dict: Détails de l'ordre
        """
        if not self.connected:
            if not self.connect():
                return None

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": amount,
            "type": mt5.ORDER_TYPE_SELL,
            "price": mt5.symbol_info_tick(symbol).bid,
            "deviation": 20,
            "magic": 234000,
            "comment": "python script open",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        if stop_loss:
            request["sl"] = stop_loss
        if take_profit:
            request["tp"] = take_profit

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Erreur d'exécution de l'ordre: {result.comment}")
            return None

        return {
            "ticket": result.order,
            "volume": amount,
            "price": result.price,
            "comment": result.comment,
        }

    def get_balance(self) -> float:
        """
        Récupère le solde du compte.

        Returns:
            float: Solde du compte
        """
        if not self.connected:
            if not self.connect():
                return 0.0

        account_info = mt5.account_info()
        return float(account_info.balance) if account_info else 0.0

    def rebalance_portfolio(self, target_weights: Dict[str, float]):
        """
        Rééquilibre le portefeuille selon les poids cibles.

        Args:
            target_weights (Dict[str, float]): Poids cibles par actif
        """
        if not self.connected:
            if not self.connect():
                return

        # Récupération des positions actuelles
        current_positions = self.get_open_positions()
        total_value = sum(pos["value"] for pos in current_positions)

        # Calcul des ajustements nécessaires
        for symbol, target_weight in target_weights.items():
            current_position = next(
                (pos for pos in current_positions if pos["symbol"] == symbol), None
            )

            target_value = total_value * target_weight

            if current_position:
                current_value = current_position["value"]
                if current_value < target_value:
                    # Achat supplémentaire
                    amount = (target_value - current_value) / mt5.symbol_info_tick(
                        symbol
                    ).ask
                    self.create_market_buy_order(symbol, amount)
                elif current_value > target_value:
                    # Vente partielle
                    amount = (current_value - target_value) / mt5.symbol_info_tick(
                        symbol
                    ).bid
                    self.create_market_sell_order(symbol, amount)
            else:
                # Nouvelle position
                amount = target_value / mt5.symbol_info_tick(symbol).ask
                self.create_market_buy_order(symbol, amount)

    def get_account_info(self) -> Dict:
        """
        Récupère les informations du compte.

        Returns:
            Dict: Informations du compte contenant balance, equity, profit, etc.
        """
        try:
            if not self.connected:
                print("Non connecté à MT5, tentative de connexion...")
                if not self.connect():
                    print("Échec de la connexion à MetaTrader 5")
                    return {}
                print("Connexion réussie!")

            account_info = mt5.account_info()
            if account_info is None:
                error = mt5.last_error()
                print(
                    f"Erreur lors de la récupération des informations du compte: {error}"
                )
                return {}

            return {
                "balance": account_info.balance,
                "equity": account_info.equity,
                "profit": account_info.profit,
                "margin": account_info.margin,
                "free_margin": account_info.margin_free,
                "leverage": account_info.leverage,
            }

        except Exception as e:
            print(
                f"Erreur lors de la récupération des informations du compte: {str(e)}"
            )
            return {}

    def get_historical_data(
        self, symbol: str, timeframe: str = "1h", limit: int = 100
    ) -> pd.DataFrame:
        """
        Récupère les données historiques d'un symbole.

        Args:
            symbol (str): Symbole à récupérer (ex: 'BTCUSD')
            timeframe (str): Intervalle de temps ('1m', '5m', '15m', '1h', '4h', '1d')
            limit (int): Nombre de barres à récupérer

        Returns:
            pd.DataFrame: DataFrame contenant les données historiques
        """
        if not self.connected:
            print("Erreur: Non connecté à MT5")
            return pd.DataFrame()

        try:
            # Conversion du timeframe en constante MT5
            timeframe_map = {
                "1m": mt5.TIMEFRAME_M1,
                "5m": mt5.TIMEFRAME_M5,
                "15m": mt5.TIMEFRAME_M15,
                "1h": mt5.TIMEFRAME_H1,
                "4h": mt5.TIMEFRAME_H4,
                "1d": mt5.TIMEFRAME_D1,
            }

            tf = timeframe_map.get(timeframe, mt5.TIMEFRAME_H1)

            # Récupération des données
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, limit)
            if rates is None:
                print(f"Erreur lors de la récupération des données: {mt5.last_error()}")
                return pd.DataFrame()

            # Conversion en DataFrame
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)

            # Sélection des colonnes nécessaires
            df = df[["open", "high", "low", "close", "tick_volume"]]
            df.rename(columns={"tick_volume": "volume"}, inplace=True)

            return df

        except Exception as e:
            print(f"Erreur lors de la récupération des données historiques: {str(e)}")
            return pd.DataFrame()
