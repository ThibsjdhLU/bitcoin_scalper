import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

import MetaTrader5 as mt5
from config.scalper_config import DEFAULT_CONFIG, ERROR_CONFIG
import pandas as pd

class BitcoinScalper:
    def __init__(self, config: Dict = DEFAULT_CONFIG):
        """Initialise la stratégie de scalping Bitcoin."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.symbol = config["symbol"]
        self.volume = config["volume"]
        self.stop_loss_pips = config["stop_loss_pips"]
        self.take_profit_pips = config["take_profit_pips"]
        self.max_spread_pips = config["max_spread_pips"]
        self.timeframe = config["timeframe"]
        self.max_positions = config["max_positions"]
        self.risk_per_trade = config["risk_per_trade"]
        self.max_daily_trades = config["max_daily_trades"]
        self.trades_today = 0
        self.last_trade_time = None

    def get_current_spread(self) -> float:
        """Récupère le spread actuel du symbole."""
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            raise ValueError(f"Impossible d'obtenir les informations pour {self.symbol}")
        return tick.ask - tick.bid

    def check_trading_conditions(self, data: dict) -> bool:
        """
        Vérifie si les conditions de trading sont remplies
        """
        try:
            # Vérification des limites quotidiennes
            if self.trades_today >= self.max_daily_trades:
                self.logger.info("Limite quotidienne de trades atteinte")
                return False
                
            # Vérification du spread
            current_spread = self.get_current_spread()
            if current_spread > self.max_spread_pips:
                self.logger.info(f"Spread trop élevé: {current_spread}")
                return False
                
            # Vérification des positions ouvertes
            positions = self.mt5_connector.get_open_positions()
            if len(positions) >= self.max_positions:
                self.logger.info("Nombre maximum de positions atteint")
                return False
                
            # Vérification de la volatilité
            if data['atr'] > self.max_atr:
                self.logger.info(f"Volatilité trop élevée: {data['atr']}")
                return False
                
            # Vérification du volume
            if data['volume'] < self.volume_threshold:
                self.logger.info(f"Volume insuffisant: {data['volume']}")
                return False
                
            # Vérification des tendances
            if not self._check_trend_alignment(data):
                self.logger.info("Les tendances ne sont pas alignées")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la vérification des conditions: {e}")
            return False
            
    def _check_trend_alignment(self, data: dict) -> bool:
        """
        Vérifie l'alignement des tendances
        """
        try:
            # Tendance à court terme (RSI)
            short_term_trend = data['rsi'] < 50
            
            # Tendance à moyen terme (MACD)
            medium_term_trend = data['macd'] > data['macd_signal']
            
            # Tendance à long terme (Bollinger Bands)
            long_term_trend = data['close'] < data['bb_middle']
            
            # Les trois tendances doivent être alignées
            return short_term_trend == medium_term_trend == long_term_trend
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la vérification des tendances: {e}")
            return False

    def calculate_position_size(self) -> float:
        """Calcule la taille de la position en fonction du risque."""
        try:
            # Récupération des informations du compte
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error("Impossible d'obtenir les informations du compte")
                return 0.0
            
            # Récupération des informations du symbole
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                self.logger.error(f"Impossible d'obtenir les informations pour {self.symbol}")
                return 0.0
            
            # Calcul du risque monétaire
            balance = account_info.balance
            risk_amount = balance * self.risk_per_trade
            
            # Calcul de la taille de position basée sur le risque
            stop_distance = self.stop_loss_pips * 10  # 10 pips = 1 point
            position_size = risk_amount / stop_distance
            
            # Vérification des limites de volume
            position_size = max(position_size, symbol_info.volume_min)
            position_size = min(position_size, symbol_info.volume_max)
            
            # Arrondi au volume step
            position_size = round(position_size / symbol_info.volume_step) * symbol_info.volume_step
            
            # Vérification finale
            if position_size <= 0:
                self.logger.warning("Volume calculé invalide")
                return 0.0
                
            return position_size
            
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul de la taille de position: {str(e)}")
            return 0.0

    def execute_trade(self) -> None:
        """Exécute la stratégie de trading."""
        try:
            if not self.check_trading_conditions():
                self.logger.info("Conditions de trading non remplies")
                return

            # Calcul de la taille de la position
            volume = self.calculate_position_size()
            if volume <= 0:
                self.logger.warning("Volume invalide calculé")
                return
            
            # Analyse technique
            signal = self.analyze_market()
            if signal == 0:  # Pas de signal
                return

            # Récupération des informations du symbole
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                self.logger.error(f"Impossible de récupérer les informations pour {self.symbol}")
                return
                
            # Vérification du volume minimum
            if volume < symbol_info.volume_min:
                self.logger.warning(f"Volume {volume} inférieur au minimum {symbol_info.volume_min}")
                return
                
            # Préparation de l'ordre
            order_type = mt5.ORDER_TYPE_BUY if signal > 0 else mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(self.symbol).ask if signal > 0 else mt5.symbol_info_tick(self.symbol).bid
            
            # Calcul des niveaux de SL et TP avec vérification des distances minimales
            min_stop_level = symbol_info.trade_stops_level * symbol_info.point
            sl = price - self.stop_loss_pips * 10 if signal > 0 else price + self.stop_loss_pips * 10
            tp = price + self.take_profit_pips * 10 if signal > 0 else price - self.take_profit_pips * 10
            
            # Vérification des distances minimales
            if abs(price - sl) < min_stop_level:
                sl = price - min_stop_level if signal > 0 else price + min_stop_level
                self.logger.warning(f"Stop loss ajusté à {sl} pour respecter la distance minimale")
            if abs(price - tp) < min_stop_level:
                tp = price + min_stop_level if signal > 0 else price - min_stop_level
                self.logger.warning(f"Take profit ajusté à {tp} pour respecter la distance minimale")

            # Envoi de l'ordre
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": 234000,
                "comment": "python scalper",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            # Log des détails de l'ordre
            self.logger.info(f"Envoi de l'ordre: {self.symbol} {order_type}")
            self.logger.info(f"Volume: {volume}")
            self.logger.info(f"Prix: {price}")
            self.logger.info(f"Stop Loss: {sl}")
            self.logger.info(f"Take Profit: {tp}")

            # Envoi de l'ordre avec gestion des erreurs
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Erreur lors de l'envoi de l'ordre: {result.comment}")
                self.logger.error(f"Code d'erreur: {result.retcode}")
                return
                
            self.logger.info(f"Ordre exécuté avec succès: {self.symbol} {order_type} Volume: {volume}")
            self.trades_today += 1
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'exécution du trade: {str(e)}")

    def analyze_market(self) -> dict:
        """
        Analyse le marché et retourne les données nécessaires
        """
        try:
            # Récupération des données
            rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 100)
            if rates is None or len(rates) == 0:
                self.logger.error(f"Impossible de récupérer les données pour {self.symbol}")
                return None
                
            # Conversion en DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Calcul des indicateurs
            rsi = self.rsi.calculate(df['close'])
            macd, signal, hist = self.macd.calculate(df['close'])
            bb_upper, bb_middle, bb_lower = self.bollinger.calculate(df['close'])
            
            # Récupération du dernier prix
            current_price = df['close'].iloc[-1]
            
            # Création du dictionnaire de données
            data = {
                'price': float(current_price),
                'time': df['time'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S'),
                'rsi': float(rsi[-1]),
                'macd': float(macd[-1]),
                'macd_signal': float(signal[-1]),
                'macd_hist': float(hist[-1]),
                'bb_upper': float(bb_upper[-1]),
                'bb_middle': float(bb_middle[-1]),
                'bb_lower': float(bb_lower[-1]),
                'volume': float(df['tick_volume'].iloc[-1]),
                'high': float(df['high'].iloc[-1]),
                'low': float(df['low'].iloc[-1]),
                'open': float(df['open'].iloc[-1]),
                'close': float(df['close'].iloc[-1])
            }
            
            self.logger.info(f"Données de marché analysées pour {self.symbol}: Prix={data['price']}, RSI={data['rsi']:.2f}")
            return data
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse du marché: {e}")
            return None
            
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcule l'Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr

    def _generate_signal(self, data: dict) -> int:
        """
        Génère un signal de trading basé sur les données du marché
        Retourne:
            1 pour signal d'achat
            -1 pour signal de vente
            0 pour pas de signal
        """
        try:
            if data is None:
                return 0
                
            # Vérification du spread
            current_spread = self.get_current_spread()
            if current_spread > self.max_spread_pips:
                self.logger.info(f"Spread trop élevé: {current_spread}")
                return 0
                
            # Conditions d'achat
            if (data['rsi'] < 30 and  # Survente
                data['macd'] > data['macd_signal'] and  # MACD croise au-dessus
                data['close'] < data['bb_lower'] and  # Prix sous la bande inférieure
                data['volume'] > self.volume_threshold):  # Volume significatif
                self.logger.info("Signal d'achat détecté")
                return 1
                
            # Conditions de vente
            if (data['rsi'] > 70 and  # Surachat
                data['macd'] < data['macd_signal'] and  # MACD croise en-dessous
                data['close'] > data['bb_upper'] and  # Prix au-dessus de la bande supérieure
                data['volume'] > self.volume_threshold):  # Volume significatif
                self.logger.info("Signal de vente détecté")
                return -1
                
            return 0
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération du signal: {e}")
            return 0

    def check_data_sufficiency(self, data: pd.DataFrame) -> bool:
        """
        Vérifie si les données sont suffisantes pour générer un signal
        
        Args:
            data: DataFrame des données
            
        Returns:
            bool: True si les données sont suffisantes
        """
        try:
            min_points = max(
                self.config.get('min_data_points', 10),  # Minimum absolu
                self.config.get('rsi_period', 14),       # Période RSI
                self.config.get('macd_slow', 26),        # Période lente MACD
                self.config.get('bb_period', 20)         # Période Bollinger
            )
            
            if len(data) < min_points:
                self.logger.warning(
                    f"Données insuffisantes pour générer un signal ({len(data)} points, "
                    f"minimum requis: {min_points})"
                )
                return False
                
            # Vérification des valeurs manquantes
            if data.isnull().any().any():
                self.logger.warning("Présence de valeurs manquantes dans les données")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la vérification des données: {str(e)}")
            return False 