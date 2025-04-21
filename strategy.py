#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module de stratégie de scalping pour le bot de trading
Implémente la logique de prise de décision basée sur les indicateurs techniques
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import os
from dotenv import load_dotenv
from PySide6.QtCore import QObject, Signal, QThread
from mt5_connector import MT5Connector
import threading
import MetaTrader5 as mt5
from api_connector import APIConnector
from indicators import RSI, MACD, BollingerBands, ATR

# Chargement des variables d'environnement
load_dotenv()

logger = logging.getLogger(__name__)

class SignalType(Enum):
    BUY = 'buy'
    SELL = 'sell'
    CLOSE = 'close'
    NONE = 'none'

@dataclass
class TradingSignal:
    type: SignalType
    symbol: str
    price: float
    timestamp: datetime
    strength: float
    indicators: Dict
    metadata: Dict

class ScalperStrategy(QObject):
    """
    Stratégie de trading
    """
    
    # Signaux pour la communication avec l'interface
    data_updated = Signal(object)  # Pour envoyer les données sous forme de dictionnaire
    error_occurred = Signal(str)   # Pour envoyer une chaîne de caractères
    log_message = Signal(str)      # Pour envoyer une chaîne de caractères
    finished = Signal()            # Signal émis lorsque la stratégie se termine
    
    def __init__(self, config: dict):
        """
        Initialise la stratégie
        
        Args:
            config (dict): Configuration de la stratégie
        """
        super().__init__()
        self.config = config
        self.running = False
        self.paused = False
        self.mt5_connector = MT5Connector()
        
        # Initialisation de l'API Connector avec les paramètres requis
        api_credentials = {
            'api_key': os.getenv('API_KEY'),
            'api_secret': os.getenv('API_SECRET')
        }
        
        # Vérification des clés API
        if not api_credentials['api_key'] or api_credentials['api_key'] == 'your_api_key_here':
            self.log_message.emit("Attention: Clé API non définie, utilisation du mode MT5 uniquement")
            self.api_connector = None
        else:
            self.api_connector = APIConnector(
                platform='mt5',
                credentials=api_credentials
            )
        
        # Données de trading
        self._price_data = None
        self._last_price = None
        self._total_pnl = 0.0
        self._lock = threading.Lock()  # Verrou pour la synchronisation
        
        # Valeurs par défaut pour la configuration
        self.config.setdefault('interval', 5)  # Intervalle par défaut de 5 secondes
        self.config.setdefault('symbol', 'BTCUSD')  # Format correct pour MT5
        self.config.setdefault('timeframe', 'M1')
        
        # Initialisation des indicateurs
        self._init_indicators()
        
    def _init_indicators(self):
        """Initialise les indicateurs techniques"""
        try:
            self.rsi = RSI(
                period=self.config['rsi_period'],
                overbought=self.config['rsi_overbought'],
                oversold=self.config['rsi_oversold']
            )
            self.macd = MACD(
                fast_period=self.config['macd_fast'],
                slow_period=self.config['macd_slow'],
                signal_period=self.config['macd_signal']
            )
            self.bollinger = BollingerBands(
                period=self.config['bb_period'],
                std_dev=self.config['bb_std']
            )
            self.atr = ATR(period=self.config['atr_period'])
        except Exception as e:
            self.error_occurred.emit(f"Erreur lors de l'initialisation des indicateurs: {str(e)}")
            
    def _get_mt5_symbol(self, symbol: str) -> str:
        """
        Vérifie et corrige le format du symbole pour MT5
        
        Args:
            symbol (str): Symbole à vérifier
            
        Returns:
            str: Symbole corrigé ou None si non trouvé
        """
        # Liste des formats possibles pour BTC/USD
        possible_formats = [
            "BTCUSD",  # Format standard
            "BTC/USD",  # Format avec slash
            "BTC-USD",  # Format avec tiret
            "BTCUSD.a",  # Format avec suffixe
            "BTCUSDm",  # Format mini
            "BTCUSDp"   # Format premium
        ]
        
        # Vérification de chaque format
        for sym in possible_formats:
            symbol_info = mt5.symbol_info(sym)
            if symbol_info is not None:
                self.log_message.emit(f"Symbole trouvé: {sym}")
                return sym
                
        self.error_occurred.emit(f"Aucun format valide trouvé pour {symbol}")
        return None

    def _trading_loop(self):
        """Boucle principale de trading"""
        self.log_message.emit("Démarrage de la boucle de trading")
        
        # Vérification et correction du symbole au démarrage
        mt5_symbol = self._get_mt5_symbol(self.config['symbol'])
        if mt5_symbol is None:
            self.error_occurred.emit("Impossible de trouver un symbole valide pour le trading")
            self.running = False
            return
            
        while self.running:
            try:
                if self.paused:
                    time.sleep(1)
                    continue
                    
                # Récupération des données de marché
                self.log_message.emit("Récupération des données de marché...")
                
                # Vérification de la connexion MT5
                if not mt5.initialize():
                    self.error_occurred.emit("Erreur: MT5 n'est pas initialisé")
                    time.sleep(5)
                    continue
                
                self.log_message.emit(f"Tentative de récupération des données pour {mt5_symbol}")
                
                # Vérification du symbole
                symbol_info = mt5.symbol_info(mt5_symbol)
                if symbol_info is None:
                    self.error_occurred.emit(f"Symbole {mt5_symbol} non trouvé dans MT5")
                    time.sleep(5)
                    continue
                
                if not symbol_info.visible:
                    self.log_message.emit(f"Activation du symbole {mt5_symbol}")
                    if not mt5.symbol_select(mt5_symbol, True):
                        self.error_occurred.emit(f"Impossible d'activer le symbole {mt5_symbol}")
                        time.sleep(5)
                        continue
                
                # Récupération des données via MT5
                self.log_message.emit("Récupération des données historiques...")
                rates = mt5.copy_rates_from_pos(
                    mt5_symbol,
                    mt5.TIMEFRAME_M1,  # Timeframe d'une minute
                    0,  # Position actuelle
                    100  # Nombre de barres
                )
                
                if rates is None:
                    self.error_occurred.emit(f"Erreur lors de la récupération des données: {mt5.last_error()}")
                    time.sleep(5)
                    continue
                    
                if len(rates) == 0:
                    self.log_message.emit("Aucune donnée historique disponible")
                    time.sleep(5)
                    continue
                    
                self.log_message.emit(f"Données récupérées avec succès: {len(rates)} barres")
                
                # Conversion en DataFrame
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                
                # Mise à jour des indicateurs
                with self._lock:
                    self._price_data = df
                    self._last_price = df['close'].iloc[-1]
                    
                    # Calcul des indicateurs
                    rsi_values = self.rsi.calculate(df)
                    macd_values = self.macd.calculate(df)
                    bb_values = self.bollinger.calculate(df)
                    atr_values = self.atr.calculate(df)
                    
                    # Génération des signaux
                    signal = self._generate_signal(
                        df.iloc[-1],
                        rsi_values[-1],
                        macd_values[-1],
                        bb_values[-1],
                        atr_values[-1]
                    )
                    
                    # Préparation des données pour l'interface
                    current_time = df.index[-1]
                    if isinstance(current_time, (int, float)):
                        current_time = pd.to_datetime(current_time, unit='s')
                    
                    data_dict = {
                        'time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'price': float(df['close'].iloc[-1]),
                        'pnl': self._total_pnl,
                        'indicators': {
                            'rsi': float(rsi_values[-1]) if len(rsi_values) > 0 else 0,
                            'macd': float(macd_values[-1]) if len(macd_values) > 0 else 0,
                            'atr': float(atr_values[-1]) if len(atr_values) > 0 else 0,
                            'bb_upper': float(bb_values['upper'][-1]) if len(bb_values['upper']) > 0 else 0,
                            'bb_lower': float(bb_values['lower'][-1]) if len(bb_values['lower']) > 0 else 0,
                            'bb_middle': float(bb_values['middle'][-1]) if len(bb_values['middle']) > 0 else 0
                        },
                        'signal': signal.type.value if signal else 'none'
                    }
                    
                    # Émission des données mises à jour
                    self.data_updated.emit(data_dict)
                    
                # Attente avant la prochaine itération
                time.sleep(self.config['interval'])
                
            except Exception as e:
                self.error_occurred.emit(f"Erreur dans la boucle de trading: {str(e)}")
                time.sleep(5)
                
        self.log_message.emit("Arrêt de la boucle de trading")

    def _generate_signal(self, data: dict, rsi: float, macd: float, bb: float, atr: float) -> Optional[TradingSignal]:
        """
        Génère un signal de trading basé sur les indicateurs techniques
        
        Args:
            data (dict): Données de marché
            rsi (float): Valeur RSI
            macd (float): Valeur MACD
            bb (float): Valeur des bandes de Bollinger
            atr (float): Valeur ATR
            
        Returns:
            Optional[TradingSignal]: Signal de trading ou None si aucun signal
        """
        try:
            if data is None:
                logger.warning("Données de marché invalides")
                return None
                
            # Vérification des conditions minimales
            min_points = self.config.get("min_data_points", 30)
            if len(data) < min_points:
                logger.warning(f"Données insuffisantes pour générer un signal ({len(data)} points, minimum requis: {min_points})")
                return None
                
            # Extraction des dernières valeurs
            current_price = data['close']
            
            # Vérification du volume
            if 'tick_volume' in data:
                current_volume = data['tick_volume']
                avg_volume = self._price_data['tick_volume'].rolling(10).mean().iloc[-1]
            elif 'real_volume' in data:
                current_volume = data['real_volume']
                avg_volume = self._price_data['real_volume'].rolling(10).mean().iloc[-1]
            else:
                logger.warning("Aucune donnée de volume disponible")
                return None
                
            if current_volume < avg_volume * self.config.get("volume_threshold", 1.1):
                return None
                
            # Analyse RSI
            if rsi < self.config.get("rsi_oversold", 35):
                return TradingSignal(
                    type=SignalType.BUY,
                    symbol=self.config["symbol"],
                    price=current_price,
                    timestamp=datetime.now(),
                    strength=1.0,
                    indicators={
                        'RSI': rsi,
                        'ATR': atr
                    },
                    metadata={"reason": "RSI oversold"}
                )
            elif rsi > self.config.get("rsi_overbought", 65):
                return TradingSignal(
                    type=SignalType.SELL,
                    symbol=self.config["symbol"],
                    price=current_price,
                    timestamp=datetime.now(),
                    strength=1.0,
                    indicators={
                        'RSI': rsi,
                        'ATR': atr
                    },
                    metadata={"reason": "RSI overbought"}
                )
                    
            # Analyse MACD
            if isinstance(macd, (list, np.ndarray)) and len(macd) >= 3:
                if macd[-1] > macd[-2] and macd[-2] <= macd[-3]:
                    return TradingSignal(
                        type=SignalType.BUY,
                        symbol=self.config["symbol"],
                        price=current_price,
                        timestamp=datetime.now(),
                        strength=0.8,
                        indicators={
                            'MACD': macd[-1],
                            'ATR': atr
                        },
                        metadata={"reason": "MACD crossover"}
                    )
                elif macd[-1] < macd[-2] and macd[-2] >= macd[-3]:
                    return TradingSignal(
                        type=SignalType.SELL,
                        symbol=self.config["symbol"],
                        price=current_price,
                        timestamp=datetime.now(),
                        strength=0.8,
                        indicators={
                            'MACD': macd[-1],
                            'ATR': atr
                        },
                        metadata={"reason": "MACD crossover"}
                    )
                    
            return None
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du signal: {e}")
            return None

    def start_trading(self):
        """Démarre la stratégie de trading dans un thread séparé."""
        try:
            self.log_message.emit("Début de start_trading")
            
            if self.running:
                self.log_message.emit("La stratégie est déjà en cours d'exécution")
                return
                
            self.running = True
            
            self.log_message.emit("Démarrage du thread")
            # Création et démarrage du thread de trading
            self.trading_thread = QThread()
            self.moveToThread(self.trading_thread)
            
            self.log_message.emit("Connexion du signal started")
            self.trading_thread.started.connect(self._trading_loop)
            
            self.log_message.emit("Démarrage du thread")
            self.trading_thread.start()
            
            self.log_message.emit("Stratégie démarrée avec succès")
            
        except Exception as e:
            self.log_message.emit(f"ERREUR dans start_trading: {str(e)}")
            self.error_occurred.emit(f"Erreur lors du démarrage de la stratégie: {str(e)}")
            self.running = False

    def stop_trading(self):
        """Arrête la stratégie de trading."""
        try:
            if not self.running:
                return
                
            self.running = False
            
            if self.trading_thread and self.trading_thread.isRunning():
                self.trading_thread.quit()
                self.trading_thread.wait()
            
            self.log_message.emit("Stratégie arrêtée")
            self.finished.emit()  # Émission du signal finished
            
        except Exception as e:
            self.error_occurred.emit(f"Erreur lors de l'arrêt de la stratégie: {str(e)}")

    def calculate_position_size(self, symbol: str, signal: TradingSignal,
                              balance: float) -> float:
        """
        Calcule la taille de position recommandée
        
        Args:
            symbol (str): Symbole de trading
            signal (TradingSignal): Signal de trading
            balance (float): Solde disponible
            
        Returns:
            float: Taille de position recommandée
        """
        # Récupération de l'ATR pour le stop loss
        atr = signal.indicators['ATR']
        
        # Calcul du risque par trade (1% du capital)
        risk_amount = balance * 0.01
        
        # Calcul de la taille basée sur l'ATR
        stop_distance = atr * self.config.get('atr_multiplier', 2)
        position_size = risk_amount / stop_distance
        
        # Ajustement en fonction de la force du signal
        position_size *= signal.strength
        
        # Limitation à 5% du capital maximum
        max_position = balance * 0.05
        position_size = min(position_size, max_position)
        
        return position_size
    
    def calculate_stop_loss(self, signal: TradingSignal, position_size: float) -> float:
        """
        Calcule le niveau de stop loss
        
        Args:
            signal (TradingSignal): Signal de trading
            position_size (float): Taille de la position
            
        Returns:
            float: Prix du stop loss
        """
        atr = signal.indicators['ATR']
        current_price = signal.price
        
        if signal.type == SignalType.BUY:
            stop_loss = current_price - (atr * self.config.get('atr_multiplier', 2))
        else:
            stop_loss = current_price + (atr * self.config.get('atr_multiplier', 2))
        
        return stop_loss
    
    def calculate_take_profit(self, signal: TradingSignal, stop_loss: float,
                            risk_reward_ratio: float = 2.0) -> float:
        """
        Calcule le niveau de take profit
        
        Args:
            signal (TradingSignal): Signal de trading
            stop_loss (float): Prix du stop loss
            risk_reward_ratio (float): Ratio risque/récompense
            
        Returns:
            float: Prix du take profit
        """
        current_price = signal.price
        
        if signal.type == SignalType.BUY:
            risk = current_price - stop_loss
            take_profit = current_price + (risk * risk_reward_ratio)
        else:
            risk = stop_loss - current_price
            take_profit = current_price - (risk * risk_reward_ratio)
        
        return take_profit
    
    def should_close_position(self, position: Dict, current_price: float,
                            current_indicators: Dict) -> bool:
        """
        Vérifie si une position doit être fermée
        
        Args:
            position (Dict): Position à vérifier
            current_price (float): Prix actuel
            current_indicators (Dict): Indicateurs actuels
            
        Returns:
            bool: True si la position doit être fermée
        """
        # Vérification du stop loss
        if position['type'] == 'buy' and current_price <= position['stop_loss']:
            return True
        if position['type'] == 'sell' and current_price >= position['stop_loss']:
            return True
            
        # Vérification du take profit
        if position['type'] == 'buy' and current_price >= position['take_profit']:
            return True
        if position['type'] == 'sell' and current_price <= position['take_profit']:
            return True
            
        return False
        
    def update_config(self, new_config: Dict):
        """
        Met à jour la configuration en temps réel
        
        Args:
            new_config (dict): Nouvelle configuration
        """
        self.log_message.emit("Mise à jour de la configuration")
        
        # Mise à jour des paramètres RSI
        if 'rsi_period' in new_config:
            self.rsi.period = new_config['rsi_period']
            self.log_message.emit(f"Nouvelle période RSI: {self.rsi.period}")
            
        if 'rsi_overbought' in new_config:
            self.rsi.overbought = new_config['rsi_overbought']
            self.log_message.emit(f"Nouveau niveau RSI surachat: {self.rsi.overbought}")
            
        if 'rsi_oversold' in new_config:
            self.rsi.oversold = new_config['rsi_oversold']
            self.log_message.emit(f"Nouveau niveau RSI survente: {self.rsi.oversold}")
            
        # Mise à jour des autres paramètres
        if 'symbols' in new_config:
            self.config['symbol'] = new_config['symbols'][0]
            
        if 'timeframe' in new_config:
            self.config['timeframe'] = new_config['timeframe']
            
        if 'min_volume' in new_config:
            self.config['min_volume'] = new_config['min_volume']
            
        if 'min_volatility' in new_config:
            self.config['min_volatility'] = new_config['min_volatility']
            
        if 'signal_strength_threshold' in new_config:
            self.config['signal_strength_threshold'] = new_config['signal_strength_threshold']
            
        # Mise à jour de la configuration complète
        self.config.update(new_config)
        
        self.log_message.emit("Configuration mise à jour avec succès")