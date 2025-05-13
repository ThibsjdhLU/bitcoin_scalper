"""
Service de gestion du dashboard et de son état.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import MetaTrader5 as mt5
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
import threading

from ..services.mt5_service import MT5Service
from ..services.storage_service import StorageService
from ..services.backtest_service import BacktestService

logger = logging.getLogger(__name__)

class DashboardService:
    """Service gérant le dashboard et coordonnant les différents services."""
    
    _instance = None
    _initialization_lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._initialization_lock:
                if cls._instance is None:
                    cls._instance = super(DashboardService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialise le service dashboard."""
        self.mt5_service = MT5Service()
        if not self.mt5_service.connected:
            if not self.mt5_service.connect():
                logger.error("Échec de la connexion MT5")
        self.storage_service = StorageService()
        self.backtest_service = BacktestService()
        self._initialize_session_state()
        if self.mt5_service.connected:
            st.session_state.bot_status = "Actif"
    
    def _initialize_session_state(self):
        """Initialise les variables de session Streamlit."""
        if 'account_stats' not in st.session_state:
            st.session_state.account_stats = {
                'balance': 0.0,
                'equity': 0.0,
                'profit': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'open_trades': 0,
                'total_trades': 0
            }
        if 'log_messages' not in st.session_state:
            st.session_state.log_messages = []
        if 'trades_history' not in st.session_state:
            st.session_state.trades_history = pd.DataFrame()
        if 'indicators' not in st.session_state:
            st.session_state.indicators = {
                'show_sma': False,
                'sma_period': 20,
                'show_ema': False,
                'ema_period': 9,
                'show_bollinger': False,
                'bollinger_period': 20,
                'show_rsi': False,
                'rsi_period': 14,
                'show_macd': False,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'bb_upper': 0,
                'bb_lower': 0
            }
        defaults = {
            'trading_params': {
                'initial_capital': 10000.0,
                'risk_per_trade': 1.0,
                'strategy': ['EMA Crossover'],
                'take_profit': 2.0,
                'stop_loss': 1.0,
                'trailing_stop': False
            },
            'selected_symbol': "BTCUSD",
            'bot_status': "Inactif",
            'confirm_action': None,
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value.copy() if isinstance(value, dict) else value
        if 'selected_symbol' not in st.session_state:
            st.session_state.selected_symbol = "BTCUSD"  # Default value
    
    @st.cache_data(ttl=60)
    def get_available_symbols(_self) -> List[str]:
        """Récupère la liste des symboles disponibles."""
        try:
            symbols = _self.mt5_service.get_available_symbols()
            if not symbols:
                # Valeurs par défaut si le service MT5 ne retourne rien
                symbols = ["BTCUSD", "ETHUSD", "XRPUSD"]
            return symbols
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des symboles: {str(e)}")
            return ["BTCUSD"]
    
    def update_data(self):
        """Met à jour toutes les données du dashboard"""
        try:
            # Mettre à jour les données brutes
            raw_data = self.get_raw_data()
            if raw_data is not None:
                st.session_state.price_history = raw_data
                
                # Mettre à jour les indicateurs
                self._update_indicators()
                
                # Mettre à jour les statistiques
                self._update_account_stats()
                
                # Mettre à jour l'historique des trades
                self._update_trades_history()
                
                # Mettre à jour les logs
                self._update_logs()
                
                logger.info("Données du dashboard mises à jour avec succès")
                return True
            return False
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des données: {e}")
            return False
    
    def _get_positions(self) -> Optional[pd.DataFrame]:
        """Récupère les positions ouvertes et fermées."""
        try:
            # Positions depuis MT5
            positions = self.mt5_service.get_positions()
            
            # Charger l'historique des  sauvegardés
            saved_trades = self.storage_service.load_trades()
            
            # Combiner les deux sources
            if saved_trades is not None and not saved_trades.empty:
                if positions is not None and not positions.empty:
                    combined = pd.concat([positions, saved_trades], ignore_index=True).drop_duplicates()
                else:
                    combined = saved_trades
            else:
                combined = positions
                
            return combined
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des positions: {str(e)}")
            return None
    
    def _get_price_history(self, symbol: str) -> Optional[pd.DataFrame]:
        """Récupère l'historique des prix pour un symbole donné."""
        try:
            data = self.mt5_service.get_price_history(symbol)
            if data is None or data.empty:
                self.add_log(f"Aucune donnée pour {symbol}", level="error")
                return pd.DataFrame()  # Retourne un DataFrame vide au lieu de None
            return data
        except Exception as e:
            self.add_log(f"Erreur historique: {str(e)}", level="error")
            return pd.DataFrame()
    
    def _update_account_stats(self):
        """Met à jour les statistiques du compte."""
        try:
            account_info = self.mt5_service.get_account_info()
            if not account_info or not isinstance(account_info, dict):
                logger.warning("Données de compte invalides")
                return
            
            balance = account_info.get('balance', 0.0)  # Use .get() to avoid KeyError
            equity = account_info.get('equity', 0.0)    # Use .get() to avoid KeyError
            if balance is None or equity is None:
                logger.warning("Les données de compte ne contiennent pas les informations nécessaires.")
                return
            
            st.session_state.account_stats['balance'] = balance
            st.session_state.account_stats['equity'] = equity
            st.session_state.account_stats['profit'] = account_info.get('profit', 0.0)
            
            logger.info(f"Statistiques du compte mises à jour: {st.session_state.account_stats}")
            
            # Calculer le taux de réussite si des trades sont disponibles
            if st.session_state.trades_history is not None and not st.session_state.trades_history.empty:
                trades = st.session_state.trades_history
                
                # Taux de réussite
                if 'profit' in trades.columns:
                    winning_trades = len(trades[trades['profit'] > 0])
                    total_trades = len(trades)
                    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                    st.session_state.account_stats['win_rate'] = win_rate
                
                # Drawdown maximum
                if 'equity' in account_info:
                    equity_values = [account_info['equity']]
                    peaks = np.maximum.accumulate(equity_values)
                    drawdowns = (peaks - equity_values) / peaks * 100
                    max_drawdown = max(drawdowns) if drawdowns else 0
                    st.session_state.account_stats['max_drawdown'] = max_drawdown
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des statistiques du compte: {str(e)}")
    
    def handle_bot_action(self, action: str):
        """Gère les actions du bot (démarrer, arrêter, réinitialiser)."""
        if action == "start" and st.session_state.confirm_action == "start":
            st.session_state.bot_status = "Actif"
            st.session_state.need_refresh = True  # Forcer le rafraîchissement
            self.add_log("Bot démarré", level="info")
            
            # Ajouter les logs des stratégies
            if isinstance(st.session_state.trading_params['strategy'], list):
                strategies = st.session_state.trading_params['strategy']
                self.add_log(f"Stratégies activées: {', '.join(strategies)}", level="info")
                
                # Logs détaillés pour chaque stratégie
                for strategy in strategies:
                    if strategy == "EMA Crossover":
                        self.add_log("📈 EMA Crossover: Surveille le croisement de moyennes mobiles exponentielles", level="info")
                    elif strategy == "RSI":
                        self.add_log("📊 RSI: Surveille les conditions de surachat/survente", level="info")
                    elif strategy == "MACD":
                        self.add_log("🔍 MACD: Surveille les croisements et divergences", level="info")
                    elif strategy == "Bollinger Bands":
                        self.add_log("📏 Bollinger Bands: Surveille les dépassements des bandes", level="info")
                    elif strategy == "Combinaison":
                        self.add_log("🔄 Combinaison: Utilise plusieurs indicateurs pour confirmer les signaux", level="info")
            else:
                self.add_log(f"Stratégie activée: {st.session_state.trading_params['strategy']}", level="info")
            
            # Ajouter les paramètres de trading
            self.add_log(f"Capital initial: ${st.session_state.trading_params['initial_capital']}", level="info")
            self.add_log(f"Risque par trade: {st.session_state.trading_params['risk_per_trade']}%", level="info")
            self.add_log(f"Take Profit: {st.session_state.trading_params['take_profit']}%", level="info")
            self.add_log(f"Stop Loss: {st.session_state.trading_params['stop_loss']}%", level="info")
            
            st.session_state.confirm_action = None
            
            # Exécuter la simulation des signaux
            self.simulate_trading_signals()
        elif action == "stop":
            if st.session_state.confirm_action == "stop":
                st.session_state.bot_status = "Inactif"
                self.add_log("Bot arrêté", level="info")
                st.session_state.confirm_action = None
                # TODO: Implémenter la logique d'arrêt du bot
            else:
                st.session_state.confirm_action = "stop"
        elif action == "reset":
            if st.session_state.confirm_action == "reset":
                st.session_state.bot_status = "Inactif"
                self.add_log("Bot réinitialisé", level="info")
                st.session_state.confirm_action = None
                # TODO: Implémenter la logique de réinitialisation du bot
            else:
                st.session_state.confirm_action = "reset"
        elif action == "cancel":
            st.session_state.confirm_action = None
    
    def add_log(self, message: str, level: str = "info"):
        """Ajoute un message au journal des logs."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] [{level.upper()}] {message}"
        
        # Ajouter au journal en mémoire pour l'UI
        st.session_state.log_messages.append(log_entry)
        
        # Limiter à 100 entrées
        if len(st.session_state.log_messages) > 100:
            st.session_state.log_messages = st.session_state.log_messages[-100:]
        
        # Nettoyer les émojis pour le logging système (pour éviter les erreurs d'encodage)
        clean_message = re.sub(r'[^\x00-\x7F]+', '', message)
        
        # Sauvegarder dans le fichier de logs avec message nettoyé
        try:
            self.storage_service.save_log(clean_message, level)
        except Exception as e:
            # Ne pas faire planter l'application si la sauvegarde échoue
            pass
        
        # Logguer avec le module logging (sans emoji)
        if level == "error":
            logger.error(clean_message)
        elif level == "warning":
            logger.warning(clean_message)
        else:
            logger.info(clean_message)
    
    def save_trading_params(self, params: Dict[str, Any]):
        """Sauvegarde les paramètres de trading."""
        st.session_state.trading_params = params
        # TODO: Sauvegarder dans un fichier de configuration
        self.add_log("Paramètres de trading sauvegardés", level="info")
    
    def create_price_chart(self, with_indicators: bool = True) -> go.Figure:
        """Crée un graphique de prix interactif avec Plotly."""
        try:
            # Vérifier les données
            if 'price_history' not in st.session_state or st.session_state.price_history is None or st.session_state.price_history.empty:
                logger.error("Aucune donnée disponible pour le graphique!")
                return go.Figure().add_annotation(
                    text="Aucune donnée disponible",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=20)
                )
            
            # Vérification des colonnes obligatoires
            required_columns = ['open', 'high', 'low', 'close']
            if not all(col in st.session_state.price_history.columns for col in required_columns):
                logger.error(f"Colonnes manquantes: {required_columns}")
                return go.Figure().add_annotation(
                    text="Format de données incorrect",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=20)
                )
            
            # Créer la figure
            if with_indicators and st.session_state.indicators.get('show_rsi', False) or st.session_state.indicators.get('show_macd', False):
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.1, 
                                    row_heights=[0.7, 0.3],
                                    specs=[[{"type": "candlestick"}],
                                           [{"type": "scatter"}]])
            else:
                fig = go.Figure()
            
            # Données de prix
            price_data = st.session_state.price_history
            
            # Chandelier japonais
            candlestick = go.Candlestick(
                x=price_data.index,
                open=price_data['open'],
                high=price_data['high'],
                low=price_data['low'],
                close=price_data['close'],
                name="OHLC",
                increasing_line_color='#00FF88',
                decreasing_line_color='#FF5588'
            )
            
            # Ajouter le chandelier
            if with_indicators and (st.session_state.indicators.get('show_rsi', False) or st.session_state.indicators.get('show_macd', False)):
                fig.add_trace(candlestick, row=1, col=1)
            else:
                fig.add_trace(candlestick)
            
            # Ajouter les indicateurs techniques
            if with_indicators:
                if st.session_state.indicators.get('show_sma', False):
                    period = st.session_state.indicators.get('sma_period', 20)
                    sma = price_data['close'].rolling(window=period).mean()
                    sma_trace = go.Scatter(
                        x=price_data.index,
                        y=sma,
                        mode='lines',
                        line=dict(color='blue', width=1),
                        name=f'SMA ({period})'
                    )
                    if with_indicators and (st.session_state.indicators.get('show_rsi', False) 
                        or st.session_state.indicators.get('show_macd', False)):
                        fig = make_subplots(sma_trace, row=1, col=1)
                    else:
                        fig = go.Figure()
                
                if st.session_state.indicators.get('show_ema', False):
                    period = st.session_state.indicators.get('ema_period', 9)
                    ema = price_data['close'].ewm(span=period, adjust=False).mean()
                    ema_trace = go.Scatter(
                        x=price_data.index,
                        y=ema,
                        mode='lines',
                        line=dict(color='orange', width=1),
                        name=f'EMA ({period})'
                    )
                    if st.session_state.indicators.get('show_rsi', False) or st.session_state.indicators.get('show_macd', False):
                        fig.add_trace(ema_trace, row=1, col=1)
                    else:
                        fig.add_trace(ema_trace)
                
                if st.session_state.indicators.get('show_bollinger', False):
                    period = st.session_state.indicators.get('bollinger_period', 20)
                    sma = price_data['close'].rolling(window=period).mean()
                    std = price_data['close'].rolling(window=period).std()
                    upper_band = sma + (std * 2)
                    lower_band = sma - (std * 2)
                    
                    upper_trace = go.Scatter(
                        x=price_data.index,
                        y=upper_band,
                        mode='lines',
                        line=dict(color='rgba(100, 100, 255, 0.5)', width=1),
                        name='Bollinger (upper)'
                    )
                    
                    lower_trace = go.Scatter(
                        x=price_data.index,
                        y=lower_band,
                        mode='lines',
                        line=dict(color='rgba(100, 100, 255, 0.5)', width=1),
                        name='Bollinger (lower)',
                        fill='tonexty',
                        fillcolor='rgba(100, 100, 255, 0.1)'
                    )
                    
                    if st.session_state.indicators.get('show_rsi', False) or st.session_state.indicators.get('show_macd', False):
                        fig.add_trace(upper_trace, row=1, col=1)
                        fig.add_trace(lower_trace, row=1, col=1)
                    else:
                        fig.add_trace(upper_trace)
                        fig.add_trace(lower_trace)
                
                if st.session_state.indicators.get('show_rsi', False):
                    period = st.session_state.indicators.get('rsi_period', 14)
                    delta = price_data['close'].diff()
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    avg_gain = gain.rolling(window=period).mean()
                    avg_loss = loss.rolling(window=period).mean()
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                    
                    rsi_trace = go.Scatter(
                        x=price_data.index,
                        y=rsi,
                        mode='lines',
                        line=dict(color='purple', width=1),
                        name=f'RSI ({period})'
                    )
                    
                    # Lignes de référence RSI
                    overbought = go.Scatter(
                        x=price_data.index,
                        y=[70] * len(price_data),
                        mode='lines',
                        line=dict(color='red', width=1, dash='dash'),
                        name='Survente'
                    )
                    
                    oversold = go.Scatter(
                        x=price_data.index,
                        y=[30] * len(price_data),
                        mode='lines',
                        line=dict(color='green', width=1, dash='dash'),
                        name='Surachat'
                    )
                    
                    fig.add_trace(rsi_trace, row=2, col=1)
                    fig.add_trace(overbought, row=2, col=1)
                    fig.add_trace(oversold, row=2, col=1)
                
                if st.session_state.indicators.get('show_macd', False):
                    fast = st.session_state.indicators.get('macd_fast', 12)
                    slow = st.session_state.indicators.get('macd_slow', 26)
                    signal = st.session_state.indicators.get('macd_signal', 9)
                    
                    ema_fast = price_data['close'].ewm(span=fast, adjust=False).mean()
                    ema_slow = price_data['close'].ewm(span=slow, adjust=False).mean()
                    macd_line = ema_fast - ema_slow
                    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
                    histogram = macd_line - signal_line
                    
                    macd_trace = go.Scatter(
                        x=price_data.index,
                        y=macd_line,
                        mode='lines',
                        line=dict(color='blue', width=1),
                        name=f'MACD ({fast},{slow})'
                    )
                    
                    signal_trace = go.Scatter(
                        x=price_data.index,
                        y=signal_line,
                        mode='lines',
                        line=dict(color='red', width=1),
                        name=f'Signal ({signal})'
                    )
                    
                    # Histogramme MACD
                    colors = ['green' if val >= 0 else 'red' for val in histogram]
                    histogram_trace = go.Bar(
                        x=price_data.index,
                        y=histogram,
                        name='Histogram',
                        marker_color=colors
                    )
                    
                    if st.session_state.indicators.get('show_rsi', False):
                        # Si RSI est déjà utilisé, ne pas afficher MACD pour éviter la confusion
                        pass
                    else:
                        fig.add_trace(histogram_trace, row=2, col=1)
                        fig.add_trace(macd_trace, row=2, col=1)
                        fig.add_trace(signal_trace, row=2, col=1)
            
            # Mise en forme du graphique
            fig.update_layout(
                title=f"Graphique {st.session_state.selected_symbol}",
                xaxis_title="Date",
                yaxis_title="Prix",
                height=600,
                xaxis_rangeslider_visible=False,
                template="plotly_dark",
                margin=dict(l=50, r=50, t=50, b=50),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Erreur lors de la création du graphique: {str(e)}")
            return go.Figure().add_annotation(
                text=f"Erreur: {str(e)}",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
    
    def calculate_statistics(self) -> Dict[str, Any]:
        """Calcule les statistiques de trading."""
        results = {
            'win_rate': 0.0,
            'winning_trades': 0,
            'total_trades': 0,
            'avg_profit': 0.0,
            'max_drawdown': st.session_state.account_stats.get('max_drawdown', 0.0)
        }
        
        trades_df = st.session_state.trades_history
        
        if trades_df is not None and not trades_df.empty and 'profit' in trades_df.columns:
            # Taux de réussite
            winning_trades = len(trades_df[trades_df['profit'] > 0])
            total_trades = len(trades_df)
            
            results['winning_trades'] = winning_trades
            results['total_trades'] = total_trades
            results['win_rate'] = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Profit moyen
            if total_trades > 0:
                results['avg_profit'] = trades_df['profit'].mean()
                
        if 'profit' in st.session_state.account_stats:
            delta_class = "positive-delta" if st.session_state.account_stats['profit'] >= 0 else "negative-delta"
        else:
            delta_class = "unknown"  # Ou une autre valeur par défaut
        
        return results
    
    def simulate_trading_signals(self):
        """Simule les signaux de trading basés sur les indicateurs."""
        try:
            # Récupérer les données de prix
            price_data = st.session_state.price_history
            if price_data is None or len(price_data) == 0:
                self.add_log("Pas de données de prix disponibles pour générer des signaux")
                return

            # Générer les signaux pour chaque stratégie
            signals = {}
            for strategy in st.session_state.trading_params['strategy']:
                if strategy == "EMA Crossover":
                    signals[strategy] = self._check_ema_crossover(price_data)
                elif strategy == "RSI":
                    signals[strategy] = self._check_rsi(price_data)
                elif strategy == "MACD":
                    signals[strategy] = self._check_macd(price_data)
                elif strategy == "Bollinger Bands":
                    signals[strategy] = self._check_bollinger_bands(price_data)
                elif strategy == "Combinaison":
                    signals[strategy] = self._check_combined_signals(price_data)

            # Analyser les signaux
            final_signal = self._analyze_signals(signals)
            
            # Ajouter le signal aux logs
            if final_signal != "CONSERVER":
                self.add_log(f"Signal généré: {final_signal}")
                
        except Exception as e:
            logger.error(f"Erreur lors de la simulation des signaux: {e}")
            self.add_log(f"Erreur lors de la génération des signaux: {e}")

    def _update_indicators(self):
        """Met à jour les indicateurs techniques."""
        try:
            price_data = st.session_state.price_history
            if price_data is not None and not price_data.empty:
                # RSI
                delta = price_data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                st.session_state.indicators['rsi'] = 100 - (100 / (1 + rs)).iloc[-1]
                
                # MACD
                exp1 = price_data['close'].ewm(span=12, adjust=False).mean()
                exp2 = price_data['close'].ewm(span=26, adjust=False).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=9, adjust=False).mean()
                st.session_state.indicators['macd'] = macd.iloc[-1]
                st.session_state.indicators['macd_signal'] = signal.iloc[-1]
                
                # Bollinger Bands
                sma = price_data['close'].rolling(window=20).mean()
                std = price_data['close'].rolling(window=20).std()
                st.session_state.indicators['bb_upper'] = (sma + (std * 2)).iloc[-1]
                st.session_state.indicators['bb_lower'] = (sma - (std * 2)).iloc[-1]
                
                # EMA
                st.session_state.indicators['ema9'] = price_data['close'].ewm(span=9, adjust=False).mean().iloc[-1]
                st.session_state.indicators['ema21'] = price_data['close'].ewm(span=21, adjust=False).mean().iloc[-1]
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des indicateurs: {e}")

    @st.cache_data(ttl=30)
    def fetch_raw_data(self):
        """Récupère les données brutes depuis MT5."""
        try:
            logger.info(f"Tentative de récupération des données pour {st.session_state.selected_symbol}")
            
            # Vérifier la connexion MT5
            if not self.mt5_service.connected:
                if not self.mt5_service.connect():
                    logger.error("Impossible de se connecter à MT5")
                    return None
            
            # Récupérer les données
            data = self.mt5_service.get_price_history(st.session_state.selected_symbol)
            
            if data is None:
                logger.error("Aucune donnée reçue de MT5Service")
                return None
                
            if data.empty:
                logger.error("DataFrame vide reçu de MT5Service")
                return None
                
            logger.info(f"Données récupérées avec succès: {len(data)} lignes")
            
            # Mettre à jour l'état de session
            st.session_state.price_history = data
            
            return data
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données: {str(e)}", exc_info=True)
            return None

    def _update_trades_history(self):
        """Met à jour l'historique des trades."""
        try:
            trades = mt5.history_deals_get(0, datetime.now())
            if not trades or len(trades) == 0:
                logger.warning("Aucun trade historique trouvé")
                return

            # Vérification supplémentaire pour les données corrompues
            valid_trades = [t for t in trades if isinstance(t, mt5.TradeDeal)]
            if not valid_trades:
                logger.error("Format de trades invalide")
                return

            trades_df = pd.DataFrame(
                (t._asdict() for t in valid_trades),
                columns=valid_trades[0]._asdict().keys() if valid_trades else []
            )
            
            if trades_df.empty:
                logger.warning("Le DataFrame des trades est vide.")
                return
            
            # Logique pour vérifier les trades gagnants
            if (trades_df['profit'] > 0).any():
                logger.info("Il y a des trades gagnants.")
            
            # Mettre à jour la session state
            st.session_state.trades_history = trades_df
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour de l'historique des trades: {str(e)}")
            
    def _update_logs(self):
        """Met à jour les logs."""
        try:
            # Sauvegarder les logs dans le fichier
            self.storage_service.save_logs(st.session_state.log_messages)
            
            # Nettoyer les logs en mémoire
            st.session_state.log_messages = []
            
            logger.info("Logs mis à jour avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des logs: {str(e)}")

    def _check_ema_crossover(self, price_data: pd.DataFrame) -> str:
        """Vérifie le signal de croisement EMA."""
        try:
            ema_short = price_data['close'].ewm(span=9, adjust=False).mean()
            ema_long = price_data['close'].ewm(span=21, adjust=False).mean()
            
            if ema_short.iloc[-1] > ema_long.iloc[-1] and ema_short.iloc[-2] <= ema_long.iloc[-2]:
                return "BUY"
            elif ema_short.iloc[-1] < ema_long.iloc[-1] and ema_short.iloc[-2] >= ema_long.iloc[-2]:
                return "SELL"
            return "HOLD"
        except Exception as e:
            logger.error(f"Erreur lors de la vérification EMA: {e}")
            return "HOLD"

    def _check_rsi(self, price_data: pd.DataFrame) -> str:
        """Vérifie le signal RSI."""
        try:
            delta = price_data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            current_rsi = rsi.iloc[-1]
            
            if current_rsi < 30:
                return "BUY"
            elif current_rsi > 70:
                return "SELL"
            return "HOLD"
        except Exception as e:
            logger.error(f"Erreur lors de la vérification RSI: {e}")
            return "HOLD"

    def _check_macd(self, price_data: pd.DataFrame) -> str:
        """Vérifie le signal MACD."""
        try:
            ema12 = price_data['close'].ewm(span=12, adjust=False).mean()
            ema26 = price_data['close'].ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            
            if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]:
                return "BUY"
            elif macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]:
                return "SELL"
            return "HOLD"
        except Exception as e:
            logger.error(f"Erreur lors de la vérification MACD: {e}")
            return "HOLD"

    def _check_bollinger_bands(self, price_data: pd.DataFrame) -> str:
        """Vérifie le signal des bandes de Bollinger."""
        try:
            period = 20
            sma = price_data['close'].rolling(window=period).mean()
            std = price_data['close'].rolling(window=period).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            
            current_price = price_data['close'].iloc[-1]
            
            if current_price < lower_band.iloc[-1]:
                return "BUY"
            elif current_price > upper_band.iloc[-1]:
                return "SELL"
            return "HOLD"
        except Exception as e:
            logger.error(f"Erreur lors de la vérification Bollinger: {e}")
            return "HOLD"

    def _check_combined_signals(self, price_data: pd.DataFrame) -> str:
        """Vérifie les signaux combinés."""
        try:
            signals = {
                "EMA": self._check_ema_crossover(price_data),
                "RSI": self._check_rsi(price_data),
                "MACD": self._check_macd(price_data),
                "Bollinger": self._check_bollinger_bands(price_data)
            }
            
            buy_count = sum(1 for signal in signals.values() if signal == "BUY")
            sell_count = sum(1 for signal in signals.values() if signal == "SELL")
            
            if buy_count > sell_count and buy_count >= 2:
                return "BUY"
            elif sell_count > buy_count and sell_count >= 2:
                return "SELL"
            return "HOLD"
        except Exception as e:
            logger.error(f"Erreur lors de la vérification des signaux combinés: {e}")
            return "HOLD"

    def _analyze_signals(self, signals: Dict[str, str]) -> str:
        """Analyse les signaux et retourne le signal final."""
        try:
            # Implémentation de la logique d'analyse des signaux
            # Cette méthode doit être implémentée en fonction de la logique de votre stratégie
            # Pour cet exemple, nous allons simplement retourner le signal le plus fréquent
            return max(signals, key=signals.get)
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse des signaux: {e}")
            return "CONSERVER"

    def _update_dashboard(self):
        """Met à jour les données du dashboard."""
        try:
            # Vérifier si nous sommes dans un thread
            if threading.current_thread() is not threading.main_thread():
                # Nous sommes dans un thread, utiliser le contexte principal
                ctx = get_script_run_ctx()
                if ctx:
                    add_script_run_ctx(threading.current_thread(), ctx)
            
            # S'assurer que les variables de session sont initialisées
            with self._initialization_lock:
                if not hasattr(st.session_state, '_initialized'):
                    self._initialize_session_state()
                    st.session_state._initialized = True
            
            # Mettre à jour les données
            self._update_logs()
            self._update_indicators()
            self._update_account_stats()
            self._update_trades_history()
            
            # Mettre à jour le timestamp de rafraîchissement
            st.session_state.last_refresh = datetime.now()
            logger.info("Données du dashboard mises à jour avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour du dashboard: {str(e)}")
            # Ne pas propager l'erreur pour éviter de casser le thread 

    def get_raw_data(self):
        """Récupère les données brutes pour le tableau de bord."""
        return self.mt5_service.get_price_history("BTCUSD")  # Exemple simplifié

    def _update_dashboard(self):
        """Met à jour les données du dashboard."""
        try:
            # Vérifier si nous sommes dans un thread
            if threading.current_thread() is not threading.main_thread():
                # Nous sommes dans un thread, utiliser le contexte principal
                ctx = get_script_run_ctx()
                if ctx:
                    add_script_run_ctx(threading.current_thread(), ctx)
            
            # S'assurer que les variables de session sont initialisées
            with self._initialization_lock:
                if not hasattr(st.session_state, '_initialized'):
                    self._initialize_session_state()
                    st.session_state._initialized = True
            
            # Mettre à jour les données
            self._update_logs()
            self._update_indicators()
            self._update_account_stats()
            self._update_trades_history()
            
            # Mettre à jour le timestamp de rafraîchissement
            st.session_state.last_refresh = datetime.now()
            logger.info("Données du dashboard mises à jour avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour du dashboard: {str(e)}")
            # Ne pas propager l'erreur pour éviter de casser le thread 

    def fetch_raw_data(self):
        """Récupère les données brutes pour le tableau de bord."""
        try:
            return self.mt5_service.get_price_history("BTCUSD")  # Remplacez "BTCUSD" par le symbole souhaité
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données brutes: {str(e)}")
            return None 