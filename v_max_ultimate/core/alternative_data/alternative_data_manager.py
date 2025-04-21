import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
import requests
from datetime import datetime, timedelta
import json

class AlternativeDataManager:
    """
    Gestionnaire de données alternatives pour le trading
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.on_chain_data = None
        self.sentiment_data = None
        self.social_metrics = None
        
    def fetch_on_chain_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Récupère les données on-chain de Glassnode
        
        Args:
            start_date: Date de début
            end_date: Date de fin
            
        Returns:
            DataFrame: Données on-chain
        """
        try:
            api_key = self.config.get('glassnode_api_key')
            if not api_key:
                raise ValueError("Glassnode API key not found in config")
                
            # Endpoints Glassnode
            endpoints = {
                'active_addresses': 'https://api.glassnode.com/v1/metrics/addresses/active_count',
                'transaction_volume': 'https://api.glassnode.com/v1/metrics/transactions/volume_sum',
                'network_profit_loss': 'https://api.glassnode.com/v1/metrics/mining/profitability',
                'exchange_flow': 'https://api.glassnode.com/v1/metrics/transactions/transfers_volume_exchanges_net'
            }
            
            data = {}
            for metric, url in endpoints.items():
                params = {
                    'api_key': api_key,
                    'a': 'BTC',
                    's': int(start_date.timestamp()),
                    'u': int(end_date.timestamp())
                }
                
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data[metric] = pd.DataFrame(response.json())
                else:
                    self.logger.error(f"Error fetching {metric}: {response.status_code}")
                    
            # Fusion des données
            self.on_chain_data = pd.concat(data.values(), axis=1)
            self.on_chain_data.columns = list(endpoints.keys())
            
            return self.on_chain_data
            
        except Exception as e:
            self.logger.error(f"Error fetching on-chain data: {str(e)}")
            raise
            
    def fetch_sentiment_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Récupère les données de sentiment des médias sociaux
        
        Args:
            start_date: Date de début
            end_date: Date de fin
            
        Returns:
            DataFrame: Données de sentiment
        """
        try:
            # Sources de données de sentiment
            sources = {
                'twitter': self._fetch_twitter_sentiment,
                'reddit': self._fetch_reddit_sentiment,
                'news': self._fetch_news_sentiment
            }
            
            data = {}
            for source, fetch_func in sources.items():
                data[source] = fetch_func(start_date, end_date)
                
            # Fusion des données
            self.sentiment_data = pd.concat(data.values(), axis=1)
            self.sentiment_data.columns = list(sources.keys())
            
            return self.sentiment_data
            
        except Exception as e:
            self.logger.error(f"Error fetching sentiment data: {str(e)}")
            raise
            
    def _fetch_twitter_sentiment(self, start_date: datetime, end_date: datetime) -> pd.Series:
        """
        Récupère le sentiment Twitter
        
        Args:
            start_date: Date de début
            end_date: Date de fin
            
        Returns:
            Series: Sentiment Twitter
        """
        # Placeholder - À implémenter avec l'API Twitter
        return pd.Series(np.random.random(len(pd.date_range(start_date, end_date))))
        
    def _fetch_reddit_sentiment(self, start_date: datetime, end_date: datetime) -> pd.Series:
        """
        Récupère le sentiment Reddit
        
        Args:
            start_date: Date de début
            end_date: Date de fin
            
        Returns:
            Series: Sentiment Reddit
        """
        # Placeholder - À implémenter avec l'API Reddit
        return pd.Series(np.random.random(len(pd.date_range(start_date, end_date))))
        
    def _fetch_news_sentiment(self, start_date: datetime, end_date: datetime) -> pd.Series:
        """
        Récupère le sentiment des actualités
        
        Args:
            start_date: Date de début
            end_date: Date de fin
            
        Returns:
            Series: Sentiment des actualités
        """
        # Placeholder - À implémenter avec une API d'actualités
        return pd.Series(np.random.random(len(pd.date_range(start_date, end_date))))
        
    def calculate_social_metrics(self) -> pd.DataFrame:
        """
        Calcule les métriques sociales agrégées
        
        Returns:
            DataFrame: Métriques sociales
        """
        try:
            if self.sentiment_data is None:
                raise ValueError("Sentiment data not available")
                
            metrics = pd.DataFrame()
            
            # Sentiment moyen
            metrics['average_sentiment'] = self.sentiment_data.mean(axis=1)
            
            # Volatilité du sentiment
            metrics['sentiment_volatility'] = self.sentiment_data.std(axis=1)
            
            # Momentum du sentiment
            metrics['sentiment_momentum'] = metrics['average_sentiment'].diff()
            
            # Indice de consensus
            metrics['consensus_index'] = (
                (self.sentiment_data > 0).mean(axis=1) - 
                (self.sentiment_data < 0).mean(axis=1)
            )
            
            self.social_metrics = metrics
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating social metrics: {str(e)}")
            raise
            
    def generate_trading_signals(self) -> Dict:
        """
        Génère des signaux de trading basés sur les données alternatives
        
        Returns:
            Dict: Signaux de trading
        """
        try:
            signals = {
                'on_chain_signals': [],
                'sentiment_signals': [],
                'composite_signals': []
            }
            
            # Signaux on-chain
            if self.on_chain_data is not None:
                for column in self.on_chain_data.columns:
                    signals['on_chain_signals'].append({
                        'metric': column,
                        'value': self.on_chain_data[column].iloc[-1],
                        'trend': self._calculate_trend(self.on_chain_data[column]),
                        'strength': self._calculate_signal_strength(self.on_chain_data[column])
                    })
                    
            # Signaux de sentiment
            if self.social_metrics is not None:
                for column in self.social_metrics.columns:
                    signals['sentiment_signals'].append({
                        'metric': column,
                        'value': self.social_metrics[column].iloc[-1],
                        'trend': self._calculate_trend(self.social_metrics[column]),
                        'strength': self._calculate_signal_strength(self.social_metrics[column])
                    })
                    
            # Signaux composites
            if self.on_chain_data is not None and self.social_metrics is not None:
                signals['composite_signals'] = self._generate_composite_signals()
                
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating trading signals: {str(e)}")
            raise
            
    def _calculate_trend(self, series: pd.Series) -> str:
        """
        Calcule la tendance d'une série
        
        Args:
            series: Série temporelle
            
        Returns:
            str: Tendance ('up', 'down', 'neutral')
        """
        if len(series) < 2:
            return 'neutral'
            
        recent = series.iloc[-5:]
        slope = np.polyfit(range(len(recent)), recent.values, 1)[0]
        
        if slope > 0.1:
            return 'up'
        elif slope < -0.1:
            return 'down'
        else:
            return 'neutral'
            
    def _calculate_signal_strength(self, series: pd.Series) -> float:
        """
        Calcule la force d'un signal
        
        Args:
            series: Série temporelle
            
        Returns:
            float: Force du signal (0-1)
        """
        if len(series) < 2:
            return 0.0
            
        # Normalisation de la série
        normalized = (series - series.min()) / (series.max() - series.min())
        
        # Calcul de la force basée sur la volatilité et la tendance
        volatility = normalized.std()
        trend = abs(normalized.iloc[-1] - normalized.iloc[0])
        
        return min(1.0, (1 - volatility) * (1 + trend))
        
    def _generate_composite_signals(self) -> List[Dict]:
        """
        Génère des signaux composites combinant données on-chain et sentiment
        
        Returns:
            List[Dict]: Signaux composites
        """
        signals = []
        
        # Indice de santé du marché
        market_health = (
            self.on_chain_data['active_addresses'].iloc[-1] * 0.3 +
            self.on_chain_data['transaction_volume'].iloc[-1] * 0.3 +
            self.social_metrics['average_sentiment'].iloc[-1] * 0.4
        )
        
        signals.append({
            'type': 'market_health',
            'value': market_health,
            'trend': self._calculate_trend(pd.Series([market_health])),
            'strength': self._calculate_signal_strength(pd.Series([market_health]))
        })
        
        # Indice de momentum social
        social_momentum = (
            self.social_metrics['sentiment_momentum'].iloc[-1] * 0.5 +
            self.on_chain_data['exchange_flow'].iloc[-1] * 0.5
        )
        
        signals.append({
            'type': 'social_momentum',
            'value': social_momentum,
            'trend': self._calculate_trend(pd.Series([social_momentum])),
            'strength': self._calculate_signal_strength(pd.Series([social_momentum]))
        })
        
        return signals 