import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import yfinance as yf

class MarketEventManager:
    """
    Gestionnaire d'événements de marché
    """
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.economic_calendar = {}
        self.market_events = {}
        self.news_impact = {
            'HIGH': 3,
            'MEDIUM': 2,
            'LOW': 1
        }
        
    def fetch_economic_calendar(self, 
                              start_date: datetime,
                              end_date: datetime) -> pd.DataFrame:
        """
        Récupère le calendrier économique
        
        Args:
            start_date: Date de début
            end_date: Date de fin
            
        Returns:
            DataFrame: Calendrier économique
        """
        try:
            # Utilisation de l'API Investing.com (exemple)
            url = f"https://www.investing.com/economic-calendar/"
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            events = []
            for row in soup.find_all('tr', class_='js-event-item'):
                event = {
                    'date': row.find('td', class_='time').text,
                    'country': row.find('td', class_='flag').text,
                    'event': row.find('td', class_='event').text,
                    'impact': row.find('td', class_='impact').text,
                    'forecast': row.find('td', class_='forecast').text,
                    'previous': row.find('td', class_='previous').text
                }
                events.append(event)
                
            return pd.DataFrame(events)
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération du calendrier économique: {str(e)}")
            raise
            
    def analyze_market_events(self, 
                            symbol: str,
                            timeframe: str = '1d') -> Dict:
        """
        Analyse les événements de marché pour un symbole
        
        Args:
            symbol: Symbole à analyser
            timeframe: Intervalle de temps
            
        Returns:
            Dict: Événements de marché détectés
        """
        try:
            # Récupération des données
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1mo', interval=timeframe)
            
            events = {
                'volume_spikes': self._detect_volume_spikes(hist),
                'price_gaps': self._detect_price_gaps(hist),
                'volatility_events': self._detect_volatility_events(hist),
                'trend_changes': self._detect_trend_changes(hist)
            }
            
            return events
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse des événements de marché: {str(e)}")
            raise
            
    def _detect_volume_spikes(self, data: pd.DataFrame) -> List[Dict]:
        """
        Détecte les pics de volume
        
        Args:
            data: DataFrame des données
            
        Returns:
            List: Liste des pics de volume détectés
        """
        try:
            volume_mean = data['Volume'].mean()
            volume_std = data['Volume'].std()
            
            spikes = []
            for idx, row in data.iterrows():
                if row['Volume'] > volume_mean + 2 * volume_std:
                    spikes.append({
                        'date': idx,
                        'volume': row['Volume'],
                        'price': row['Close'],
                        'magnitude': (row['Volume'] - volume_mean) / volume_std
                    })
                    
            return spikes
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la détection des pics de volume: {str(e)}")
            raise
            
    def _detect_price_gaps(self, data: pd.DataFrame) -> List[Dict]:
        """
        Détecte les gaps de prix
        
        Args:
            data: DataFrame des données
            
        Returns:
            List: Liste des gaps détectés
        """
        try:
            gaps = []
            for i in range(1, len(data)):
                prev_close = data['Close'].iloc[i-1]
                curr_open = data['Open'].iloc[i]
                
                gap_size = (curr_open - prev_close) / prev_close
                if abs(gap_size) > 0.02:  # Gap > 2%
                    gaps.append({
                        'date': data.index[i],
                        'gap_size': gap_size,
                        'direction': 'up' if gap_size > 0 else 'down'
                    })
                    
            return gaps
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la détection des gaps: {str(e)}")
            raise
            
    def _detect_volatility_events(self, data: pd.DataFrame) -> List[Dict]:
        """
        Détecte les événements de volatilité
        
        Args:
            data: DataFrame des données
            
        Returns:
            List: Liste des événements de volatilité
        """
        try:
            returns = data['Close'].pct_change()
            volatility = returns.rolling(window=20).std()
            
            events = []
            for idx, vol in volatility.items():
                if vol > volatility.mean() + 2 * volatility.std():
                    events.append({
                        'date': idx,
                        'volatility': vol,
                        'price': data['Close'][idx]
                    })
                    
            return events
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la détection des événements de volatilité: {str(e)}")
            raise
            
    def _detect_trend_changes(self, data: pd.DataFrame) -> List[Dict]:
        """
        Détecte les changements de tendance
        
        Args:
            data: DataFrame des données
            
        Returns:
            List: Liste des changements de tendance
        """
        try:
            # Calcul des moyennes mobiles
            ma20 = data['Close'].rolling(window=20).mean()
            ma50 = data['Close'].rolling(window=50).mean()
            
            changes = []
            for i in range(1, len(data)):
                prev_diff = ma20.iloc[i-1] - ma50.iloc[i-1]
                curr_diff = ma20.iloc[i] - ma50.iloc[i]
                
                if (prev_diff < 0 and curr_diff > 0) or (prev_diff > 0 and curr_diff < 0):
                    changes.append({
                        'date': data.index[i],
                        'direction': 'bullish' if curr_diff > 0 else 'bearish',
                        'price': data['Close'].iloc[i]
                    })
                    
            return changes
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la détection des changements de tendance: {str(e)}")
            raise
            
    def calculate_event_impact(self, 
                             event: Dict,
                             market_data: pd.DataFrame) -> float:
        """
        Calcule l'impact d'un événement sur le marché
        
        Args:
            event: Événement à analyser
            market_data: Données de marché
            
        Returns:
            float: Score d'impact
        """
        try:
            impact_score = 0
            
            # Impact de l'importance de l'événement
            impact_score += self.news_impact.get(event.get('impact', 'LOW'), 1)
            
            # Impact sur le volume
            event_date = pd.to_datetime(event['date'])
            if event_date in market_data.index:
                volume_change = market_data.loc[event_date, 'Volume'] / market_data['Volume'].mean()
                impact_score += min(volume_change - 1, 2)  # Max +2 points
                
            # Impact sur la volatilité
            returns = market_data['Close'].pct_change()
            volatility = returns.rolling(window=20).std()
            if event_date in volatility.index:
                vol_change = volatility[event_date] / volatility.mean()
                impact_score += min(vol_change - 1, 2)  # Max +2 points
                
            return impact_score
            
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul de l'impact: {str(e)}")
            raise
            
    def generate_market_alerts(self,
                             events: Dict,
                             impact_threshold: float = 5.0) -> List[Dict]:
        """
        Génère des alertes de marché
        
        Args:
            events: Événements détectés
            impact_threshold: Seuil d'impact minimum
            
        Returns:
            List: Liste des alertes générées
        """
        try:
            alerts = []
            
            for event_type, event_list in events.items():
                for event in event_list:
                    impact = self.calculate_event_impact(event, pd.DataFrame())  # À adapter selon les données disponibles
                    
                    if impact >= impact_threshold:
                        alert = {
                            'type': event_type,
                            'date': event['date'],
                            'impact': impact,
                            'description': f"Événement {event_type} détecté avec un impact de {impact:.2f}"
                        }
                        alerts.append(alert)
                        
            return alerts
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération des alertes: {str(e)}")
            raise 