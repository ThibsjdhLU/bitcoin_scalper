import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta
import yfinance as yf
from scipy import stats

class CorrelationManager:
    """
    Gestionnaire de corrélations entre actifs
    """
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.correlation_matrix = pd.DataFrame()
        self.pairs_correlation = {}
        self.cointegration_pairs = []
        
    def calculate_correlation_matrix(self,
                                   symbols: List[str],
                                   start_date: datetime,
                                   end_date: datetime,
                                   timeframe: str = '1d') -> pd.DataFrame:
        """
        Calcule la matrice de corrélation entre les actifs
        
        Args:
            symbols: Liste des symboles
            start_date: Date de début
            end_date: Date de fin
            timeframe: Intervalle de temps
            
        Returns:
            DataFrame: Matrice de corrélation
        """
        try:
            # Récupération des données
            data = {}
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date, interval=timeframe)
                data[symbol] = hist['Close']
                
            # Création du DataFrame
            df = pd.DataFrame(data)
            
            # Calcul des corrélations
            self.correlation_matrix = df.corr()
            
            return self.correlation_matrix
            
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul de la matrice de corrélation: {str(e)}")
            raise
            
    def find_highly_correlated_pairs(self,
                                   correlation_threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """
        Trouve les paires d'actifs fortement corrélés
        
        Args:
            correlation_threshold: Seuil de corrélation
            
        Returns:
            List: Liste des paires corrélées
        """
        try:
            pairs = []
            
            for i in range(len(self.correlation_matrix.columns)):
                for j in range(i+1, len(self.correlation_matrix.columns)):
                    corr = self.correlation_matrix.iloc[i,j]
                    if abs(corr) >= correlation_threshold:
                        symbol1 = self.correlation_matrix.columns[i]
                        symbol2 = self.correlation_matrix.columns[j]
                        pairs.append((symbol1, symbol2, corr))
                        
            return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la recherche des paires corrélées: {str(e)}")
            raise
            
    def check_cointegration(self,
                          symbol1: str,
                          symbol2: str,
                          start_date: datetime,
                          end_date: datetime) -> Tuple[bool, float]:
        """
        Vérifie la cointégration entre deux actifs
        
        Args:
            symbol1: Premier symbole
            symbol2: Deuxième symbole
            start_date: Date de début
            end_date: Date de fin
            
        Returns:
            Tuple: (Est cointégré, Score de cointégration)
        """
        try:
            # Récupération des données
            ticker1 = yf.Ticker(symbol1)
            ticker2 = yf.Ticker(symbol2)
            
            hist1 = ticker1.history(start=start_date, end=end_date)['Close']
            hist2 = ticker2.history(start=start_date, end=end_date)['Close']
            
            # Test de cointégration
            score, pvalue, _ = stats.coint(hist1, hist2)
            
            return pvalue < 0.05, score
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la vérification de la cointégration: {str(e)}")
            raise
            
    def find_cointegration_pairs(self,
                               symbols: List[str],
                               start_date: datetime,
                               end_date: datetime) -> List[Dict]:
        """
        Trouve les paires d'actifs cointégrés
        
        Args:
            symbols: Liste des symboles
            start_date: Date de début
            end_date: Date de fin
            
        Returns:
            List: Liste des paires cointégrées
        """
        try:
            pairs = []
            
            for i in range(len(symbols)):
                for j in range(i+1, len(symbols)):
                    is_cointegrated, score = self.check_cointegration(
                        symbols[i], symbols[j], start_date, end_date
                    )
                    
                    if is_cointegrated:
                        pairs.append({
                            'symbol1': symbols[i],
                            'symbol2': symbols[j],
                            'score': score
                        })
                        
            return sorted(pairs, key=lambda x: abs(x['score']), reverse=True)
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la recherche des paires cointégrées: {str(e)}")
            raise
            
    def calculate_spread(self,
                        symbol1: str,
                        symbol2: str,
                        start_date: datetime,
                        end_date: datetime) -> pd.Series:
        """
        Calcule le spread entre deux actifs
        
        Args:
            symbol1: Premier symbole
            symbol2: Deuxième symbole
            start_date: Date de début
            end_date: Date de fin
            
        Returns:
            Series: Série temporelle du spread
        """
        try:
            # Récupération des données
            ticker1 = yf.Ticker(symbol1)
            ticker2 = yf.Ticker(symbol2)
            
            hist1 = ticker1.history(start=start_date, end=end_date)['Close']
            hist2 = ticker2.history(start=start_date, end=end_date)['Close']
            
            # Calcul du spread
            spread = hist1 - hist2
            
            return spread
            
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul du spread: {str(e)}")
            raise
            
    def find_arbitrage_opportunities(self,
                                   symbol1: str,
                                   symbol2: str,
                                   start_date: datetime,
                                   end_date: datetime,
                                   zscore_threshold: float = 2.0) -> List[Dict]:
        """
        Trouve les opportunités d'arbitrage entre deux actifs
        
        Args:
            symbol1: Premier symbole
            symbol2: Deuxième symbole
            start_date: Date de début
            end_date: Date de fin
            zscore_threshold: Seuil de Z-score
            
        Returns:
            List: Liste des opportunités d'arbitrage
        """
        try:
            # Calcul du spread
            spread = self.calculate_spread(symbol1, symbol2, start_date, end_date)
            
            # Calcul du Z-score
            zscore = (spread - spread.mean()) / spread.std()
            
            opportunities = []
            for date, score in zscore.items():
                if abs(score) > zscore_threshold:
                    opportunities.append({
                        'date': date,
                        'zscore': score,
                        'spread': spread[date],
                        'action': 'buy' if score < -zscore_threshold else 'sell'
                    })
                    
            return sorted(opportunities, key=lambda x: abs(x['zscore']), reverse=True)
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la recherche des opportunités d'arbitrage: {str(e)}")
            raise
            
    def calculate_hedge_ratio(self,
                            symbol1: str,
                            symbol2: str,
                            start_date: datetime,
                            end_date: datetime) -> float:
        """
        Calcule le ratio de couverture optimal
        
        Args:
            symbol1: Premier symbole
            symbol2: Deuxième symbole
            start_date: Date de début
            end_date: Date de fin
            
        Returns:
            float: Ratio de couverture
        """
        try:
            # Récupération des données
            ticker1 = yf.Ticker(symbol1)
            ticker2 = yf.Ticker(symbol2)
            
            hist1 = ticker1.history(start=start_date, end=end_date)['Close']
            hist2 = ticker2.history(start=start_date, end=end_date)['Close']
            
            # Calcul du ratio de couverture
            returns1 = hist1.pct_change()
            returns2 = hist2.pct_change()
            
            covariance = returns1.cov(returns2)
            variance = returns2.var()
            
            hedge_ratio = covariance / variance
            
            return hedge_ratio
            
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul du ratio de couverture: {str(e)}")
            raise 