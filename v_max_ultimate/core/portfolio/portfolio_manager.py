import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
from scipy.optimize import minimize

class PortfolioManager:
    """
    Gestionnaire de portefeuille
    """
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.positions = {}
        self.allocations = {}
        self.correlations = {}
        self.risk_metrics = {}
        
    def calculate_optimal_allocation(self, 
                                  returns: pd.DataFrame,
                                  risk_free_rate: float = 0.02) -> Dict:
        """
        Calcule l'allocation optimale selon Markowitz
        
        Args:
            returns: DataFrame des rendements
            risk_free_rate: Taux sans risque
            
        Returns:
            Dict: Allocations optimales
        """
        try:
            # Calcul de la matrice de covariance
            cov_matrix = returns.cov()
            
            # Calcul des rendements moyens
            mean_returns = returns.mean()
            
            # Fonction objectif (ratio de Sharpe)
            def objective(weights):
                portfolio_return = np.sum(mean_returns * weights)
                portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
                return -sharpe_ratio
                
            # Contraintes
            n_assets = len(returns.columns)
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Somme = 1
            )
            bounds = tuple((0, 1) for _ in range(n_assets))  # Poids entre 0 et 1
            
            # Optimisation
            result = minimize(
                objective,
                x0=np.array([1/n_assets] * n_assets),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            # Création du dictionnaire d'allocations
            allocations = {
                asset: weight for asset, weight in zip(returns.columns, result.x)
            }
            
            self.allocations = allocations
            return allocations
            
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul de l'allocation: {str(e)}")
            raise
            
    def calculate_correlations(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule la matrice de corrélation
        
        Args:
            returns: DataFrame des rendements
            
        Returns:
            pd.DataFrame: Matrice de corrélation
        """
        try:
            correlations = returns.corr()
            self.correlations = correlations
            return correlations
            
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul des corrélations: {str(e)}")
            raise
            
    def update_positions(self, current_prices: Dict[str, float]):
        """
        Met à jour les positions en fonction des prix actuels
        
        Args:
            current_prices: Prix actuels des actifs
        """
        try:
            for asset, price in current_prices.items():
                if asset in self.positions:
                    position = self.positions[asset]
                    position['current_value'] = position['size'] * price
                    position['unrealized_pnl'] = (
                        position['current_value'] - position['entry_value']
                    )
                    
            self.logger.info("Positions mises à jour avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la mise à jour des positions: {str(e)}")
            raise
            
    def calculate_portfolio_metrics(self) -> Dict:
        """
        Calcule les métriques du portefeuille
        
        Returns:
            Dict: Métriques du portefeuille
        """
        try:
            metrics = {
                'total_value': 0,
                'total_pnl': 0,
                'positions': {}
            }
            
            for asset, position in self.positions.items():
                metrics['total_value'] += position['current_value']
                metrics['total_pnl'] += position['unrealized_pnl']
                metrics['positions'][asset] = {
                    'size': position['size'],
                    'current_value': position['current_value'],
                    'unrealized_pnl': position['unrealized_pnl']
                }
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul des métriques: {str(e)}")
            raise
            
class MultiTimeframeManager:
    """
    Gestionnaire multi-timeframes
    """
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.timeframes = config.get('timeframes', ['1h', '4h', '1d'])
        self.weights = config.get('timeframe_weights', {'1h': 0.3, '4h': 0.3, '1d': 0.4})
        
    def combine_signals(self, signals: Dict[str, float]) -> float:
        """
        Combine les signaux de différents timeframes
        
        Args:
            signals: Signaux par timeframe
            
        Returns:
            float: Signal combiné
        """
        try:
            combined_signal = 0
            for timeframe, signal in signals.items():
                if timeframe in self.weights:
                    combined_signal += signal * self.weights[timeframe]
                    
            return combined_signal
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la combinaison des signaux: {str(e)}")
            raise
            
    def adjust_position_size(self, base_size: float, timeframe_signals: Dict[str, float]) -> float:
        """
        Ajuste la taille de position en fonction des signaux multi-timeframes
        
        Args:
            base_size: Taille de base
            timeframe_signals: Signaux par timeframe
            
        Returns:
            float: Taille ajustée
        """
        try:
            # Calcul du signal combiné
            combined_signal = self.combine_signals(timeframe_signals)
            
            # Ajustement de la taille
            adjusted_size = base_size * (1 + combined_signal)
            
            # Limites
            min_size = self.config.get('min_position_size', 0.1)
            max_size = self.config.get('max_position_size', 2.0)
            
            return np.clip(adjusted_size, min_size, max_size)
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'ajustement de la taille: {str(e)}")
            raise
            
class CrossAssetManager:
    """
    Gestionnaire cross-asset
    """
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.asset_groups = config.get('asset_groups', {})
        self.correlation_threshold = config.get('correlation_threshold', 0.7)
        
    def analyze_correlations(self, returns: pd.DataFrame) -> Dict:
        """
        Analyse les corrélations entre les actifs
        
        Args:
            returns: DataFrame des rendements
            
        Returns:
            Dict: Groupes d'actifs corrélés
        """
        try:
            # Calcul de la matrice de corrélation
            correlations = returns.corr()
            
            # Identification des groupes corrélés
            correlated_groups = {}
            processed = set()
            
            for asset in returns.columns:
                if asset in processed:
                    continue
                    
                # Recherche des actifs corrélés
                correlated = []
                for other_asset in returns.columns:
                    if other_asset not in processed and other_asset != asset:
                        if abs(correlations.loc[asset, other_asset]) > self.correlation_threshold:
                            correlated.append(other_asset)
                            processed.add(other_asset)
                            
                if correlated:
                    correlated_groups[asset] = correlated
                    processed.add(asset)
                    
            return correlated_groups
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse des corrélations: {str(e)}")
            raise
            
    def adjust_exposure(self, positions: Dict, correlations: pd.DataFrame) -> Dict:
        """
        Ajuste l'exposition en fonction des corrélations
        
        Args:
            positions: Positions actuelles
            correlations: Matrice de corrélation
            
        Returns:
            Dict: Positions ajustées
        """
        try:
            adjusted_positions = positions.copy()
            
            for asset, position in positions.items():
                # Calcul de l'exposition corrélée
                correlated_exposure = 0
                for other_asset, other_position in positions.items():
                    if other_asset != asset:
                        correlation = correlations.loc[asset, other_asset]
                        if abs(correlation) > self.correlation_threshold:
                            correlated_exposure += other_position['size'] * correlation
                            
                # Ajustement de la position
                if correlated_exposure > 0:
                    adjustment_factor = 1 / (1 + correlated_exposure)
                    adjusted_positions[asset]['size'] *= adjustment_factor
                    
            return adjusted_positions
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'ajustement de l'exposition: {str(e)}")
            raise 