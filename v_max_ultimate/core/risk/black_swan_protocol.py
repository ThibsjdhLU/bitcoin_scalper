import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
from scipy import stats

class BlackSwanProtocol:
    """
    Protocole de gestion des événements extrêmes
    """
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.volatility_threshold = config.get('volatility_threshold', 3.0)
        self.price_change_threshold = config.get('price_change_threshold', 0.1)
        self.correlation_threshold = config.get('correlation_threshold', 0.8)
        self.position_limits = config.get('position_limits', {})
        self.circuit_breakers = {}
        
    def detect_extreme_events(self, 
                            returns: pd.DataFrame,
                            window: int = 20) -> Dict[str, bool]:
        """
        Détecte les événements extrêmes
        
        Args:
            returns: DataFrame des rendements
            window: Fenêtre de calcul
            
        Returns:
            Dict: Événements extrêmes détectés par actif
        """
        try:
            extreme_events = {}
            
            for asset in returns.columns:
                # Calcul de la volatilité
                volatility = returns[asset].rolling(window).std()
                current_vol = volatility.iloc[-1]
                
                # Calcul du changement de prix
                price_change = abs(returns[asset].iloc[-1])
                
                # Détection d'événements extrêmes
                is_extreme = (
                    current_vol > self.volatility_threshold or
                    price_change > self.price_change_threshold
                )
                
                extreme_events[asset] = is_extreme
                
            return extreme_events
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la détection d'événements extrêmes: {str(e)}")
            raise
            
    def calculate_stress_metrics(self, 
                               returns: pd.DataFrame,
                               positions: Dict) -> Dict:
        """
        Calcule les métriques de stress
        
        Args:
            returns: DataFrame des rendements
            positions: Positions actuelles
            
        Returns:
            Dict: Métriques de stress
        """
        try:
            metrics = {
                'var': {},
                'expected_shortfall': {},
                'correlation_stress': {},
                'position_stress': {}
            }
            
            for asset in returns.columns:
                # Value at Risk (VaR)
                var_95 = np.percentile(returns[asset], 5)
                metrics['var'][asset] = var_95
                
                # Expected Shortfall (ES)
                es_95 = returns[asset][returns[asset] <= var_95].mean()
                metrics['expected_shortfall'][asset] = es_95
                
                # Stress de corrélation
                correlations = returns.corr()[asset]
                stress_corr = correlations[correlations > self.correlation_threshold]
                metrics['correlation_stress'][asset] = len(stress_corr)
                
                # Stress de position
                if asset in positions:
                    position = positions[asset]
                    position_stress = abs(position['size']) * abs(returns[asset].iloc[-1])
                    metrics['position_stress'][asset] = position_stress
                    
            return metrics
            
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul des métriques de stress: {str(e)}")
            raise
            
    def activate_circuit_breaker(self, asset: str, reason: str):
        """
        Active un circuit breaker pour un actif
        
        Args:
            asset: Actif concerné
            reason: Raison de l'activation
        """
        try:
            self.circuit_breakers[asset] = {
                'active': True,
                'timestamp': datetime.now(),
                'reason': reason
            }
            
            self.logger.warning(f"Circuit breaker activé pour {asset}: {reason}")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'activation du circuit breaker: {str(e)}")
            raise
            
    def deactivate_circuit_breaker(self, asset: str):
        """
        Désactive un circuit breaker pour un actif
        
        Args:
            asset: Actif concerné
        """
        try:
            if asset in self.circuit_breakers:
                self.circuit_breakers[asset]['active'] = False
                self.logger.info(f"Circuit breaker désactivé pour {asset}")
                
        except Exception as e:
            self.logger.error(f"Erreur lors de la désactivation du circuit breaker: {str(e)}")
            raise
            
    def calculate_hedge_ratio(self, 
                            asset: str,
                            hedge_asset: str,
                            returns: pd.DataFrame,
                            window: int = 20) -> float:
        """
        Calcule le ratio de couverture optimal
        
        Args:
            asset: Actif à couvrir
            hedge_asset: Actif de couverture
            returns: DataFrame des rendements
            window: Fenêtre de calcul
            
        Returns:
            float: Ratio de couverture optimal
        """
        try:
            # Calcul de la covariance
            cov = returns[[asset, hedge_asset]].cov().iloc[0, 1]
            
            # Calcul de la variance de l'actif de couverture
            var_hedge = returns[hedge_asset].var()
            
            # Ratio de couverture optimal
            hedge_ratio = -cov / var_hedge
            
            return hedge_ratio
            
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul du ratio de couverture: {str(e)}")
            raise
            
    def generate_hedge_signals(self,
                             positions: Dict,
                             returns: pd.DataFrame,
                             stress_metrics: Dict) -> Dict:
        """
        Génère des signaux de couverture
        
        Args:
            positions: Positions actuelles
            returns: DataFrame des rendements
            stress_metrics: Métriques de stress
            
        Returns:
            Dict: Signaux de couverture par actif
        """
        try:
            hedge_signals = {}
            
            for asset, position in positions.items():
                if position['size'] != 0:  # Position ouverte
                    # Vérification des métriques de stress
                    var = stress_metrics['var'][asset]
                    es = stress_metrics['expected_shortfall'][asset]
                    position_stress = stress_metrics['position_stress'][asset]
                    
                    # Conditions pour la couverture
                    needs_hedge = (
                        var < -0.05 or  # VaR élevée
                        es < -0.1 or    # ES élevée
                        position_stress > self.position_limits.get(asset, 1.0)  # Stress de position élevé
                    )
                    
                    if needs_hedge:
                        # Recherche de l'actif de couverture optimal
                        correlations = returns.corr()[asset]
                        hedge_candidates = correlations[correlations < -0.7].index
                        
                        if len(hedge_candidates) > 0:
                            hedge_asset = hedge_candidates[0]
                            hedge_ratio = self.calculate_hedge_ratio(
                                asset, hedge_asset, returns
                            )
                            
                            hedge_signals[asset] = {
                                'hedge_asset': hedge_asset,
                                'hedge_ratio': hedge_ratio,
                                'reason': 'Stress metrics exceeded thresholds'
                            }
                            
            return hedge_signals
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération des signaux de couverture: {str(e)}")
            raise 