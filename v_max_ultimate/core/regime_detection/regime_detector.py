import numpy as np
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
import pandas as pd
from typing import Dict, List, Tuple
import logging

class MarketRegimeDetector:
    """
    Détecteur de régime de marché utilisant HMM et GMM
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.hmm_model = None
        self.gmm_model = None
        self.current_regime = None
        self.regime_history = []
        
    def initialize_models(self):
        """Initialise les modèles HMM et GMM"""
        try:
            # Initialisation du HMM
            n_states = self.config.get('hmm_states', 3)
            self.hmm_model = hmm.GaussianHMM(
                n_components=n_states,
                covariance_type="full",
                n_iter=1000,
                random_state=42
            )
            
            # Initialisation du GMM
            n_components = self.config.get('gmm_components', 3)
            self.gmm_model = GaussianMixture(
                n_components=n_components,
                covariance_type='full',
                random_state=42
            )
            
            self.logger.info("Models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            raise
            
    def detect_regime(self, data: pd.DataFrame) -> Dict:
        """
        Détecte le régime de marché actuel
        
        Args:
            data: DataFrame contenant les données de marché
            
        Returns:
            Dict: Informations sur le régime détecté
        """
        try:
            if self.hmm_model is None or self.gmm_model is None:
                self.initialize_models()
                
            # Préparation des features
            features = self._prepare_features(data)
            
            # Détection HMM
            hmm_regime = self.hmm_model.predict(features)[-1]
            hmm_prob = self.hmm_model.predict_proba(features)[-1]
            
            # Détection GMM
            gmm_regime = self.gmm_model.predict(features)[-1]
            gmm_prob = self.gmm_model.predict_proba(features)[-1]
            
            # Combinaison des résultats
            regime_info = {
                'hmm_regime': hmm_regime,
                'hmm_probability': hmm_prob[hmm_regime],
                'gmm_regime': gmm_regime,
                'gmm_probability': gmm_prob[gmm_regime],
                'regime_type': self._classify_regime(hmm_regime, gmm_regime),
                'confidence': np.mean([hmm_prob[hmm_regime], gmm_prob[gmm_regime]])
            }
            
            self.current_regime = regime_info
            self.regime_history.append(regime_info)
            
            return regime_info
            
        except Exception as e:
            self.logger.error(f"Error detecting regime: {str(e)}")
            raise
            
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prépare les features pour la détection de régime
        
        Args:
            data: DataFrame contenant les données brutes
            
        Returns:
            np.ndarray: Features préparées
        """
        # Calcul des rendements
        returns = data['close'].pct_change().dropna()
        
        # Calcul de la volatilité
        volatility = returns.rolling(window=20).std()
        
        # Calcul du momentum
        momentum = data['close'].pct_change(periods=20)
        
        # Calcul du volume relatif
        volume_ma = data['volume'].rolling(window=20).mean()
        relative_volume = data['volume'] / volume_ma
        
        # Combinaison des features
        features = np.column_stack([
            returns,
            volatility,
            momentum,
            relative_volume
        ])
        
        return features.dropna().values
        
    def _classify_regime(self, hmm_regime: int, gmm_regime: int) -> str:
        """
        Classifie le régime de marché
        
        Args:
            hmm_regime: Régime détecté par HMM
            gmm_regime: Régime détecté par GMM
            
        Returns:
            str: Type de régime
        """
        # Logique de classification basée sur les deux modèles
        if hmm_regime == gmm_regime:
            if hmm_regime == 0:
                return 'BULL'
            elif hmm_regime == 1:
                return 'BEAR'
            else:
                return 'VOLATILE'
        else:
            return 'MIXED'
            
    def get_regime_parameters(self) -> Dict:
        """
        Retourne les paramètres optimaux pour le régime actuel
        
        Returns:
            Dict: Paramètres adaptés au régime
        """
        if self.current_regime is None:
            return self.config.get('default_parameters', {})
            
        regime_type = self.current_regime['regime_type']
        
        # Paramètres spécifiques au régime
        parameters = {
            'BULL': {
                'atr_multiplier': 2.0,
                'breakout_threshold': 0.02,
                'position_size': 1.0
            },
            'BEAR': {
                'atr_multiplier': 2.5,
                'breakout_threshold': 0.03,
                'position_size': 0.7
            },
            'VOLATILE': {
                'atr_multiplier': 3.0,
                'breakout_threshold': 0.04,
                'position_size': 0.5
            },
            'MIXED': {
                'atr_multiplier': 2.2,
                'breakout_threshold': 0.025,
                'position_size': 0.8
            }
        }
        
        return parameters.get(regime_type, self.config.get('default_parameters', {})) 