import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

class HMMRegimeDetector:
    """
    Détecteur de régimes de marché utilisant un modèle HMM (Hidden Markov Model).
    """
    
    def __init__(self, n_regimes: int = 3, lookback_period: int = 100):
        """
        Initialise le détecteur de régimes.
        
        Args:
            n_regimes (int): Nombre de régimes à détecter
            lookback_period (int): Période de lookback pour l'analyse
        """
        self.n_regimes = n_regimes
        self.lookback_period = lookback_period
        self.model = hmm.GaussianHMM(n_components=n_regimes, covariance_type="full")
        self.scaler = StandardScaler()
        self.current_regime = None
        self.regime_probabilities = None
        
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prépare les caractéristiques pour l'analyse HMM.
        
        Args:
            data (pd.DataFrame): DataFrame avec les colonnes ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            np.ndarray: Matrice de caractéristiques normalisée
        """
        # Créer un DataFrame pour stocker toutes les caractéristiques
        features_df = pd.DataFrame(index=data.index)
        
        # Calculer les rendements
        features_df['returns'] = data['close'].pct_change()
        
        # Calculer la volatilité sur une fenêtre glissante
        features_df['volatility'] = features_df['returns'].rolling(window=20).std()
        
        # Calculer le volume relatif
        features_df['relative_volume'] = data['volume'].pct_change()
        
        # Supprimer les lignes avec des valeurs NaN
        features_df = features_df.dropna()
        
        # Normaliser les caractéristiques
        scaler = StandardScaler()
        features = scaler.fit_transform(features_df)
        
        return features
        
    def fit(self, data: pd.DataFrame):
        """
        Entraîne le modèle HMM sur les données historiques.
        
        Args:
            data (pd.DataFrame): Données historiques
        """
        features = self._prepare_features(data)
        self.model.fit(features)
        
    def detect_regime(self, data: pd.DataFrame = None) -> int:
        """
        Détecte le régime de marché actuel.
        
        Args:
            data (pd.DataFrame): Données récentes du marché
            
        Returns:
            int: Indice du régime détecté
        """
        if data is None:
            return self.current_regime if self.current_regime is not None else 0
            
        features = self._prepare_features(data)
        
        # Prédiction du régime
        self.current_regime = self.model.predict(features)[-1]
        self.regime_probabilities = self.model.predict_proba(features)[-1]
        
        return self.current_regime
        
    def get_regime_probabilities(self) -> np.ndarray:
        """
        Retourne les probabilités de chaque régime.
        
        Returns:
            np.ndarray: Probabilités des régimes
        """
        return self.regime_probabilities if self.regime_probabilities is not None else np.zeros(self.n_regimes) 