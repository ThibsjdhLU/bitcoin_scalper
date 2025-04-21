import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.base import BaseEstimator
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.model_selection import cross_val_score
from ..deep_learning.models import LSTMPredictor, TransformerPredictor

class EnsembleManager:
    """
    Gestionnaire des méthodes d'ensemble
    """
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.ensemble = None
        
    def initialize_models(self):
        """
        Initialise les modèles individuels
        """
        try:
            # Initialisation des modèles de base
            self.models['lstm'] = LSTMPredictor(self.config)
            self.models['transformer'] = TransformerPredictor(self.config)
            
            # Configuration du voting
            estimators = [
                ('lstm', self.models['lstm']),
                ('transformer', self.models['transformer'])
            ]
            
            # Création du voting classifier
            self.ensemble = VotingClassifier(
                estimators=estimators,
                voting='soft',
                weights=[1, 1]
            )
            
            self.logger.info("Modèles d'ensemble initialisés avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation des modèles: {str(e)}")
            raise
            
    def train_ensemble(self, train_data: np.ndarray, train_labels: np.ndarray) -> Dict:
        """
        Entraîne l'ensemble de modèles
        
        Args:
            train_data: Données d'entraînement
            train_labels: Labels d'entraînement
            
        Returns:
            Dict: Métriques d'entraînement
        """
        try:
            metrics = {}
            
            # Entraînement des modèles individuels
            for name, model in self.models.items():
                model_metrics = model.train(train_data, train_labels)
                metrics[name] = model_metrics
                
            # Entraînement de l'ensemble
            self.ensemble.fit(train_data, train_labels)
            
            # Évaluation de l'ensemble
            ensemble_score = cross_val_score(
                self.ensemble,
                train_data,
                train_labels,
                cv=5
            ).mean()
            
            metrics['ensemble'] = {
                'cv_score': ensemble_score
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'entraînement: {str(e)}")
            raise
            
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Fait des prédictions avec l'ensemble
        
        Args:
            data: Données pour la prédiction
            
        Returns:
            np.ndarray: Prédictions
        """
        try:
            return self.ensemble.predict(data)
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la prédiction: {str(e)}")
            raise
            
class StackingEnsemble:
    """
    Ensemble utilisant le stacking
    """
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.base_models = {}
        self.meta_model = None
        
    def initialize_models(self):
        """
        Initialise les modèles de base et le meta-modèle
        """
        try:
            # Modèles de base
            self.base_models['lstm'] = LSTMPredictor(self.config)
            self.base_models['transformer'] = TransformerPredictor(self.config)
            
            # Meta-modèle (LSTM)
            self.meta_model = LSTMPredictor(self.config)
            
            self.logger.info("Modèles de stacking initialisés avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation des modèles: {str(e)}")
            raise
            
    def train(self, train_data: np.ndarray, train_labels: np.ndarray) -> Dict:
        """
        Entraîne le modèle de stacking
        
        Args:
            train_data: Données d'entraînement
            train_labels: Labels d'entraînement
            
        Returns:
            Dict: Métriques d'entraînement
        """
        try:
            metrics = {}
            
            # Entraînement des modèles de base
            base_predictions = {}
            for name, model in self.base_models.items():
                model_metrics = model.train(train_data, train_labels)
                metrics[name] = model_metrics
                
                # Prédictions pour le meta-modèle
                base_predictions[name] = model.predict(train_data)
                
            # Combinaison des prédictions
            meta_features = np.column_stack(list(base_predictions.values()))
            
            # Entraînement du meta-modèle
            meta_metrics = self.meta_model.train(meta_features, train_labels)
            metrics['meta_model'] = meta_metrics
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'entraînement: {str(e)}")
            raise
            
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Fait des prédictions avec le modèle de stacking
        
        Args:
            data: Données pour la prédiction
            
        Returns:
            np.ndarray: Prédictions
        """
        try:
            # Prédictions des modèles de base
            base_predictions = {}
            for name, model in self.base_models.items():
                base_predictions[name] = model.predict(data)
                
            # Combinaison des prédictions
            meta_features = np.column_stack(list(base_predictions.values()))
            
            # Prédiction finale
            return self.meta_model.predict(meta_features)
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la prédiction: {str(e)}")
            raise 