import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import logging
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm

class BayesianOptimizer:
    """
    Optimiseur bayésien pour les hyperparamètres
    """
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.X = []  # Points évalués
        self.y = []  # Valeurs objectives
        self.gp = None
        self.bounds = config.get('bounds', {})
        
    def initialize(self):
        """
        Initialise l'optimiseur bayésien
        """
        try:
            # Initialisation du GP
            kernel = Matern(nu=2.5)
            self.gp = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                n_restarts_optimizer=10,
                random_state=42
            )
            
            self.logger.info("Optimiseur bayésien initialisé avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation: {str(e)}")
            raise
            
    def suggest_next_point(self) -> Dict:
        """
        Suggère le prochain point à évaluer
        
        Returns:
            Dict: Hyperparamètres suggérés
        """
        try:
            if len(self.X) < 2:
                # Points initiaux aléatoires
                return self._random_point()
                
            # Conversion en arrays numpy
            X = np.array(self.X)
            y = np.array(self.y)
            
            # Entraînement du GP
            self.gp.fit(X, y)
            
            # Génération de points candidats
            n_candidates = 1000
            candidates = self._generate_candidates(n_candidates)
            
            # Calcul de l'acquisition (Expected Improvement)
            mu, sigma = self.gp.predict(candidates, return_std=True)
            best_f = np.max(y)
            
            with np.errstate(divide='warn'):
                imp = mu - best_f
                Z = imp / sigma
                ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
                ei[sigma == 0.0] = 0.0
                
            # Sélection du meilleur point
            best_idx = np.argmax(ei)
            next_point = candidates[best_idx]
            
            # Conversion en dictionnaire
            return self._array_to_dict(next_point)
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la suggestion: {str(e)}")
            raise
            
    def update(self, x: Dict, y: float):
        """
        Met à jour l'optimiseur avec un nouveau point évalué
        
        Args:
            x: Hyperparamètres évalués
            y: Valeur objective
        """
        try:
            # Conversion en array
            x_array = self._dict_to_array(x)
            
            # Ajout aux observations
            self.X.append(x_array)
            self.y.append(y)
            
            self.logger.info(f"Nouveau point ajouté: {x}, valeur: {y}")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la mise à jour: {str(e)}")
            raise
            
    def get_best_point(self) -> Tuple[Dict, float]:
        """
        Retourne le meilleur point trouvé
        
        Returns:
            Tuple[Dict, float]: Meilleurs hyperparamètres et valeur
        """
        try:
            if not self.X:
                raise ValueError("Aucun point n'a encore été évalué")
                
            best_idx = np.argmax(self.y)
            return self._array_to_dict(self.X[best_idx]), self.y[best_idx]
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération du meilleur point: {str(e)}")
            raise
            
    def _random_point(self) -> Dict:
        """
        Génère un point aléatoire dans les bounds
        
        Returns:
            Dict: Point aléatoire
        """
        point = {}
        for param, (low, high) in self.bounds.items():
            point[param] = np.random.uniform(low, high)
        return point
        
    def _generate_candidates(self, n_candidates: int) -> np.ndarray:
        """
        Génère des points candidats
        
        Args:
            n_candidates: Nombre de candidats
            
        Returns:
            np.ndarray: Points candidats
        """
        candidates = []
        for _ in range(n_candidates):
            point = self._random_point()
            candidates.append(self._dict_to_array(point))
        return np.array(candidates)
        
    def _dict_to_array(self, x: Dict) -> np.ndarray:
        """
        Convertit un dictionnaire en array
        
        Args:
            x: Dictionnaire de paramètres
            
        Returns:
            np.ndarray: Array de paramètres
        """
        return np.array([x[param] for param in self.bounds.keys()])
        
    def _array_to_dict(self, x: np.ndarray) -> Dict:
        """
        Convertit un array en dictionnaire
        
        Args:
            x: Array de paramètres
            
        Returns:
            Dict: Dictionnaire de paramètres
        """
        return {param: val for param, val in zip(self.bounds.keys(), x)}
        
class HyperparameterOptimizer:
    """
    Gestionnaire d'optimisation des hyperparamètres
    """
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.optimizer = BayesianOptimizer(config)
        
    def optimize(self, objective_func: Callable, n_iterations: int = 50) -> Dict:
        """
        Optimise les hyperparamètres
        
        Args:
            objective_func: Fonction objectif à optimiser
            n_iterations: Nombre d'itérations
            
        Returns:
            Dict: Meilleurs hyperparamètres
        """
        try:
            # Initialisation
            self.optimizer.initialize()
            
            # Boucle d'optimisation
            for i in range(n_iterations):
                # Suggestion du prochain point
                next_point = self.optimizer.suggest_next_point()
                
                # Évaluation
                value = objective_func(next_point)
                
                # Mise à jour
                self.optimizer.update(next_point, value)
                
                self.logger.info(f"Itération {i+1}/{n_iterations}, valeur: {value}")
                
            # Récupération du meilleur point
            best_point, best_value = self.optimizer.get_best_point()
            
            return {
                'best_params': best_point,
                'best_value': best_value
            }
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'optimisation: {str(e)}")
            raise 