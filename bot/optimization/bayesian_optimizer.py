"""
Module d'optimisation bayésienne des hyperparamètres
"""

from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import optuna
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


class BayesianOptimizer:
    """
    Classe pour l'optimisation bayésienne des hyperparamètres utilisant Optuna.
    """

    def __init__(self, param_space: Dict[str, Tuple], n_trials: int = 100):
        """
        Initialise l'optimiseur bayésien.

        Args:
            param_space (Dict[str, Tuple]): Espace des paramètres à optimiser.
                Format: {'param_name': (min_value, max_value)}
            n_trials (int): Nombre d'essais d'optimisation.
        """
        self.param_space = param_space
        self.n_trials = n_trials
        self.study = None
        self.best_params = None
        self.optimization_history = []

    def objective(self, trial: optuna.Trial, objective_function: Callable) -> float:
        """
        Fonction objective pour Optuna.

        Args:
            trial: Instance d'essai Optuna
            objective_function: Fonction à optimiser

        Returns:
            float: Score d'optimisation
        """
        params = {}
        for param_name, (min_val, max_val) in self.param_space.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                params[param_name] = trial.suggest_int(param_name, min_val, max_val)
            else:
                params[param_name] = trial.suggest_float(param_name, min_val, max_val)

        return objective_function(params)

    def optimize(self, objective_function: Callable) -> Dict[str, Any]:
        """
        Exécute l'optimisation bayésienne.

        Args:
            objective_function: Fonction à optimiser

        Returns:
            Dict[str, Any]: Meilleurs paramètres trouvés
        """
        self.study = optuna.create_study(direction="maximize")

        objective = lambda trial: self.objective(trial, objective_function)
        self.study.optimize(objective, n_trials=self.n_trials)

        self.best_params = self.study.best_params
        self.optimization_history = self.study.trials

        return self.best_params

    def get_best_value(self) -> float:
        """
        Retourne la meilleure valeur trouvée.

        Returns:
            float: Meilleure valeur de la fonction objective
        """
        if self.study is None:
            raise ValueError("L'optimisation n'a pas encore été exécutée.")
        return self.study.best_value

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """
        Retourne l'historique d'optimisation.

        Returns:
            List[Dict[str, Any]]: Historique des essais d'optimisation
        """
        if self.study is None:
            raise ValueError("L'optimisation n'a pas encore été exécutée.")

        history = []
        for trial in self.optimization_history:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                history.append(
                    {
                        "params": trial.params,
                        "value": trial.value,
                        "datetime": trial.datetime,
                    }
                )
        return history
