import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
import numpy as np
import logging
from copy import deepcopy

class MetaLearner:
    """
    Meta-learning framework utilisant MAML et Reptile
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.meta_optimizer = None
        self.inner_optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def initialize_model(self):
        """Initialise le modèle de base et les optimiseurs"""
        try:
            # Architecture du modèle
            self.model = nn.Sequential(
                nn.Linear(self.config['input_size'], 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.config['output_size'])
            ).to(self.device)
            
            # Optimiseurs
            self.meta_optimizer = optim.Adam(self.model.parameters(), lr=self.config['meta_lr'])
            self.inner_optimizer = optim.SGD(self.model.parameters(), lr=self.config['inner_lr'])
            
            self.logger.info("Model initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing model: {str(e)}")
            raise
            
    def maml_update(self, tasks: List[Dict]) -> Dict:
        """
        Met à jour le modèle en utilisant MAML
        
        Args:
            tasks: Liste des tâches d'apprentissage
            
        Returns:
            Dict: Métriques de performance
        """
        try:
            if self.model is None:
                self.initialize_model()
                
            meta_loss = 0
            for task in tasks:
                # Copie du modèle pour la tâche
                task_model = deepcopy(self.model)
                task_optimizer = optim.SGD(task_model.parameters(), lr=self.config['inner_lr'])
                
                # Adaptation interne
                for _ in range(self.config['inner_steps']):
                    task_optimizer.zero_grad()
                    loss = self._compute_loss(task_model, task['train_data'])
                    loss.backward()
                    task_optimizer.step()
                    
                # Évaluation sur les données de test
                with torch.no_grad():
                    test_loss = self._compute_loss(task_model, task['test_data'])
                    meta_loss += test_loss
                    
            # Mise à jour meta
            meta_loss /= len(tasks)
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_optimizer.step()
            
            return {'meta_loss': meta_loss.item()}
            
        except Exception as e:
            self.logger.error(f"Error in MAML update: {str(e)}")
            raise
            
    def reptile_update(self, tasks: List[Dict]) -> Dict:
        """
        Met à jour le modèle en utilisant Reptile
        
        Args:
            tasks: Liste des tâches d'apprentissage
            
        Returns:
            Dict: Métriques de performance
        """
        try:
            if self.model is None:
                self.initialize_model()
                
            meta_loss = 0
            for task in tasks:
                # Copie du modèle pour la tâche
                task_model = deepcopy(self.model)
                task_optimizer = optim.SGD(task_model.parameters(), lr=self.config['inner_lr'])
                
                # Adaptation interne
                for _ in range(self.config['inner_steps']):
                    task_optimizer.zero_grad()
                    loss = self._compute_loss(task_model, task['train_data'])
                    loss.backward()
                    task_optimizer.step()
                    
                # Mise à jour Reptile
                for p, p_task in zip(self.model.parameters(), task_model.parameters()):
                    p.data.add_(p_task.data - p.data, alpha=self.config['reptile_alpha'])
                    
                meta_loss += loss.item()
                
            meta_loss /= len(tasks)
            return {'meta_loss': meta_loss}
            
        except Exception as e:
            self.logger.error(f"Error in Reptile update: {str(e)}")
            raise
            
    def _compute_loss(self, model: nn.Module, data: Dict) -> torch.Tensor:
        """
        Calcule la perte pour un modèle donné
        
        Args:
            model: Modèle à évaluer
            data: Données d'entraînement/test
            
        Returns:
            torch.Tensor: Valeur de la perte
        """
        x = torch.FloatTensor(data['x']).to(self.device)
        y = torch.FloatTensor(data['y']).to(self.device)
        
        predictions = model(x)
        loss = nn.MSELoss()(predictions, y)
        
        return loss
        
    def adapt_to_new_task(self, task_data: Dict) -> nn.Module:
        """
        Adapte le modèle à une nouvelle tâche
        
        Args:
            task_data: Données de la nouvelle tâche
            
        Returns:
            nn.Module: Modèle adapté
        """
        try:
            if self.model is None:
                self.initialize_model()
                
            # Copie du modèle
            adapted_model = deepcopy(self.model)
            optimizer = optim.SGD(adapted_model.parameters(), lr=self.config['inner_lr'])
            
            # Adaptation
            for _ in range(self.config['adaptation_steps']):
                optimizer.zero_grad()
                loss = self._compute_loss(adapted_model, task_data)
                loss.backward()
                optimizer.step()
                
            return adapted_model
            
        except Exception as e:
            self.logger.error(f"Error adapting to new task: {str(e)}")
            raise
            
    def predict(self, model: nn.Module, data: np.ndarray) -> np.ndarray:
        """
        Fait des prédictions avec un modèle
        
        Args:
            model: Modèle à utiliser
            data: Données d'entrée
            
        Returns:
            np.ndarray: Prédictions
        """
        try:
            x = torch.FloatTensor(data).to(self.device)
            with torch.no_grad():
                predictions = model(x)
            return predictions.cpu().numpy()
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise 