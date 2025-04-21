import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

class LSTMPredictor(nn.Module):
    """
    Modèle LSTM pour la prédiction de séries temporelles
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Paramètres du modèle
        self.input_size = config.get('input_size', 5)  # OHLCV
        self.hidden_size = config.get('hidden_size', 64)
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.2)
        
        # Architecture LSTM
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Couche de prédiction
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass du modèle
        
        Args:
            x: Tenseur d'entrée de forme (batch_size, seq_length, input_size)
            
        Returns:
            torch.Tensor: Prédictions
        """
        # Passage LSTM
        lstm_out, _ = self.lstm(x)
        
        # On prend la dernière sortie
        last_output = lstm_out[:, -1, :]
        
        # Prédiction finale
        predictions = self.fc(last_output)
        
        return predictions
        
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Fait des prédictions sur de nouvelles données
        
        Args:
            data: Données de forme (seq_length, input_size)
            
        Returns:
            np.ndarray: Prédictions
        """
        self.eval()
        with torch.no_grad():
            # Préparation des données
            x = torch.FloatTensor(data).unsqueeze(0)
            
            # Prédiction
            predictions = self.forward(x)
            
        return predictions.numpy()

class TransformerPredictor(nn.Module):
    """
    Modèle Transformer pour la prédiction de séries temporelles
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Paramètres du modèle
        self.input_size = config.get('input_size', 5)
        self.d_model = config.get('d_model', 64)
        self.nhead = config.get('nhead', 4)
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.1)
        
        # Couche d'embedding
        self.embedding = nn.Linear(self.input_size, self.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout)
        
        # Couches Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dropout=self.dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        # Couche de prédiction
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass du modèle
        
        Args:
            x: Tenseur d'entrée de forme (batch_size, seq_length, input_size)
            
        Returns:
            torch.Tensor: Prédictions
        """
        # Embedding
        x = self.embedding(x)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer
        x = x.permute(1, 0, 2)  # (seq_length, batch_size, d_model)
        transformer_out = self.transformer_encoder(x)
        
        # On prend la dernière sortie
        last_output = transformer_out[-1]
        
        # Prédiction finale
        predictions = self.fc(last_output)
        
        return predictions
        
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Fait des prédictions sur de nouvelles données
        
        Args:
            data: Données de forme (seq_length, input_size)
            
        Returns:
            np.ndarray: Prédictions
        """
        self.eval()
        with torch.no_grad():
            # Préparation des données
            x = torch.FloatTensor(data).unsqueeze(0)
            
            # Prédiction
            predictions = self.forward(x)
            
        return predictions.numpy()

class PositionalEncoding(nn.Module):
    """
    Encodage positionnel pour les Transformers
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ajoute l'encodage positionnel aux embeddings
        
        Args:
            x: Tenseur d'entrée de forme (batch_size, seq_length, d_model)
            
        Returns:
            torch.Tensor: Tenseur avec encodage positionnel
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class DeepLearningManager:
    """
    Gestionnaire des modèles de deep learning
    """
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def initialize_models(self):
        """
        Initialise les modèles de deep learning
        """
        try:
            # Initialisation du LSTM
            self.models['lstm'] = LSTMPredictor(self.config).to(self.device)
            
            # Initialisation du Transformer
            self.models['transformer'] = TransformerPredictor(self.config).to(self.device)
            
            self.logger.info("Modèles de deep learning initialisés avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation des modèles: {str(e)}")
            raise
            
    def train_model(self, model_name: str, train_data: torch.Tensor, 
                   train_labels: torch.Tensor, epochs: int = 100) -> Dict:
        """
        Entraîne un modèle spécifique
        
        Args:
            model_name: Nom du modèle ('lstm' ou 'transformer')
            train_data: Données d'entraînement
            train_labels: Labels d'entraînement
            epochs: Nombre d'époques
            
        Returns:
            Dict: Métriques d'entraînement
        """
        try:
            model = self.models[model_name]
            optimizer = torch.optim.Adam(model.parameters())
            criterion = nn.MSELoss()
            
            metrics = {
                'loss': [],
                'val_loss': []
            }
            
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                
                # Forward pass
                predictions = model(train_data)
                loss = criterion(predictions, train_labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Métriques
                metrics['loss'].append(loss.item())
                
                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
                    
            return metrics
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'entraînement: {str(e)}")
            raise
            
    def predict(self, model_name: str, data: np.ndarray) -> np.ndarray:
        """
        Fait des prédictions avec un modèle spécifique
        
        Args:
            model_name: Nom du modèle ('lstm' ou 'transformer')
            data: Données pour la prédiction
            
        Returns:
            np.ndarray: Prédictions
        """
        try:
            model = self.models[model_name]
            return model.predict(data)
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la prédiction: {str(e)}")
            raise 