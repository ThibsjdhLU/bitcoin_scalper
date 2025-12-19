"""
PyTorch model wrapper implementing BaseModel interface.

This module provides a generic wrapper for PyTorch models that handles:
- Training loops (epochs, batches, optimizer)
- Early stopping
- GPU/CPU management
- Model persistence
- Integration with BaseModel interface

The wrapper allows any torch.nn.Module to be used with the standard interface.
"""

from typing import Optional, Dict, Any, Tuple, Union
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import json

from src.bitcoin_scalper.models.base import BaseModel

logger = logging.getLogger(__name__)

# Optional import
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    logger.warning("PyTorch not available. Install with: pip install torch")


class TorchModelWrapper(BaseModel):
    """
    Wrapper for PyTorch models implementing BaseModel interface.
    
    This class handles the training loop, optimization, and persistence for
    any torch.nn.Module, allowing it to work seamlessly with the rest of the
    pipeline.
    
    Attributes:
        model: The underlying torch.nn.Module
        device: Device to run computations on (cuda/cpu)
        optimizer: Optimizer for training
        criterion: Loss function
        
    Example:
        >>> # Create a PyTorch model
        >>> lstm = LSTMModel(input_size=50, hidden_size=128, num_layers=2)
        >>> 
        >>> # Wrap it
        >>> model = TorchModelWrapper(
        ...     model=lstm,
        ...     task_type='classification',
        ...     learning_rate=0.001
        ... )
        >>> 
        >>> # Train using standard interface
        >>> model.train(
        ...     X_train, y_train,
        ...     eval_set=(X_val, y_val),
        ...     epochs=50,
        ...     batch_size=32
        ... )
        >>> 
        >>> # Predict
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(
        self,
        model: Optional['nn.Module'] = None,
        task_type: str = 'classification',
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        optimizer_type: str = 'adam',
        use_gpu: bool = True,
        **kwargs
    ):
        """
        Initialize PyTorch model wrapper.
        
        Args:
            model: PyTorch nn.Module to wrap. If None, must be set before training.
            task_type: 'classification' or 'regression'
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization strength
            optimizer_type: 'adam', 'sgd', or 'adamw'
            use_gpu: Whether to use GPU if available
            **kwargs: Additional parameters
        """
        super().__init__()
        
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required but not installed")
        
        self.model = model
        self.task_type = task_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type
        
        # Setup device
        self.device = self._setup_device(use_gpu)
        
        # Move model to device if provided
        if self.model is not None:
            self.model = self.model.to(self.device)
        
        # Will be initialized during training
        self.optimizer = None
        self.criterion = None
        
        logger.info(f"Initialized TorchModelWrapper (device: {self.device}, "
                   f"task: {task_type})")
    
    def _setup_device(self, use_gpu: bool) -> torch.device:
        """Setup computation device (GPU/CPU)."""
        if use_gpu and torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU")
        return device
    
    def _setup_optimizer(self) -> None:
        """Setup optimizer for training."""
        if self.optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")
    
    def _setup_criterion(self) -> None:
        """Setup loss function based on task type."""
        if self.task_type == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        elif self.task_type == 'regression':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
    
    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sample_weights: Optional[Union[pd.Series, np.ndarray]] = None,
        eval_set: Optional[Tuple[Union[pd.DataFrame, np.ndarray],
                                 Union[pd.Series, np.ndarray]]] = None,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
        verbose: bool = True,
        **kwargs
    ) -> 'TorchModelWrapper':
        """
        Train PyTorch model.
        
        Args:
            X: Training features
            y: Training labels
            sample_weights: Optional sample weights (Note: Not yet implemented for PyTorch
                          models. Will be ignored if provided.)
            eval_set: Optional (X_val, y_val) for early stopping
            epochs: Number of training epochs
            batch_size: Batch size for training
            early_stopping_patience: Stop if no improvement for N epochs
            verbose: Whether to print training progress
            **kwargs: Additional parameters
            
        Returns:
            Self for method chaining
            
        Note:
            Sample weights support for PyTorch models is planned for a future release.
        """
        if sample_weights is not None:
            logger.warning("Sample weights are not yet supported for PyTorch models and will be ignored")
        if self.model is None:
            raise ValueError("Model must be set before training")
        
        # Validate inputs
        self.validate_inputs(X, y)
        
        # Extract feature names
        self.feature_names = self._extract_feature_names(X)
        self.n_features = len(self.feature_names)
        
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y
        
        # For classification, map labels to indices
        if self.task_type == 'classification':
            self.classes_ = np.unique(y_array)
            label_map = {label: i for i, label in enumerate(self.classes_)}
            y_array = np.array([label_map[label] for label in y_array])
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_array).to(self.device)
        y_tensor = torch.LongTensor(y_array).to(self.device) if self.task_type == 'classification' \
                   else torch.FloatTensor(y_array).to(self.device)
        
        # Create data loader
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Prepare validation set
        val_loader = None
        if eval_set is not None:
            X_val, y_val = eval_set
            self.validate_inputs(X_val, y_val)
            
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if isinstance(y_val, pd.Series):
                y_val = y_val.values
            
            if self.task_type == 'classification':
                y_val = np.array([label_map[label] for label in y_val])
            
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.LongTensor(y_val).to(self.device) if self.task_type == 'classification' \
                          else torch.FloatTensor(y_val).to(self.device)
            
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup optimizer and criterion
        self._setup_optimizer()
        self._setup_criterion()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_X)
                
                # Compute loss
                if self.task_type == 'regression':
                    outputs = outputs.squeeze()
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X)
                        if self.task_type == 'regression':
                            outputs = outputs.squeeze()
                        loss = self.criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} - "
                              f"Train Loss: {train_loss:.4f}, "
                              f"Val Loss: {val_loss:.4f}")
                
                # Check early stopping
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")
        
        self.is_fitted = True
        logger.info("Training completed")
        
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict labels or values.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        self.validate_inputs(X)
        
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_array).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            
            if self.task_type == 'classification':
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                # Map back to original labels
                return self.classes_[predictions]
            else:
                return outputs.squeeze().cpu().numpy()
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities (classification only).
        
        Args:
            X: Features to predict on
            
        Returns:
            Class probabilities
        """
        if self.task_type != 'classification':
            raise NotImplementedError("predict_proba only available for classification")
        
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        self.validate_inputs(X)
        
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_array).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            proba = torch.softmax(outputs, dim=1).cpu().numpy()
        
        return proba
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save model to disk.
        
        Args:
            path: File path to save model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'classes': self.classes_.tolist() if hasattr(self, 'classes_') else None,
            'feature_names': self.feature_names,
            'n_features': self.n_features,
            'task_type': self.task_type
        }, str(path))
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: Union[str, Path]) -> 'TorchModelWrapper':
        """
        Load model from disk.
        
        Args:
            path: File path to load model from
            
        Returns:
            Self for method chaining
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        if self.model is None:
            raise ValueError("Model architecture must be provided before loading weights")
        
        # Load checkpoint
        checkpoint = torch.load(str(path), map_location=self.device)
        
        # Restore model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        
        # Restore metadata
        if checkpoint.get('classes') is not None:
            self.classes_ = np.array(checkpoint['classes'])
        self.feature_names = checkpoint.get('feature_names')
        self.n_features = checkpoint.get('n_features')
        self.task_type = checkpoint.get('task_type', 'classification')
        
        self.is_fitted = True
        logger.info(f"Model loaded from {path}")
        
        return self
