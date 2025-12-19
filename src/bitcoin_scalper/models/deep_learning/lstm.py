"""
LSTM model for time series prediction in Bitcoin trading.

LSTM (Long Short-Term Memory) networks are well-suited for sequential data
and can capture temporal dependencies in market data.

Features:
- Multi-layer LSTM architecture
- Dropout for regularization
- Bidirectional option for better context
- Integration with TorchModelWrapper

References:
    Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory.
    Neural computation, 9(8), 1735-1780.
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    logger.warning("PyTorch not available for LSTM model")


class LSTMModel(nn.Module):
    """
    LSTM model for sequential prediction.
    
    This model processes sequential market data (prices, volumes, indicators)
    to predict future market direction or returns.
    
    Architecture:
        Input -> LSTM Layers (with dropout) -> Fully Connected -> Output
        
    Attributes:
        input_size: Number of input features
        hidden_size: Number of hidden units in LSTM
        num_layers: Number of LSTM layers
        output_size: Number of output units
        dropout: Dropout rate for regularization
        bidirectional: Whether to use bidirectional LSTM
        
    Example:
        >>> # For classification (3 classes: Long, Neutral, Short)
        >>> model = LSTMModel(
        ...     input_size=50,  # 50 features
        ...     hidden_size=128,
        ...     num_layers=2,
        ...     output_size=3,
        ...     dropout=0.2,
        ...     bidirectional=False
        ... )
        >>> 
        >>> # Wrap and train
        >>> wrapper = TorchModelWrapper(model, task_type='classification')
        >>> wrapper.train(X_train, y_train, epochs=50)
        
        >>> # For regression (predict returns)
        >>> model = LSTMModel(input_size=50, hidden_size=128, output_size=1)
        >>> wrapper = TorchModelWrapper(model, task_type='regression')
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 3,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        """
        Initialize LSTM model.
        
        Args:
            input_size: Number of input features per timestep
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of stacked LSTM layers
            output_size: Number of output units (classes or 1 for regression)
            dropout: Dropout rate between LSTM layers
            bidirectional: Whether to use bidirectional LSTM
            
        Notes:
            - For classification: output_size = number of classes
            - For regression: output_size = 1
            - Bidirectional doubles the hidden size
            - More layers = more capacity but slower training
        """
        super(LSTMModel, self).__init__()
        
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required but not installed")
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Calculate final hidden size (doubled if bidirectional)
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Fully connected layer
        self.fc = nn.Linear(lstm_output_size, output_size)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        logger.info(f"Initialized LSTM model: {input_size} -> "
                   f"{hidden_size}x{num_layers} -> {output_size}")
    
    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor. Shape (batch_size, sequence_length, input_size)
               For non-sequential data: (batch_size, input_size)
               
        Returns:
            Output tensor. Shape (batch_size, output_size)
            
        Notes:
            - Input can be 2D (batch, features) or 3D (batch, sequence, features)
            - If 2D, will be reshaped to 3D with sequence_length=1
            - LSTM processes the full sequence and we use the last output
        """
        # Handle 2D input (batch, features) -> (batch, 1, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # LSTM forward pass
        # lstm_out shape: (batch, seq, hidden_size * num_directions)
        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last output from the sequence
        # Shape: (batch, hidden_size * num_directions)
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout
        last_output = self.dropout_layer(last_output)
        
        # Fully connected layer
        # Shape: (batch, output_size)
        output = self.fc(last_output)
        
        return output
    
    def get_model_info(self) -> dict:
        """Get model architecture information."""
        return {
            'model_type': 'LSTM',
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_size': self.output_size,
            'dropout': self.dropout,
            'bidirectional': self.bidirectional,
            'total_parameters': sum(p.numel() for p in self.parameters())
        }


class BidirectionalLSTM(LSTMModel):
    """
    Bidirectional LSTM model.
    
    This is a convenience class that sets bidirectional=True by default.
    Bidirectional LSTMs process sequences in both forward and backward
    directions, which can capture more context.
    
    Note: Not suitable for real-time prediction where future data is unknown.
          Use only for offline training/analysis.
    
    Example:
        >>> model = BidirectionalLSTM(input_size=50, hidden_size=128)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 3,
        dropout: float = 0.2
    ):
        """Initialize Bidirectional LSTM."""
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout,
            bidirectional=True
        )


# Placeholder for future GRU implementation
class GRUModel(nn.Module):
    """
    GRU (Gated Recurrent Unit) model - PLACEHOLDER.
    
    GRU is similar to LSTM but with fewer parameters (faster training).
    To be implemented in the future if needed.
    
    Advantages over LSTM:
    - Faster training (fewer parameters)
    - Less prone to overfitting on small datasets
    - Similar performance to LSTM in many tasks
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError(
            "GRU model not yet implemented. Use LSTMModel for now."
        )
