"""
Deep Learning models for Bitcoin trading using PyTorch.

This submodule provides PyTorch-based models including:
- LSTM for sequential pattern learning
- Transformer for attention-based learning
- Hybrid architectures (Transformer-XGBoost)

All models implement the BaseModel interface for consistency.
"""

from src.bitcoin_scalper.models.deep_learning.torch_wrapper import TorchModelWrapper
from src.bitcoin_scalper.models.deep_learning.lstm import LSTMModel
from src.bitcoin_scalper.models.deep_learning.transformer import TransformerModel

__all__ = [
    'TorchModelWrapper',
    'LSTMModel',
    'TransformerModel'
]
