"""
Deep Learning models for Bitcoin trading using PyTorch.

This submodule provides PyTorch-based models including:
- LSTM for sequential pattern learning
- Transformer for attention-based learning (placeholder)
- Hybrid architectures (Transformer-XGBoost) - planned

All models implement the BaseModel interface for consistency.

Note: TransformerModel is a placeholder and will raise NotImplementedError
      if instantiated. Use LSTMModel or XGBoostClassifier for now.
"""

from src.bitcoin_scalper.models.deep_learning.torch_wrapper import TorchModelWrapper
from src.bitcoin_scalper.models.deep_learning.lstm import LSTMModel

# TransformerModel is intentionally not imported here as it's a placeholder
# that raises NotImplementedError. Import directly if needed for type hints.

__all__ = [
    'TorchModelWrapper',
    'LSTMModel',
]
