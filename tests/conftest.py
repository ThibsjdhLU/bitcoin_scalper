"""
Configuration des tests pour pytest.
"""
import pytest
import sys
from pathlib import Path

# Ajouter le répertoire racine au PYTHONPATH
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir)) 