"""
Script to update imports in test files to use src.bitcoin_scalper package.
"""

import os
import re

def update_imports(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Mapping of old imports to new imports
    import_mapping = {
        'from core.': 'from src.bitcoin_scalper.core.',
        'from utils.': 'from src.bitcoin_scalper.utils.',
        'from backtest.': 'from src.bitcoin_scalper.backtest.',
        'from strategies.': 'from src.bitcoin_scalper.strategies.',
        'from services.': 'from src.bitcoin_scalper.services.',
        'from analysis.': 'from src.bitcoin_scalper.analysis.',
        'from dashboard.': 'from src.bitcoin_scalper.dashboard.',
        'from ui.': 'from src.bitcoin_scalper.ui.',
    }
    
    # Replace imports
    for old, new in import_mapping.items():
        content = content.replace(old, new)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    # Update all test files
    test_dir = 'tests'
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                print(f"Updating imports in {file_path}")
                update_imports(file_path)

if __name__ == '__main__':
    main() 