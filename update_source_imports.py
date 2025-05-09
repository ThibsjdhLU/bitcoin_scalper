"""
Script to update imports in source files to use relative imports.
"""

import os
import re

def update_imports(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Mapping of old imports to new imports
    import_mapping = {
        'from core.': 'from ..core.',
        'from utils.': 'from ..utils.',
        'from backtest.': 'from ..backtest.',
        'from strategies.': 'from ..strategies.',
        'from services.': 'from ..services.',
        'from analysis.': 'from ..analysis.',
        'from dashboard.': 'from ..dashboard.',
        'from ui.': 'from ..ui.',
    }
    
    # Replace imports
    for old, new in import_mapping.items():
        content = content.replace(old, new)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    # Update all source files
    src_dir = 'src/bitcoin_scalper'
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                print(f"Updating imports in {file_path}")
                update_imports(file_path)

if __name__ == '__main__':
    main() 