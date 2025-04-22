from setuptools import setup, find_packages

setup(
    name="bitcoin_scalper",
    version="2.0.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        'scikit-learn>=0.24.2',
        'torch>=1.9.0',
        'hmmlearn>=0.2.7',
        'arch>=5.0.0',
        'statsmodels>=0.13.0',
        'optuna>=2.10.0',
        'python-binance>=1.0.15',
        'MetaTrader5>=5.0.34',
        'plotly>=5.3.0',
        'python-dotenv>=0.19.0',
    ],
    author="V-Max Ultimate",
    description="Bot de scalping Bitcoin avancé avec adversarial testing et apprentissage méta-continu",
    python_requires=">=3.8",
) 