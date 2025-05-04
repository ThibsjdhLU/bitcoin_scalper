"""
Exemple d'utilisation du moteur de backtest.
"""
import json
import pandas as pd
from datetime import datetime
from loguru import logger

from backtest.backtest_engine import BacktestEngine
from strategies.ema_crossover import EMACrossoverStrategy
from strategies.rsi_strategy import RSIStrategy
from strategies.macd_strategy import MACDStrategy
from strategies.bollinger_bands_reversal import BollingerBandsReversalStrategy
from core.data_fetcher import DataFetcher, TimeFrame

def load_config() -> dict:
    """Charge la configuration du backtest."""
    with open('config/backtest_config.json', 'r') as f:
        return json.load(f)

def load_data(config: dict) -> dict:
    """
    Charge les données pour le backtest.
    
    Args:
        config: Configuration du backtest
        
    Returns:
        dict: Données OHLCV par symbole
    """
    data_fetcher = DataFetcher()
    data = {}
    
    for symbol in config['data']['symbols']:
        for timeframe in config['data']['timeframes']:
            # Convertir le timeframe en TimeFrame
            tf = TimeFrame[timeframe.upper()]
            
            # Récupérer les données
            symbol_data = data_fetcher.get_historical_data(
                symbol=symbol,
                timeframe=tf,
                start_date=datetime.strptime(config['data']['start_date'], '%Y-%m-%d'),
                end_date=datetime.strptime(config['data']['end_date'], '%Y-%m-%d')
            )
            
            if symbol_data is not None:
                data[f"{symbol}_{timeframe}"] = symbol_data
                
    return data

def create_strategies(config: dict) -> list:
    """
    Crée les stratégies à tester.
    
    Args:
        config: Configuration du backtest
        
    Returns:
        list: Liste des stratégies
    """
    strategies = []
    
    # EMA Crossover
    if config['strategies']['ema_crossover']['enabled']:
        strategies.append(
            EMACrossoverStrategy(
                data_fetcher=None,  # Non nécessaire pour le backtest
                order_executor=None,  # Non nécessaire pour le backtest
                symbols=[],
                timeframe=None,
                params=config['strategies']['ema_crossover']['params']
            )
        )
        
    # RSI
    if config['strategies']['rsi']['enabled']:
        strategies.append(
            RSIStrategy(
                data_fetcher=None,
                order_executor=None,
                symbols=[],
                timeframe=None,
                params=config['strategies']['rsi']['params']
            )
        )
        
    # MACD
    if config['strategies']['macd']['enabled']:
        strategies.append(
            MACDStrategy(
                data_fetcher=None,
                order_executor=None,
                symbols=[],
                timeframe=None,
                params=config['strategies']['macd']['params']
            )
        )
        
    # Bollinger Bands
    if config['strategies']['bollinger_bands']['enabled']:
        strategies.append(
            BollingerBandsReversalStrategy(
                data_fetcher=None,
                order_executor=None,
                symbols=[],
                timeframe=None,
                params=config['strategies']['bollinger_bands']['params']
            )
        )
        
    return strategies

def main():
    """Point d'entrée principal."""
    # Charger la configuration
    config = load_config()
    
    # Charger les données
    logger.info("Chargement des données...")
    data = load_data(config)
    
    if not data:
        logger.error("Aucune donnée trouvée")
        return
        
    # Créer les stratégies
    logger.info("Création des stratégies...")
    strategies = create_strategies(config)
    
    if not strategies:
        logger.error("Aucune stratégie activée")
        return
        
    # Créer le moteur de backtest
    engine = BacktestEngine(
        data=data,
        strategies=strategies,
        initial_capital=config['general']['initial_capital'],
        commission=config['general']['commission'],
        slippage=config['general']['slippage']
    )
    
    # Exécuter le backtest
    logger.info("Démarrage du backtest...")
    results = engine.run()
    
    # Afficher les résultats
    logger.info("\nRésultats du backtest:")
    logger.info(f"Nombre total de trades: {results['metrics']['total_trades']}")
    logger.info(f"Win rate: {results['metrics']['win_rate']:.2%}")
    logger.info(f"P&L total: {results['metrics']['total_pnl']:.2f}")
    logger.info(f"Ratio de Sharpe: {results['metrics']['sharpe_ratio']:.2f}")
    logger.info(f"Drawdown maximum: {results['metrics']['max_drawdown']:.2%}")
    
    # Sauvegarder les résultats
    if config['output']['save_trades']:
        engine.save_results(f"backtest_results/backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
    # Afficher les graphiques
    if config['output']['plot_results']:
        engine.plot_results()

if __name__ == '__main__':
    main() 