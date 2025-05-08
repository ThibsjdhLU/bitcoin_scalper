import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from pathlib import Path
import json
import random
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Dashboard:
    def __init__(self):
        logger.info("Initializing Dashboard")
        self.data_dir = Path("data")
        self.logs_dir = Path("logs")
        self.trades_df = None
        self.performance_df = None
        self.errors_df = None
        
        # Charger la configuration
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
                self.demo_mode = config['trading'].get('demo_mode', True)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            self.demo_mode = True  # Fallback sur le mode démo
        
    def load_data(self):
        """Charge les données depuis les fichiers CSV ou génère des données de démo."""
        try:
            logger.info("Loading data")
            if self.demo_mode:
                logger.info("Using demo mode")
                self._generate_demo_data()
            else:
                logger.info("Loading from files")
                self.trades_df = pd.read_csv(self.data_dir / "trades.csv")
                self.performance_df = pd.read_csv(self.data_dir / "performance.csv")
                self.load_error_logs()
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            st.error(f"Erreur lors du chargement des données: {str(e)}")
            self._generate_demo_data()  # Fallback sur les données de démo
            
    def _generate_demo_data(self):
        """Génère des données de démo pour le dashboard."""
        logger.info("Generating demo data")
        # Générer des données de trading
        dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
        trades_data = {
            'timestamp': dates,
            'symbol': ['BTCUSD'] * 100,
            'strategy': ['EMA'] * 50 + ['RSI'] * 50,
            'side': ['BUY', 'SELL'] * 50,
            'volume': [random.uniform(0.01, 1.0) for _ in range(100)],
            'price': [random.uniform(40000, 50000) for _ in range(100)],
            'profit': [random.uniform(-100, 100) for _ in range(100)]
        }
        self.trades_df = pd.DataFrame(trades_data)
        
        # Générer des données de performance
        balance = 10000
        performance_data = {
            'timestamp': dates,
            'balance': [balance + sum(random.uniform(-100, 100) for _ in range(i)) for i in range(100)],
            'equity': [balance + sum(random.uniform(-100, 100) for _ in range(i)) for i in range(100)]
        }
        self.performance_df = pd.DataFrame(performance_data)
        
        # Générer des logs d'erreurs
        error_types = ['Connection', 'Order', 'Data', 'System']
        errors = []
        for i in range(20):
            error = {
                'timestamp': (datetime.now() - timedelta(hours=i)).isoformat(),
                'error': f'{error_types[i % 4]} error',
                'details': f'Error details {i}'
            }
            errors.append(error)
        self.errors_df = pd.DataFrame(errors)
        logger.info("Demo data generated successfully")
            
    def load_error_logs(self):
        """Charge les logs d'erreurs."""
        logger.info("Loading error logs")
        error_files = list(self.logs_dir.glob("error_*.json"))
        errors = []
        for file in error_files:
            try:
                with open(file, 'r') as f:
                    errors.append(pd.read_json(f))
            except Exception as e:
                logger.warning(f"Error loading {file}: {str(e)}")
                st.warning(f"Erreur lors du chargement de {file}: {str(e)}")
        if errors:
            self.errors_df = pd.concat(errors, ignore_index=True)
            
    def display_metrics(self):
        """Affiche les métriques principales."""
        logger.info("Displaying metrics")
        if self.performance_df is not None:
            latest = self.performance_df.iloc[-1]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Balance", f"${latest['balance']:.2f}")
            with col2:
                st.metric("Équité", f"${latest['equity']:.2f}")
            with col3:
                daily_change = latest['equity'] - self.performance_df.iloc[-2]['equity']
                st.metric("Variation Journalière", f"${daily_change:.2f}")
                
    def display_performance_chart(self):
        """Affiche le graphique de performance."""
        logger.info("Displaying performance chart")
        if self.performance_df is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=self.performance_df['timestamp'],
                y=self.performance_df['equity'],
                name='Équité'
            ))
            fig.update_layout(
                title='Performance du Trading',
                xaxis_title='Date',
                yaxis_title='Équité ($)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
    def display_trades(self):
        """Affiche la table des trades récents."""
        logger.info("Displaying trades")
        if self.trades_df is not None:
            st.subheader("Trades Récents")
            recent_trades = self.trades_df.tail(10)
            st.dataframe(recent_trades)
            
    def display_errors(self):
        """Affiche les erreurs récentes."""
        logger.info("Displaying errors")
        if self.errors_df is not None:
            with st.expander("Logs d'Erreurs"):
                st.dataframe(self.errors_df)

def main():
    logger.info("Starting main function")
    st.set_page_config(
        page_title="Trading Bot Dashboard",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("Dashboard Trading Bot")
    
    dashboard = Dashboard()
    
    # Afficher un avertissement en mode démo
    if dashboard.demo_mode:
        st.warning("Mode Démo Actif - Les données affichées sont simulées")
    
    dashboard.load_data()
    
    dashboard.display_metrics()
    dashboard.display_performance_chart()
    
    col1, col2 = st.columns(2)
    with col1:
        dashboard.display_trades()
    with col2:
        dashboard.display_errors()
    
    logger.info("Dashboard rendered successfully")

if __name__ == "__main__":
    main() 