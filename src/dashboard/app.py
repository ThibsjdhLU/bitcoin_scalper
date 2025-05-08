import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from pathlib import Path
import json
import logging
from src.exchange.avatrader_mt5 import AvatraderMT5

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
                self.config = json.load(f)
                self.demo_mode = self.config['trading'].get('demo_mode', False)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            self.demo_mode = False
        
        # Initialiser la connexion MT5
        try:
            self.mt5 = AvatraderMT5(
                login=self.config['exchange']['login'],
                password=self.config['exchange']['password'],
                server=self.config['exchange']['server']
            )
        except Exception as e:
            logger.error(f"Error initializing MT5: {str(e)}")
            st.error(f"Erreur de connexion à MT5: {str(e)}")
        
    def load_data(self):
        """Charge les données depuis MT5."""
        try:
            logger.info("Loading data from MT5")
            
            # Récupérer les positions ouvertes
            positions = self.mt5.get_positions()
            if positions:
                self.trades_df = pd.DataFrame(positions)
            
            # Récupérer les informations du compte
            account_info = self.mt5.get_account_info()
            if account_info:
                self.performance_df = pd.DataFrame([{
                    'timestamp': datetime.now(),
                    'balance': account_info['balance'],
                    'equity': account_info['equity'],
                    'profit': account_info['profit']
                }])
            
            # Charger les logs d'erreurs
            self.load_error_logs()
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            st.error(f"Erreur lors du chargement des données: {str(e)}")
            
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
                st.metric("Profit", f"${latest['profit']:.2f}")
                
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
            st.subheader("Positions Ouvertes")
            st.dataframe(self.trades_df)
            
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