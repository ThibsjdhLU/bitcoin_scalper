import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from pathlib import Path

class Dashboard:
    def __init__(self):
        self.data_dir = Path("data")
        self.logs_dir = Path("logs")
        self.trades_df = None
        self.performance_df = None
        self.errors_df = None
        
    def load_data(self):
        """Charge les données depuis les fichiers CSV."""
        try:
            self.trades_df = pd.read_csv(self.data_dir / "trades.csv")
            self.performance_df = pd.read_csv(self.data_dir / "performance.csv")
            self.load_error_logs()
        except Exception as e:
            st.error(f"Erreur lors du chargement des données: {str(e)}")
            
    def load_error_logs(self):
        """Charge les logs d'erreurs."""
        error_files = list(self.logs_dir.glob("error_*.json"))
        errors = []
        for file in error_files:
            try:
                with open(file, 'r') as f:
                    errors.append(pd.read_json(f))
            except Exception as e:
                st.warning(f"Erreur lors du chargement de {file}: {str(e)}")
        if errors:
            self.errors_df = pd.concat(errors, ignore_index=True)
            
    def display_metrics(self):
        """Affiche les métriques principales."""
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
        if self.trades_df is not None:
            st.subheader("Trades Récents")
            recent_trades = self.trades_df.tail(10)
            st.dataframe(recent_trades)
            
    def display_errors(self):
        """Affiche les erreurs récentes."""
        if self.errors_df is not None:
            with st.expander("Logs d'Erreurs"):
                st.dataframe(self.errors_df)

def main():
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

if __name__ == "__main__":
    main() 