import streamlit as st
import pandas as pd

def statistics():
    """Affiche les statistiques de trading."""
    st.subheader("Statistiques de Trading")
    # Exemple de données
    data = pd.DataFrame({
        'Date': ['2023-01-01', '2023-01-02'],
        'Profit': [100, 150]
    })
    st.line_chart(data.set_index('Date'))  # Graphique des profits

def trades_history():
    """Affiche l'historique des transactions."""
    st.subheader("Historique des Transactions")
    st.write("Historique des transactions ici.")
