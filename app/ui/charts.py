import streamlit as st
import pandas as pd

def price_chart():
    """Affiche le graphique des prix."""
    # Exemple de données
    data = pd.DataFrame({
        'Prix': [1, 2, 3, 4, 5],
        'Volume': [10, 20, 30, 40, 50]
    })
    st.line_chart(data['Prix'])  # Graphique des prix
    st.bar_chart(data['Volume'])  # Graphique des volumes
