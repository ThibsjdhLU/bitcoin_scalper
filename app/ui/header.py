import streamlit as st

def header():
    """Affiche l'en-tête de l'application."""
    st.image("path/to/logo.png", width=100)  # Ajout d'un logo
    st.title("Bitcoin Trading Bot")
    st.subheader("Suivi et contrôle du bot de trading")
    st.markdown("---")  # Ligne de séparation

def check_critical_alerts():
    """Vérifie et affiche les alertes critiques."""
    # Logique pour vérifier les alertes critiques
    pass
