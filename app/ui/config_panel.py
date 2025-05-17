import streamlit as st

def config_panel():
    """Affiche le panneau de configuration."""
    st.sidebar.header("Configuration")
    # Ajoutez ici les éléments de configuration que vous souhaitez afficher
    st.sidebar.text_input("Paramètre 1", value="Valeur par défaut")
