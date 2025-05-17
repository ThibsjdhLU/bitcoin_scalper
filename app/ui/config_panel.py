import streamlit as st

def config_panel():
    """Affiche le panneau de configuration."""
    st.sidebar.header("Configuration")
    
    # Paramètres de trading
    st.sidebar.subheader("Paramètres de Trading")
    st.sidebar.text_input("Paramètre 1", value="Valeur par défaut")
    st.sidebar.slider("Intervalle de rafraîchissement (s)", min_value=1, max_value=60, value=10)
    
    # Autres options
    st.sidebar.subheader("Options supplémentaires")
    st.sidebar.checkbox("Activer les alertes critiques", value=True)
    st.sidebar.selectbox("Sélectionnez le thème", ["Clair", "Sombre"])
