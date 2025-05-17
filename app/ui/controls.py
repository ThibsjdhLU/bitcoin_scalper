import streamlit as st

def refresh_controls():
    """Affiche les contrôles de rafraîchissement."""
    if st.button("Rafraîchir"):
        # Logique de rafraîchissement
        pass

def symbol_selector():
    """Permet de sélectionner un symbole."""
    symbols = ["BTCUSD", "ETHUSD"]  # Exemple de symboles
    selected_symbol = st.selectbox("Sélectionnez un symbole", symbols)
    st.session_state.selected_symbol = selected_symbol
