import streamlit as st

def refresh_controls():
    """Affiche les contrôles de rafraîchissement et d'exécution."""
    if st.button("Rafraîchir"):
        # Logique de rafraîchissement
        pass

    if st.button("Démarrer le bot"):
        # Logique pour démarrer le bot
        st.success("Bot démarré avec succès!")
    elif st.button("Arrêter le bot"):
        # Logique pour arrêter le bot
        st.warning("Bot arrêté.")

def symbol_selector():
    """Permet de sélectionner un symbole."""
    symbols = ["BTCUSD", "ETHUSD", "LTCUSD"]  # Ajout d'un symbole supplémentaire
    selected_symbol = st.selectbox("Sélectionnez un symbole", symbols)
    st.session_state.selected_symbol = selected_symbol
