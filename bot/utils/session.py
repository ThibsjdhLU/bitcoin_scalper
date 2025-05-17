import streamlit as st

def get_session_state(key, default=None):
    """Récupère une valeur de session_state avec une valeur par défaut."""
    return st.session_state.get(key, default)

def set_session_state(key, value):
    """Définit une valeur dans session_state."""
    st.session_state[key] = value
