import streamlit as st

def logs_console():
    """Affiche la console des logs."""
    if 'log_messages' in st.session_state:
        for message in st.session_state.log_messages:
            st.write(message)
