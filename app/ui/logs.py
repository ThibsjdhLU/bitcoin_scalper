import streamlit as st

def logs_console():
    """Affiche la console des logs."""
    st.subheader("Console des Logs")
    if 'log_messages' in st.session_state:
        for message in st.session_state.log_messages:
            st.write(f"🔹 {message}")
