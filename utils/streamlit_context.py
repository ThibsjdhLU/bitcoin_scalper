import threading
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx


def safe_streamlit_call(func):
    """Wrapper pour exécuter des calls Streamlit hors du main thread"""
    def wrapper(*args, **kwargs):
        if threading.current_thread() != threading.main_thread():
            with st.spinner("Processing..."):
                return st.runtime.scriptrunner.add_script_run_ctx(func)(*args, **kwargs)
        return func(*args, **kwargs)
    return wrapper

# Si vous avez besoin d'une fonction safe_mode, définissez-la ici
def safe_mode():
    # Implémentez la logique de votre mode sécurisé ici
    pass
