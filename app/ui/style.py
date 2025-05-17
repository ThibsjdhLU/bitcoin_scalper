import streamlit as st

def apply_css():
    """Applique le style CSS personnalisé."""
    st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: white;  /* Texte en blanc pour le contraste */
    }
    .header-container {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .bot-status {
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    .refresh-info {
        font-size: 0.8rem;
        color: #AAAAAA;
    }
    .stButton>button:disabled {
        background-color: #FF5555 !important;
        color: white !important;
    }
    .stButton>button:enabled {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
