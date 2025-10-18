# config.py
# Чете secrets от Streamlit (при деплой) или от environment (локално)
import os

def get_openai_key():
    try:
        import streamlit as st
        # първо опитай st.secrets (Streamlit Cloud)
        key = st.secrets.get("OPENAI_KEY")
        if key:
            return key
    except Exception:
        # ако Streamlit не е наличен (напр. unit tests), продължаваме към env
        pass
    # fallback към environment variable
    return os.getenv("OPENAI_KEY")
