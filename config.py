# config.py
# Чете OPENAI ключове от Streamlit secrets (напр. Streamlit Cloud) с fallback,
# или от environment variables за локална разработка.
import os

def get_openai_key():
    """
    Последователност за взимане на ключ:
      1) st.secrets['OPENAI_KEY_PRIMARY']
      2) st.secrets['OPENAI_KEY_FALLBACK']
      3) ENV OPENAI_KEY_PRIMARY
      4) ENV OPENAI_KEY_FALLBACK
      5) ENV OPENAI_KEY (legacy)
    Връща None ако не е намерен.
    """
    # Опитай st.secrets (Streamlit Cloud)
    try:
        import streamlit as st
        # предпочитаме primary, след това fallback
        key = st.secrets.get("OPENAI_KEY_PRIMARY") or st.secrets.get("OPENAI_KEY_FALLBACK")
        if key:
            return key
    except Exception:
        # ако streamlit не е наличен (напр. unit tests), продължаваме към env
        pass

    # Fallback към environment variables (локално или CI)
    return (
        os.getenv("OPENAI_KEY_PRIMARY")
        or os.getenv("OPENAI_KEY_FALLBACK")
        or os.getenv("OPENAI_KEY")
    )
