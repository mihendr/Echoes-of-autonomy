# app.py
import streamlit as st
from config import get_openai_key

st.title("My Streamlit app — secure keys example")

OPENAI_KEY = get_openai_key()

st.write("OpenAI key loaded:", bool(OPENAI_KEY))

if OPENAI_KEY:
    st.text("Key preview: " + (OPENAI_KEY[:6] + "..." + OPENAI_KEY[-4:]))
else:
    st.warning("No OPENAI_KEY found. Add it in Streamlit Cloud Secrets or set environment variable OPENAI_KEY.")

# Демонстрационен бутон — тук сложи реалния call към OpenAI/друга услуга
if st.button("Demo use key"):
    if not OPENAI_KEY:
        st.error("Няма ключ.")
    else:
        st.success("Ще извикам API тук с ключ, който завършва на: " + OPENAI_KEY[-4:])
