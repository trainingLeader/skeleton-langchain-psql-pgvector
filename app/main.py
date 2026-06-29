import os

import streamlit as st


st.set_page_config(
    page_title="Python LangChain Dev",
    page_icon=":robot_face:",
    layout="wide",
)

st.title("Python LangChain Dev")
st.write("Entorno base listo para Streamlit, LangChain y PostgreSQL.")

database_url = os.getenv("DATABASE_URL", "No configurada")
postgres_host = os.getenv("POSTGRES_HOST", "No configurado")
openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

status_rows = [
    ("DATABASE_URL", "Configurada" if database_url != "No configurada" else "Falta"),
    ("POSTGRES_HOST", postgres_host),
    ("OPENAI_API_KEY", "Configurada" if openai_api_key else "Falta"),
    ("GOOGLE_API_KEY", "Configurada" if google_api_key else "Falta"),
]

st.subheader("Estado de configuracion")
st.table(
    {
        "Variable": [row[0] for row in status_rows],
        "Estado": [row[1] for row in status_rows],
    }
)

st.subheader("Siguientes pasos")
st.markdown(
    """
1. Agregar la logica de negocio en `app/main.py` o modularizarla en nuevos archivos.
2. Conectar LangChain a tu proveedor de modelos y a PostgreSQL.
3. Levantar el entorno con `docker compose up --build`.
"""
)
