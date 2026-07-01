from pathlib import Path
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, text

load_dotenv()
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
client = OpenAI()

def get_apiKey():
    load_dotenv(ENV_PATH, override=True)
    return os.getenv("OPENAI_API_KEY")

def get_postgres_connection_params():
    load_dotenv(ENV_PATH, override=True)
    PG_HOST = os.getenv("POSTGRES_HOST")
    PG_PORT = os.getenv("POSTGRES_PORT")
    PG_USER = os.getenv("POSTGRES_USER")
    PG_PASSWORD = os.getenv("POSTGRES_PASSWORD")
    PG_DATABASE = os.getenv("POSTGRES_DB")
    return PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DATABASE

def get_postgres_connection_url() -> str:
    """
    Construye la URL de conexión a PostgreSQL usando psycopg2.
    """
    PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DATABASE = get_postgres_connection_params()
    return f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}" 

def get_embedding(texto: str) -> list:
    """
    Genera un embedding para el texto de entrada usando OpenAI.
    """
    respuesta = client.embeddings.create(
        model="text-embedding-3-small",
        input=texto
    )
    return respuesta.data[0].embedding
def search_similar_products(texto_busqueda: str,k: int = 5) -> pd.DataFrame:
    """
    Busca productos similares a partir de una descripción en lenguaje natural.

    Retorna un DataFrame con:
    - producto_id
    - nombre_producto
    - categoria
    - precio_referencia_usd
    - distancia (cuanto menor, más parecido)
    """
    engine = create_engine(get_postgres_connection_url())

    # 1. Embedding del texto de búsqueda
    emb = get_embedding(texto_busqueda)

    # 2. Convertir a string compatible con pgvector: [0.1,0.2,...]
    vector_str = "[" + ",".join(str(x) for x in emb) + "]"

    # 3. Query de similitud usando el operador <-> de pgvector
    similaridad_sql = f"""
    SELECT
        p.producto_id,
        p.nombre_producto,
        p.categoria,
        p.precio_referencia_usd,
        (pe.embedding <-> '{vector_str}'::vector) AS distancia
    FROM productos_embeddings pe
    JOIN productos p
      ON p.producto_id = pe.producto_id
    ORDER BY distancia ASC
    LIMIT {k};
    """

    resultados = pd.read_sql(similaridad_sql, engine)
    return resultados

def get_product_context(product_ids: list[int]) -> pd.DataFrame:
    engine = create_engine(get_postgres_connection_url())
    """
    Dado un listado de producto_id, arma un contexto combinando:
    - productos
    - inventario
    - detalle_orden / orden

    Devuelve un DataFrame listo para pasar al LLM.
    """
    if not product_ids:
        return pd.DataFrame()

    # Construimos el IN (...) de manera sencilla
    ids_str = ",".join(str(pid) for pid in product_ids)

    contexto_sql = f"""
    SELECT
        p.producto_id,
        p.nombre_producto,
        p.categoria,
        p.precio_referencia_usd,
        COALESCE(i.stock_disponible, 0) AS stock_disponible,
        COALESCE(SUM(d.cantidad), 0)    AS unidades_vendidas
    FROM productos p
    LEFT JOIN inventario i
           ON i.producto_id = p.producto_id
    LEFT JOIN detalle_orden d
           ON d.producto_id = p.producto_id
    LEFT JOIN orden o
           ON o.orden_id = d.orden_id
    WHERE p.producto_id IN ({ids_str})
    GROUP BY
        p.producto_id,
        p.nombre_producto,
        p.categoria,
        p.precio_referencia_usd,
        i.stock_disponible
    ORDER BY p.producto_id;
    """

    contexto = pd.read_sql(contexto_sql, engine)
    return contexto
def responder_consulta_usuario(texto_busqueda: str, k: int = 5) -> str:
    """
    Orquesta todo el flujo:
    - Búsqueda semántica
    - Enriquecimiento relacional
    - Redacción de la respuesta con un LLM
    """
    # 1. Buscar candidatos por similitud
    candidatos = search_similar_products(texto_busqueda, k=k)

    if candidatos.empty:
        return "No encontré productos similares a tu descripción."

    # 2. Obtener el contexto relacional para esos productos
    product_ids = candidatos["producto_id"].tolist()
    contexto = get_product_context(product_ids)

    # 3. Convertir el contexto a un formato simple (CSV) para el modelo
    contexto_csv = contexto.to_csv(index=False)

    # 4. Prompt para el modelo de lenguaje
    system_prompt = (
        "Eres un asesor de regalos corporativos para una tienda en línea. "
        "Usa estrictamente la información de la tabla de productos que te voy a dar. "
        "Escribe en español latinoamericano, con un tono profesional pero cercano. "
        "Cuando recomiendes, menciona nombre del producto, categoría, precio aproximado "
        "y si hay stock suficiente."
    )

    user_prompt = f"""
    Consulta del usuario:
    \"\"\"{texto_busqueda}\"\"\"

    Datos de productos (formato CSV):
    {contexto_csv}

    Con base en estos datos:
    1. Elige 1 a 3 opciones que calcen bien con la consulta.
    2. Explica brevemente por qué las recomiendas.
    3. Si el presupuesto es muy bajo para las opciones encontradas, sugiere el producto más cercano explicando la diferencia de precio.
    4. Si ves stock muy bajo, menciónalo como advertencia.
    """

    respuesta = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.5,
    )

    return respuesta.choices[0].message.content.strip()
