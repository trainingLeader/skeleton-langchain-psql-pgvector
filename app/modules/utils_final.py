from pathlib import Path
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, text
import json
import unicodedata
from datetime import date, timedelta

load_dotenv()
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
client = OpenAI()

def normalizar(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s

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
def extraer_datos_orden(texto_usuario: str, model: str = "gpt-4.1-mini") -> dict:
    system_prompt = """
    Eres un asistente que extrae datos estructurados para crear una orden en una tienda de regalos.

    Devuelve SOLO un JSON válido (sin texto adicional) con estas llaves:
    - nombre_producto (string)
    - cantidad (int)
    - zona_entrega (string o null)
    - alcaldia (string o null)
    - codigo_postal (string o null)
    - fecha_entrega (string: "hoy", "mañana" o null)
    - prioridad (string: "URGENTE" o "NORMAL" o null)
    - notas (string o null)
    - dedicatoria (string o null)

    Reglas:
    - Si el texto contiene "urgente", prioridad = "URGENTE".
    - Si fecha_entrega es "hoy" o "mañana", prioridad = "URGENTE".
    - Si la cantidad no se especifica, usa 1.
    """

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": texto_usuario.strip()},
        ],
        # Esto fuerza salida JSON (si tu modelo lo soporta)
        response_format={"type": "json_object"},
        temperature=0.0
    )

    data = json.loads(resp.choices[0].message.content)

    # Normalización mínima
    if not data.get("cantidad"):
        data["cantidad"] = 1
    data["cantidad"] = int(data["cantidad"])

    # Regla prioridad por fecha_entrega
    fe = (data.get("fecha_entrega") or "").strip().lower()
    pr = (data.get("prioridad") or "").strip().upper()

    if fe in ("hoy", "mañana"):
        data["prioridad"] = "URGENTE"
    elif pr in ("URGENTE", "NORMAL"):
        data["prioridad"] = pr
    else:
        data["prioridad"] = "NORMAL"

    return data

def obtener_producto_id(nombre_producto: str) -> int | None:
    # Probamos con el texto original y luego sin tildes
    engine = create_engine(get_postgres_connection_url())
    candidatos = [
        f"%{nombre_producto.strip()}%",
        f"%{normalizar(nombre_producto)}%",
    ]

    q = text("""
        SELECT producto_id, nombre_producto
        FROM productos
        WHERE nombre_producto ILIKE :patron
        ORDER BY LENGTH(nombre_producto) ASC
        LIMIT 1;
    """)

    with engine.connect() as conn:
        for patron in candidatos:
            row = conn.execute(q, {"patron": patron}).fetchone()
            if row:
                return int(row[0])

    return None
def crear_orden(datos: dict) -> dict:
    engine = create_engine(get_postgres_connection_url())
    nombre = datos["nombre_producto"]
    cantidad = int(datos["cantidad"])

    zona = datos.get("zona_entrega") or "No especificada"
    alcaldia = datos.get("alcaldia") or "No especificada"
    codigo_postal = datos.get("codigo_postal") or "00000"
    prioridad = (datos.get("prioridad") or "NORMAL").upper()
    notas = datos.get("notas")
    dedicatoria = datos.get("dedicatoria")

    # Convertimos "hoy"/"mañana" a nota trazable (sin complicar calendario)
    fecha_entrega_txt = datos.get("fecha_entrega")
    if fecha_entrega_txt:
        extra = f"Fecha entrega solicitada: {fecha_entrega_txt}"
        notas = f"{notas} | {extra}" if notas else extra

    producto_id = obtener_producto_id(nombre)
    if producto_id is None:
        raise ValueError(f"No se encontró el producto en tabla productos: '{nombre}'")

    with engine.begin() as conn:
        # 1) Bloquear fila de inventario para evitar condiciones de carrera
        stock = conn.execute(
            text("""
                SELECT stock_disponible
                FROM inventario
                WHERE producto_id = :pid
                FOR UPDATE;
            """),
            {"pid": producto_id}
        ).scalar_one_or_none()

        if stock is None:
            raise ValueError(f"No existe inventario para producto_id={producto_id}")
        if int(stock) < cantidad:
            raise ValueError(f"Stock insuficiente. Disponible={stock}, solicitado={cantidad}")

        # 2) Insert orden (requiere orden_id SERIAL/IDENTITY en BD)
        orden_id = conn.execute(
            text("""
                INSERT INTO orden (
                    fecha_orden, canal_venta, zona_entrega, alcaldia, codigo_postal,
                    estado_orden, prioridad, notas_entrega
                )
                VALUES (
                    :fecha_orden, :canal_venta, :zona_entrega, :alcaldia, :codigo_postal,
                    :estado_orden, :prioridad, :notas_entrega
                )
                RETURNING orden_id;
            """),
            {
                "fecha_orden": date.today(),
                "canal_venta": "chatbot",
                "zona_entrega": zona,
                "alcaldia": alcaldia,
                "codigo_postal": codigo_postal,
                "estado_orden": "CREADA",
                "prioridad": prioridad,
                "notas_entrega": notas,
            }
        ).scalar_one()

        # 3) Insert detalle_orden
        conn.execute(
            text("""
                INSERT INTO detalle_orden (orden_id, producto_id, cantidad, dedicatoria)
                VALUES (:orden_id, :producto_id, :cantidad, :dedicatoria);
            """),
            {
                "orden_id": int(orden_id),
                "producto_id": int(producto_id),
                "cantidad": int(cantidad),
                "dedicatoria": dedicatoria,
            }
        )

        # 4) Descontar inventario
        conn.execute(
            text("""
                UPDATE inventario
                SET stock_disponible = stock_disponible - :cantidad
                WHERE producto_id = :producto_id;
            """),
            {"cantidad": cantidad, "producto_id": producto_id}
        )

        # 5) Confirmación (lectura rápida)
        resumen = conn.execute(
            text("""
                SELECT
                    o.orden_id,
                    o.fecha_orden,
                    o.zona_entrega,
                    o.alcaldia,
                    o.prioridad,
                    p.nombre_producto,
                    d.cantidad,
                    i.stock_disponible
                FROM orden o
                JOIN detalle_orden d ON d.orden_id = o.orden_id
                JOIN productos p ON p.producto_id = d.producto_id
                JOIN inventario i ON i.producto_id = p.producto_id
                WHERE o.orden_id = :oid;
            """),
            {"oid": int(orden_id)}
        ).mappings().one()

    return dict(resumen)
def procesar_orden(texto_usuario: str, debug: bool = True) -> str:
    datos = extraer_datos_orden(texto_usuario)

    if debug:
        print("=== JSON extraído ===")
        print(datos)
        print("=====================\n")

    resultado = crear_orden(datos)

    return (
        f"Orden creada correctamente.\n"
        f"- orden_id: {resultado['orden_id']}\n"
        f"- producto: {resultado['nombre_producto']}\n"
        f"- cantidad: {resultado['cantidad']}\n"
        f"- prioridad: {resultado['prioridad']}\n"
        f"- entrega: {resultado['zona_entrega']}, {resultado['alcaldia']}\n"
        f"- stock restante: {resultado['stock_disponible']}\n"
    )