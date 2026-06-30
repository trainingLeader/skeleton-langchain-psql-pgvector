import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
from sqlalchemy import create_engine
from sqlalchemy import text
from openai import OpenAI
from dotenv import load_dotenv

client = OpenAI()

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding
def alcohol_label(x):
    return "Sí tiene alcohol" if x == 1 else "No tiene alcohol"
load_dotenv()

PG_HOST = os.getenv("POSTGRES_HOST")
PG_PORT = os.getenv("POSTGRES_PORT")
PG_USER = os.getenv("POSTGRES_USER")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD")
PG_DATABASE = os.getenv("POSTGRES_DB")
conn = psycopg2.connect(
    host=PG_HOST,
    port=PG_PORT,
    user=PG_USER,
    password=PG_PASSWORD,
    dbname=PG_DATABASE
)

#print("✅ Conectado a PostgreSQL correctamente.")

engine = create_engine(
    f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"
)
try:
    with engine.connect() as conn:
        print("Conexión exitosa a Postgres ✔️")
except Exception as e:
    print("Error de conexión ❌:", e)
    
query = """
SELECT 
    producto_id,
    nombre_producto,
    categoria,
    descripcion_larga,
    contiene_alcohol,
    precio_referencia_usd
FROM productos;
"""

df = pd.read_sql(query, engine)
print(df.head())



df["text_for_embedding"] = (
    df["nombre_producto"] + ". " +
    df["categoria"] + ". " +
    df["descripcion_larga"] + ". " +
    "Contiene alcohol: " + df["contiene_alcohol"].apply(alcohol_label) + ". " +
    "Precio USD: " + df["precio_referencia_usd"].astype(str)
)

df.head()
print(df.head())

insert_sql = """
INSERT INTO productos_embeddings (producto_id, embedding)
VALUES (:id, :emb)
ON CONFLICT (producto_id) DO UPDATE
SET embedding = EXCLUDED.embedding;
"""

with engine.begin() as conn:
    for _, row in df.iterrows():
        emb = get_embedding(row["text_for_embedding"])
        conn.execute(
            text(insert_sql),
            {"id": int(row["producto_id"]), "emb": emb}
        )

print("Embeddings generados y guardados correctamente ✔️")

# 1. Crear embedding de prueba
test_embedding = get_embedding("regalo elegante corporativo con vino tinto")

# 2. Convertir embedding a un formato compatible con PostgreSQL
vector_str = "[" + ",".join([str(x) for x in test_embedding]) + "]"

# 3. Consulta SQL de similitud usando el operador <-> de pgvector
similaridad_sql = f"""
SELECT
    p.producto_id,
    p.nombre_producto,
    p.categoria,
    (pe.embedding <-> '{vector_str}'::vector) AS distancia
FROM productos_embeddings pe
JOIN productos p
  ON p.producto_id = pe.producto_id
ORDER BY distancia ASC
LIMIT 5;
"""

# 4. Ejecutar consulta y mostrar resultados
resultados = pd.read_sql(similaridad_sql, engine)
print(resultados)

# Paso 10: preparar query UPDATE
update_sql = text("""
    UPDATE productos
    SET text_for_embedding = :texto
    WHERE producto_id = :producto_id;
""")
with engine.begin() as conn:   # engine.begin() abre conexión + hace commit automáticamente
    for _, fila in df.iterrows():
        conn.execute(
            update_sql,
            {
                "texto": fila["text_for_embedding"],
                "producto_id": int(fila["producto_id"])
            }
        )

print("✔️ Todas las filas fueron actualizadas correctamente en PostgreSQL.")