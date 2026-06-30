-- =========================================
-- 1. Activar extensión pgvector
-- =========================================
CREATE EXTENSION IF NOT EXISTS vector;

-- =========================================
-- 2. Crear tabla principal de productos
--    (solo estructura, sin datos)
-- =========================================
DROP TABLE IF EXISTS productos CASCADE;

CREATE TABLE productos (
    producto_id           INTEGER PRIMARY KEY,
    nombre_producto       TEXT        NOT NULL,
    categoria             TEXT        NOT NULL,
    descripcion_larga     TEXT        NOT NULL,
    contiene_alcohol      BOOLEAN     NOT NULL,
    precio_referencia_usd NUMERIC(10,2) NOT NULL
);

-- Columna opcional para el texto que luego
-- vamos a mandar a embeddings desde Python
ALTER TABLE productos
ADD COLUMN IF NOT EXISTS text_for_embedding TEXT;

-- =========================================
-- 3. Tabla para guardar los embeddings
-- =========================================
DROP TABLE IF EXISTS productos_embeddings CASCADE;

CREATE TABLE productos_embeddings (
    producto_id INTEGER PRIMARY KEY
        REFERENCES productos (producto_id) ON DELETE CASCADE,
    embedding   vector(1536)  -- dimensiones del modelo de embeddings
);

-- =========================================
-- 4. Consultas de verificación
-- =========================================

-- ¿Está instalada la extensión pgvector?
SELECT *
FROM pg_extension
WHERE extname = 'vector';

-- ¿Existen las tablas que necesitamos?
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name IN ('productos', 'productos_embeddings');

-- Ver algunas filas de productos una vez importado el CSV
SELECT *
FROM productos
LIMIT 5;

REATE TABLE orden (
    orden_iCd       INTEGER      NOT NULL,
    fecha_orden    DATE         NOT NULL,
    canal_venta    VARCHAR(50)  NOT NULL,
    zona_entrega   VARCHAR(100) NOT NULL,
    alcaldia       VARCHAR(100) NOT NULL,
    codigo_postal  VARCHAR(10)  NOT NULL,
    estado_orden   VARCHAR(30)  NOT NULL,
    prioridad      VARCHAR(20)  NOT NULL,
    notas_entrega  TEXT,

    CONSTRAINT pk_orden PRIMARY KEY (orden_id)
);

-- 5.2 Tabla DETALLE_ORDEN
CREATE TABLE detalle_orden (
    orden_id    INTEGER     NOT NULL,
    producto_id INTEGER     NOT NULL,
    cantidad    INTEGER     NOT NULL,
    dedicatoria TEXT,

    -- Clave primaria compuesta
    CONSTRAINT pk_detalle_orden PRIMARY KEY (orden_id, producto_id),

    -- Relaciones
    CONSTRAINT fk_detalle_orden_orden
        FOREIGN KEY (orden_id)
        REFERENCES orden (orden_id),

    CONSTRAINT fk_detalle_orden_producto
        FOREIGN KEY (producto_id)
        REFERENCES productos (producto_id)
);

-- 5.3 Tabla INVENTARIO
CREATE TABLE inventario (
    producto_id      INTEGER     NOT NULL,
    stock_disponible INTEGER     NOT NULL DEFAULT 0,

    CONSTRAINT pk_inventario PRIMARY KEY (producto_id),

    CONSTRAINT fk_inventario_producto
        FOREIGN KEY (producto_id)
        REFERENCES productos (producto_id)
);
