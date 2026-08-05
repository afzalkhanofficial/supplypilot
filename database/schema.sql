-- =====================================================================
-- SupplyPilot database schema
-- Target: PostgreSQL 14+
-- Each "Store" from the Rossmann dataset is treated as one "Product"
-- in this inventory system (see README for the rationale).
-- =====================================================================

-- Clean slate if re-running (order matters due to foreign keys)
DROP TABLE IF EXISTS agent_interactions CASCADE;
DROP TABLE IF EXISTS risk_alerts CASCADE;
DROP TABLE IF EXISTS purchase_orders CASCADE;
DROP TABLE IF EXISTS inventory CASCADE;
DROP TABLE IF EXISTS sales_history CASCADE;
DROP TABLE IF EXISTS products CASCADE;

-- ------------------------------------------------------------------
-- products
-- One row per store/product. product_id matches the Rossmann Store ID.
-- ------------------------------------------------------------------
CREATE TABLE products (
    product_id          INTEGER         PRIMARY KEY,
    product_name        VARCHAR(100)    NOT NULL,
    store_type          CHAR(1),                        -- a, b, c, or d
    assortment          CHAR(1),                        -- a=basic, b=extra, c=extended
    competition_distance NUMERIC(10, 2)                 -- metres to nearest competitor
);

-- ------------------------------------------------------------------
-- sales_history
-- Daily sales figures for each product. Rows where Open=0 are kept
-- for historical completeness but excluded from model training.
-- ------------------------------------------------------------------
CREATE TABLE sales_history (
    id              BIGSERIAL       PRIMARY KEY,
    product_id      INTEGER         NOT NULL REFERENCES products (product_id) ON DELETE CASCADE,
    date            DATE            NOT NULL,
    sales           INTEGER         NOT NULL,
    customers       INTEGER         NOT NULL,
    open            SMALLINT        NOT NULL,            -- 1 = store was open
    promo           SMALLINT        NOT NULL,            -- 1 = promotion active that day
    state_holiday   CHAR(1)         NOT NULL,            -- 0=none, a=public, b=Easter, c=Christmas
    school_holiday  SMALLINT        NOT NULL,            -- 1 = school holiday
    CONSTRAINT uq_product_date UNIQUE (product_id, date)
);

CREATE INDEX idx_sales_history_product_date ON sales_history (product_id, date);

-- ------------------------------------------------------------------
-- inventory
-- One row per product. current_stock is updated whenever a purchase
-- order is approved or a manual adjustment is made.
-- ------------------------------------------------------------------
CREATE TABLE inventory (
    product_id      INTEGER         PRIMARY KEY REFERENCES products (product_id) ON DELETE CASCADE,
    current_stock   NUMERIC(12, 2)  NOT NULL CHECK (current_stock >= 0),
    lead_time_days  INTEGER         NOT NULL CHECK (lead_time_days > 0),
    unit_cost       NUMERIC(10, 2)  NOT NULL CHECK (unit_cost > 0),
    supplier_name   VARCHAR(200)    NOT NULL,
    last_updated    TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- ------------------------------------------------------------------
-- purchase_orders
-- Created by the agent (status='pending'). A human approves or
-- rejects via the dashboard, which sets decided_at and flips status.
-- ------------------------------------------------------------------
CREATE TABLE purchase_orders (
    id              SERIAL          PRIMARY KEY,
    product_id      INTEGER         NOT NULL REFERENCES products (product_id) ON DELETE CASCADE,
    quantity        INTEGER         NOT NULL CHECK (quantity > 0),
    supplier_name   VARCHAR(200)    NOT NULL,
    estimated_cost  NUMERIC(12, 2)  NOT NULL CHECK (estimated_cost >= 0),
    status          VARCHAR(20)     NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'approved', 'rejected')),
    agent_reasoning TEXT,                               -- full reasoning chain from the agent
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    decided_at      TIMESTAMPTZ                         -- NULL until a human acts on it
);

CREATE INDEX idx_purchase_orders_status ON purchase_orders (status);

-- ------------------------------------------------------------------
-- risk_alerts
-- Populated by the risk-checking tools. product_id is nullable so
-- we can store system-wide alerts that are not product-specific.
-- ------------------------------------------------------------------
CREATE TABLE risk_alerts (
    id          SERIAL          PRIMARY KEY,
    product_id  INTEGER         REFERENCES products (product_id) ON DELETE SET NULL,
    alert_type  VARCHAR(100)    NOT NULL,
    message     TEXT            NOT NULL,
    severity    VARCHAR(20)     NOT NULL CHECK (severity IN ('low', 'medium', 'high')),
    created_at  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_risk_alerts_created_at ON risk_alerts (created_at DESC);

-- ------------------------------------------------------------------
-- agent_interactions
-- Audit trail: every question asked through the dashboard and the
-- agent's full response, including which tools were called.
-- ------------------------------------------------------------------
CREATE TABLE agent_interactions (
    id              SERIAL          PRIMARY KEY,
    user_question   TEXT            NOT NULL,
    agent_answer    TEXT            NOT NULL,
    tools_used      TEXT,           -- comma-separated list of tool names called
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_agent_interactions_created_at ON agent_interactions (created_at DESC);

-- ------------------------------------------------------------------
-- documents  (Phase 8 — Supplier Document Intelligence)
-- One row per uploaded supplier document.
-- sha256_hex is a hex-encoded SHA-256 of the raw file bytes, used to
-- reject duplicate uploads without re-embedding.
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id              SERIAL          PRIMARY KEY,
    filename        VARCHAR(500)    NOT NULL,
    supplier_name   VARCHAR(200)    NOT NULL,
    doc_type        VARCHAR(50)     NOT NULL,   -- 'sla' | 'contract' | 'policy'
    sha256_hex      CHAR(64)        NOT NULL UNIQUE,
    page_count      INTEGER,
    uploaded_at     TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_supplier ON documents (supplier_name);
CREATE INDEX IF NOT EXISTS idx_documents_uploaded  ON documents (uploaded_at DESC);

-- ------------------------------------------------------------------
-- document_chunks  (Phase 8)
-- Each document is split into overlapping text windows.  The embedding
-- column stores a 384-dimensional vector produced by all-MiniLM-L6-v2.
-- chunk_index is 0-based within the parent document.
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS document_chunks (
    id              BIGSERIAL       PRIMARY KEY,
    document_id     INTEGER         NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    chunk_index     INTEGER         NOT NULL,
    chunk_text      TEXT            NOT NULL,
    embedding       VECTOR(384)     NOT NULL,
    CONSTRAINT uq_chunk_doc_index UNIQUE (document_id, chunk_index)
);

-- IVFFlat index — 50 centroids is a good default for a corpus up to
-- ~50 000 chunks; rebuild with more lists if the corpus grows beyond that.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON document_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 50);

