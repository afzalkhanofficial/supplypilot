# SupplyPilot — Project Audit

**Audit date:** 2026-08-04  
**Auditor:** AI-assisted (Antigravity / Gemini)  
**Author:** Afzal Khan \<afzalkhan10802@gmail.com\>  
**Repository:** https://github.com/afzalkhanofficial/supplypilot  
**Last commit:** `6d4dc0b` — all changes committed and pushed to `origin/main`

---

## 1. Directory Tree

```
supplypilot/
├── agent/
│   ├── agent.py
│   ├── prompts.py
│   ├── tools.py
│   └── __init__.py
├── api/
│   ├── main.py
│   ├── schemas.py
│   ├── __init__.py
│   └── routers/
│       ├── agent.py
│       ├── inventory.py
│       ├── orders.py
│       ├── products.py
│       └── __init__.py
├── backend/                        ← EMPTY STUB (see §8)
│   ├── __init__.py
│   └── routers/
│       └── __init__.py
├── dashboard/
│   ├── api_client.py
│   ├── app.py
│   └── __init__.py
├── data/
│   ├── processed/                  ← empty directory
│   └── raw/
│       ├── store.csv               ← Rossmann store metadata (not committed)
│       └── train.csv               ← Rossmann sales history (not committed)
├── database/
│   ├── db.py
│   ├── schema.sql
│   └── __init__.py
├── forecasting/
│   ├── evaluate.py
│   ├── model_comparison.md
│   ├── naive_baseline_results.json
│   ├── predict.py
│   ├── train_naive_baseline.py
│   ├── train_prophet.py
│   ├── __init__.py
│   └── models/
│       └── prophet/
│           ├── product_85.json
│           ├── product_259.json
│           ├── product_262.json
│           ├── product_274.json
│           ├── product_310.json
│           ├── product_335.json
│           ├── product_353.json
│           ├── product_423.json
│           ├── product_494.json
│           ├── product_530.json
│           ├── product_562.json
│           ├── product_578.json
│           ├── product_676.json
│           ├── product_682.json
│           ├── product_733.json
│           ├── product_769.json
│           ├── product_863.json
│           ├── product_948.json
│           ├── product_1097.json
│           └── product_1099.json
├── inventory/
│   ├── calculator.py
│   └── __init__.py
├── scripts/
│   ├── run_api.py
│   ├── run_dashboard.py
│   ├── seed_data.py
│   ├── test_agent.py
│   ├── test_inventory.py
│   └── __init__.py
├── tests/
│   ├── conftest.py
│   ├── test_health.py
│   ├── test_inventory.py
│   ├── test_orders.py
│   ├── test_products.py
│   └── __init__.py
├── .env.example
├── .gitignore
├── Procfile
├── PROJECT_AUDIT.md                ← this file
├── README.md
├── render.yaml
└── requirements.txt
```

**Not committed to git (intentional):**
- `.env` (contains secrets)
- `data/raw/train.csv` and `data/raw/store.csv` (Kaggle dataset, ~40 MB each)
- `.venv/` (virtual environment)

---

## 2. Source Files

### `database/db.py`

```python
"""
Database engine, session factory, ORM models, and utility functions.

All connection parameters are read from the DATABASE_URL environment
variable — never hardcoded.  The engine is configured for compatibility
with Supabase's PgBouncer transaction-mode pooler (port 6543).
"""

import logging
import os
from datetime import datetime, timezone
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import (
    BigInteger, Boolean, CheckConstraint, Column, Date, DateTime,
    ForeignKey, Integer, Numeric, SmallInteger, String, Text,
    UniqueConstraint, create_engine, text,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

load_dotenv()

logger = logging.getLogger(__name__)

def _build_database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL environment variable is not set. "
            "Copy .env.example to .env and fill in your Supabase connection string."
        )
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    if "supabase.com" in url and "sslmode" not in url:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}sslmode=require"
    return url


_DATABASE_URL: str = _build_database_url()

engine = create_engine(
    _DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    connect_args={"options": "-c timezone=utc"},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class Product(Base):
    __tablename__ = "products"
    product_id: int = Column(Integer, primary_key=True)
    product_name: str = Column(String(100), nullable=False)
    store_type: str = Column(String(1))
    assortment: str = Column(String(1))
    competition_distance: float = Column(Numeric(10, 2))
    sales_history = relationship("SalesHistory", back_populates="product", cascade="all, delete-orphan")
    inventory = relationship("Inventory", back_populates="product", uselist=False, cascade="all, delete-orphan")
    purchase_orders = relationship("PurchaseOrder", back_populates="product", cascade="all, delete-orphan")
    risk_alerts = relationship("RiskAlert", back_populates="product")


class SalesHistory(Base):
    __tablename__ = "sales_history"
    __table_args__ = (UniqueConstraint("product_id", "date", name="uq_product_date"),)
    id: int = Column(BigInteger, primary_key=True, autoincrement=True)
    product_id: int = Column(Integer, ForeignKey("products.product_id", ondelete="CASCADE"), nullable=False)
    date = Column(Date, nullable=False)
    sales: int = Column(Integer, nullable=False)
    customers: int = Column(Integer, nullable=False)
    open: int = Column(SmallInteger, nullable=False)
    promo: int = Column(SmallInteger, nullable=False)
    state_holiday: str = Column(String(1), nullable=False)
    school_holiday: int = Column(SmallInteger, nullable=False)
    product = relationship("Product", back_populates="sales_history")


class Inventory(Base):
    __tablename__ = "inventory"
    product_id: int = Column(Integer, ForeignKey("products.product_id", ondelete="CASCADE"), primary_key=True)
    current_stock: float = Column(Numeric(12, 2), nullable=False)
    lead_time_days: int = Column(Integer, nullable=False)
    unit_cost: float = Column(Numeric(10, 2), nullable=False)
    supplier_name: str = Column(String(200), nullable=False)
    last_updated = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    product = relationship("Product", back_populates="inventory")


class PurchaseOrder(Base):
    __tablename__ = "purchase_orders"
    id: int = Column(Integer, primary_key=True, autoincrement=True)
    product_id: int = Column(Integer, ForeignKey("products.product_id", ondelete="CASCADE"), nullable=False)
    quantity: int = Column(Integer, nullable=False)
    supplier_name: str = Column(String(200), nullable=False)
    estimated_cost: float = Column(Numeric(12, 2), nullable=False)
    status: str = Column(String(20), nullable=False, default="pending")
    agent_reasoning: str = Column(Text)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    decided_at = Column(DateTime(timezone=True), nullable=True)
    product = relationship("Product", back_populates="purchase_orders")


class RiskAlert(Base):
    __tablename__ = "risk_alerts"
    id: int = Column(Integer, primary_key=True, autoincrement=True)
    product_id: int = Column(Integer, ForeignKey("products.product_id", ondelete="SET NULL"), nullable=True)
    alert_type: str = Column(String(100), nullable=False)
    message: str = Column(Text, nullable=False)
    severity: str = Column(String(20), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    product = relationship("Product", back_populates="risk_alerts")


class AgentInteraction(Base):
    __tablename__ = "agent_interactions"
    id: int = Column(Integer, primary_key=True, autoincrement=True)
    user_question: str = Column(Text, nullable=False)
    agent_answer: str = Column(Text, nullable=False)
    tools_used: str = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    logger.info("Running init_db — creating missing tables if any...")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("init_db complete — all tables are present.")
    except Exception:
        logger.exception("init_db failed — could not create tables.")
        raise


def test_connection() -> bool:
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
            if result == 1:
                logger.info("Database connection test passed.")
                return True
            logger.warning("Database connection test returned unexpected result: %s", result)
            return False
    except Exception:
        logger.exception("Database connection test failed.")
        return False
```

---

### `inventory/calculator.py`

*(475 lines — key sections shown; full file at `inventory/calculator.py`)*

```python
"""
Inventory optimization calculator.

Implements EOQ, Safety Stock, and Reorder Point formulas.
Lead-time uncertainty is NOT modelled (fixed lead times only).
Service level → z-score mapping is a lookup table, not scipy.
"""

DEFAULT_ORDER_COST: float = 50.0        # $ per purchase order
DEFAULT_HOLDING_COST_RATE: float = 0.25  # fraction of unit cost per year
DAYS_PER_YEAR: int = 365

_Z_TABLE: dict[float, float] = {
    0.80: 0.842, 0.85: 1.036, 0.90: 1.282, 0.95: 1.645,
    0.97: 1.881, 0.99: 2.326, 0.999: 3.090,
}

_CRITICAL_DAYS_OF_COVER: int = 3
_WARNING_DAYS_OF_COVER: int = 7

def z_score(service_level: float) -> float: ...          # linear interpolation
def calculate_eoq(...) -> float: ...                      # Wilson's formula: sqrt(2DS/H)
def calculate_safety_stock(...) -> float: ...             # z * σ_d * sqrt(LT)
def calculate_reorder_point(...) -> float: ...            # avg_d * LT + safety_stock
def get_inventory_recommendation(product_id, ...) -> dict: ...  # full pipeline
```

Full source: see `inventory/calculator.py` (475 lines).

---

### `forecasting/predict.py`

```python
"""
Reusable forecast interface. Loads trained Prophet models, returns
structured day-by-day predictions. Pure read-only; no DB writes.

Notable: monkey-patches cmdstanpy.set_cmdstan_path with a safe wrapper
that silently swallows ValueError raised by Prophet's internal wheel
path detection on Windows (see §8 — Bugs & Hacks).
"""

import cmdstanpy
# HACK: Prophet's model_from_json internally calls set_cmdstan_path with a
# path baked into the wheel that may not exist in the local environment.
# We patch the function to suppress the resulting ValueError.
_real_set_cmdstan_path = cmdstanpy.set_cmdstan_path
def _safe_set_cmdstan_path(path: str) -> None:
    try:
        _real_set_cmdstan_path(path)
    except ValueError:
        pass
cmdstanpy.set_cmdstan_path = _safe_set_cmdstan_path

_MODELS_DIR = Path(__file__).resolve().parent.parent / "forecasting" / "models" / "prophet"
_model_cache: dict[int, Any] = {}   # module-level in-memory cache

class ModelNotFoundError(Exception): ...
class ForecastRangeError(ValueError): ...

def _load_model(product_id: int) -> Any:
    """Load and cache Prophet model from JSON on first call."""
    ...

def get_forecast(product_id: int, days_ahead: int) -> dict[str, Any]:
    """
    Returns: {product_id, days_ahead, dates, yhat, yhat_lower, yhat_upper, training_end}
    Regressors default to 0 (no promo, no holidays) — conservative baseline.
    Negative yhat clipped to 0.
    """
    ...

def list_available_products() -> list[int]: ...
```

Full source: see `forecasting/predict.py` (218 lines).

---

### `forecasting/train_prophet.py`

*(313 lines)*

- Loads open-day sales from DB per product.
- Trains Prophet with `weekly_seasonality=True`, `yearly_seasonality=True`, `daily_seasonality=False`, `seasonality_mode="multiplicative"`.
- Three extra regressors: `promo`, `state_holiday` (binary), `school_holiday`.
- Holds out last 42 calendar days from training.
- Persists each model as `forecasting/models/prophet/product_{id}.json`.
- Includes the same `_safe_set_cmdstan_path` monkey-patch as `predict.py`.

Full source: see `forecasting/train_prophet.py` (313 lines).

---

### `forecasting/train_naive_baseline.py`

*(274 lines)*

- 1-day lag naive baseline: `predicted[t] = actual[t-1]` on open-day records.
- Same 42-day holdout as Prophet.
- Writes results to `forecasting/naive_baseline_results.json`.

Full source: see `forecasting/train_naive_baseline.py` (274 lines).

---

### `forecasting/evaluate.py`

*(392 lines)*

- Loads every trained Prophet model, generates holdout predictions using **actual** regressor values (fair evaluation), computes MAE/RMSE/MAPE, compares vs naive baseline.
- Writes `forecasting/model_comparison.md`.
- Includes the same `_safe_set_cmdstan_path` monkey-patch.

Full source: see `forecasting/evaluate.py` (392 lines).

---

### `agent/prompts.py`

```python
SYSTEM_PROMPT = """You are SupplyPilot, an AI supply chain optimization advisor...
CAPABILITIES: list_products, get_demand_forecast, get_inventory_status,
              scan_all_inventory, create_purchase_order, get_recent_risk_alerts
RULES:
  1. Always call at least one tool before drawing a conclusion.
  2. When risk is CRITICAL/WARNING, always recommend a specific action.
  3. Confirm product_id and quantity before calling create_purchase_order.
  4. Format numbers clearly.
  5. Keep answers concise.
  6. Never fabricate data.
"""

def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
```

Full source: see `agent/prompts.py` (81 lines).

---

### `agent/tools.py`

*(365 lines)*

Six `@tool`-decorated functions:

| Tool | What it does |
|---|---|
| `list_products` | SQL `SELECT` from `products` table |
| `get_demand_forecast` | Calls `forecasting.predict.get_forecast()` |
| `get_inventory_status` | Calls `inventory.calculator.get_inventory_recommendation()` |
| `scan_all_inventory` | Loops all products through the recommendation function |
| `create_purchase_order` | INSERTs a pending order into `purchase_orders` |
| `get_recent_risk_alerts` | SQL `SELECT` from `risk_alerts` table |

All tools: catch all exceptions and return JSON error strings (agent never sees a Python traceback). All DB access uses lazy `_engine()` imports to avoid circular imports at module load.

Full source: see `agent/tools.py` (365 lines).

---

### `agent/agent.py`

```python
_MODEL_NAME = "llama-3.3-70b-versatile"   # Groq
_TEMPERATURE = 0
_MAX_ITERATIONS = 8
_MAX_EXECUTION_TIME = 120  # seconds

def _build_agent_executor() -> AgentExecutor:
    """Builds a fresh AgentExecutor on every call (stateless by design)."""
    ...

def run_agent(user_question: str, chat_history=None) -> dict:
    """
    Returns: {"answer": str, "tools_used": list[str], "steps": int}
    Logs every interaction to agent_interactions table (failures silently swallowed).
    """
    ...

def build_chat_history(turns: list[tuple[str, str]]) -> list[BaseMessage]: ...
```

Full source: see `agent/agent.py` (233 lines).

---

### `api/main.py`

```python
# Lifespan: DB ping on startup (non-fatal if it fails).
# CORS: defaults to localhost:8501,localhost:3000 from env var CORS_ORIGINS.
# Global error handler: catches unhandled exceptions → 500 JSON.
# Health endpoint: GET /health → {status, db_connected, version}
# Routers: products, inventory, orders, agent, alerts

app = FastAPI(title="SupplyPilot API", version="1.0.0", lifespan=lifespan)
```

Full source: see `api/main.py` (163 lines).

---

### `api/schemas.py`

*(200 lines — all Pydantic v2 models)*

Key models:
- `ProductOut`, `ProductListResponse`
- `ForecastResponse` (dates, yhat, yhat_lower, yhat_upper, total_forecast)
- `InventoryStatusResponse`, `InventoryScanItem`, `InventoryScanResponse`
- `CreateOrderRequest` (product_id gt=0, quantity gt=0, reason 5–500 chars)
- `OrderStatusUpdate` (pattern `^(approved|rejected)$`)
- `AgentChatRequest`, `AgentChatResponse`
- `RiskAlertOut`, `RiskAlertListResponse`

Full source: see `api/schemas.py` (200 lines).

---

### `api/routers/products.py`

*(119 lines)*

- `GET /products` — list all, ordered by product_id.
- `GET /products/{id}` — single product, 404 if missing.
- `GET /products/{id}/forecast?days_ahead=N` — Prophet forecast, 404 if no model, 422 if days_ahead ∉ [1,90].

---

### `api/routers/inventory.py`

*(116 lines)*

- `GET /inventory/scan` — runs `get_inventory_recommendation()` for all products; sorts CRITICAL→WARNING→OK.
- `GET /inventory/{id}` — single product recommendation; 404 via `ValueError` from calculator.

---

### `api/routers/orders.py`

*(229 lines)*

- `GET /orders` — list with optional status filter, limit/offset.
- `POST /orders` — create pending order (looks up unit_cost/supplier from inventory table).
- `GET /orders/{id}` — single order, 404 if missing.
- `PATCH /orders/{id}/status` — approve/reject; 409 if order not in `pending` state.

---

### `api/routers/agent.py`

*(168 lines)*

- `POST /agent/chat` — calls `run_agent()`; 503 if GROQ_API_KEY missing.
- `GET /agent/history` — paginated agent_interactions log.
- `GET /alerts` — paginated risk_alerts (severity filter optional).

---

### `dashboard/api_client.py`

*(147 lines)*

- `_BASE_URL` from `API_BASE_URL` env var (default `http://localhost:8000`).
- `_TIMEOUT = 120s` (agent calls can be slow).
- Unified `_get` / `_post` / `_patch` helpers that raise `APIError` on connection failures or non-2xx responses.
- Thin wrappers: `health()`, `list_products()`, `get_forecast()`, `get_inventory()`, `scan_inventory()`, `list_orders()`, `create_order()`, `approve_order()`, `reject_order()`, `agent_chat()`, `agent_history()`, `list_alerts()`.

Full source: see `dashboard/api_client.py` (147 lines).

---

### `dashboard/app.py`

*(642 lines)*

Single-file Streamlit multi-page app. Five pages driven by `st.radio` sidebar navigation:

| Page | Key features |
|---|---|
| **Overview** | 4 KPI cards (total/critical/warning/pending), Plotly risk bar, colour-coded risk table |
| **Inventory** | Product selectbox, 6 KPI cards, Plotly bar (stock vs ROP/SS/EOQ), action sentence |
| **Demand Forecast** | Product + days slider, Plotly line+CI chart, raw data expander |
| **Purchase Orders** | 3 tabs: Pending (approve/reject buttons), All Orders (filter), Create Order (form) |
| **Agent Chat** | Suggested prompts, multi-turn history, tool pills per response, clear button |

Dark glassmorphism CSS injected via `st.markdown(..., unsafe_allow_html=True)`. Plotly dark theme.

Full source: see `dashboard/app.py` (642 lines).

---

### `scripts/seed_data.py`

*(403 lines)*

- Reads `data/raw/train.csv` and `data/raw/store.csv` (Rossmann dataset).
- Selects 20 stores with the most open-day records.
- Inserts: `products`, `sales_history` (batches of 5,000), `inventory` (with random lead times and unit costs, seed=42).
- Entire load in a single transaction; rolls back on any failure.
- Truncates existing data before re-seeding (idempotent).

Full source: see `scripts/seed_data.py` (403 lines).

---

### `scripts/run_api.py`

*(43 lines)* — `argparse` wrapper around `uvicorn.run("api.main:app", ...)`.

### `scripts/run_dashboard.py`

*(34 lines)* — `subprocess.run(["streamlit", "run", ...])` launcher.

### `scripts/test_agent.py`

*(61 lines)* — ad-hoc smoke test script (not pytest). Exercises `list_products`, `get_inventory_status`, `build_prompt`, and a live Groq call. Requires `GROQ_API_KEY` and a running DB.

### `scripts/test_inventory.py`

*(49 lines)* — ad-hoc smoke test script (not pytest). Verifies formula outputs and `get_inventory_recommendation` for 4 products.

---

### `tests/conftest.py`

```python
@pytest.fixture(scope="session")
def client() -> TestClient:
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c

KNOWN_PRODUCT_IDS = [85, 259, 262, 274, 310]
FIRST_PRODUCT_ID = KNOWN_PRODUCT_IDS[0]
```

**Note:** tests run against the **production/development database** (same `DATABASE_URL`). There is no isolated test database. The one write test (purchase orders) cleans up after itself via PATCH/reject.

---

### `tests/test_health.py`

*(24 lines)* — 3 tests: status 200, schema validation, `db_connected == true`.

### `tests/test_inventory.py`

*(full file — 5 test classes, 22 tests)*

Tests:
- `TestZScore` (6 tests): table lookups, edge cases, interpolation.
- `TestEOQ` (5 tests): standard formula, from unit cost, zero-demand/zero-cost guard.
- `TestSafetyStock` (4 tests): known values, zero std, service level monotonicity, lead time sensitivity.
- `TestReorderPoint` (3 tests): known values, ROP > cycle stock, zero demand.
- `TestInventoryScan` (5 tests): GET /inventory/scan 200, all products returned, schema, count totals, sort order.
- `TestInventorySingle` (6 tests): GET /inventory/{pid} 200, schema fields, valid risk level, positive EOQ/ROP, 404 for unknown product, parametrized across 5 known products.

### `tests/test_orders.py`

*(full file — 9 tests)*

Tests: GET /orders 200/schema/status-filter/invalid-422, POST /orders 201/schema/cost-positive, GET /orders/{id}, GET /orders/unknown-404, approve-then-reject-409-conflict, unknown-product-404, zero-quantity-422.

### `tests/test_products.py`

*(full file — 11 tests)*

Tests: GET /products 200/total=20/schema/ordered, GET /products/{id} known/404, GET /products/{id}/forecast default horizon/custom/non-negative values/total=sum/out-of-range-422/unknown-404, parametrized forecast across 5 known products.

---

## 3. Configuration Files

### `requirements.txt`

```
# Web framework & server
fastapi==0.115.6
uvicorn[standard]==0.32.1
python-multipart==0.0.20

# Dashboard
streamlit==1.41.1
plotly==5.24.1

# Data processing
pandas==2.2.3
numpy==1.26.4

# Forecasting
prophet==1.1.6
scikit-learn==1.5.2

# Statistics
scipy==1.14.1

# Database
sqlalchemy==2.0.36
psycopg2-binary==2.9.10

# Environment & config
python-dotenv==1.0.1

# LangChain agent stack
langchain-core==0.3.63
langchain==0.3.13
langchain-groq==0.2.3
langchain-community==0.3.13

# HTTP & feeds
requests==2.32.3
feedparser==6.0.11

# Data validation
pydantic==2.10.3

# Testing
pytest==8.3.4
httpx==0.28.1
```

**Note:** `scipy` is listed as a dependency ("used in safety stock calculation") but is **not actually imported** anywhere in the codebase. The z-score is computed via a lookup table in `inventory/calculator.py`. `feedparser` is listed but also not used — it was included speculatively for a weather/news feed tool that was never implemented.

---

### `.env.example`

```ini
# PostgreSQL connection string
DATABASE_URL=postgresql://postgres:your_db_password@db.xxxxxxxxxxxx.supabase.co:5432/postgres

# Groq LLM API key
GROQ_API_KEY=gsk_your_groq_api_key_here

# OpenWeatherMap API key (not used — see §8)
OPENWEATHER_API_KEY=your_openweathermap_api_key_here

# FastAPI base URL for the dashboard
BACKEND_URL=http://localhost:8000
```

**Note:** `OPENWEATHER_API_KEY` and `BACKEND_URL` appear in `.env.example` but are **never read** anywhere in the code. The dashboard reads `API_BASE_URL` (not `BACKEND_URL`). See §8.

---

### `Procfile`

```
web: uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

**Note:** Only declares the API process. The Streamlit dashboard is a separate Render service defined in `render.yaml` and has no `Procfile` entry.

---

### `render.yaml`

```yaml
services:
  - type: web
    name: supplypilot-api
    env: python
    region: oregon
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn api.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: DATABASE_URL
        fromDatabase:
          name: supplypilot-db
          property: connectionString
      - key: GROQ_API_KEY
        sync: false
      - key: CORS_ORIGINS
        value: "*"           # ← permissive; should be locked to dashboard URL in prod

  - type: web
    name: supplypilot-dashboard
    env: python
    region: oregon
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0 --theme.base dark
    envVars:
      - key: API_BASE_URL
        fromService:
          type: web
          name: supplypilot-api
          property: url

databases:
  - name: supplypilot-db
    databaseName: supplypilot
    user: supplypilot_user
    plan: free
```

---

### `database/schema.sql`

```sql
-- Clean slate (DROP TABLE ... CASCADE for all 6 tables)

CREATE TABLE products (
    product_id          INTEGER         PRIMARY KEY,
    product_name        VARCHAR(100)    NOT NULL,
    store_type          CHAR(1),
    assortment          CHAR(1),
    competition_distance NUMERIC(10, 2)
);

CREATE TABLE sales_history (
    id              BIGSERIAL       PRIMARY KEY,
    product_id      INTEGER         NOT NULL REFERENCES products (product_id) ON DELETE CASCADE,
    date            DATE            NOT NULL,
    sales           INTEGER         NOT NULL,
    customers       INTEGER         NOT NULL,
    open            SMALLINT        NOT NULL,
    promo           SMALLINT        NOT NULL,
    state_holiday   CHAR(1)         NOT NULL,
    school_holiday  SMALLINT        NOT NULL,
    CONSTRAINT uq_product_date UNIQUE (product_id, date)
);
CREATE INDEX idx_sales_history_product_date ON sales_history (product_id, date);

CREATE TABLE inventory (
    product_id      INTEGER         PRIMARY KEY REFERENCES products (product_id) ON DELETE CASCADE,
    current_stock   NUMERIC(12, 2)  NOT NULL CHECK (current_stock >= 0),
    lead_time_days  INTEGER         NOT NULL CHECK (lead_time_days > 0),
    unit_cost       NUMERIC(10, 2)  NOT NULL CHECK (unit_cost > 0),
    supplier_name   VARCHAR(200)    NOT NULL,
    last_updated    TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE TABLE purchase_orders (
    id              SERIAL          PRIMARY KEY,
    product_id      INTEGER         NOT NULL REFERENCES products (product_id) ON DELETE CASCADE,
    quantity        INTEGER         NOT NULL CHECK (quantity > 0),
    supplier_name   VARCHAR(200)    NOT NULL,
    estimated_cost  NUMERIC(12, 2)  NOT NULL CHECK (estimated_cost >= 0),
    status          VARCHAR(20)     NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'approved', 'rejected')),
    agent_reasoning TEXT,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    decided_at      TIMESTAMPTZ
);
CREATE INDEX idx_purchase_orders_status ON purchase_orders (status);

CREATE TABLE risk_alerts (
    id          SERIAL          PRIMARY KEY,
    product_id  INTEGER         REFERENCES products (product_id) ON DELETE SET NULL,
    alert_type  VARCHAR(100)    NOT NULL,
    message     TEXT            NOT NULL,
    severity    VARCHAR(20)     NOT NULL CHECK (severity IN ('low', 'medium', 'high')),
    created_at  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_risk_alerts_created_at ON risk_alerts (created_at DESC);

CREATE TABLE agent_interactions (
    id              SERIAL          PRIMARY KEY,
    user_question   TEXT            NOT NULL,
    agent_answer    TEXT            NOT NULL,
    tools_used      TEXT,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_agent_interactions_created_at ON agent_interactions (created_at DESC);
```

---

## 4. Test Results

**Command:** `.venv\Scripts\pytest.exe tests\ -v --tb=long`  
**Python:** 3.12.3  
**Platform:** win32  
**Plugins:** anyio-4.14.2  

**Result: 65 passed, 0 failed in 197.61s (3m 17s)**

```
tests/test_health.py::test_health_returns_200                                 PASSED
tests/test_health.py::test_health_schema                                      PASSED
tests/test_health.py::test_health_db_connected                                PASSED
tests/test_inventory.py::TestZScore::test_95th_percentile                     PASSED
tests/test_inventory.py::TestZScore::test_99th_percentile                     PASSED
tests/test_inventory.py::TestZScore::test_80th_percentile                     PASSED
tests/test_inventory.py::TestZScore::test_interpolation_between_table_entries PASSED
tests/test_inventory.py::TestZScore::test_invalid_zero_raises                 PASSED
tests/test_inventory.py::TestZScore::test_invalid_one_raises                  PASSED
tests/test_inventory.py::TestEOQ::test_standard_formula                       PASSED
tests/test_inventory.py::TestEOQ::test_from_unit_cost_and_rate                PASSED
tests/test_inventory.py::TestEOQ::test_zero_demand_raises                     PASSED
tests/test_inventory.py::TestEOQ::test_zero_order_cost_raises                 PASSED
tests/test_inventory.py::TestEOQ::test_result_is_positive_int                 PASSED
tests/test_inventory.py::TestSafetyStock::test_known_values                   PASSED
tests/test_inventory.py::TestSafetyStock::test_zero_std_gives_zero            PASSED
tests/test_inventory.py::TestSafetyStock::test_higher_service_level_gives_larger_buffer PASSED
tests/test_inventory.py::TestSafetyStock::test_longer_lead_time_gives_larger_buffer PASSED
tests/test_inventory.py::TestReorderPoint::test_known_values                  PASSED
tests/test_inventory.py::TestReorderPoint::test_rop_greater_than_cycle_stock  PASSED
tests/test_inventory.py::TestReorderPoint::test_zero_demand_and_std           PASSED
tests/test_inventory.py::TestInventoryScan::test_scan_200                     PASSED
tests/test_inventory.py::TestInventoryScan::test_scan_returns_all_products    PASSED
tests/test_inventory.py::TestInventoryScan::test_scan_schema                  PASSED
tests/test_inventory.py::TestInventoryScan::test_scan_count_totals_match_scanned PASSED
tests/test_inventory.py::TestInventoryScan::test_scan_sorted_by_risk_then_cover PASSED
tests/test_inventory.py::TestInventorySingle::test_known_product_200          PASSED
tests/test_inventory.py::TestInventorySingle::test_schema_fields_present      PASSED
tests/test_inventory.py::TestInventorySingle::test_risk_level_valid           PASSED
tests/test_inventory.py::TestInventorySingle::test_eoq_and_rop_positive       PASSED
tests/test_inventory.py::TestInventorySingle::test_unknown_product_404        PASSED
tests/test_inventory.py::TestInventorySingle::test_all_known_products[85]     PASSED
tests/test_inventory.py::TestInventorySingle::test_all_known_products[259]    PASSED
tests/test_inventory.py::TestInventorySingle::test_all_known_products[262]    PASSED
tests/test_inventory.py::TestInventorySingle::test_all_known_products[274]    PASSED
tests/test_inventory.py::TestInventorySingle::test_all_known_products[310]    PASSED
tests/test_orders.py::test_list_orders_200                                    PASSED
tests/test_orders.py::test_list_orders_schema                                 PASSED
tests/test_orders.py::test_list_orders_status_filter_pending                  PASSED
tests/test_orders.py::test_list_orders_invalid_status_422                     PASSED
tests/test_orders.py::test_create_order_201                                   PASSED
tests/test_orders.py::test_create_order_schema                                PASSED
tests/test_orders.py::test_create_order_estimated_cost_positive               PASSED
tests/test_orders.py::test_get_order_by_id                                    PASSED
tests/test_orders.py::test_get_order_unknown_404                              PASSED
tests/test_orders.py::test_approve_then_reject_conflict                       PASSED
tests/test_orders.py::test_create_order_unknown_product_404                   PASSED
tests/test_orders.py::test_create_order_zero_quantity_422                     PASSED
tests/test_products.py::test_list_products_200                                PASSED
tests/test_products.py::test_list_products_returns_all                        PASSED
tests/test_products.py::test_list_products_schema                             PASSED
tests/test_products.py::test_list_products_ordered                            PASSED
tests/test_products.py::test_get_product_known                                PASSED
tests/test_products.py::test_get_product_unknown_404                          PASSED
tests/test_products.py::test_forecast_default_horizon                         PASSED
tests/test_products.py::test_forecast_custom_horizon                          PASSED
tests/test_products.py::test_forecast_values_non_negative                     PASSED
tests/test_products.py::test_forecast_total_matches_sum                       PASSED
tests/test_products.py::test_forecast_out_of_range_422                        PASSED
tests/test_products.py::test_forecast_unknown_product_404                     PASSED
tests/test_products.py::test_forecast_all_known_products[85]                  PASSED
tests/test_products.py::test_forecast_all_known_products[259]                 PASSED
tests/test_products.py::test_forecast_all_known_products[262]                 PASSED
tests/test_products.py::test_forecast_all_known_products[274]                 PASSED
tests/test_products.py::test_forecast_all_known_products[310]                 PASSED

==================== 65 passed in 197.61s (0:03:17) ====================
```

**Coverage gaps:** No tests for `agent/` (would require mocking Groq API), `dashboard/` (Streamlit testing is complex), or `forecasting/train_*.py` (training scripts). No load/stress tests.

**Test run time:** 3m 17s — dominated by Prophet model loading (~3–5s per product on first call) and DB query latency. The session-scoped client fixture avoids repeated startup but models are loaded lazily per test.

---

## 5. Git Commit History

```
6d4dc0b feat: add integration test suite, render iac, procfile, env template, and complete readme
a1fa848 feat: add streamlit dashboard with overview, inventory, forecast, orders, and agent chat pages
bccc7e5 feat: add fastapi rest api with products, inventory, orders, and agent routes
074c781 feat: add langchain agent with tools, prompts, and groq llm runner
64d440f feat: add inventory calculator with eoq, safety stock, and reorder point logic
3925ff9 feat: add trained prophet models and evaluation results
385ebe0 fix: apply cmdstanpy set_cmdstan_path safety patch to evaluate.py and predict.py
1e13b71 fix: add cmdstanpy set_cmdstan_path safety patch for prophet internal wheel path issue
67c04e2 fix: pass stan_backend='CMDSTANPY' explicitly to prophet constructor
cef3fca fix: resolve prophet stan_backend initialization error by enforcing cmdstanpy path
08db7bd feat: add reusable forecast prediction interface
9ffcd76 feat: add naive baseline and model evaluation pipeline
8184661 feat: train per-product prophet models with holiday regressors
7791031 feat: add data seeding pipeline for rossmann dataset
ed53a3c feat: add postgres schema and sqlalchemy models
95e8153 chore: initialize project structure and dependencies
```

**Total commits:** 16 across a single `main` branch. No feature branches. No pull requests — development was done directly on `main`.

Working tree: **clean** (`git status` shows nothing to commit).

---

## 6. Forecasting Benchmark Results

Prophet vs. 1-day lag naive baseline, 42-day holdout, 20 products:

| Product | Prophet MAE | Baseline MAE | MAE Δ | Prophet MAPE | Baseline MAPE | MAPE Δ |
|---|---|---|---|---|---|---|
| Product-85 | 443.3 | 2219.4 | **-80.0%** | 6.4% | 27.3% | -76.7% |
| Product-259 | 701.8 | 2320.9 | -69.8% | 5.8% | 17.5% | -67.0% |
| Product-262 | 1650.4 | 4472.0 | -63.1% | 7.5% | 19.4% | -61.5% |
| Product-274 | 1276.1 | 1110.7 | **+14.9%** | 21.6% | 18.5% | **+17.1%** |
| Product-310 | 837.8 | 2795.4 | -70.0% | 10.8% | 44.0% | -75.5% |
| Product-335 | 1381.8 | 3058.9 | -54.8% | 9.9% | 22.6% | -56.1% |
| Product-353 | 457.1 | 1548.0 | -70.5% | 6.4% | 21.7% | -70.5% |
| Product-423 | 657.1 | 2388.6 | -72.5% | 5.1% | 19.2% | -73.5% |
| Product-494 | 619.3 | 1187.7 | -47.9% | 8.1% | 17.1% | -52.4% |
| Product-530 | 745.0 | 958.7 | -22.3% | 10.7% | 13.7% | -22.2% |
| Product-562 | 1056.2 | 2617.6 | -59.7% | 6.0% | 15.9% | -62.3% |
| Product-578 | 1034.8 | 2305.6 | -55.1% | 9.4% | 23.7% | -60.2% |
| Product-676 | 474.9 | 2327.3 | -79.6% | 5.2% | 23.7% | -78.1% |
| Product-682 | 581.6 | 1912.8 | -69.6% | 5.4% | 17.5% | -69.2% |
| Product-733 | 882.3 | 1191.4 | -25.9% | 6.3% | 8.2% | -23.8% |
| Product-769 | 910.8 | 1322.2 | -31.1% | 7.4% | 10.8% | -30.9% |
| Product-863 | 655.3 | 2735.1 | -76.0% | 8.0% | 46.7% | **-82.8%** |
| Product-948 | 406.8 | 1744.5 | -76.7% | 4.6% | 19.7% | -76.4% |
| Product-1097 | 491.6 | 1719.6 | -71.4% | 4.4% | 15.7% | -72.0% |
| Product-1099 | 802.5 | 2053.1 | -60.9% | 8.8% | 25.9% | -65.9% |
| **AVERAGE** | **803.3** | **2099.5** | **-61.7%** | **7.9%** | **21.4%** | **-63.2%** |

**Outlier:** Product-274 is the only product where Prophet performs **worse** than the naive baseline (MAE +14.9%, MAPE +17.1%). This product likely has a volatile, low-signal demand pattern that Prophet's seasonality components overfit.

---

## 7. Trained Model Inventory

**Location:** `forecasting/models/prophet/`  
**Count:** 20 JSON files  
**Total size:** 3.69 MB

| File | Size (KB) |
|---|---|
| product_562.json | 192.7 |
| product_733.json | 192.7 |
| product_262.json | 192.6 |
| product_335.json | 192.4 |
| product_423.json | 192.3 |
| product_682.json | 192.3 |
| product_769.json | 192.3 |
| product_85.json | 191.9 |
| product_494.json | 191.8 |
| product_676.json | 187.0 |
| product_530.json | 184.5 |
| product_578.json | 183.6 |
| product_310.json | 182.8 |
| product_1099.json | 181.9 |
| product_863.json | 180.8 |
| product_948.json | 188.9 |
| product_274.json | 189.8 |
| product_353.json | 189.5 |
| product_259.json | 190.4 |
| product_1097.json | 192.1 |

Models are committed to git, which is acceptable at ~190 KB each. A `.gitattributes` LFS configuration would be cleaner at scale.

---

## 8. Known Bugs, Hacks, and Unfinished Pieces

### 8.1 CmdStanPy Monkey-Patch (Hack)

**Files:** `forecasting/predict.py` (L30–41), `forecasting/evaluate.py` (L37–48), `forecasting/train_prophet.py` (L38–49)

**Description:** Prophet's `model_from_json()` internally calls `cmdstanpy.set_cmdstan_path()` with a path baked into the Python wheel (e.g., `site-packages/prophet/stan_model/cmdstan-...`). On Windows, this path may not exist in the local environment, causing a `ValueError` that crashes deserialization. The patch replaces `set_cmdstan_path` with a version that silently swallows `ValueError`.

**Risk:** This patch is a global monkey-patch applied at module import time. If Prophet is updated and the internal path changes or the behavior changes, the silent swallow could mask a real error. The correct long-term fix is to file an issue with the Prophet library or pin to a version that doesn't exhibit this behavior.

**Note:** The patch appears in **three separate files** with identical code, not shared through a common utility module. This is a maintenance risk.

---

### 8.2 No Test Database Isolation

**File:** `tests/conftest.py`

**Description:** The test suite runs against the development/production database. This means:
- Tests can see and affect real data.
- The order test (`test_create_order_201`) writes a real `purchase_order` row and relies on cleanup via `PATCH /orders/{id}/status` (reject). If that cleanup step fails (e.g. network error, test timeout), the DB is left with a dirty pending order.
- Running tests twice quickly could cause the `approve_then_reject_conflict` test to interfere with the cleanup fixture's reject call if ordering is unlucky.

**Mitigation in place:** The `created_order` fixture is `scope="module"` and cleans up in the fixture teardown. This is reasonable but brittle.

---

### 8.3 Empty `backend/` Directory

**Files:** `backend/__init__.py`, `backend/routers/__init__.py`

**Description:** Both files are empty. This directory appears to be a leftover from an earlier design iteration where the API package was named `backend`. It was renamed to `api` but the `backend/` stub was never deleted. It has no functional impact but adds noise to the directory tree.

---

### 8.4 Unused Dependencies in `requirements.txt`

**Packages:** `scipy`, `feedparser`

- `scipy`: Listed with comment "used in safety stock calculation" but the z-score is computed via a lookup table in `inventory/calculator.py`. `scipy` is **never imported** anywhere in the codebase. Including it adds ~35 MB to the install.
- `feedparser`: Listed but never imported anywhere. Was presumably included for a planned weather/news feed tool that was never implemented.

---

### 8.5 Stale Env Var Names in `.env.example`

**File:** `.env.example`

Two variables defined in `.env.example` are never consumed:
- `OPENWEATHER_API_KEY` — was for a weather tool that was never built.
- `BACKEND_URL` — the dashboard reads `API_BASE_URL` (set correctly in `render.yaml`), not `BACKEND_URL`.

This creates confusion for anyone following the setup instructions — they will set `BACKEND_URL` and it will be silently ignored.

---

### 8.6 Forecast Regressors Always Default to Zero

**File:** `forecasting/predict.py` (L175–178)

**Description:** When generating future forecasts, all three regressors (`promo`, `state_holiday`, `school_holiday`) are hard-coded to `0`. This means:
- The model assumes no future promotions or holidays.
- For the 42-day holdout evaluation in `evaluate.py`, **actual** regressor values are used — which makes the evaluation optimistic relative to production where actual future promo/holiday schedules are unknown.
- In practice, for inventory planning, this conservative baseline is reasonable (it assumes the worst: no promotions to boost demand). However, the mismatch between evaluation and inference is not documented in the code comments.

---

### 8.7 Inventory `current_stock` is Never Updated

**File:** `api/routers/orders.py` — `PATCH /orders/{id}/status`

**Description:** When a purchase order is `approved`, the code sets `decided_at` and flips `status` to `approved` but **does not add the ordered quantity to `inventory.current_stock`**. This is a logical gap: in a real system, approval would trigger restocking. As implemented, `current_stock` is only set at seeding time and never changes through the application lifecycle. The agent's inventory recommendations are therefore stale from day one of production use.

**Severity:** Medium. The system functions correctly for demonstration purposes (the stock level shown is the seeded initial value), but would require this fix for real operational use.

---

### 8.8 `risk_alerts` Table is Never Populated

**File:** Various

**Description:** The `risk_alerts` table exists in the schema and ORM, and is exposed via `GET /alerts` and the agent's `get_recent_risk_alerts` tool. However, there is no code path that **writes** to this table. The agent's risk assessment results are computed on-the-fly and returned as structured data but never persisted as alerts. The `GET /alerts` endpoint will always return an empty list until alerts are manually inserted.

**Impact:** The "Alerts" feature in the dashboard sidebar and the `get_recent_risk_alerts` agent tool are functionally inert.

---

### 8.9 CORS Set to Wildcard `"*"` in `render.yaml`

**File:** `render.yaml` (line: `value: "*"`)

**Description:** The Render deployment sets `CORS_ORIGINS=*`, allowing any origin to call the API. In the local environment, `api/main.py` defaults to `localhost:8501,localhost:3000`. This is noted in the code comment but the production config does not restrict origins to the dashboard's URL.

**Mitigation:** Low risk for a demo/portfolio project, but should be locked down before handling real business data.

---

### 8.10 Product-274 Prophet Model Underperforms the Baseline

**See §6.** Prophet's MAE on Product-274 is 14.9% worse than the naive baseline. No special handling or fallback exists — the model is used as-is for inventory recommendations for this product.

---

### 8.11 Single-Database Pooling Risk with PgBouncer

**File:** `database/db.py` (L88)

```python
connect_args={"options": "-c timezone=utc"},
```

The engine is configured with `pool_pre_ping=True` and `pool_recycle=300`, which are the correct settings for PgBouncer transaction-mode pooling. However, the `connect_args` note in the code mentions PgBouncer does not support prepared statements, but `psycopg2` is **not** explicitly configured to disable server-side prepared statements (`statement_cache_size=0`). For PgBouncer transaction-mode, this can cause `ERROR: prepared statement "..." does not exist` errors under concurrent load.

---

### 8.12 No Auth/AuthZ on Any Endpoint

**Description:** The entire API is unauthenticated. Any client with network access can:
- Read all inventory and forecast data.
- Create, approve, or reject purchase orders.
- Submit arbitrary questions to the agent (which has a $120s timeout cap).

This is acceptable for a portfolio demo but not for a production deployment.

---

### 8.13 Agent Executor Rebuilt on Every Request

**File:** `agent/agent.py` (L63–96, `_build_agent_executor`)

**Description:** A new `AgentExecutor` (including LLM client initialization) is constructed on every `POST /agent/chat` call. The code notes this is intentional for statelessness, but it adds latency to every agent call. The `ChatGroq` client instantiation makes an implicit HTTPS connection. For high-traffic scenarios, the executor should be cached (e.g., at the FastAPI app-level, one per worker process).

---

### 8.14 Test Suite Runs Against Live DB With Real Prophet Models

**Description:** The `test_forecast_*` tests in `test_products.py` actually invoke `forecasting.predict.get_forecast()` which loads Prophet JSON models and runs inference. Each parametrized product triggers model deserialization. This is why the test suite takes 3m 17s. Faster alternatives (mocking `get_forecast` or using `@pytest.mark.slow`) were not implemented.

---

## 9. Environment Variables Reference

| Variable | Where read | Default | Notes |
|---|---|---|---|
| `DATABASE_URL` | `database/db.py` | none (hard fail) | Must be PostgreSQL URL |
| `GROQ_API_KEY` | `agent/agent.py` | none (raises EnvironmentError) | Groq inference API |
| `API_BASE_URL` | `dashboard/api_client.py` | `http://localhost:8000` | Dashboard → API URL |
| `CORS_ORIGINS` | `api/main.py` | `http://localhost:8501,...` | Comma-separated origins |
| `OPENWEATHER_API_KEY` | **Nowhere** | — | In .env.example only; unused |
| `BACKEND_URL` | **Nowhere** | — | In .env.example only; unused |

---

## 10. Deployment Status

| Component | Status |
|---|---|
| FastAPI API | `render.yaml` configured; not yet deployed (requires Render account + DB provisioning) |
| Streamlit Dashboard | `render.yaml` configured; not yet deployed |
| PostgreSQL (Supabase) | Running (dev instance); connected and tested |
| Procfile | Present (API only) |
| CI/CD | None — no GitHub Actions or similar |

The project is **ready for Render deployment** — all infra-as-code is in place. The manual steps remaining are: create a Render account, connect the GitHub repo, and provision the services.

---

## 11. Architecture Summary

```
User/Browser
     │
     ▼
Streamlit Dashboard (port 8501)
  dashboard/app.py
  dashboard/api_client.py
     │  HTTP (requests)
     ▼
FastAPI API (port 8000)
  api/main.py
  api/routers/{products,inventory,orders,agent}.py
  api/schemas.py
     ├── GET /products* → database/db.py (SQLAlchemy)
     ├── GET /inventory* → inventory/calculator.py
     │                    → forecasting/predict.py (Prophet)
     │                    → database/db.py
     ├── POST /agent/chat → agent/agent.py
     │                     → agent/tools.py
     │                     → LangChain + Groq API (llama-3.3-70b-versatile)
     └── CRUD /orders → database/db.py
          │
          ▼
     PostgreSQL (Supabase)
       products, sales_history, inventory,
       purchase_orders, risk_alerts, agent_interactions
```

**Data flow for a forecast request:**
1. Dashboard calls `GET /products/{id}/forecast?days_ahead=N`
2. `products.py` router calls `forecasting.predict.get_forecast(id, N)`
3. `predict.py` loads (or returns cached) Prophet JSON model from disk
4. Prophet generates `N` future rows with regressor defaults = 0
5. `yhat` is clipped to ≥ 0, returned as JSON
6. Dashboard renders Plotly chart

**Data flow for an agent chat:**
1. Dashboard POSTs `{"question": "...", "chat_history": [...]}` to `/agent/chat`
2. `agent.py` builds `AgentExecutor` with Groq LLM + 6 tools
3. LLM decides which tools to call (0–8 iterations, 120s cap)
4. Tools query DB or Prophet models and return JSON strings
5. LLM synthesizes a natural-language answer
6. Interaction is logged to `agent_interactions` table
7. Response returned to dashboard
