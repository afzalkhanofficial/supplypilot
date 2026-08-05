# 🚀 SupplyPilot — AI Supply Chain Optimization System

SupplyPilot is an end-to-end AI supply chain optimization platform. It combines **time-series demand forecasting** (Facebook Prophet), **classical inventory optimization math** (EOQ, Safety Stock, Reorder Point), **supplier document RAG** (`pgvector` + `sentence-transformers`), a **LangChain tool-calling AI agent** powered by Groq (`llama-3.3-70b-versatile`), a **FastAPI REST backend**, and a modern **Streamlit interactive dashboard**.

---

## 🌟 Key Features

- **Multi-Product Demand Forecasting**: Trained Facebook Prophet models for 20 retail products evaluated against a 42-day holdout set. Outperforms 1-day lag naive baseline by **-61.7% MAE**, **-65.1% RMSE**, and **-63.2% MAPE**.
- **Deterministic Inventory Optimization**:
  - **EOQ (Economic Order Quantity)**: Wilson's formula balancing order setup costs vs holding costs.
  - **Safety Stock**: Calibrated to target service level ($Z$-score lookup).
  - **Reorder Point (ROP)**: Dynamic demand-during-lead-time thresholding.
- **Supplier Document Intelligence (RAG)**:
  - Vector similarity search over supplier contracts, SLAs, and policies stored in Supabase PostgreSQL using `pgvector`.
  - Local CPU-based embedding generation using `all-MiniLM-L6-v2` (384-dimensional normalized vectors).
  - Character-window chunking (1 000 chars / 200 overlap) with word-boundary snapping.
  - Automatic PDF (PyPDF) and plain text extraction with SHA-256 deduplication.
  - IVFFlat index (`lists=50`) for scalable approximate nearest neighbor (ANN) search.
- **Autonomous Tool-Calling Agent**: LangChain agent equipped with **10 custom tools** (`list_products`, `get_demand_forecast`, `get_inventory_status`, `scan_all_inventory`, `create_purchase_order`, `get_recent_risk_alerts`, `check_weather_risk`, `check_supplier_news_risk`, `search_supplier_docs`, `list_supplier_documents`).
- **RESTful FastAPI Service**: Clean OpenAPI specs with CORS, DB lifespan probes, purchase order approval workflows that update live inventory stock, document RAG endpoints (`/documents/ingest`, `/documents/search`, `/documents`), and Pydantic validation.
- **Interactive Streamlit Dashboard**: Dark glassmorphism UI with 6 main pages:
  1. 📊 **Overview**: Fleet KPIs & risk summary table.
  2. 📦 **Inventory**: Per-product stock analysis & gauge metrics.
  3. 📈 **Demand Forecast**: Prophet forecast visualization with 80% CI band.
  4. 🛒 **Purchase Orders**: Human-in-the-loop approval workflow (automatically restocking inventory on approval).
  5. 🤖 **Agent Chat**: Multi-turn conversational AI interface with tool audit logs and document citation capabilities.
  6. 📄 **Supplier Intelligence**: Semantic document search, drag-and-drop document upload/ingest, and indexed document library.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    User([User / Operations Manager]) <--> Dashboard[Streamlit Dashboard\n:8501]
    Dashboard <--> API[FastAPI Backend\n:8000]

    subgraph Core Engines
        API <--> Agent[LangChain Tool-Calling Agent\nGroq llama-3.3-70b-versatile]
        API <--> Inv[Inventory Calculator\nEOQ / Safety Stock / ROP]
        API <--> Forecaster[Prophet Predictor\n20 Trained Model JSONs]
        API <--> RAG[RAG Engine\nSentenceTransformers all-MiniLM-L6-v2]
    end

    Agent <--> Tools[10 Agent Tools]
    Tools <--> Inv
    Tools <--> Forecaster
    Tools <--> RAG
    Tools <--> External[Weather & News RSS APIs]
    Tools <--> DB[(Supabase PostgreSQL\n+ pgvector)]
    API <--> DB
```

---

## 📂 Project Structure

```
supplypilot/
├── agent/                  # LangChain AI agent, tools, and prompts
│   ├── agent.py            # Agent executor runner & DB interaction logger
│   ├── prompts.py          # System prompt & ChatPromptTemplate
│   └── tools.py            # 10 LangChain @tool wrappers (including RAG search)
├── api/                    # FastAPI REST API backend
│   ├── main.py             # FastAPI app, CORS, lifespan probe & error handlers
│   ├── schemas.py          # Pydantic request/response models
│   └── routers/            # API endpoint routers
│       ├── agent.py        # /agent/chat, /agent/history, /alerts
│       ├── documents.py    # /documents/ingest, /documents/search, /documents
│       ├── inventory.py    # /inventory/scan, /inventory/{id}
│       ├── orders.py       # /orders (list, create, approve/reject)
│       └── products.py     # /products, /products/{id}/forecast
├── dashboard/              # Streamlit frontend dashboard
│   ├── app.py              # 6-page Streamlit web app with custom dark CSS
│   └── api_client.py       # Requests-based HTTP client for API backend
├── data/                   # Data files and sample supplier documents
│   └── supplier_docs/      # Sample SLAs, contracts, and policy documents
├── database/               # PostgreSQL schema & SQLAlchemy ORM
│   ├── db.py               # Engine, pgvector adapter, session setup & ORM models
│   └── schema.sql          # PostgreSQL DDL script (including pgvector tables & IVFFlat index)
├── forecasting/            # Time-series forecasting pipeline
│   ├── models/prophet/     # 20 trained Prophet model JSON artifacts
│   ├── evaluate.py         # Prophet vs baseline evaluation pipeline
│   ├── model_comparison.md # Benchmark results table
│   ├── prophet_patch.py    # CmdStanPy path safety patch
│   ├── predict.py          # Read-only Prophet forecast interface
│   ├── train_naive_baseline.py # 1-day lag baseline script
│   └── train_prophet.py    # Prophet training script
├── inventory/              # Classical inventory mathematics
│   └── calculator.py       # EOQ, Safety Stock, ROP & recommendation logic
├── rag/                    # Supplier Document Intelligence (RAG) package
│   ├── __init__.py         # Package entry point
│   ├── chunker.py          # Text chunker with word boundary snapping
│   ├── embedder.py         # Thread-safe all-MiniLM-L6-v2 singleton embedder
│   ├── ingestor.py         # Text/PDF extraction, SHA-256 dedup, & pgvector insert
│   └── search.py           # Cosine-similarity ANN search & list_documents
├── scripts/                # Helper scripts & CLI runners
│   ├── ingest_docs.py      # Bulk document ingestion CLI
│   ├── run_api.py          # FastAPI launcher
│   ├── run_dashboard.py    # Streamlit launcher
│   ├── seed_data.py        # Rossmann dataset seed script
│   ├── test_agent.py       # Agent verification test
│   └── test_inventory.py   # Inventory math verification script
├── tests/                  # Integration & unit test suite (pytest)
│   ├── conftest.py         # Session TestClient fixture
│   ├── test_health.py      # Health endpoint tests
│   ├── test_inventory.py   # Inventory math & endpoint tests
│   ├── test_orders.py      # Purchase order API tests
│   ├── test_products.py    # Product & forecast API tests
│   └── test_rag.py         # Phase 8 RAG unit & integration test suite
├── .env.example            # Environment variable template
├── render.yaml             # Render.com infrastructure as code specification
├── Procfile                # Web process configuration
└── requirements.txt        # Pinned Python dependencies
```

---

## 📊 Forecasting Benchmark Results

Prophet was benchmarked against a 1-day lag Naive Baseline on a 42-day holdout across 20 products:

| Metric | Naive Baseline | Prophet Model | Improvement |
|---|---|---|---|
| **Mean Absolute Error (MAE)** | 2,099.5 units | **803.3 units** | **-61.7%** |
| **Root Mean Squared Error (RMSE)** | 3,032.4 units | **1,058.8 units** | **-65.1%** |
| **Mean Absolute Percentage Error (MAPE)** | 21.4% | **7.9%** | **-63.2%** |

*Note: Evaluated on open-day sales data over the last 42 calendar days for each product.*

---

## ⚡ Quickstart & Local Setup

### 1. Prerequisites
- Python 3.11+
- PostgreSQL database with `pgvector` enabled (e.g., Supabase PostgreSQL)
- Groq API Key ([Get one free at console.groq.com](https://console.groq.com))

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/afzalkhanofficial/supplypilot.git
cd supplypilot

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Database Vector Setup
In your Supabase SQL Editor:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 4. Environment Configuration
Copy `.env.example` to `.env` and fill in your credentials:
```bash
cp .env.example .env
```
Edit `.env`:
```env
DATABASE_URL=postgresql://postgres:your_password@db.xxxx.supabase.co:6543/postgres
GROQ_API_KEY=gsk_your_groq_key_here
API_BASE_URL=http://localhost:8000
HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

### 5. Seed Database & Ingest Supplier Documents
```bash
# Seed product catalog & sales history
python scripts/seed_data.py

# Ingest sample supplier contracts, SLAs & policies into vector database
python scripts/ingest_docs.py data/supplier_docs/
```

### 6. Running the Application

```bash
# Terminal 1 — Start Backend API (Port 8000)
python scripts/run_api.py

# Terminal 2 — Start Streamlit Dashboard (Port 8501)
python scripts/run_dashboard.py
```

Open **http://localhost:8501** in your browser.

---

## 🧪 Running Tests

```bash
# Run complete test suite (80/80 passing tests across 5 modules)
pytest tests/ -v
```

---

## 🚀 Deployment (Render.com)

The project includes a ready-to-use `render.yaml` specification for Render deployment:

1. Connect your GitHub repository to Render.
2. Select **New Blueprint Instance**.
3. Render automatically provisions:
   - **FastAPI Backend Web Service** (`supplypilot-api`)
   - **Streamlit Dashboard Web Service** (`supplypilot-dashboard`)
4. Supply your `DATABASE_URL` and `GROQ_API_KEY` under Environment Variables in the Render dashboard.

---

## 📜 License

MIT License. Author: Afzal Khan.
