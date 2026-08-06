<div align="center">

# 🚀 SupplyPilot — AI Supply Chain Optimization System

An end-to-end supply chain decision-support platform combining **time-series demand forecasting** (Facebook Prophet), **deterministic inventory optimization** (EOQ, Safety Stock, Reorder Point), **supplier document intelligence** (RAG via `pgvector` and `sentence-transformers`), an **autonomous tool-calling AI agent** (LangChain + Groq `llama-3.3-70b-versatile`), and a **Unified FastAPI Web Application**.

</div>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-CSS3--JS-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU--build-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)

</div>

---

## 👤 Author

<div align="center">

| <img src="https://github.com/afzalkhanofficial.png" width="100px" style="border-radius: 50%;" /><br><sub><b>Afzal Khan</b></sub> |
| :---: |

</div>

---

## 📌 About This Project

SupplyPilot was built to design and evaluate an end-to-end AI system that helps operations managers make data-backed inventory decisions. Instead of relying on an LLM to guess stock numbers or forecast demand from memory, SupplyPilot decouples deterministic math and time-series modeling into specialized tools, using the LLM strictly as an orchestrator and natural-language interface.

**Key Design & Modeling Choices:**
- **Dataset & Domain Mapping**: Uses real retail sales data from the Rossmann Store Sales dataset. To model a multi-product retail inventory system, 20 distinct store IDs are mapped as 20 separate products in the supply chain.
- **Measured Forecasting Performance**: Prophet models were trained per product on historical open-day sales and evaluated against a 42-day holdout set against a 1-day lag Naive Baseline.
- **Grounded AI Reasoning**: The agent is equipped with 10 deterministic tools. Every stock query, order creation, or contract lookup executes real backend logic, returning JSON payloads that feed the agent's chain-of-thought.
- **Local & Open-Weight Embeddings**: Supplier document RAG uses `sentence-transformers` (`all-MiniLM-L6-v2`) running locally on CPU with `pgvector` in PostgreSQL — requiring zero external embedding API keys.

---

## 🔑 Key Features

- **Multi-Product Demand Forecasting**: 20 trained Prophet models incorporating weekly/yearly seasonality and promotional events, outperforming the naive baseline by **-61.7% MAE**, **-65.1% RMSE**, and **-63.2% MAPE**.
- **Deterministic Inventory Mathematics**:
  - **EOQ (Economic Order Quantity)**: Wilson's formula balancing order setup costs vs holding costs.
  - **Safety Stock**: Buffer stock calibrated to a 95% target service level using standard normal distribution $Z$-score lookups.
  - **Reorder Point (ROP)**: Dynamic threshold triggering replenishment orders when stock falls below demand-during-lead-time + safety stock.
- **Supplier Document Intelligence (RAG)**:
  - Vector similarity search over supplier contracts, SLAs, and policies stored in PostgreSQL using `pgvector` (`VECTOR(384)` with IVFFlat index).
  - Character-window chunking (1,000 characters / 200 overlap) snapped to word boundaries.
  - Plain text and PDF text extraction (`pypdf`) with SHA-256 deduplication.
- **Autonomous Tool-Calling Agent**: LangChain agent equipped with 10 custom tools (`list_products`, `get_demand_forecast`, `get_inventory_status`, `scan_all_inventory`, `create_purchase_order`, `get_recent_risk_alerts`, `check_weather_risk`, `check_supplier_news_risk`, `search_supplier_docs`, `list_supplier_documents`).
- **Human-in-the-Loop Purchase Orders**: Orders created by the agent enter a `pending` state and require explicit human approval via the web interface before updating live inventory stock.
- **Unified FastAPI Single-Page Web App**: High-performance dark glassmorphism interface across 6 dedicated pages: Overview, Inventory, Demand Forecast, Purchase Orders, Agent Chat, and Supplier Intelligence.

---

## 🛠️ Technology Stack

- **Backend Framework**: FastAPI (Uvicorn, Pydantic v2, CORS middleware, StaticFiles hosting)
- **Frontend**: Single-Page Web Application (HTML5, Vanilla CSS3 glassmorphism, ES6 JavaScript, Chart.js)
- **Database & Vector Store**: Supabase PostgreSQL with `pgvector` extension (`psycopg2-binary`, SQLAlchemy 2.0 ORM)
- **Time-Series Forecasting**: Facebook Prophet (`prophet==1.1.6`), `scikit-learn`
- **Vector Embeddings & RAG**: `sentence-transformers` (`all-MiniLM-L6-v2`), PyTorch (CPU build), `pypdf`
- **AI Agent Stack**: LangChain (`langchain-core`, `langchain-community`), Groq API (`llama-3.3-70b-versatile`)
- **Testing**: `pytest` (80 unit & integration tests)

---

## 🔄 Workflow & Architecture

```mermaid
flowchart TD
    User([User / Operations Manager]) <--> WebApp[Unified FastAPI Web Application\n:8000 / Single Render Service]

    subgraph Backend & Engines
        WebApp <--> Agent[LangChain Tool-Calling Agent\nGroq llama-3.3-70b-versatile]
        WebApp <--> Inv[Inventory Calculator\nEOQ / Safety Stock / ROP]
        WebApp <--> Forecaster[Prophet Predictor\n20 Trained Model Artifacts]
        WebApp <--> RAG[RAG Vector Search\nall-MiniLM-L6-v2 Embeddings]
    end

    Agent <--> Tools[10 Agent Tools]
    Tools <--> Inv
    Tools <--> Forecaster
    Tools <--> RAG
    Tools <--> External[Weather & News RSS APIs]
    Tools <--> DB[(PostgreSQL + pgvector)]
    WebApp <--> DB
```

---

## 📁 Repository Structure

```
supplypilot/
├── agent/                  # LangChain AI agent, tools, and prompts
│   ├── agent.py            # Agent executor runner & DB interaction logger
│   ├── prompts.py          # System prompt & ChatPromptTemplate
│   └── tools.py            # 10 LangChain @tool wrappers
├── api/                    # FastAPI REST API backend & static server
│   ├── main.py             # FastAPI app, static files mount, lifespan probe & error handlers
│   ├── schemas.py          # Pydantic request/response models
│   └── routers/            # API endpoint routers
│       ├── agent.py        # /agent/chat, /agent/history, /alerts
│       ├── documents.py    # /documents/ingest, /documents/search, /documents
│       ├── inventory.py    # /inventory/scan, /inventory/{id}
│       ├── orders.py       # /orders (list, create, approve/reject)
│       └── products.py     # /products, /products/{id}/forecast
├── data/                   # Data directory
│   └── supplier_docs/      # Sample SLAs, contracts, and policy text documents
├── database/               # PostgreSQL schema & SQLAlchemy ORM
│   ├── db.py               # Engine, pgvector adapter, session setup & ORM models
│   └── schema.sql          # PostgreSQL DDL script (tables & IVFFlat index)
├── forecasting/            # Time-series forecasting pipeline
│   ├── models/prophet/     # 20 trained Prophet model JSON artifacts
│   ├── evaluate.py         # Prophet vs baseline evaluation pipeline
│   ├── model_comparison.md # Benchmark results documentation
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
│   ├── ingestor.py         # Text/PDF parsing, SHA-256 dedup, & pgvector insert
│   └── search.py           # Cosine-similarity ANN search & document listing
├── scripts/                # Helper scripts & CLI runners
│   ├── ingest_docs.py      # Bulk document ingestion CLI
│   ├── run_api.py          # FastAPI application launcher
│   ├── seed_data.py        # Rossmann dataset seed script
│   ├── test_agent.py       # Agent verification test
│   └── test_inventory.py   # Inventory math verification script
├── tests/                  # Integration & unit test suite (pytest)
│   ├── conftest.py         # Session TestClient fixture
│   ├── test_health.py      # Health endpoint tests
│   ├── test_inventory.py   # Inventory math & endpoint tests
│   ├── test_orders.py      # Purchase order API tests
│   ├── test_products.py    # Product & forecast API tests
│   └── test_rag.py         # Phase 8 RAG test suite
├── web/                    # Native Single-Page Web Frontend
│   ├── app.js              # Single-page router, fetch API client & Chart.js renderer
│   ├── index.html          # 6-page responsive Single-Page Application layout
│   └── style.css           # Glassmorphism dark design system
├── .env.example            # Environment variable template
├── render.yaml             # Render.com single web service specification
├── Procfile                # Web process configuration
└── requirements.txt        # Pinned Python dependencies
```

---

## 📈 Results: Demand Forecasting Benchmark

Prophet models were evaluated against a 1-day lag Naive Baseline across a 42-day holdout period for 20 retail products:

| Metric | Naive Baseline | Prophet Model | Improvement |
|---|---|---|---|
| **Mean Absolute Error (MAE)** | 2,099.5 units | **803.3 units** | **-61.7%** |
| **Root Mean Squared Error (RMSE)** | 3,032.4 units | **1,058.8 units** | **-65.1%** |
| **Mean Absolute Percentage Error (MAPE)** | 21.4% | **7.9%** | **-63.2%** |

*Evaluated on open-day sales data over the 42-day holdout set.*

---

## 💻 Running This Project

### 1. Prerequisites
- Python 3.11+
- PostgreSQL instance with `pgvector` enabled (e.g. Supabase)
- Groq API Key ([Get one free at console.groq.com](https://console.groq.com))

### 2. Installation
```bash
# Clone repository
git clone https://github.com/afzalkhanofficial/supplypilot.git
cd supplypilot

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install pinned dependencies
pip install -r requirements.txt
```

### 3. Database Vector Setup
In your Supabase SQL Editor:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 4. Environment Setup
Copy `.env.example` to `.env` and fill in credentials:
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

### 5. Seed Database & Ingest Documents
```bash
# Seed product catalog & sales history
python scripts/seed_data.py

# Ingest sample supplier contracts, SLAs & policies into vector database
python scripts/ingest_docs.py data/supplier_docs/
```

### 6. Start Unified Application
```bash
# Start Unified FastAPI Web Application (Port 8000)
python scripts/run_api.py
```

Access the Web Application UI at **http://localhost:8000** and Swagger API docs at **http://localhost:8000/docs**.

---

## 🧪 Automated Test Suite

```bash
# Run all 80 unit & integration tests
pytest tests/ -v
```

---

## 📜 License

This project is distributed under the MIT License — see the [LICENSE](LICENSE) file for details.
