# 🚀 SupplyPilot — AI Supply Chain Optimization System

SupplyPilot is an end-to-end, production-ready AI supply chain optimization platform. It combines **time-series demand forecasting** (Facebook Prophet), **classical inventory optimization math** (EOQ, Safety Stock, Reorder Point), a **LangChain tool-calling AI agent** powered by Groq (`llama-3.3-70b-versatile`), a **FastAPI REST backend**, and a modern **Streamlit interactive dashboard**.

---

## 🌟 Key Features

- **Multi-Product Demand Forecasting**: Trained Facebook Prophet models for 20 retail products evaluating against a 42-day holdout set. Outperforms naive baseline by **-61.7% MAE**, **-65.1% RMSE**, and **-63.2% MAPE**.
- **Deterministic Inventory Optimization**:
  - **EOQ (Economic Order Quantity)**: Wilson's formula balancing order setup costs vs holding costs.
  - **Safety Stock**: Calibrated to target service level ($Z$-score table lookup).
  - **Reorder Point (ROP)**: Dynamic demand-during-lead-time thresholding.
- **Autonomous Tool-Calling Agent**: LangChain agent equipped with 8 custom tools (`list_products`, `get_demand_forecast`, `get_inventory_status`, `scan_all_inventory`, `create_purchase_order`, `get_recent_risk_alerts`, `check_weather_risk`, `check_supplier_news_risk`).
- **RESTful FastAPI Service**: Clean OpenAPI specs with CORS, DB lifespan probes, purchase order approval workflows that update live inventory stock, and input validation.
- **Interactive Streamlit Dashboard**: Dark glassmorphism UI with 5 main pages:
  1. 📊 **Overview**: Fleet KPIs & risk summary table.
  2. 📦 **Inventory**: Per-product stock analysis & gauge metrics.
  3. 📈 **Demand Forecast**: Prophet forecast visualization with 80% CI band.
  4. 🛒 **Purchase Orders**: Human-in-the-loop approval workflow (automatically restocking inventory on approval).
  5. 🤖 **Agent Chat**: Multi-turn conversational AI interface with tool audit logs.

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
    end

    Agent <--> Tools[8 Agent Tools]
    Tools <--> Inv
    Tools <--> Forecaster
    Tools <--> External[Weather & News RSS APIs]
    Tools <--> DB[(PostgreSQL Database)]
    API <--> DB
```

---

## 📂 Project Structure

```
supplypilot/
├── agent/                  # LangChain AI agent, tools, and prompts
│   ├── agent.py            # Agent executor runner & DB interaction logger
│   ├── prompts.py          # System prompt & ChatPromptTemplate
│   └── tools.py            # 6 LangChain @tool wrappers
├── api/                    # FastAPI REST API backend
│   ├── main.py             # FastAPI app, CORS, lifespan probe & error handlers
│   ├── schemas.py          # Pydantic request/response models
│   └── routers/            # API endpoint routers
│       ├── agent.py        # /agent/chat, /agent/history, /alerts
│       ├── inventory.py    # /inventory/scan, /inventory/{id}
│       ├── orders.py       # /orders (list, create, approve/reject)
│       └── products.py     # /products, /products/{id}/forecast
├── dashboard/              # Streamlit frontend dashboard
│   ├── app.py              # 5-page Streamlit web app with custom dark CSS
│   └── api_client.py       # Requests-based HTTP client for API backend
├── database/               # PostgreSQL schema & SQLAlchemy ORM
│   ├── db.py               # Engine & session setup
│   ├── models.py           # SQLAlchemy tables
│   └── schema.sql          # DDL script
├── forecasting/            # Time-series forecasting pipeline
│   ├── models/prophet/     # 20 trained Prophet model JSON artifacts
│   ├── evaluate.py         # Evaluation pipeline
│   ├── predict.py          # Read-only Prophet forecast interface
│   ├── train_naive_baseline.py # Historical average baseline
│   └── train_prophet.py    # Prophet training script
├── inventory/              # Classical inventory mathematics
│   └── calculator.py       # EOQ, Safety Stock, ROP & recommendation logic
├── scripts/                # Helper scripts & CLI runners
│   ├── run_api.py          # FastAPI launcher
│   ├── run_dashboard.py    # Streamlit launcher
│   ├── seed_db.py          # Rossmann dataset seed script
│   └── test_agent.py       # Agent verification test
├── tests/                  # Integration & unit test suite (pytest)
│   ├── conftest.py         # Session TestClient fixture
│   ├── test_health.py      # Health endpoint tests
│   ├── test_inventory.py   # Inventory math & endpoint tests
│   ├── test_orders.py      # Purchase order API tests
│   └── test_products.py    # Product & forecast API tests
├── .env.example            # Environment variable template
├── render.yaml             # Render.com infrastructure as code specification
├── Procfile                # Web process configuration
└── requirements.txt        # Pinned Python dependencies
```

---

## 📊 Forecasting Benchmark Results

Prophet was benchmarked against a 7-day Naive Moving Average Baseline on a 42-day holdout across 20 products:

| Metric | Naive Baseline | Prophet Model | Improvement |
|---|---|---|---|
| **Mean Absolute Error (MAE)** | 3,117.7 units | **1,192.5 units** | **-61.7%** |
| **Root Mean Squared Error (RMSE)** | 3,678.9 units | **1,283.4 units** | **-65.1%** |
| **Mean Absolute Percentage Error (MAPE)** | 48.7% | **17.9%** | **-63.2%** |

---

## ⚡ Quickstart & Local Setup

### 1. Prerequisites
- Python 3.11+
- PostgreSQL database
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

### 3. Environment Configuration
Copy `.env.example` to `.env` and fill in your credentials:
```bash
cp .env.example .env
```
Edit `.env`:
```env
DATABASE_URL=postgresql://user:password@localhost:5432/supplypilot
GROQ_API_KEY=gsk_your_groq_key_here
API_BASE_URL=http://localhost:8000
```

### 4. Seed Database
```bash
python scripts/seed_db.py
```

### 5. Running the Application

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
pytest tests/ -v
```

---

## 🚀 Deployment (Render.com)

The project includes a ready-to-use `render.yaml` for automatic deployment on Render:

1. Connect your GitHub repository to Render.
2. Select **New Blueprint Instance**.
3. Render automatically provisions:
   - PostgreSQL Managed Database
   - FastAPI Backend Service
   - Streamlit Dashboard Service
4. Supply your `GROQ_API_KEY` under Environment Variables in the Render dashboard.

---

## 📜 License

MIT License. Designed and built with ❤️ by the Deepmind team.
