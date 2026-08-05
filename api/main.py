"""
SupplyPilot FastAPI application entry point.

Start the server
----------------
    uvicorn api.main:app --reload --port 8000

Or via the helper script:
    python scripts/run_api.py

API documentation
-----------------
Once running, open http://localhost:8000/docs for the interactive Swagger UI
or http://localhost:8000/redoc for the ReDoc layout.

Architecture notes
------------------
- All business logic lives in the domain modules (forecasting/, inventory/,
  agent/).  This file only wires routes and handles cross-cutting concerns
  (CORS, lifespan, error handlers).
- The DB engine is initialised lazily inside each domain module; the startup
  probe just runs a lightweight ping to fail fast on misconfiguration.
- CORS is permissive for development.  Restrict ``allow_origins`` for
  production deployment.
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown events
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run a DB ping on startup so misconfiguration surfaces immediately."""
    from database.db import engine

    logger.info("SupplyPilot API starting up...")
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection verified.")
    except Exception as exc:
        logger.error("Database connection failed on startup: %s", exc)
        # Not a hard abort — the API can still serve non-DB endpoints.

    yield  # Application runs here.

    logger.info("SupplyPilot API shutting down.")


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SupplyPilot API",
    description=(
        "REST API for the SupplyPilot supply-chain optimization system. "
        "Provides demand forecasting, inventory analysis, purchase order "
        "management, and AI-powered agent chat."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

_ALLOWED_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:8501,http://localhost:3000,http://127.0.0.1:8501",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global error handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred."},
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    tags=["Health"],
    summary="API health check",
    response_model=dict,
)
def health():
    """
    Lightweight health probe.

    Returns ``{"status": "ok", "db_connected": true/false, "version": "1.0.0"}``.
    Safe to poll from a load balancer or monitoring system.
    """
    from database.db import engine

    db_ok = False
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    return {"status": "ok", "db_connected": db_ok, "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

from api.routers.agent import agent_router, alerts_router  # noqa: E402
from api.routers.documents import router as documents_router  # noqa: E402
from api.routers.inventory import router as inventory_router  # noqa: E402
from api.routers.orders import router as orders_router  # noqa: E402
from api.routers.products import router as products_router  # noqa: E402

app.include_router(products_router)
app.include_router(inventory_router)
app.include_router(orders_router)
app.include_router(agent_router)
app.include_router(alerts_router)
app.include_router(documents_router)

