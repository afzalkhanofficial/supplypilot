"""
HTTP client for the SupplyPilot FastAPI backend.

All dashboard pages import from this module — it is the single place where
the base URL, timeout, and error handling are configured.  Every method
returns parsed Python objects (dicts / lists) or raises ``APIError`` so
that pages never have to parse responses themselves.
"""

from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
_TIMEOUT = 120  # seconds — agent calls can be slow


class APIError(Exception):
    """Raised when the backend returns an error or is unreachable."""
    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


def _get(path: str, params: dict | None = None) -> Any:
    try:
        r = requests.get(f"{_BASE_URL}{path}", params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        raise APIError("Cannot reach the API server. Is it running on port 8000?")
    except requests.exceptions.HTTPError as exc:
        detail = exc.response.json().get("detail", str(exc))
        raise APIError(detail, status_code=exc.response.status_code)


def _post(path: str, body: dict) -> Any:
    try:
        r = requests.post(f"{_BASE_URL}{path}", json=body, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        raise APIError("Cannot reach the API server. Is it running on port 8000?")
    except requests.exceptions.HTTPError as exc:
        detail = exc.response.json().get("detail", str(exc))
        raise APIError(detail, status_code=exc.response.status_code)


def _patch(path: str, body: dict) -> Any:
    try:
        r = requests.patch(f"{_BASE_URL}{path}", json=body, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        raise APIError("Cannot reach the API server. Is it running on port 8000?")
    except requests.exceptions.HTTPError as exc:
        detail = exc.response.json().get("detail", str(exc))
        raise APIError(detail, status_code=exc.response.status_code)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def health() -> dict:
    return _get("/health")


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------

def list_products() -> list[dict]:
    return _get("/products")["products"]


def get_product(product_id: int) -> dict:
    return _get(f"/products/{product_id}")


def get_forecast(product_id: int, days_ahead: int = 14) -> dict:
    return _get(f"/products/{product_id}/forecast", params={"days_ahead": days_ahead})


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

def get_inventory(product_id: int) -> dict:
    return _get(f"/inventory/{product_id}")


def scan_inventory() -> dict:
    return _get("/inventory/scan")


# ---------------------------------------------------------------------------
# Purchase Orders
# ---------------------------------------------------------------------------

def list_orders(status: str | None = None, limit: int = 100) -> dict:
    params: dict = {"limit": limit}
    if status:
        params["status"] = status
    return _get("/orders", params=params)


def create_order(product_id: int, quantity: int, reason: str) -> dict:
    return _post("/orders", {"product_id": product_id, "quantity": quantity, "reason": reason})


def approve_order(order_id: int) -> dict:
    return _patch(f"/orders/{order_id}/status", {"status": "approved"})


def reject_order(order_id: int) -> dict:
    return _patch(f"/orders/{order_id}/status", {"status": "rejected"})


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

def agent_chat(question: str, chat_history: list[dict] | None = None) -> dict:
    return _post("/agent/chat", {
        "question": question,
        "chat_history": chat_history or [],
    })


def agent_history(limit: int = 20) -> dict:
    return _get("/agent/history", params={"limit": limit})


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

def list_alerts(limit: int = 20) -> dict:
    return _get("/alerts", params={"limit": limit})
