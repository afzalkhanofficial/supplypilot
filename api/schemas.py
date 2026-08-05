"""
Pydantic request and response schemas for the SupplyPilot REST API.

Keeping all schemas in one module makes them easy to import from any
router and ensures the OpenAPI documentation stays consistent.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' when the service is healthy.")
    db_connected: bool = Field(..., description="True if a DB ping succeeded.")
    version: str = Field(default="1.0.0")


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------

class ProductOut(BaseModel):
    product_id: int
    product_name: str
    store_type: Optional[str] = None
    assortment: Optional[str] = None
    competition_distance: Optional[float] = None


class ProductListResponse(BaseModel):
    products: list[ProductOut]
    total: int


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------

class ForecastResponse(BaseModel):
    product_id: int
    days_ahead: int
    training_end: str = Field(..., description="Last date the model was trained on (ISO-8601).")
    dates: list[str]
    yhat: list[float] = Field(..., description="Point forecast per day (units).")
    yhat_lower: list[float] = Field(..., description="80% confidence lower bound.")
    yhat_upper: list[float] = Field(..., description="80% confidence upper bound.")
    total_forecast: float = Field(..., description="Sum of yhat across all forecast days.")


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

class InventoryStatusResponse(BaseModel):
    product_id: int
    current_stock: float
    avg_daily_demand: float
    daily_demand_std: float
    lead_time_days: int
    safety_stock: int
    reorder_point: int
    eoq: int
    days_of_cover: float
    forecast_demand_in_lead_time: float
    units_at_risk: int
    risk_level: str = Field(..., description="One of 'OK', 'WARNING', 'CRITICAL'.")
    action: str
    service_level: float


class InventoryScanItem(BaseModel):
    product_id: int
    current_stock: float
    days_of_cover: float
    reorder_point: int
    eoq: int
    risk_level: str
    action: str


class InventoryScanCounts(BaseModel):
    CRITICAL: int
    WARNING: int
    OK: int


class InventoryScanResponse(BaseModel):
    summary: list[InventoryScanItem]
    counts: InventoryScanCounts
    scanned: int
    errors: list[int]


# ---------------------------------------------------------------------------
# Purchase Orders
# ---------------------------------------------------------------------------

class CreateOrderRequest(BaseModel):
    product_id: int = Field(..., gt=0)
    quantity: int = Field(..., gt=0, description="Number of units to order.")
    reason: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Brief explanation for placing this order.",
    )


class OrderStatusUpdate(BaseModel):
    status: str = Field(..., pattern="^(approved|rejected)$")


class PurchaseOrderOut(BaseModel):
    id: int
    product_id: int
    quantity: int
    supplier_name: str
    estimated_cost: float
    status: str
    agent_reasoning: Optional[str] = None
    created_at: datetime
    decided_at: Optional[datetime] = None


class OrderListResponse(BaseModel):
    orders: list[PurchaseOrderOut]
    total: int


class CreateOrderResponse(BaseModel):
    order_id: int
    product_id: int
    quantity: int
    supplier_name: str
    estimated_cost: float
    status: str
    message: str


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ChatTurn(BaseModel):
    human: str
    ai: str


class AgentChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    chat_history: list[ChatTurn] = Field(
        default_factory=list,
        description="Prior conversation turns for multi-turn context.",
    )


class AgentChatResponse(BaseModel):
    answer: str
    tools_used: list[str]
    steps: int


class AgentInteractionOut(BaseModel):
    id: int
    user_question: str
    agent_answer: str
    tools_used: Optional[str] = None
    created_at: datetime


class AgentHistoryResponse(BaseModel):
    interactions: list[AgentInteractionOut]
    total: int


# ---------------------------------------------------------------------------
# Risk Alerts
# ---------------------------------------------------------------------------

class RiskAlertOut(BaseModel):
    id: int
    product_id: Optional[int] = None
    alert_type: str
    message: str
    severity: str
    created_at: datetime


class RiskAlertListResponse(BaseModel):
    alerts: list[RiskAlertOut]
    total: int


# ---------------------------------------------------------------------------
# Documents / RAG (Phase 8)
# ---------------------------------------------------------------------------

class DocumentIngestResponse(BaseModel):
    status: str
    document_id: Optional[int] = None
    filename: Optional[str] = None
    supplier_name: Optional[str] = None
    doc_type: Optional[str] = None
    chunks_stored: Optional[int] = None
    page_count: Optional[int] = None
    message: Optional[str] = None
    sha256_hex: Optional[str] = None


class SearchResultChunk(BaseModel):
    rank: int
    similarity: float
    chunk_text: str
    document_id: int
    filename: str
    supplier_name: str
    doc_type: str
    chunk_index: int


class DocumentSearchResponse(BaseModel):
    status: str
    query: str
    results: Optional[list[SearchResultChunk]] = None
    message: Optional[str] = None


class DocumentInfo(BaseModel):
    id: int
    filename: str
    supplier_name: str
    doc_type: str
    page_count: Optional[int] = None
    uploaded_at: Optional[str] = None


class DocumentListResponse(BaseModel):
    status: str
    count: int
    documents: list[DocumentInfo]

