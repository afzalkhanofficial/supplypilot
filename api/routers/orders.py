"""
Purchase order routes.

Endpoints
---------
GET  /orders                    — List purchase orders (with status filter).
POST /orders                    — Create a new pending purchase order.
GET  /orders/{order_id}         — Get a single order.
PATCH /orders/{order_id}/status — Approve or reject an order.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text

from api.schemas import (
    CreateOrderRequest,
    CreateOrderResponse,
    OrderListResponse,
    OrderStatusUpdate,
    PurchaseOrderOut,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/orders", tags=["Purchase Orders"])


def _engine():
    from database.db import engine
    return engine


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("", response_model=OrderListResponse, summary="List purchase orders")
def list_orders(
    status: Optional[str] = Query(
        default=None,
        description="Filter by status: 'pending', 'approved', or 'rejected'.",
        pattern="^(pending|approved|rejected)$",
    ),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """Return purchase orders, newest first. Optionally filter by status."""
    base_sql = """
        SELECT id, product_id, quantity, supplier_name,
               estimated_cost, status, agent_reasoning,
               created_at, decided_at
        FROM   purchase_orders
    """
    count_sql = "SELECT COUNT(*) FROM purchase_orders"

    params: dict = {"lim": limit, "off": offset}
    where = ""
    if status:
        where = " WHERE status = :status"
        params["status"] = status

    with _engine().connect() as conn:
        total = conn.execute(text(count_sql + where), params).scalar()
        rows = conn.execute(
            text(base_sql + where + " ORDER BY created_at DESC LIMIT :lim OFFSET :off"),
            params,
        ).fetchall()

    orders = [
        PurchaseOrderOut(
            id=r[0],
            product_id=r[1],
            quantity=r[2],
            supplier_name=r[3],
            estimated_cost=float(r[4]),
            status=r[5],
            agent_reasoning=r[6],
            created_at=r[7],
            decided_at=r[8],
        )
        for r in rows
    ]
    return OrderListResponse(orders=orders, total=int(total))


@router.post(
    "",
    response_model=CreateOrderResponse,
    status_code=201,
    summary="Create a new purchase order",
)
def create_order(body: CreateOrderRequest):
    """
    Insert a new purchase order with status='pending'.

    The order is NOT dispatched to the supplier — it waits for human
    approval via PATCH /orders/{id}/status.
    """
    with _engine().connect() as conn:
        inv_row = conn.execute(
            text("""
                SELECT unit_cost, supplier_name
                FROM   inventory
                WHERE  product_id = :pid
            """),
            {"pid": body.product_id},
        ).fetchone()

    if inv_row is None:
        raise HTTPException(
            status_code=404,
            detail=f"No inventory record for product_id={body.product_id}.",
        )

    unit_cost = float(inv_row[0])
    supplier_name = str(inv_row[1])
    estimated_cost = round(body.quantity * unit_cost, 2)

    with _engine().begin() as conn:
        result = conn.execute(
            text("""
                INSERT INTO purchase_orders
                    (product_id, quantity, supplier_name,
                     estimated_cost, status, agent_reasoning, created_at)
                VALUES
                    (:pid, :qty, :supplier, :cost, 'pending', :reason, NOW())
                RETURNING id
            """),
            {
                "pid": body.product_id,
                "qty": body.quantity,
                "supplier": supplier_name,
                "cost": estimated_cost,
                "reason": body.reason,
            },
        )
        order_id = result.fetchone()[0]

    return CreateOrderResponse(
        order_id=order_id,
        product_id=body.product_id,
        quantity=body.quantity,
        supplier_name=supplier_name,
        estimated_cost=estimated_cost,
        status="pending",
        message=(
            f"Purchase order #{order_id} created for {body.quantity} units of "
            f"product {body.product_id} from {supplier_name} "
            f"(est. ${estimated_cost:,.2f}). Awaiting approval."
        ),
    )


@router.get(
    "/{order_id}",
    response_model=PurchaseOrderOut,
    summary="Get a single purchase order",
)
def get_order(order_id: int):
    """Return one purchase order by its ID."""
    with _engine().connect() as conn:
        row = conn.execute(
            text("""
                SELECT id, product_id, quantity, supplier_name,
                       estimated_cost, status, agent_reasoning,
                       created_at, decided_at
                FROM   purchase_orders
                WHERE  id = :oid
            """),
            {"oid": order_id},
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found.")

    return PurchaseOrderOut(
        id=row[0],
        product_id=row[1],
        quantity=row[2],
        supplier_name=row[3],
        estimated_cost=float(row[4]),
        status=row[5],
        agent_reasoning=row[6],
        created_at=row[7],
        decided_at=row[8],
    )


@router.patch(
    "/{order_id}/status",
    response_model=PurchaseOrderOut,
    summary="Approve or reject a purchase order",
)
def update_order_status(order_id: int, body: OrderStatusUpdate):
    """
    Transition a pending order to 'approved' or 'rejected'.

    Sets decided_at to the current timestamp.  Only orders in 'pending'
    status can be transitioned; others return 409 Conflict.
    """
    with _engine().connect() as conn:
        current = conn.execute(
            text("SELECT status FROM purchase_orders WHERE id = :oid"),
            {"oid": order_id},
        ).fetchone()

    if current is None:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found.")

    if current[0] != "pending":
        raise HTTPException(
            status_code=409,
            detail=f"Order {order_id} is already '{current[0]}' and cannot be changed.",
        )

    with _engine().begin() as conn:
        conn.execute(
            text("""
                UPDATE purchase_orders
                SET    status = :status, decided_at = NOW()
                WHERE  id = :oid
            """),
            {"status": body.status, "oid": order_id},
        )

    return get_order(order_id)
