"""
Inventory status routes.

Endpoints
---------
GET /inventory/scan             — Full fleet inventory scan (all products).
GET /inventory/{product_id}     — Single-product inventory status and recommendation.
"""

import logging

from fastapi import APIRouter, HTTPException

from api.schemas import (
    InventoryScanCounts,
    InventoryScanItem,
    InventoryScanResponse,
    InventoryStatusResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/inventory", tags=["Inventory"])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/scan",
    response_model=InventoryScanResponse,
    summary="Scan inventory health for all products",
)
def scan_all():
    """
    Evaluate every product and return a risk-ranked summary.

    Products are ordered CRITICAL → WARNING → OK, then by days_of_cover
    ascending within each group.  This is the primary endpoint for the
    dashboard's overview table.
    """
    from inventory.calculator import get_inventory_recommendation
    from database.db import engine
    from sqlalchemy import text

    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT product_id FROM products ORDER BY product_id")
            ).fetchall()
        product_ids = [r[0] for r in rows]
    except Exception as exc:
        logger.exception("scan_all: failed to load product list")
        raise HTTPException(status_code=500, detail=str(exc))

    results = []
    errors = []

    for pid in product_ids:
        try:
            rec = get_inventory_recommendation(pid)
            results.append(
                InventoryScanItem(
                    product_id=pid,
                    current_stock=rec["current_stock"],
                    days_of_cover=rec["days_of_cover"],
                    reorder_point=rec["reorder_point"],
                    eoq=rec["eoq"],
                    risk_level=rec["risk_level"],
                    action=rec["action"],
                )
            )
        except Exception:
            logger.warning("scan_all: product %d failed", pid)
            errors.append(pid)

    _order = {"CRITICAL": 0, "WARNING": 1, "OK": 2}
    results.sort(key=lambda r: (_order.get(r.risk_level, 9), r.days_of_cover))

    counts = InventoryScanCounts(
        CRITICAL=sum(1 for r in results if r.risk_level == "CRITICAL"),
        WARNING=sum(1 for r in results if r.risk_level == "WARNING"),
        OK=sum(1 for r in results if r.risk_level == "OK"),
    )

    return InventoryScanResponse(
        summary=results,
        counts=counts,
        scanned=len(results),
        errors=errors,
    )


@router.get(
    "/{product_id}",
    response_model=InventoryStatusResponse,
    summary="Get inventory status for a single product",
)
def get_inventory(product_id: int):
    """
    Return full inventory recommendation for one product: current stock,
    reorder point, EOQ, safety stock, days of cover, risk label, and
    the recommended action sentence.
    """
    from inventory.calculator import get_inventory_recommendation

    try:
        rec = get_inventory_recommendation(product_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("get_inventory: product %d failed", product_id)
        raise HTTPException(status_code=500, detail=str(exc))

    return InventoryStatusResponse(**rec)
