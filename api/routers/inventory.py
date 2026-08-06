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
    Optimized bulk query for sub-50ms execution.
    """
    from database.db import engine
    from sqlalchemy import text
    from inventory.calculator import (
        calculate_safety_stock,
        calculate_reorder_point,
        calculate_eoq,
        DEFAULT_ORDER_COST,
        DEFAULT_HOLDING_COST_RATE,
        _CRITICAL_DAYS_OF_COVER,
        _WARNING_DAYS_OF_COVER,
    )

    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT 
                        i.product_id,
                        i.current_stock,
                        i.lead_time_days,
                        i.unit_cost,
                        i.supplier_name,
                        COALESCE(s.avg_demand, 0.0) AS avg_demand,
                        COALESCE(s.demand_std, 0.0) AS demand_std
                    FROM inventory i
                    LEFT JOIN (
                        SELECT product_id, AVG(sales)::float AS avg_demand, STDDEV(sales)::float AS demand_std
                        FROM sales_history
                        GROUP BY product_id
                    ) s ON i.product_id = s.product_id
                    ORDER BY i.product_id;
                """)
            ).fetchall()
    except Exception as exc:
        logger.exception("scan_all: failed to execute bulk inventory query")
        raise HTTPException(status_code=500, detail=str(exc))

    results = []
    errors = []

    for r in rows:
        pid = int(r[0])
        on_hand = float(r[1])
        lead_time = int(r[2])
        unit_cost = float(r[3])
        supplier = str(r[4])
        avg_demand = float(r[5])
        demand_std = float(r[6])

        try:
            ss = calculate_safety_stock(demand_std, lead_time, 0.95)
            rop = calculate_reorder_point(avg_demand, lead_time, demand_std, 0.95)
            eoq = calculate_eoq(avg_demand * 365, DEFAULT_ORDER_COST, unit_cost, DEFAULT_HOLDING_COST_RATE)

            days_of_cover = (on_hand / avg_demand) if avg_demand > 0 else float("inf")

            if days_of_cover < _CRITICAL_DAYS_OF_COVER or on_hand <= 0:
                risk_level = "CRITICAL"
                action = (
                    f"Immediate replenishment required. Place an emergency order of {eoq} units now. "
                    f"Current stock ({on_hand:.1f} units) covers only {days_of_cover:.1f} day(s) of demand."
                )
            elif on_hand <= rop or days_of_cover < _WARNING_DAYS_OF_COVER:
                risk_level = "WARNING"
                action = (
                    f"Stock is below the reorder point ({rop} units). Place an order of {eoq} units with supplier {supplier}. "
                    f"Current stock ({on_hand:.1f} units) covers {days_of_cover:.1f} day(s) of demand."
                )
            else:
                risk_level = "OK"
                action = f"Stock is healthy ({on_hand:.1f} units, {days_of_cover:.1f} days of cover). Reorder when stock falls to {rop} units."

            results.append(
                InventoryScanItem(
                    product_id=pid,
                    current_stock=on_hand,
                    days_of_cover=days_of_cover,
                    reorder_point=rop,
                    eoq=eoq,
                    risk_level=risk_level,
                    action=action,
                )
            )
        except Exception:
            logger.warning("scan_all: processing product %d failed", pid)
            errors.append(pid)

    _order = {"CRITICAL": 0, "WARNING": 1, "OK": 2}
    results.sort(key=lambda item: (_order.get(item.risk_level, 9), item.days_of_cover))

    counts = InventoryScanCounts(
        CRITICAL=sum(1 for item in results if item.risk_level == "CRITICAL"),
        WARNING=sum(1 for item in results if item.risk_level == "WARNING"),
        OK=sum(1 for item in results if item.risk_level == "OK"),
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
