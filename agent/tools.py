"""
LangChain tools that the SupplyPilot agent can call.

Each tool is a thin, well-documented wrapper around either:
  - ``forecasting.predict.get_forecast``  (demand forecasting)
  - ``inventory.calculator.get_inventory_recommendation``  (inventory math)
  - Direct SQL queries  (purchase orders, risk alerts, product list)

Design rules
------------
- Every tool returns a plain string.  JSON strings are used for structured
  data; the agent can parse and quote specific fields in its response.
- Tools never raise exceptions that reach the agent.  All errors are caught
  and returned as a descriptive error string so the agent can report them
  gracefully.
- Tools are stateless: they read from the DB or the model files on each call.
  No caching is done here — the model cache in predict.py handles that.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from langchain_core.tools import tool
from sqlalchemy import text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports — resolved at call time to avoid circular import issues and
# to keep module-level load time fast.
# ---------------------------------------------------------------------------

def _engine():
    from database.db import engine
    return engine


def _get_forecast(product_id: int, days_ahead: int):
    from forecasting.predict import get_forecast
    return get_forecast(product_id, days_ahead)


def _get_recommendation(product_id: int):
    from inventory.calculator import get_inventory_recommendation
    return get_inventory_recommendation(product_id)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@tool
def list_products(query: str = "") -> str:
    """
    List all products tracked in the system.

    Returns a JSON array where each entry contains product_id,
    product_name, store_type, and assortment.  Pass any non-empty
    string as query (it is ignored; the argument exists because
    LangChain requires all tools to accept at least one parameter).

    Use this tool first when the user asks about "all products" or
    does not specify a product_id.
    """
    try:
        with _engine().connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT product_id, product_name, store_type, assortment
                    FROM   products
                    ORDER  BY product_id
                """)
            ).fetchall()

        products = [
            {
                "product_id": r[0],
                "product_name": r[1],
                "store_type": r[2],
                "assortment": r[3],
            }
            for r in rows
        ]
        return json.dumps({"products": products, "total": len(products)})
    except Exception as exc:
        logger.exception("list_products failed")
        return json.dumps({"error": str(exc)})


@tool
def get_demand_forecast(product_id: int, days_ahead: int = 14) -> str:
    """
    Return a day-by-day demand forecast for a product using the trained
    Prophet model.

    Parameters
    ----------
    product_id : int
        The product to forecast.  Use list_products() to find valid IDs.
    days_ahead : int
        Number of future days to forecast (1–90).  Defaults to 14.

    Returns a JSON object with fields:
    - product_id, days_ahead, training_end
    - dates: list of ISO date strings
    - yhat: point forecast per day (units, clipped ≥ 0)
    - yhat_lower / yhat_upper: 80% confidence interval
    - total_forecast: sum of yhat across all days

    If the product has no trained model, returns an error message.
    """
    try:
        result = _get_forecast(product_id, days_ahead)
        result["total_forecast"] = round(sum(result["yhat"]), 1)
        return json.dumps(result)
    except Exception as exc:
        logger.exception("get_demand_forecast failed for product %d", product_id)
        return json.dumps({"error": str(exc), "product_id": product_id})


@tool
def get_inventory_status(product_id: int) -> str:
    """
    Return the full inventory status and replenishment recommendation for
    a single product.

    Parameters
    ----------
    product_id : int
        The product to analyse.

    Returns a JSON object with fields:
    - current_stock, avg_daily_demand, daily_demand_std
    - lead_time_days, safety_stock, reorder_point, eoq
    - days_of_cover, forecast_demand_in_lead_time, units_at_risk
    - risk_level: "OK", "WARNING", or "CRITICAL"
    - action: one-sentence human-readable recommendation

    Use this tool whenever the user asks about stock levels, reorder
    points, or whether a product needs ordering.
    """
    try:
        result = _get_recommendation(product_id)
        return json.dumps(result)
    except Exception as exc:
        logger.exception("get_inventory_status failed for product %d", product_id)
        return json.dumps({"error": str(exc), "product_id": product_id})


@tool
def scan_all_inventory(query: str = "") -> str:
    """
    Scan every product in the system and return a risk-ranked inventory
    summary.

    Products are ordered: CRITICAL first, then WARNING, then OK.
    Within each group they are sorted by days_of_cover ascending (most
    urgent first).

    Returns a JSON object with:
    - summary: list of lightweight inventory dicts (no daily forecast detail)
    - counts: {"CRITICAL": N, "WARNING": N, "OK": N}
    - scanned: total number of products evaluated
    - errors: list of product_ids that could not be evaluated

    Use this tool when the user asks "which products need attention?" or
    "show me the overall inventory health".
    """
    from inventory.calculator import get_inventory_recommendation

    try:
        with _engine().connect() as conn:
            rows = conn.execute(
                text("SELECT product_id FROM products ORDER BY product_id")
            ).fetchall()
        product_ids = [r[0] for r in rows]
    except Exception as exc:
        return json.dumps({"error": f"Could not load product list: {exc}"})

    results = []
    errors = []

    for pid in product_ids:
        try:
            rec = get_inventory_recommendation(pid)
            results.append({
                "product_id": pid,
                "current_stock": rec["current_stock"],
                "days_of_cover": rec["days_of_cover"],
                "reorder_point": rec["reorder_point"],
                "eoq": rec["eoq"],
                "risk_level": rec["risk_level"],
                "action": rec["action"],
            })
        except Exception as exc:
            logger.warning("scan_all_inventory: product %d failed — %s", pid, exc)
            errors.append(pid)

    # Sort: CRITICAL < WARNING < OK, then ascending days_of_cover.
    _order = {"CRITICAL": 0, "WARNING": 1, "OK": 2}
    results.sort(key=lambda r: (_order.get(r["risk_level"], 9), r["days_of_cover"]))

    counts = {
        "CRITICAL": sum(1 for r in results if r["risk_level"] == "CRITICAL"),
        "WARNING":  sum(1 for r in results if r["risk_level"] == "WARNING"),
        "OK":       sum(1 for r in results if r["risk_level"] == "OK"),
    }

    return json.dumps({
        "summary": results,
        "counts": counts,
        "scanned": len(results),
        "errors": errors,
    })


@tool
def create_purchase_order(product_id: int, quantity: int, reason: str) -> str:
    """
    Create a new purchase order for a product with status='pending'.

    The order will appear in the dashboard for a human to approve or
    reject.  It is NOT dispatched to the supplier automatically.

    Parameters
    ----------
    product_id : int
        The product to order.
    quantity : int
        Number of units to order.  Must be > 0.
    reason : str
        Brief explanation of why the order is being placed (e.g.
        "Stock below reorder point, 7 days of cover remaining").
        This is stored in agent_reasoning for audit purposes.

    Returns a JSON object with:
    - order_id: the newly created purchase order ID
    - product_id, quantity, estimated_cost, supplier_name, status
    - message: confirmation string

    Only call this tool after confirming the product_id and quantity
    with the user.
    """
    try:
        # Look up unit cost and supplier for the product.
        with _engine().connect() as conn:
            row = conn.execute(
                text("""
                    SELECT i.unit_cost, i.supplier_name
                    FROM   inventory i
                    WHERE  i.product_id = :pid
                """),
                {"pid": product_id},
            ).fetchone()

        if row is None:
            return json.dumps({"error": f"No inventory record for product_id={product_id}."})

        unit_cost = float(row[0])
        supplier_name = str(row[1])
        estimated_cost = round(quantity * unit_cost, 2)

        with _engine().begin() as conn:
            result = conn.execute(
                text("""
                    INSERT INTO purchase_orders
                        (product_id, quantity, supplier_name,
                         estimated_cost, status, agent_reasoning, created_at)
                    VALUES
                        (:pid, :qty, :supplier,
                         :cost, 'pending', :reason, NOW())
                    RETURNING id
                """),
                {
                    "pid": product_id,
                    "qty": quantity,
                    "supplier": supplier_name,
                    "cost": estimated_cost,
                    "reason": reason,
                },
            )
            order_id = result.fetchone()[0]

        return json.dumps({
            "order_id": order_id,
            "product_id": product_id,
            "quantity": quantity,
            "supplier_name": supplier_name,
            "estimated_cost": estimated_cost,
            "status": "pending",
            "message": (
                f"Purchase order #{order_id} created for {quantity} units of "
                f"product {product_id} from {supplier_name} "
                f"(est. ${estimated_cost:,.2f}). Awaiting human approval."
            ),
        })
    except Exception as exc:
        logger.exception("create_purchase_order failed for product %d", product_id)
        return json.dumps({"error": str(exc), "product_id": product_id})


@tool
def get_recent_risk_alerts(limit: int = 20) -> str:
    """
    Retrieve the most recent risk alerts stored in the database.

    Parameters
    ----------
    limit : int
        Maximum number of alerts to return (default 20, max 100).

    Returns a JSON object with:
    - alerts: list of {id, product_id, alert_type, message, severity, created_at}
    - total: number of alerts returned

    Use this tool when the user asks "any alerts?" or "what's the latest
    risk summary?".
    """
    cap = min(int(limit), 100)
    try:
        with _engine().connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT id, product_id, alert_type, message, severity,
                           created_at
                    FROM   risk_alerts
                    ORDER  BY created_at DESC
                    LIMIT  :lim
                """),
                {"lim": cap},
            ).fetchall()

        alerts = [
            {
                "id": r[0],
                "product_id": r[1],
                "alert_type": r[2],
                "message": r[3],
                "severity": r[4],
                "created_at": r[5].isoformat() if r[5] else None,
            }
            for r in rows
        ]
        return json.dumps({"alerts": alerts, "total": len(alerts)})
    except Exception as exc:
        logger.exception("get_recent_risk_alerts failed")
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Exported tool list
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    list_products,
    get_demand_forecast,
    get_inventory_status,
    scan_all_inventory,
    create_purchase_order,
    get_recent_risk_alerts,
]
