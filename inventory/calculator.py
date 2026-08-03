"""
Inventory optimization calculator.

Implements the three core inventory formulas used by the SupplyPilot agent:

1.  Economic Order Quantity (EOQ) — the order size that minimises the sum of
    ordering cost and holding cost.

2.  Safety Stock — the buffer held to absorb demand uncertainty and lead-time
    variability, calibrated to a target service level.

3.  Reorder Point (ROP) — the on-hand quantity threshold at which a
    replenishment order should be placed.

Each formula accepts plain numeric inputs so it can be unit-tested in
isolation.  The higher-level function `get_inventory_recommendation()` wires
everything together by pulling live data from the database and the trained
Prophet models.

Key design decisions
--------------------
- Holding cost is expressed as a *fraction* of unit cost per year (default
  25 %), which is the standard warehouse convention.
- Order cost defaults to $50 per purchase order, a reasonable assumption for
  a mid-size retailer placing electronic orders.
- Service level → z-score mapping is looked up from a table rather than
  computed via scipy, so this module has no heavy statistical dependencies.
- Lead-time uncertainty is NOT modelled (lead times are treated as fixed)
  because the seed data has a single deterministic lead_time_days per product.
  If variable lead times are added later, extend safety_stock with the
  lead-time variance term: z * sqrt(LT * σ_d² + σ_LT² * d²).

Usage
-----
    from inventory.calculator import get_inventory_recommendation
    rec = get_inventory_recommendation(product_id=85)
    print(rec)
"""

import logging
import math
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_ORDER_COST: float = 50.0        # $ per purchase order (S)
DEFAULT_HOLDING_COST_RATE: float = 0.25  # fraction of unit cost per year (h)
DAYS_PER_YEAR: int = 365

# z-scores for common service levels (one-sided normal).
# Key: service level as a float between 0 and 1.
_Z_TABLE: dict[float, float] = {
    0.80: 0.842,
    0.85: 1.036,
    0.90: 1.282,
    0.95: 1.645,
    0.97: 1.881,
    0.99: 2.326,
    0.999: 3.090,
}

# Stock-status thresholds used to generate the risk label.
_CRITICAL_DAYS_OF_COVER: int = 3   # on-hand stock < 3 days' demand → CRITICAL
_WARNING_DAYS_OF_COVER: int = 7    # on-hand stock < 7 days' demand → WARNING


# ---------------------------------------------------------------------------
# Pure formula functions
# ---------------------------------------------------------------------------

def z_score(service_level: float) -> float:
    """
    Return the one-sided standard-normal z-score for a service level.

    Performs linear interpolation between the two nearest table entries so
    that any service level in [0.80, 0.999] is handled cleanly.

    Parameters
    ----------
    service_level:
        Target fill rate, e.g. 0.95 for 95 %.  Must be in (0, 1).

    Returns
    -------
    float
        The z-score corresponding to the requested service level.

    Raises
    ------
    ValueError
        If service_level is outside the range covered by the look-up table.
    """
    if not (0 < service_level < 1):
        raise ValueError(
            f"service_level must be in (0, 1); got {service_level}."
        )

    keys = sorted(_Z_TABLE)
    if service_level <= keys[0]:
        return _Z_TABLE[keys[0]]
    if service_level >= keys[-1]:
        return _Z_TABLE[keys[-1]]

    # Linear interpolation between bracketing entries.
    for lo, hi in zip(keys, keys[1:]):
        if lo <= service_level <= hi:
            t = (service_level - lo) / (hi - lo)
            return _Z_TABLE[lo] + t * (_Z_TABLE[hi] - _Z_TABLE[lo])

    return _Z_TABLE[keys[-1]]  # unreachable, satisfies type checker


def calculate_eoq(
    annual_demand: float,
    order_cost: float = DEFAULT_ORDER_COST,
    holding_cost_per_unit: float | None = None,
    unit_cost: float = 10.0,
    holding_cost_rate: float = DEFAULT_HOLDING_COST_RATE,
) -> float:
    """
    Economic Order Quantity (Wilson's formula).

    EOQ = sqrt(2 · D · S / H)

    where:
      D = annual demand (units / year)
      S = cost per order placed ($)
      H = holding cost per unit per year ($)

    If ``holding_cost_per_unit`` is not provided, H is derived from
    ``unit_cost × holding_cost_rate``.

    Parameters
    ----------
    annual_demand:
        Expected total units demanded over one year.
    order_cost:
        Fixed cost incurred each time an order is placed, in $.
    holding_cost_per_unit:
        Annual holding cost per unit in $.  If None, computed from
        unit_cost × holding_cost_rate.
    unit_cost:
        Per-unit purchase price.  Used only when holding_cost_per_unit is
        None.
    holding_cost_rate:
        Fraction of unit_cost charged per year as holding cost.  Defaults
        to 0.25 (25 %).

    Returns
    -------
    float
        Optimal order quantity, rounded up to the nearest whole unit.

    Raises
    ------
    ValueError
        If annual_demand, order_cost, or the computed holding cost is ≤ 0.
    """
    if annual_demand <= 0:
        raise ValueError(f"annual_demand must be > 0; got {annual_demand}.")
    if order_cost <= 0:
        raise ValueError(f"order_cost must be > 0; got {order_cost}.")

    h = holding_cost_per_unit if holding_cost_per_unit is not None else (unit_cost * holding_cost_rate)
    if h <= 0:
        raise ValueError(
            f"Effective holding cost must be > 0; got {h}. "
            "Check unit_cost and holding_cost_rate."
        )

    eoq = math.sqrt((2 * annual_demand * order_cost) / h)
    return math.ceil(eoq)


def calculate_safety_stock(
    daily_demand_std: float,
    lead_time_days: int,
    service_level: float = 0.95,
) -> float:
    """
    Safety stock for a fixed lead time.

    Safety Stock = z(SL) × σ_d × sqrt(LT)

    where σ_d is the standard deviation of daily demand and LT is the
    fixed lead time in days.

    Parameters
    ----------
    daily_demand_std:
        Standard deviation of daily sales (open days only).
    lead_time_days:
        Supplier lead time in days (treated as deterministic).
    service_level:
        Target service level as a fraction, e.g. 0.95.

    Returns
    -------
    float
        Safety stock quantity, rounded up to the nearest whole unit.
    """
    if daily_demand_std < 0:
        raise ValueError("daily_demand_std must be >= 0.")
    if lead_time_days < 1:
        raise ValueError("lead_time_days must be >= 1.")

    z = z_score(service_level)
    ss = z * daily_demand_std * math.sqrt(lead_time_days)
    return math.ceil(ss)


def calculate_reorder_point(
    avg_daily_demand: float,
    lead_time_days: int,
    daily_demand_std: float,
    service_level: float = 0.95,
) -> float:
    """
    Reorder Point (ROP).

    ROP = (avg_daily_demand × lead_time_days) + safety_stock

    The reorder point is the quantity of on-hand stock at which a new
    purchase order should be placed so that stock arrives before the
    current inventory is exhausted.

    Parameters
    ----------
    avg_daily_demand:
        Mean daily sales over open days.
    lead_time_days:
        Supplier lead time in days.
    daily_demand_std:
        Standard deviation of daily sales.
    service_level:
        Target service level as a fraction.

    Returns
    -------
    float
        Reorder point, rounded up to the nearest whole unit.
    """
    if avg_daily_demand < 0:
        raise ValueError("avg_daily_demand must be >= 0.")

    ss = calculate_safety_stock(daily_demand_std, lead_time_days, service_level)
    rop = (avg_daily_demand * lead_time_days) + ss
    return math.ceil(rop)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _load_demand_stats(product_id: int, engine: Any) -> tuple[float, float, int]:
    """
    Compute mean and standard deviation of daily demand from sales_history,
    using only open-day rows (the same rows used for Prophet training).

    Returns
    -------
    tuple of (avg_daily_demand, daily_demand_std, n_rows)
    """
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT sales
                FROM   sales_history
                WHERE  product_id = :pid AND open = 1
                ORDER  BY date
            """),
            {"pid": product_id},
        ).fetchall()

    if not rows:
        raise ValueError(f"No open-day sales rows for product_id={product_id}.")

    sales = np.array([r[0] for r in rows], dtype=float)
    return float(np.mean(sales)), float(np.std(sales, ddof=1)), len(sales)


def _load_inventory_record(product_id: int, engine: Any) -> dict[str, Any]:
    """
    Load the current inventory record for a product.

    Returns a dict with keys: current_stock, lead_time_days, unit_cost,
    supplier_name.

    Raises
    ------
    ValueError
        If no inventory row exists for the product.
    """
    with engine.connect() as conn:
        row = conn.execute(
            text("""
                SELECT current_stock, lead_time_days, unit_cost, supplier_name
                FROM   inventory
                WHERE  product_id = :pid
            """),
            {"pid": product_id},
        ).fetchone()

    if row is None:
        raise ValueError(f"No inventory record for product_id={product_id}.")

    return {
        "current_stock": float(row[0]),
        "lead_time_days": int(row[1]),
        "unit_cost": float(row[2]),
        "supplier_name": str(row[3]),
    }



# ---------------------------------------------------------------------------
# High-level recommendation interface
# ---------------------------------------------------------------------------

def get_inventory_recommendation(
    product_id: int,
    forecast_days: int = 42,
    service_level: float = 0.95,
    order_cost: float = DEFAULT_ORDER_COST,
    holding_cost_rate: float = DEFAULT_HOLDING_COST_RATE,
) -> dict[str, Any]:
    """
    Generate a complete inventory recommendation for a product.

    Steps:
      1. Load current on-hand quantity, lead time, and unit cost from DB.
      2. Compute historical demand mean and std from sales_history.
      3. Retrieve the Prophet forecast for the next ``forecast_days`` days.
      4. Compute EOQ, safety stock, and reorder point.
      5. Derive a risk label and action recommendation.

    Parameters
    ----------
    product_id:
        The product to analyse.
    forecast_days:
        Number of days to forecast demand (used for projected stock-out date
        and projected demand in the lead-time window).
    service_level:
        Target service level for safety stock and reorder point, e.g. 0.95.
    order_cost:
        Fixed cost per purchase order in $.
    holding_cost_rate:
        Annual holding cost as a fraction of unit cost.

    Returns
    -------
    dict with keys:

    - ``product_id`` (int)
    - ``quantity_on_hand`` (int): current stock from DB
    - ``avg_daily_demand`` (float): historical mean daily sales
    - ``daily_demand_std`` (float): historical std dev of daily sales
    - ``lead_time_days`` (int): supplier lead time from DB
    - ``safety_stock`` (int): computed safety stock
    - ``reorder_point`` (int): computed reorder point
    - ``eoq`` (int): economic order quantity
    - ``days_of_cover`` (float): current stock / avg daily demand
    - ``forecast_demand_in_lead_time`` (float): expected demand during lead time
    - ``units_at_risk`` (int): max(0, forecast_demand_in_lead_time - quantity_on_hand)
    - ``risk_level`` (str): one of "OK", "WARNING", "CRITICAL"
    - ``action`` (str): human-readable recommendation sentence
    - ``service_level`` (float): the service level used

    Raises
    ------
    ModelNotFoundError
        If no trained Prophet model exists for the product.
    ValueError
        If the product has no inventory or sales records in the database.
    """
    # Deferred import to avoid circular dependency; engine and prophet loaded
    # only when this function is called, not at module import time.
    from database.db import engine
    from forecasting.predict import ModelNotFoundError, get_forecast

    inv = _load_inventory_record(product_id, engine)
    avg_demand, demand_std, n_history = _load_demand_stats(product_id, engine)

    lead_time = inv["lead_time_days"]
    unit_cost = inv["unit_cost"]
    on_hand = inv["current_stock"]

    # Safety stock and reorder point based on historical demand statistics.
    ss = calculate_safety_stock(demand_std, lead_time, service_level)
    rop = calculate_reorder_point(avg_demand, lead_time, demand_std, service_level)

    # EOQ based on annualised historical demand.
    annual_demand = avg_demand * DAYS_PER_YEAR
    eoq = calculate_eoq(
        annual_demand=annual_demand,
        order_cost=order_cost,
        unit_cost=unit_cost,
        holding_cost_rate=holding_cost_rate,
    )

    # Prophet forecast for lead-time window (demand expected while waiting for
    # a replenishment order to arrive).
    try:
        forecast = get_forecast(product_id, days_ahead=min(forecast_days, lead_time))
        forecast_lead_time_demand = float(sum(forecast["yhat"]))
    except ModelNotFoundError:
        # If no model is trained, fall back to historical average.
        logger.warning(
            "No Prophet model for product %d — using historical average for forecast.",
            product_id,
        )
        forecast_lead_time_demand = avg_demand * lead_time

    # Days of cover: how many days the current stock will last at average demand.
    days_of_cover = (on_hand / avg_demand) if avg_demand > 0 else float("inf")

    # Units at risk: shortfall between forecasted lead-time demand and on-hand.
    units_at_risk = max(0, int(math.ceil(forecast_lead_time_demand - on_hand)))

    # Risk label.
    if days_of_cover < _CRITICAL_DAYS_OF_COVER or on_hand <= 0:
        risk_level = "CRITICAL"
    elif on_hand <= rop or days_of_cover < _WARNING_DAYS_OF_COVER:
        risk_level = "WARNING"
    else:
        risk_level = "OK"

    # Human-readable action sentence.
    if risk_level == "CRITICAL":
        action = (
            f"Immediate replenishment required. "
            f"Place an emergency order of {eoq} units now. "
            f"Current stock ({on_hand} units) covers only "
            f"{days_of_cover:.1f} day(s) of demand."
        )
    elif risk_level == "WARNING":
        action = (
            f"Stock is below the reorder point ({rop} units). "
            f"Place an order of {eoq} units with supplier {inv['supplier_name']}. "
            f"Current stock ({on_hand} units) covers "
            f"{days_of_cover:.1f} day(s) of demand."
        )
    else:
        action = (
            f"Stock is healthy ({on_hand} units, {days_of_cover:.1f} days of cover). "
            f"Reorder when stock falls to {rop} units. "
            f"Recommended order quantity: {eoq} units."
        )

    return {
        "product_id": product_id,
        "current_stock": on_hand,
        "avg_daily_demand": round(avg_demand, 2),
        "daily_demand_std": round(demand_std, 2),
        "lead_time_days": lead_time,
        "safety_stock": int(ss),
        "reorder_point": int(rop),
        "eoq": int(eoq),
        "days_of_cover": round(days_of_cover, 2),
        "forecast_demand_in_lead_time": round(forecast_lead_time_demand, 2),
        "units_at_risk": units_at_risk,
        "risk_level": risk_level,
        "action": action,
        "service_level": service_level,
    }
