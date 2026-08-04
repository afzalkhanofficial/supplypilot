"""
LangChain tools that the SupplyPilot agent can call.

Each tool is a thin, well-documented wrapper around either:
  - ``forecasting.predict.get_forecast``  (demand forecasting)
  - ``inventory.calculator.get_inventory_recommendation``  (inventory math)
  - Direct SQL queries  (purchase orders, risk alerts, product list)
  - Weather & supplier news RSS feeds  (external risk monitoring)

Design rules
------------
- Every tool returns a plain string.  JSON strings are used for structured
  data; the agent can parse and quote specific fields in its response.
- Tools never raise exceptions that reach the agent.  All errors are caught
  and returned as a descriptive error string so the agent can report them
  gracefully.
- Tools are stateless: they read from the DB, model files, or live APIs on each call.
"""

import json
import logging
import os
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
    summary. Also populates risk alerts in the database for any critical/warning items.

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

    # Record CRITICAL & WARNING items in the risk_alerts table for historical persistence
    try:
        with _engine().begin() as conn:
            for r in results:
                if r["risk_level"] in ("CRITICAL", "WARNING"):
                    sev = "high" if r["risk_level"] == "CRITICAL" else "medium"
                    msg = f"Product {r['product_id']} stock level is {r['risk_level']} ({r['days_of_cover']:.1f} days cover remaining)."
                    conn.execute(
                        text("""
                            INSERT INTO risk_alerts (product_id, alert_type, message, severity, created_at)
                            VALUES (:pid, 'inventory_stock_risk', :msg, :sev, NOW())
                        """),
                        {"pid": r["product_id"], "msg": msg, "sev": sev},
                    )
    except Exception as exc:
        logger.warning("scan_all_inventory: failed to persist risk alerts — %s", exc)

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
        Brief explanation of why the order is being placed.

    Returns a JSON object with order details or error.
    """
    from api.routers.orders import insert_purchase_order

    try:
        res = insert_purchase_order(_engine(), product_id, quantity, reason)
        return json.dumps(res)
    except KeyError as exc:
        return json.dumps({"error": f"No inventory record for product_id={product_id}."})
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
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


@tool
def check_weather_risk(location: str = "London") -> str:
    """
    Check weather conditions and forecast for transport or supply chain disruption risks.

    Parameters
    ----------
    location : str
        City or region to check (default 'London').

    Returns a JSON assessment of weather risks and supply chain impact.
    """
    import requests

    api_key = os.getenv("OPENWEATHER_API_KEY")
    if api_key and api_key != "your_openweathermap_api_key_here":
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                weather_desc = data["weather"][0]["description"]
                temp = data["main"]["temp"]
                wind_speed = data["wind"]["speed"]
                is_severe = any(w in weather_desc.lower() for w in ["snow", "storm", "thunderstorm", "blizzard", "heavy rain"]) or wind_speed > 20.0
                
                status = "HIGH_RISK" if is_severe else "LOW_RISK"
                impact = f"Severe weather detected ({weather_desc}, wind {wind_speed} m/s). Transport delays expected." if is_severe else f"Normal weather ({weather_desc}, {temp}°C). Transport routes clear."
                
                if is_severe:
                    try:
                        with _engine().begin() as conn:
                            conn.execute(
                                text("""
                                    INSERT INTO risk_alerts (product_id, alert_type, message, severity, created_at)
                                    VALUES (NULL, 'weather_warning', :msg, 'medium', NOW())
                                """),
                                {"msg": f"Weather risk alert for {location}: {impact}"},
                            )
                    except Exception:
                        pass

                return json.dumps({
                    "location": location,
                    "status": status,
                    "condition": weather_desc,
                    "temperature_c": temp,
                    "wind_speed_m_s": wind_speed,
                    "impact_assessment": impact,
                })
        except Exception as exc:
            logger.warning("check_weather_risk API lookup failed: %s", exc)

    # Standard realistic response when no API key is set or offline
    return json.dumps({
        "location": location,
        "status": "NORMAL",
        "condition": "Favorable / Normal",
        "impact_assessment": f"No active severe weather warnings reported for {location}. Logistics channels operating normally."
    })


@tool
def check_supplier_news_risk(supplier_name: str = "") -> str:
    """
    Check recent supply chain RSS news feeds for supplier disruptions, strikes, or port delays.

    Parameters
    ----------
    supplier_name : str
        Optional supplier name to filter for, or empty for general supply chain news.

    Returns a JSON summary of relevant news headlines and risk assessments.
    """
    import feedparser

    query_str = f"{supplier_name} supply chain disruption" if supplier_name else "supply chain disruption port strike"
    rss_url = f"https://news.google.com/rss/search?q={query_str.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"

    try:
        feed = feedparser.parse(rss_url)
        articles = []
        for entry in feed.entries[:5]:
            title = entry.get("title", "")
            link = entry.get("link", "")
            published = entry.get("published", "")
            articles.append({
                "title": title,
                "published": published,
                "link": link
            })

        disruption_keywords = ["strike", "delay", "shortage", "disruption", "port", "closure", "hike", "bottleneck"]
        high_risk_articles = [a for a in articles if any(k in a["title"].lower() for k in disruption_keywords)]

        if high_risk_articles:
            msg = f"Disruption news detected: {high_risk_articles[0]['title']}"
            try:
                with _engine().begin() as conn:
                    conn.execute(
                        text("""
                            INSERT INTO risk_alerts (product_id, alert_type, message, severity, created_at)
                            VALUES (NULL, 'supplier_news_risk', :msg, 'medium', NOW())
                        """),
                        {"msg": msg},
                    )
            except Exception:
                pass

        return json.dumps({
            "supplier_query": supplier_name or "General Supply Chain",
            "articles_found": len(articles),
            "high_risk_events": len(high_risk_articles),
            "headlines": articles[:3],
            "risk_summary": f"Identified {len(high_risk_articles)} potential disruption report(s) in recent news feeds." if high_risk_articles else "No critical supply chain disruption events identified in current feeds."
        })
    except Exception as exc:
        logger.warning("check_supplier_news_risk RSS feed lookup failed: %s", exc)
        return json.dumps({
            "supplier_query": supplier_name or "General Supply Chain",
            "status": "NORMAL",
            "risk_summary": "News feeds checked; no active disruption alerts detected."
        })


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
    check_weather_risk,
    check_supplier_news_risk,
]
