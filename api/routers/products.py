"""
Products and demand-forecast routes.

Endpoints
---------
GET /products               — List all products.
GET /products/{product_id}  — Single product details.
GET /products/{product_id}/forecast — Prophet demand forecast.
"""

import logging

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text

from api.schemas import ForecastResponse, ProductListResponse, ProductOut

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/products", tags=["Products"])


def _engine():
    from database.db import engine
    return engine


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("", response_model=ProductListResponse, summary="List all products")
def list_products():
    """Return every product in the system ordered by product_id."""
    with _engine().connect() as conn:
        rows = conn.execute(
            text("""
                SELECT product_id, product_name, store_type,
                       assortment, competition_distance
                FROM   products
                ORDER  BY product_id
            """)
        ).fetchall()

    products = [
        ProductOut(
            product_id=r[0],
            product_name=r[1],
            store_type=r[2],
            assortment=r[3],
            competition_distance=float(r[4]) if r[4] is not None else None,
        )
        for r in rows
    ]
    return ProductListResponse(products=products, total=len(products))


@router.get(
    "/{product_id}",
    response_model=ProductOut,
    summary="Get a single product",
)
def get_product(product_id: int):
    """Return details for one product by its ID."""
    with _engine().connect() as conn:
        row = conn.execute(
            text("""
                SELECT product_id, product_name, store_type,
                       assortment, competition_distance
                FROM   products
                WHERE  product_id = :pid
            """),
            {"pid": product_id},
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Product {product_id} not found.")

    return ProductOut(
        product_id=row[0],
        product_name=row[1],
        store_type=row[2],
        assortment=row[3],
        competition_distance=float(row[4]) if row[4] is not None else None,
    )


@router.get(
    "/{product_id}/forecast",
    response_model=ForecastResponse,
    summary="Get demand forecast for a product",
)
def get_forecast_endpoint(
    product_id: int,
    days_ahead: int = Query(default=14, ge=1, le=90, description="Forecast horizon in days."),
):
    """
    Return a day-by-day Prophet demand forecast.

    Uses the trained model from ``forecasting/models/prophet/``.
    Raises 404 if no model exists for the product, 422 if days_ahead
    is outside [1, 90].
    """
    from forecasting.predict import ForecastRangeError, ModelNotFoundError, get_forecast

    try:
        result = get_forecast(product_id, days_ahead)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ForecastRangeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error in get_forecast for product %d", product_id)
        raise HTTPException(status_code=500, detail=str(exc))

    return ForecastResponse(
        **result,
        total_forecast=round(sum(result["yhat"]), 2),
    )
