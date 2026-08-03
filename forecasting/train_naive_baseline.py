"""
Naive baseline model — predicts tomorrow's sales using today's actual sales.

This 1-day lag baseline is the standard sanity check for any time-series
forecasting project.  If a model can't beat this simple rule, it isn't
adding value.

Evaluates the baseline on the same 42-day holdout window used by the
Prophet training script and writes the per-product metrics to
forecasting/naive_baseline_results.json.

Usage:
    python forecasting/train_naive_baseline.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from database.db import engine, test_connection  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HOLDOUT_DAYS = 42
OUTPUT_PATH = PROJECT_ROOT / "forecasting" / "naive_baseline_results.json"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_product_ids() -> list[int]:
    """
    Retrieve all product IDs from the products table.

    Returns:
        Sorted list of product IDs.

    Raises:
        RuntimeError: If no products are found.
    """
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT product_id FROM products ORDER BY product_id")).fetchall()
    if not rows:
        raise RuntimeError("No products found. Run scripts/seed_data.py first.")
    return [r[0] for r in rows]


def load_open_day_sales(product_id: int) -> pd.DataFrame:
    """
    Load sales for open days only, sorted by date ascending.

    Parameters:
        product_id: The product's primary key.

    Returns:
        DataFrame with columns: date (datetime64), sales (int).

    Raises:
        ValueError: If fewer than HOLDOUT_DAYS + 2 rows are available.
    """
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT date, sales
                FROM   sales_history
                WHERE  product_id = :pid AND open = 1
                ORDER  BY date
            """),
            {"pid": product_id},
        ).fetchall()

    df = pd.DataFrame(rows, columns=["date", "sales"])
    df["date"] = pd.to_datetime(df["date"])

    if len(df) < HOLDOUT_DAYS + 2:
        raise ValueError(
            f"Product {product_id}: too few open-day rows ({len(df)}) to evaluate."
        )
    return df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Mean Absolute Error.

    Parameters:
        actual: Array of true values.
        predicted: Array of predicted values.

    Returns:
        MAE as a float.
    """
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Root Mean Squared Error.

    Parameters:
        actual: Array of true values.
        predicted: Array of predicted values.

    Returns:
        RMSE as a float.
    """
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error, skipping rows where actual == 0 to
    avoid division-by-zero.

    Parameters:
        actual: Array of true values.
        predicted: Array of predicted values.

    Returns:
        MAPE as a percentage (e.g., 12.3 means 12.3%).  Returns NaN if
        all actual values are zero.
    """
    nonzero = actual != 0
    if not nonzero.any():
        return float("nan")
    return float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100)


# ---------------------------------------------------------------------------
# Baseline evaluation
# ---------------------------------------------------------------------------

def evaluate_product(product_id: int) -> dict[str, Any]:
    """
    Evaluate the 1-day lag naive baseline on the 42-day holdout for one
    product.

    The baseline rule: predicted_sales[t] = actual_sales[t-1].
    The lag is computed over open-day records only (not calendar days),
    so the "previous day" is the previous open day.

    Parameters:
        product_id: The product to evaluate.

    Returns:
        Dict with keys: product_id, mae, rmse, mape, n_holdout_rows.
    """
    df = load_open_day_sales(product_id)

    # Identify the holdout cutoff using calendar days (same logic as
    # train_prophet.py so both are evaluated on identical date ranges).
    max_date = df["date"].max()
    cutoff = max_date - pd.Timedelta(days=HOLDOUT_DAYS - 1)

    holdout = df[df["date"] >= cutoff].copy()
    # We need the last training row to get the lag for the first holdout day.
    train_tail = df[df["date"] < cutoff]

    if train_tail.empty:
        raise ValueError(f"Product {product_id}: no training rows before cutoff {cutoff.date()}.")

    # Build a single series containing the last training row + all holdout rows.
    # The naive prediction for holdout row i is the actual value at i-1.
    combined_sales = pd.concat([train_tail.tail(1), holdout], ignore_index=True)["sales"].values

    actual = combined_sales[1:].astype(float)
    predicted = combined_sales[:-1].astype(float)

    return {
        "product_id": product_id,
        "mae": round(mae(actual, predicted), 4),
        "rmse": round(rmse(actual, predicted), 4),
        "mape": round(mape(actual, predicted), 4),
        "n_holdout_rows": len(holdout),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Evaluate the naive baseline for all products and write results to JSON.
    """
    logger.info("Naive baseline evaluation starting.")

    if not test_connection():
        logger.error("Cannot reach the database — aborting.")
        sys.exit(1)

    product_ids = get_product_ids()
    logger.info("Evaluating %d products...", len(product_ids))

    results: list[dict[str, Any]] = []
    failed: list[int] = []

    for pid in product_ids:
        try:
            metrics = evaluate_product(pid)
            results.append(metrics)
            logger.info(
                "  Product %d — MAE=%.2f  RMSE=%.2f  MAPE=%.2f%%",
                pid, metrics["mae"], metrics["rmse"], metrics["mape"],
            )
        except Exception:
            logger.exception("Product %d: evaluation failed — skipping.", pid)
            failed.append(pid)

    # Aggregate averages across all evaluated products
    if results:
        avg_mae = round(float(np.mean([r["mae"] for r in results])), 4)
        avg_rmse = round(float(np.mean([r["rmse"] for r in results])), 4)
        avg_mape = round(float(np.mean([r["mape"] for r in results if not np.isnan(r["mape"])])), 4)
    else:
        avg_mae = avg_rmse = avg_mape = float("nan")

    output = {
        "model": "naive_1day_lag",
        "holdout_days": HOLDOUT_DAYS,
        "per_product": results,
        "aggregate": {
            "mean_mae": avg_mae,
            "mean_rmse": avg_rmse,
            "mean_mape_pct": avg_mape,
        },
        "failed_products": failed,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)

    logger.info("=" * 60)
    logger.info("Naive baseline results:")
    logger.info("  Mean MAE  : %.2f", avg_mae)
    logger.info("  Mean RMSE : %.2f", avg_rmse)
    logger.info("  Mean MAPE : %.2f%%", avg_mape)
    logger.info("  Saved to  : %s", OUTPUT_PATH)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
