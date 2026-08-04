"""
Model evaluation — Prophet vs naive baseline, side by side.

Loads each trained Prophet model from forecasting/models/prophet/,
generates predictions for the 42-day holdout window using the actual
historical regressor values for that period, then computes and compares
MAE, RMSE, and MAPE against the naive baseline results produced by
train_naive_baseline.py.

Outputs a comparison table to:
  - the console (plain text)
  - forecasting/model_comparison.md (markdown table)

Usage:
    python forecasting/evaluate.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

import cmdstanpy
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from prophet.serialize import model_from_json
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from database.db import engine, test_connection  # noqa: E402

import forecasting.prophet_patch  # noqa: F401


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS_DIR = PROJECT_ROOT / "forecasting" / "models" / "prophet"
BASELINE_RESULTS_PATH = PROJECT_ROOT / "forecasting" / "naive_baseline_results.json"
COMPARISON_OUTPUT_PATH = PROJECT_ROOT / "forecasting" / "model_comparison.md"
HOLDOUT_DAYS = 42


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error, skipping zero-actual rows.

    Returns:
        MAPE as a percentage float, or NaN if all actuals are zero.
    """
    nonzero = actual != 0
    if not nonzero.any():
        return float("nan")
    return float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100)


# ---------------------------------------------------------------------------
# Prophet evaluation
# ---------------------------------------------------------------------------

def load_holdout_with_regressors(product_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return the training set and the 42-day holdout with actual regressor
    values for the product.

    Prophet requires regressor values when calling predict(), so we supply
    the real historical values for the holdout period.  This gives a fair
    evaluation: the model knows what actually happened with promotions and
    holidays during those 42 days, just as it did during training.

    Parameters:
        product_id: The product's primary key.

    Returns:
        Tuple of (full_df, holdout_df) both in Prophet format (ds, y,
        promo, state_holiday, school_holiday).  state_holiday is encoded
        as a binary integer.

    Raises:
        ValueError: If the product has no open-day history.
    """
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT date, sales, promo, state_holiday, school_holiday
                FROM   sales_history
                WHERE  product_id = :pid AND open = 1
                ORDER  BY date
            """),
            {"pid": product_id},
        ).fetchall()

    if not rows:
        raise ValueError(f"No open-day rows for product_id={product_id}.")

    df = pd.DataFrame(rows, columns=["date", "sales", "promo", "state_holiday", "school_holiday"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"date": "ds", "sales": "y"})
    df["state_holiday"] = (df["state_holiday"] != "0").astype(int)
    df["promo"] = df["promo"].astype(int)
    df["school_holiday"] = df["school_holiday"].astype(int)

    max_date = df["ds"].max()
    cutoff = max_date - pd.Timedelta(days=HOLDOUT_DAYS - 1)
    holdout = df[df["ds"] >= cutoff].copy()

    return df, holdout


def evaluate_prophet_product(product_id: int) -> dict[str, Any]:
    """
    Load the trained Prophet model for a product, generate holdout
    predictions, and return error metrics.

    Parameters:
        product_id: The product to evaluate.

    Returns:
        Dict with keys: product_id, mae, rmse, mape, n_holdout_rows.

    Raises:
        FileNotFoundError: If no trained model exists for this product.
        ValueError: If the holdout is empty.
    """
    model_path = MODELS_DIR / f"product_{product_id}.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. "
            "Run forecasting/train_prophet.py first."
        )

    with open(model_path, "r", encoding="utf-8") as fh:
        model = model_from_json(fh.read())

    _, holdout = load_holdout_with_regressors(product_id)

    if holdout.empty:
        raise ValueError(f"Product {product_id}: holdout DataFrame is empty.")

    # Predict on the holdout dates using actual regressor values.
    predictions = model.predict(holdout[["ds", "promo", "state_holiday", "school_holiday"]])

    actual = holdout["y"].values.astype(float)
    predicted = predictions["yhat"].values.astype(float)
    # Clip negative predictions to 0 — sales cannot be negative.
    predicted = np.clip(predicted, 0, None)

    return {
        "product_id": product_id,
        "mae": round(mae(actual, predicted), 4),
        "rmse": round(rmse(actual, predicted), 4),
        "mape": round(mape(actual, predicted), 4),
        "n_holdout_rows": len(holdout),
    }


# ---------------------------------------------------------------------------
# Comparison table generation
# ---------------------------------------------------------------------------

def build_comparison_table(
    prophet_results: list[dict],
    baseline_results: list[dict],
) -> str:
    """
    Build a markdown table comparing Prophet vs the naive baseline for
    every product that both models evaluated.

    Parameters:
        prophet_results: List of per-product Prophet metric dicts.
        baseline_results: List of per-product naive baseline metric dicts.

    Returns:
        A markdown-formatted string containing the comparison table.
    """
    prophet_by_id = {r["product_id"]: r for r in prophet_results}
    baseline_by_id = {r["product_id"]: r for r in baseline_results}
    common_ids = sorted(set(prophet_by_id) & set(baseline_by_id))

    rows = []
    for pid in common_ids:
        p = prophet_by_id[pid]
        b = baseline_by_id[pid]

        def pct_change(prophet_val: float, baseline_val: float) -> str:
            """Return improvement % (negative = Prophet is better)."""
            if baseline_val == 0 or np.isnan(baseline_val):
                return "N/A"
            delta = (prophet_val - baseline_val) / baseline_val * 100
            sign = "+" if delta > 0 else ""
            return f"{sign}{delta:.1f}%"

        rows.append({
            "Product": f"Product-{pid}",
            "Prophet MAE": f"{p['mae']:.1f}",
            "Baseline MAE": f"{b['mae']:.1f}",
            "MAE Δ": pct_change(p["mae"], b["mae"]),
            "Prophet RMSE": f"{p['rmse']:.1f}",
            "Baseline RMSE": f"{b['rmse']:.1f}",
            "RMSE Δ": pct_change(p["rmse"], b["rmse"]),
            "Prophet MAPE": f"{p['mape']:.1f}%",
            "Baseline MAPE": f"{b['mape']:.1f}%",
            "MAPE Δ": pct_change(p["mape"], b["mape"]),
        })

    # Aggregates
    def avg_metric(results: list[dict], key: str) -> float:
        vals = [r[key] for r in results if not np.isnan(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    p_avg_mae = avg_metric(prophet_results, "mae")
    b_avg_mae = avg_metric(baseline_results, "mae")
    p_avg_rmse = avg_metric(prophet_results, "rmse")
    b_avg_rmse = avg_metric(baseline_results, "rmse")
    p_avg_mape = avg_metric(prophet_results, "mape")
    b_avg_mape = avg_metric(baseline_results, "mape")

    def pct_change_raw(a: float, b_val: float) -> str:
        if b_val == 0 or np.isnan(b_val):
            return "N/A"
        delta = (a - b_val) / b_val * 100
        sign = "+" if delta > 0 else ""
        return f"{sign}{delta:.1f}%"

    rows.append({
        "Product": "**AVERAGE**",
        "Prophet MAE": f"**{p_avg_mae:.1f}**",
        "Baseline MAE": f"**{b_avg_mae:.1f}**",
        "MAE Δ": f"**{pct_change_raw(p_avg_mae, b_avg_mae)}**",
        "Prophet RMSE": f"**{p_avg_rmse:.1f}**",
        "Baseline RMSE": f"**{b_avg_rmse:.1f}**",
        "RMSE Δ": f"**{pct_change_raw(p_avg_rmse, b_avg_rmse)}**",
        "Prophet MAPE": f"**{p_avg_mape:.1f}%**",
        "Baseline MAPE": f"**{b_avg_mape:.1f}%**",
        "MAPE Δ": f"**{pct_change_raw(p_avg_mape, b_avg_mape)}**",
    })

    # Build markdown
    headers = list(rows[0].keys())
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body_lines = ["| " + " | ".join(str(r[h]) for h in headers) + " |" for r in rows]

    note = (
        "\n> **Notes:** Δ columns show Prophet's metric relative to the naive baseline "
        "(negative % = Prophet is better). MAPE skips days with zero actual sales. "
        f"Holdout window: {HOLDOUT_DAYS} days.\n"
    )

    return "\n".join([
        "# Model Performance: Prophet vs Naive Baseline",
        "",
        f"Holdout period: last **{HOLDOUT_DAYS} calendar days** of available data per product.",
        "",
        header_line,
        separator,
        *body_lines,
        note,
    ])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Evaluate all trained Prophet models against the naive baseline and
    write the comparison table to disk.
    """
    logger.info("Model evaluation starting.")

    if not test_connection():
        logger.error("Cannot reach the database — aborting.")
        sys.exit(1)

    # Load naive baseline results (must have been generated first).
    if not BASELINE_RESULTS_PATH.exists():
        logger.error(
            "Naive baseline results not found at %s. "
            "Run forecasting/train_naive_baseline.py first.",
            BASELINE_RESULTS_PATH,
        )
        sys.exit(1)

    with open(BASELINE_RESULTS_PATH, "r", encoding="utf-8") as fh:
        baseline_data = json.load(fh)
    baseline_results = baseline_data["per_product"]

    # Discover all trained Prophet models.
    model_paths = sorted(MODELS_DIR.glob("product_*.json"))
    if not model_paths:
        logger.error(
            "No trained Prophet models found in %s. "
            "Run forecasting/train_prophet.py first.",
            MODELS_DIR,
        )
        sys.exit(1)

    logger.info("Evaluating %d Prophet models...", len(model_paths))

    prophet_results: list[dict] = []
    failed: list[int] = []

    for model_path in model_paths:
        pid_str = model_path.stem.replace("product_", "")
        try:
            pid = int(pid_str)
        except ValueError:
            logger.warning("Unexpected model file name: %s — skipping.", model_path.name)
            continue

        try:
            metrics = evaluate_prophet_product(pid)
            prophet_results.append(metrics)
            logger.info(
                "  Product %d — MAE=%.2f  RMSE=%.2f  MAPE=%.2f%%",
                pid, metrics["mae"], metrics["rmse"], metrics["mape"],
            )
        except Exception:
            logger.exception("Product %d: evaluation failed — skipping.", pid)
            failed.append(pid)

    if not prophet_results:
        logger.error("No Prophet models could be evaluated — check logs above.")
        sys.exit(1)

    # Build and save the comparison table.
    table_md = build_comparison_table(prophet_results, baseline_results)

    COMPARISON_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(COMPARISON_OUTPUT_PATH, "w", encoding="utf-8") as fh:
        fh.write(table_md)

    # Also print a summary to the console.
    logger.info("=" * 60)
    logger.info("Evaluation complete.")
    logger.info("  Prophet models evaluated : %d", len(prophet_results))
    logger.info("  Failed                   : %d", len(failed))
    logger.info("  Results saved to         : %s", COMPARISON_OUTPUT_PATH)
    logger.info("=" * 60)
    print("\n" + table_md)


if __name__ == "__main__":
    main()
