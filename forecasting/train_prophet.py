"""
Prophet model training script — one model per product.

Loads each product's sales history from the database (open days only),
trains a separate Prophet model with weekly/yearly seasonality and three
extra regressors (promo, state_holiday, school_holiday), and saves each
trained model as a JSON file in forecasting/models/prophet/.

The last 42 days of each product's history are held out and NOT used
during training.  The same holdout window is used in evaluate.py so
results are directly comparable.

Usage:
    python forecasting/train_prophet.py
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cmdstanpy
import pandas as pd
from dotenv import load_dotenv
from prophet import Prophet
from prophet.serialize import model_to_json
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from database.db import engine, test_connection  # noqa: E402


def _ensure_stan_backend() -> None:
    """Ensure CmdStan is installed and properly pointed to by cmdstanpy."""
    try:
        current_path = Path(cmdstanpy.cmdstan_path())
        makefile = current_path / "makefile"
        if not makefile.exists():
            raise FileNotFoundError(f"CmdStan makefile missing at {current_path}")
    except Exception:
        logger.info("CmdStan backend missing or invalid — installing CmdStan...")
        try:
            installed_path = cmdstanpy.install_cmdstan()
            cmdstanpy.set_cmdstan_path(installed_path)
            logger.info("CmdStan installed and set to: %s", installed_path)
        except Exception as exc:
            logger.warning("CmdStan installation warning: %s", exc)



# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Suppress Prophet's verbose Stan output
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS_DIR = PROJECT_ROOT / "forecasting" / "models" / "prophet"
HOLDOUT_DAYS = 42   # calendar days withheld from training for evaluation


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_product_ids() -> list[int]:
    """
    Retrieve all product IDs from the products table.

    Returns:
        Sorted list of product IDs.

    Raises:
        RuntimeError: If the query fails or returns no products.
    """
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT product_id FROM products ORDER BY product_id")).fetchall()
    if not rows:
        raise RuntimeError(
            "No products found in the database. Run scripts/seed_data.py first."
        )
    return [r[0] for r in rows]


def load_product_sales(product_id: int) -> pd.DataFrame:
    """
    Load open-day sales history for one product from the database.

    Only rows where open = 1 are returned because the model is trained to
    forecast sales on days the store/product is active.  Closed days with
    zero sales would distort seasonality patterns.

    Parameters:
        product_id: The product's primary key.

    Returns:
        DataFrame with columns: date (datetime64), sales (int), promo (int),
        state_holiday (str), school_holiday (int).  Sorted ascending by date.

    Raises:
        ValueError: If no open-day rows exist for this product.
    """
    query = text("""
        SELECT date, sales, promo, state_holiday, school_holiday
        FROM   sales_history
        WHERE  product_id = :pid
          AND  open = 1
        ORDER  BY date
    """)
    with engine.connect() as conn:
        rows = conn.execute(query, {"pid": product_id}).fetchall()

    if not rows:
        raise ValueError(
            f"No open-day sales rows found for product_id={product_id}."
        )

    df = pd.DataFrame(rows, columns=["date", "sales", "promo", "state_holiday", "school_holiday"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def prepare_prophet_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename and encode columns for Prophet's expected format.

    Prophet requires the time column to be named 'ds' and the target
    column 'y'.  state_holiday is encoded as a binary flag (0 = no
    holiday, 1 = any state holiday) because Prophet's add_regressor
    expects numeric inputs.

    Parameters:
        df: Raw sales DataFrame from load_product_sales().

    Returns:
        Prophet-ready DataFrame with columns: ds, y, promo,
        state_holiday (binary int), school_holiday.
    """
    prophet_df = df.rename(columns={"date": "ds", "sales": "y"}).copy()
    # '0' → 0 (no holiday);  'a', 'b', 'c' → 1 (some state holiday)
    prophet_df["state_holiday"] = (prophet_df["state_holiday"] != "0").astype(int)
    prophet_df["promo"] = prophet_df["promo"].astype(int)
    prophet_df["school_holiday"] = prophet_df["school_holiday"].astype(int)
    return prophet_df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def compute_cutoff(df: pd.DataFrame) -> pd.Timestamp:
    """
    Calculate the training cutoff date (max date minus HOLDOUT_DAYS).

    Parameters:
        df: Prophet-ready DataFrame with a 'ds' column.

    Returns:
        The exclusive cutoff timestamp — training uses dates strictly
        before this value; holdout uses dates on or after it.
    """
    max_date = df["ds"].max()
    cutoff = max_date - pd.Timedelta(days=HOLDOUT_DAYS - 1)
    return cutoff


def train_and_save(product_id: int, df: pd.DataFrame) -> Optional[pd.Timestamp]:
    """
    Train a Prophet model on data before the holdout window and persist
    it to disk as a JSON file.

    Extra regressors added:
    - promo: binary, 1 when a promotion is running
    - state_holiday: binary, 1 on public/Easter/Christmas holidays
    - school_holiday: binary, 1 during school holiday periods

    Parameters:
        product_id: Used to name the output file.
        df: Full Prophet-ready DataFrame (training + holdout combined).

    Returns:
        The cutoff timestamp used to split training vs holdout, or None
        if training was skipped due to insufficient data.
    """
    cutoff = compute_cutoff(df)
    train_df = df[df["ds"] < cutoff].copy()

    if len(train_df) < 50:
        logger.warning(
            "Product %d: only %d training rows (need >=50) — skipping.",
            product_id, len(train_df),
        )
        return None

    logger.info(
        "Product %d: training on %d rows (cutoff %s, holdout %d days).",
        product_id, len(train_df), cutoff.date(), HOLDOUT_DAYS,
    )

    try:
        valid_path = cmdstanpy.cmdstan_path()
        if (Path(valid_path) / "makefile").exists():
            cmdstanpy.set_cmdstan_path(valid_path)
    except Exception:
        pass

    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",  # multiplicative fits retail sales well
    )
    model.add_regressor("promo")
    model.add_regressor("state_holiday")
    model.add_regressor("school_holiday")

    model.fit(train_df[["ds", "y", "promo", "state_holiday", "school_holiday"]])

    model_path = MODELS_DIR / f"product_{product_id}.json"
    with open(model_path, "w", encoding="utf-8") as fh:
        fh.write(model_to_json(model))

    logger.info("Product %d: model saved → %s", product_id, model_path.name)
    return cutoff


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Train Prophet models for all products in the database and report
    total elapsed time.
    """
    logger.info("Prophet training pipeline starting.")

    if not test_connection():
        logger.error("Cannot reach the database — aborting.")
        sys.exit(1)

    _ensure_stan_backend()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    product_ids = get_product_ids()
    logger.info("Training %d models...", len(product_ids))

    wall_start = time.time()
    trained = 0
    skipped = 0

    for pid in product_ids:
        try:
            t0 = time.time()
            df = load_product_sales(pid)
            prophet_df = prepare_prophet_df(df)
            cutoff = train_and_save(pid, prophet_df)
            elapsed = time.time() - t0

            if cutoff is not None:
                trained += 1
                logger.info("  Product %d done in %.1fs.", pid, elapsed)
            else:
                skipped += 1
        except Exception:
            logger.exception("Product %d: training failed — skipping.", pid)
            skipped += 1

    total_elapsed = time.time() - wall_start
    logger.info("=" * 60)
    logger.info("Training complete.")
    logger.info("  Models trained : %d", trained)
    logger.info("  Skipped        : %d", skipped)
    logger.info("  Total time     : %.1fs (%.1f min)", total_elapsed, total_elapsed / 60)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
