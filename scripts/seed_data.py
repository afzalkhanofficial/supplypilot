"""
Data seeding pipeline for the SupplyPilot project.

Reads the Rossmann Store Sales dataset from data/raw/, selects the 20
stores with the most complete sales histories, and loads them into the
PostgreSQL database as products, sales records, and initial inventory
rows.

The entire load runs inside a single database transaction — any failure
rolls back all changes so the database is never left in a partial state.

Usage:
    python scripts/seed_data.py
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Ensure the project root is on sys.path so sibling packages resolve.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from database.db import (  # noqa: E402 — import after sys.path patch
    AgentInteraction,
    Inventory,
    Product,
    PurchaseOrder,
    RiskAlert,
    SalesHistory,
    SessionLocal,
    init_db,
    test_connection,
)

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

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_CSV = RAW_DATA_DIR / "train.csv"
STORE_CSV = RAW_DATA_DIR / "store.csv"

TOP_N_STORES = 20          # number of stores (products) to keep
RANDOM_SEED = 42           # reproducibility for lead times and unit costs
STOCK_COVER_DAYS = 7       # initial stock = this many days of average sales
MIN_LEAD_TIME = 3          # days
MAX_LEAD_TIME = 10         # days
MIN_UNIT_COST = 5.0        # USD
MAX_UNIT_COST = 50.0       # USD

SUPPLIER_NAMES = [
    "Apex Supply Co.",
    "Meridian Distributors",
    "Northgate Wholesale",
    "Clearline Logistics",
    "Pinnacle Trade Group",
]


# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------

def load_and_clean(train_path: Path, store_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train.csv and store.csv, apply cleaning rules, and return them
    as (train_df, store_df).

    Cleaning applied:
    - CompetitionDistance NaN → median value (most common imputation for
      this dataset; missing usually means no nearby competitor, not truly
      unknown, so the median is a conservative safe choice)
    - Date parsed to datetime.date
    - StateHoliday cast to string (the raw file mixes int 0 and str codes)
    - Rows with NaN in any critical column (Sales, Customers, Open) dropped

    Parameters:
        train_path: Path to train.csv
        store_path: Path to store.csv

    Returns:
        Tuple of (cleaned training DataFrame, cleaned store DataFrame)

    Raises:
        FileNotFoundError: If either CSV file is missing.
        ValueError: If required columns are absent from either file.
    """
    for path in (train_path, store_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Required data file not found: {path}\n"
                "Download train.csv and store.csv from the Kaggle Rossmann "
                "Store Sales competition and place them in data/raw/."
            )

    logger.info("Loading %s ...", train_path.name)
    train = pd.read_csv(
        train_path,
        dtype={
            "Store": int,
            "DayOfWeek": int,
            "Sales": int,
            "Customers": int,
            "Open": int,
            "Promo": int,
            "SchoolHoliday": int,
        },
        parse_dates=["Date"],
        low_memory=False,
    )

    logger.info("Loading %s ...", store_path.name)
    store = pd.read_csv(store_path, low_memory=False)

    # --- Train cleaning ---
    required_train_cols = {"Store", "Date", "Sales", "Customers", "Open", "Promo",
                           "StateHoliday", "SchoolHoliday"}
    missing = required_train_cols - set(train.columns)
    if missing:
        raise ValueError(f"train.csv is missing expected columns: {missing}")

    # StateHoliday can be '0' (string) or 0 (int) — normalise to string.
    train["StateHoliday"] = train["StateHoliday"].astype(str).str.strip()
    # Replace integer zero-string with the canonical '0' code.
    train["StateHoliday"] = train["StateHoliday"].replace({"0": "0"})

    before = len(train)
    train = train.dropna(subset=["Sales", "Customers", "Open"])
    dropped = before - len(train)
    if dropped:
        logger.warning("Dropped %d rows with null Sales/Customers/Open.", dropped)

    # --- Store cleaning ---
    required_store_cols = {"Store", "StoreType", "Assortment", "CompetitionDistance"}
    missing = required_store_cols - set(store.columns)
    if missing:
        raise ValueError(f"store.csv is missing expected columns: {missing}")

    median_dist = store["CompetitionDistance"].median()
    null_count = store["CompetitionDistance"].isna().sum()
    if null_count:
        logger.info(
            "Imputing %d missing CompetitionDistance values with median %.1f m.",
            null_count, median_dist,
        )
    store["CompetitionDistance"] = store["CompetitionDistance"].fillna(median_dist)

    logger.info(
        "Loaded %d training rows across %d stores; %d store metadata rows.",
        len(train), train["Store"].nunique(), len(store),
    )
    return train, store


def select_top_stores(train: pd.DataFrame, n: int = TOP_N_STORES) -> list[int]:
    """
    Select the N stores whose sales histories have the fewest missing days.

    'Completeness' is measured as the number of distinct dates on which
    the store was open (Open == 1).  Stores with the most open-day records
    are preferred because Prophet requires a dense time series for reliable
    training.

    Parameters:
        train: Cleaned training DataFrame.
        n: Number of stores to select.

    Returns:
        Sorted list of store IDs (ascending) with the most complete records.
    """
    open_day_counts = (
        train[train["Open"] == 1]
        .groupby("Store")["Date"]
        .nunique()
        .sort_values(ascending=False)
    )
    selected = open_day_counts.head(n).index.tolist()
    selected_sorted = sorted(selected)
    logger.info(
        "Selected %d stores by open-day completeness: %s",
        n, selected_sorted,
    )
    logger.info(
        "Open-day counts — min: %d, max: %d",
        open_day_counts.iloc[n - 1],
        open_day_counts.iloc[0],
    )
    return selected_sorted


# ---------------------------------------------------------------------------
# Database insertion
# ---------------------------------------------------------------------------

def build_inventory_row(
    product_id: int,
    avg_daily_sales: float,
    rng: np.random.Generator,
    supplier_index: int,
) -> Inventory:
    """
    Construct an Inventory ORM object for a product.

    Parameters:
        product_id: The product's primary key.
        avg_daily_sales: Historical average daily sales for this product.
        rng: A seeded NumPy random generator for reproducible randomness.
        supplier_index: Index into SUPPLIER_NAMES (round-robin assignment).

    Returns:
        An unsaved Inventory instance ready to be added to a session.
    """
    lead_time = int(rng.integers(MIN_LEAD_TIME, MAX_LEAD_TIME + 1))
    unit_cost = round(float(rng.uniform(MIN_UNIT_COST, MAX_UNIT_COST)), 2)
    initial_stock = round(avg_daily_sales * STOCK_COVER_DAYS, 2)
    supplier = SUPPLIER_NAMES[supplier_index % len(SUPPLIER_NAMES)]

    return Inventory(
        product_id=product_id,
        current_stock=max(initial_stock, 1.0),  # never start at zero
        lead_time_days=lead_time,
        unit_cost=unit_cost,
        supplier_name=supplier,
    )


def seed(train: pd.DataFrame, store: pd.DataFrame, selected_store_ids: list[int]) -> None:
    """
    Insert products, sales history, and inventory rows into the database.

    All inserts run inside a single transaction — if anything fails the
    entire transaction is rolled back, leaving the database unchanged.

    Parameters:
        train: Cleaned training DataFrame (all stores).
        store: Cleaned store metadata DataFrame.
        selected_store_ids: List of store IDs to load.

    Raises:
        Exception: Any database or data-processing error (after rollback).
    """
    train_subset = train[train["Store"].isin(selected_store_ids)].copy()
    store_subset = store[store["Store"].isin(selected_store_ids)].copy()

    rng = np.random.default_rng(RANDOM_SEED)

    db = SessionLocal()
    try:
        # Truncate existing data so re-seeding is safe.
        logger.info("Clearing any existing seeded data...")
        for model in (AgentInteraction, RiskAlert, PurchaseOrder, Inventory, SalesHistory, Product):
            db.query(model).delete()
        db.flush()

        # --- Products ---
        logger.info("Inserting %d product rows...", len(selected_store_ids))
        store_meta = store_subset.set_index("Store")

        for store_id in selected_store_ids:
            meta = store_meta.loc[store_id]
            product = Product(
                product_id=int(store_id),
                product_name=f"Product-{store_id}",
                store_type=str(meta.get("StoreType", "")).strip() or None,
                assortment=str(meta.get("Assortment", "")).strip() or None,
                competition_distance=float(meta["CompetitionDistance"]),
            )
            db.add(product)

        db.flush()  # ensure products exist before FK inserts

        # --- Sales history ---
        logger.info("Inserting sales history rows — this may take a minute...")
        total_rows = 0
        BATCH_SIZE = 5_000

        for store_id in selected_store_ids:
            store_rows = train_subset[train_subset["Store"] == store_id]
            batch = []
            for _, row in store_rows.iterrows():
                batch.append(
                    SalesHistory(
                        product_id=int(store_id),
                        date=row["Date"].date(),
                        sales=int(row["Sales"]),
                        customers=int(row["Customers"]),
                        open=int(row["Open"]),
                        promo=int(row["Promo"]),
                        state_holiday=str(row["StateHoliday"]),
                        school_holiday=int(row["SchoolHoliday"]),
                    )
                )
                if len(batch) >= BATCH_SIZE:
                    db.bulk_save_objects(batch)
                    db.flush()
                    total_rows += len(batch)
                    batch = []

            if batch:
                db.bulk_save_objects(batch)
                db.flush()
                total_rows += len(batch)

            logger.info("  Store %d — %d rows inserted.", store_id, len(store_rows))

        # --- Inventory ---
        logger.info("Computing initial inventory levels...")
        avg_sales_per_store = (
            train_subset[train_subset["Open"] == 1]
            .groupby("Store")["Sales"]
            .mean()
        )

        for idx, store_id in enumerate(selected_store_ids):
            avg_daily = avg_sales_per_store.get(store_id, 100.0)
            inv_row = build_inventory_row(store_id, avg_daily, rng, idx)
            db.add(inv_row)

        db.flush()

        # --- Commit everything ---
        db.commit()

        # --- Summary ---
        date_range_start = train_subset["Date"].min().date()
        date_range_end = train_subset["Date"].max().date()

        logger.info("=" * 60)
        logger.info("Seeding complete.")
        logger.info("  Products loaded  : %d", len(selected_store_ids))
        logger.info("  Sales rows       : %d", total_rows)
        logger.info("  Date range       : %s → %s", date_range_start, date_range_end)
        logger.info("=" * 60)

    except Exception:
        db.rollback()
        logger.exception("Seeding failed — all changes have been rolled back.")
        raise
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Orchestrate the full seeding pipeline: connect, load, clean, select,
    seed.
    """
    logger.info("SupplyPilot — data seeding pipeline starting.")

    # 1. Verify database connectivity before doing any file I/O.
    logger.info("Testing database connection...")
    if not test_connection():
        logger.error(
            "Cannot reach the database. Check DATABASE_URL in .env and ensure "
            "your Supabase project is active."
        )
        sys.exit(1)

    # 2. Initialise schema (creates tables if they don't exist).
    init_db()

    # 3. Load and clean source data.
    try:
        train, store = load_and_clean(TRAIN_CSV, STORE_CSV)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Data loading failed: %s", exc)
        sys.exit(1)

    # 4. Select the 20 most complete stores.
    selected = select_top_stores(train, TOP_N_STORES)

    # 5. Seed the database inside a single transaction.
    seed(train, store, selected)

    logger.info("Done. Run 'python -m pytest tests/' to verify the setup.")


if __name__ == "__main__":
    main()
