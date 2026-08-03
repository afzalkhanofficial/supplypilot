"""
Database engine, session factory, ORM models, and utility functions.

All connection parameters are read from the DATABASE_URL environment
variable — never hardcoded.  The engine is configured for compatibility
with Supabase's PgBouncer transaction-mode pooler (port 6543).
"""

import logging
import os
from datetime import datetime, timezone
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine construction
# ---------------------------------------------------------------------------

def _build_database_url() -> str:
    """
    Read DATABASE_URL from the environment and ensure it uses the
    psycopg2 driver scheme.  Also appends sslmode=require when the URL
    targets Supabase (identified by .supabase.com in the hostname) so
    that connections are always encrypted.

    Returns:
        A fully-qualified database URL string.

    Raises:
        RuntimeError: If DATABASE_URL is not set.
    """
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL environment variable is not set. "
            "Copy .env.example to .env and fill in your Supabase connection string."
        )

    # SQLAlchemy expects postgresql+psycopg2:// or postgresql://
    # Both are treated identically by psycopg2; normalise to the standard form.
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    # Append sslmode=require for Supabase hosted connections when not already set.
    if "supabase.com" in url and "sslmode" not in url:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}sslmode=require"

    return url


_DATABASE_URL: str = _build_database_url()

# pool_pre_ping re-checks connections before handing them to the app,
# which is important when using PgBouncer because idle connections can
# be dropped by the pooler between requests.
# pool_recycle=300 closes and reopens connections every 5 minutes to
# avoid issues with long-lived connections through the pooler.
engine = create_engine(
    _DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    # PgBouncer transaction mode does not support prepared statements;
    # psycopg2 uses them by default, so we disable them here.
    connect_args={"options": "-c timezone=utc"},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ---------------------------------------------------------------------------
# ORM base & models
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


class Product(Base):
    """
    One row per product (= one Rossmann store).

    product_id matches the original Store ID from the dataset, which
    makes it straightforward to cross-reference the raw data.
    """

    __tablename__ = "products"

    product_id: int = Column(Integer, primary_key=True)
    product_name: str = Column(String(100), nullable=False)
    store_type: str = Column(String(1))             # a, b, c, or d
    assortment: str = Column(String(1))             # a=basic, b=extra, c=extended
    competition_distance: float = Column(Numeric(10, 2))

    # Relationships
    sales_history = relationship(
        "SalesHistory", back_populates="product", cascade="all, delete-orphan"
    )
    inventory = relationship(
        "Inventory", back_populates="product", uselist=False, cascade="all, delete-orphan"
    )
    purchase_orders = relationship(
        "PurchaseOrder", back_populates="product", cascade="all, delete-orphan"
    )
    risk_alerts = relationship(
        "RiskAlert", back_populates="product"
    )


class SalesHistory(Base):
    """
    Daily sales record for one product.

    Rows where open=0 are stored for completeness but are excluded from
    Prophet training (where open == 1 is enforced by the training script).
    """

    __tablename__ = "sales_history"
    __table_args__ = (
        UniqueConstraint("product_id", "date", name="uq_product_date"),
    )

    id: int = Column(BigInteger, primary_key=True, autoincrement=True)
    product_id: int = Column(Integer, ForeignKey("products.product_id", ondelete="CASCADE"), nullable=False)
    date = Column(Date, nullable=False)
    sales: int = Column(Integer, nullable=False)
    customers: int = Column(Integer, nullable=False)
    open: int = Column(SmallInteger, nullable=False)
    promo: int = Column(SmallInteger, nullable=False)
    state_holiday: str = Column(String(1), nullable=False)
    school_holiday: int = Column(SmallInteger, nullable=False)

    product = relationship("Product", back_populates="sales_history")


class Inventory(Base):
    """
    Current inventory state for one product.

    current_stock is initialised during seeding and updated whenever a
    purchase order is approved or a manual stock adjustment is made.
    """

    __tablename__ = "inventory"

    product_id: int = Column(
        Integer, ForeignKey("products.product_id", ondelete="CASCADE"), primary_key=True
    )
    current_stock: float = Column(Numeric(12, 2), nullable=False)
    lead_time_days: int = Column(Integer, nullable=False)
    unit_cost: float = Column(Numeric(10, 2), nullable=False)
    supplier_name: str = Column(String(200), nullable=False)
    last_updated = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    product = relationship("Product", back_populates="inventory")


class PurchaseOrder(Base):
    """
    A purchase order drafted by the agent and awaiting human approval.

    status transitions: pending → approved or pending → rejected.
    decided_at is set when a human acts on the order via the dashboard.
    """

    __tablename__ = "purchase_orders"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    product_id: int = Column(Integer, ForeignKey("products.product_id", ondelete="CASCADE"), nullable=False)
    quantity: int = Column(Integer, nullable=False)
    supplier_name: str = Column(String(200), nullable=False)
    estimated_cost: float = Column(Numeric(12, 2), nullable=False)
    status: str = Column(String(20), nullable=False, default="pending")
    agent_reasoning: str = Column(Text)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    decided_at = Column(DateTime(timezone=True), nullable=True)

    product = relationship("Product", back_populates="purchase_orders")


class RiskAlert(Base):
    """
    A risk signal raised by one of the agent's monitoring tools.

    product_id is nullable so system-wide alerts (not tied to a specific
    product) can also be stored here.
    """

    __tablename__ = "risk_alerts"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    product_id: int = Column(
        Integer, ForeignKey("products.product_id", ondelete="SET NULL"), nullable=True
    )
    alert_type: str = Column(String(100), nullable=False)
    message: str = Column(Text, nullable=False)
    severity: str = Column(String(20), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    product = relationship("Product", back_populates="risk_alerts")


class AgentInteraction(Base):
    """
    Audit trail for every question answered by the agent.

    tools_used is a comma-separated list of the tool names called during
    the agent's reasoning, in order. This lets us show users exactly how
    the agent reached its conclusion.
    """

    __tablename__ = "agent_interactions"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    user_question: str = Column(Text, nullable=False)
    agent_answer: str = Column(Text, nullable=False)
    tools_used: str = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a database session and closes it when
    the request is finished, whether it succeeded or raised an exception.

    Yields:
        An active SQLAlchemy Session bound to the configured engine.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Create all tables that do not yet exist in the database.

    This is safe to call on every startup — SQLAlchemy's create_all uses
    IF NOT EXISTS semantics, so existing tables and data are never dropped.

    Raises:
        Exception: Propagates any database connection or DDL error so the
            caller can decide whether to abort startup.
    """
    logger.info("Running init_db — creating missing tables if any...")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("init_db complete — all tables are present.")
    except Exception:
        logger.exception("init_db failed — could not create tables.")
        raise


def test_connection() -> bool:
    """
    Execute a trivial query to verify the database is reachable.

    Returns:
        True if the connection succeeded and the query returned a result,
        False otherwise.  Logs the outcome either way so the caller can
        check application logs without raising an exception.
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
            if result == 1:
                logger.info("Database connection test passed.")
                return True
            logger.warning("Database connection test returned unexpected result: %s", result)
            return False
    except Exception:
        logger.exception("Database connection test failed.")
        return False
