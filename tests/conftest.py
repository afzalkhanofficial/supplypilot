"""
Shared pytest fixtures for the SupplyPilot API test suite.

The TestClient wraps the FastAPI app in a real ASGI lifespan cycle,
so the database startup probe runs just like production.  All tests
share a single session-scoped client to avoid repeated startup costs.

Environment
-----------
Tests require a running PostgreSQL instance pointed to by DATABASE_URL
in .env (the same DB used for local development).  There is no separate
test DB — the tests are read-heavy and the one write test (purchase
orders) cleans up after itself.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure project root is on sys.path so all domain modules resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.main import app  # noqa: E402 (must come after sys.path insert)


@pytest.fixture(scope="session")
def client() -> TestClient:
    """
    Session-scoped FastAPI TestClient.

    Using scope="session" means the app boots once for the entire test
    run, which avoids repeated DB connection overhead and Prophet model
    loading delays.
    """
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# Known product IDs that exist in the seeded DB.
KNOWN_PRODUCT_IDS = [85, 259, 262, 274, 310]
FIRST_PRODUCT_ID = KNOWN_PRODUCT_IDS[0]
