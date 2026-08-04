"""
Tests for purchase order endpoints.

GET  /orders
POST /orders
GET  /orders/{id}
PATCH /orders/{id}/status

The POST test creates a real order and cleans it up via PATCH (reject)
so the DB is left in the same state it started in.
"""

import pytest
from tests.conftest import FIRST_PRODUCT_ID


# ---------------------------------------------------------------------------
# GET /orders
# ---------------------------------------------------------------------------

def test_list_orders_200(client):
    r = client.get("/orders")
    assert r.status_code == 200


def test_list_orders_schema(client):
    body = client.get("/orders").json()
    assert "orders" in body
    assert "total" in body
    assert isinstance(body["orders"], list)


def test_list_orders_status_filter_pending(client):
    r = client.get("/orders?status=pending")
    assert r.status_code == 200
    body = r.json()
    for order in body["orders"]:
        assert order["status"] == "pending"


def test_list_orders_invalid_status_422(client):
    r = client.get("/orders?status=shipped")
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# POST /orders  →  GET /orders/{id}  →  PATCH /orders/{id}/status
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def created_order(client):
    """
    Create a real purchase order for testing.  Yields the response body.
    The order is rejected after the test module finishes so the DB stays clean.
    """
    payload = {
        "product_id": FIRST_PRODUCT_ID,
        "quantity": 50,
        "reason": "Integration test order — will be rejected immediately.",
    }
    r = client.post("/orders", json=payload)
    assert r.status_code == 201, r.text
    yield r.json()

    # Cleanup: reject the order so it doesn't pollute the pending queue.
    order_id = r.json()["order_id"]
    client.patch(f"/orders/{order_id}/status", json={"status": "rejected"})


def test_create_order_201(client, created_order):
    assert created_order["status"] == "pending"


def test_create_order_schema(client, created_order):
    for field in ("order_id", "product_id", "quantity", "supplier_name",
                  "estimated_cost", "status", "message"):
        assert field in created_order, f"Missing field: {field}"


def test_create_order_estimated_cost_positive(client, created_order):
    assert created_order["estimated_cost"] > 0


def test_get_order_by_id(client, created_order):
    oid = created_order["order_id"]
    r = client.get(f"/orders/{oid}")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == oid
    assert body["product_id"] == FIRST_PRODUCT_ID


def test_get_order_unknown_404(client):
    r = client.get("/orders/9999999")
    assert r.status_code == 404


def test_approve_then_reject_conflict(client):
    """
    Create a fresh order, approve it, then try to reject it.
    The second transition must return 409.
    """
    payload = {
        "product_id": FIRST_PRODUCT_ID,
        "quantity": 10,
        "reason": "Conflict test — approve then reject.",
    }
    oid = client.post("/orders", json=payload).json()["order_id"]

    # Approve — should succeed
    r_approve = client.patch(f"/orders/{oid}/status", json={"status": "approved"})
    assert r_approve.status_code == 200
    assert r_approve.json()["status"] == "approved"

    # Reject the already-approved order — should conflict
    r_reject = client.patch(f"/orders/{oid}/status", json={"status": "rejected"})
    assert r_reject.status_code == 409


def test_create_order_unknown_product_404(client):
    payload = {
        "product_id": 999999,
        "quantity": 10,
        "reason": "Should 404 because product does not exist.",
    }
    r = client.post("/orders", json=payload)
    assert r.status_code == 404


def test_create_order_zero_quantity_422(client):
    payload = {
        "product_id": FIRST_PRODUCT_ID,
        "quantity": 0,
        "reason": "Zero qty should be rejected by Pydantic.",
    }
    r = client.post("/orders", json=payload)
    assert r.status_code == 422


def test_approve_order_increments_inventory_stock(client):
    """Verifies that approving a purchase order increments the product's current_stock."""
    # 1. Get initial inventory
    inv_before = client.get(f"/inventory/{FIRST_PRODUCT_ID}").json()
    stock_before = inv_before["current_stock"]

    # 2. Create order for 15 units
    order_qty = 15
    payload = {
        "product_id": FIRST_PRODUCT_ID,
        "quantity": order_qty,
        "reason": "Test stock increment on approval.",
    }
    oid = client.post("/orders", json=payload).json()["order_id"]

    # 3. Approve order
    r_approve = client.patch(f"/orders/{oid}/status", json={"status": "approved"})
    assert r_approve.status_code == 200

    # 4. Check inventory stock after approval
    inv_after = client.get(f"/inventory/{FIRST_PRODUCT_ID}").json()
    assert inv_after["current_stock"] == stock_before + order_qty

