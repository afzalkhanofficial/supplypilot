"""
Tests for product and forecast endpoints.

GET /products
GET /products/{id}
GET /products/{id}/forecast
"""

import pytest
from tests.conftest import FIRST_PRODUCT_ID, KNOWN_PRODUCT_IDS


# ---------------------------------------------------------------------------
# GET /products
# ---------------------------------------------------------------------------

def test_list_products_200(client):
    r = client.get("/products")
    assert r.status_code == 200


def test_list_products_returns_all(client):
    body = client.get("/products").json()
    assert body["total"] == 20
    assert len(body["products"]) == 20


def test_list_products_schema(client):
    products = client.get("/products").json()["products"]
    first = products[0]
    assert "product_id" in first
    assert "product_name" in first
    assert isinstance(first["product_id"], int)


def test_list_products_ordered(client):
    products = client.get("/products").json()["products"]
    ids = [p["product_id"] for p in products]
    assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# GET /products/{id}
# ---------------------------------------------------------------------------

def test_get_product_known(client):
    r = client.get(f"/products/{FIRST_PRODUCT_ID}")
    assert r.status_code == 200
    body = r.json()
    assert body["product_id"] == FIRST_PRODUCT_ID


def test_get_product_unknown_404(client):
    r = client.get("/products/999999")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET /products/{id}/forecast
# ---------------------------------------------------------------------------

def test_forecast_default_horizon(client):
    r = client.get(f"/products/{FIRST_PRODUCT_ID}/forecast")
    assert r.status_code == 200
    body = r.json()
    assert body["days_ahead"] == 14
    assert len(body["dates"]) == 14
    assert len(body["yhat"]) == 14


def test_forecast_custom_horizon(client):
    r = client.get(f"/products/{FIRST_PRODUCT_ID}/forecast?days_ahead=7")
    assert r.status_code == 200
    body = r.json()
    assert body["days_ahead"] == 7
    assert len(body["dates"]) == 7


def test_forecast_values_non_negative(client):
    body = client.get(f"/products/{FIRST_PRODUCT_ID}/forecast?days_ahead=14").json()
    assert all(v >= 0 for v in body["yhat"])
    assert all(v >= 0 for v in body["yhat_lower"])
    assert all(v >= 0 for v in body["yhat_upper"])


def test_forecast_total_matches_sum(client):
    body = client.get(f"/products/{FIRST_PRODUCT_ID}/forecast?days_ahead=14").json()
    assert abs(body["total_forecast"] - sum(body["yhat"])) < 0.1


def test_forecast_out_of_range_422(client):
    r = client.get(f"/products/{FIRST_PRODUCT_ID}/forecast?days_ahead=0")
    assert r.status_code == 422


def test_forecast_unknown_product_404(client):
    r = client.get("/products/999999/forecast?days_ahead=7")
    assert r.status_code == 404


@pytest.mark.parametrize("pid", KNOWN_PRODUCT_IDS)
def test_forecast_all_known_products(client, pid):
    r = client.get(f"/products/{pid}/forecast?days_ahead=7")
    assert r.status_code == 200
