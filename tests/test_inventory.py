"""
Tests for inventory endpoints and the pure calculator formulas.

GET /inventory/scan
GET /inventory/{id}

Unit tests for:
  inventory.calculator.calculate_eoq
  inventory.calculator.calculate_safety_stock
  inventory.calculator.calculate_reorder_point
  inventory.calculator.z_score
"""

import math

import pytest
from tests.conftest import FIRST_PRODUCT_ID, KNOWN_PRODUCT_IDS

from inventory.calculator import (
    calculate_eoq,
    calculate_reorder_point,
    calculate_safety_stock,
    z_score,
)


# ---------------------------------------------------------------------------
# Pure formula unit tests
# ---------------------------------------------------------------------------

class TestZScore:
    def test_95th_percentile(self):
        assert z_score(0.95) == pytest.approx(1.645, abs=0.001)

    def test_99th_percentile(self):
        assert z_score(0.99) == pytest.approx(2.326, abs=0.001)

    def test_80th_percentile(self):
        assert z_score(0.80) == pytest.approx(0.842, abs=0.001)

    def test_interpolation_between_table_entries(self):
        # 0.90 is in the table; result should be between 0.85 and 0.95 values.
        z = z_score(0.90)
        assert 1.036 < z < 1.645

    def test_invalid_zero_raises(self):
        with pytest.raises(ValueError):
            z_score(0)

    def test_invalid_one_raises(self):
        with pytest.raises(ValueError):
            z_score(1)


class TestEOQ:
    def test_standard_formula(self):
        # EOQ = ceil(sqrt(2 * 10000 * 50 / 5)) = ceil(447.2) = 448
        result = calculate_eoq(10000, 50, holding_cost_per_unit=5.0)
        assert result == 448

    def test_from_unit_cost_and_rate(self):
        # H = 20 * 0.25 = 5.0 → same as above
        result = calculate_eoq(10000, 50, unit_cost=20.0, holding_cost_rate=0.25)
        assert result == 448

    def test_zero_demand_raises(self):
        with pytest.raises(ValueError):
            calculate_eoq(0, 50)

    def test_zero_order_cost_raises(self):
        with pytest.raises(ValueError):
            calculate_eoq(1000, 0)

    def test_result_is_positive_int(self):
        result = calculate_eoq(5000, 40, unit_cost=15.0)
        assert isinstance(result, int)
        assert result > 0


class TestSafetyStock:
    def test_known_values(self):
        # ss = ceil(1.645 * 500 * sqrt(7)) = ceil(2176.5) = 2177
        result = calculate_safety_stock(500, 7, 0.95)
        assert result == 2177

    def test_zero_std_gives_zero(self):
        result = calculate_safety_stock(0.0, 7, 0.95)
        assert result == 0

    def test_higher_service_level_gives_larger_buffer(self):
        ss_95 = calculate_safety_stock(200, 5, 0.95)
        ss_99 = calculate_safety_stock(200, 5, 0.99)
        assert ss_99 > ss_95

    def test_longer_lead_time_gives_larger_buffer(self):
        ss_3 = calculate_safety_stock(200, 3, 0.95)
        ss_7 = calculate_safety_stock(200, 7, 0.95)
        assert ss_7 > ss_3


class TestReorderPoint:
    def test_known_values(self):
        # ROP = ceil(1000 * 7 + 2177) = ceil(9177) = 9177
        result = calculate_reorder_point(1000, 7, 500, 0.95)
        assert result == 9177

    def test_rop_greater_than_cycle_stock(self):
        avg, lt, std = 100, 5, 20
        rop = calculate_reorder_point(avg, lt, std, 0.95)
        cycle_stock = avg * lt
        assert rop > cycle_stock

    def test_zero_demand_and_std(self):
        result = calculate_reorder_point(0.0, 7, 0.0, 0.95)
        assert result == 0


# ---------------------------------------------------------------------------
# Inventory API endpoint tests
# ---------------------------------------------------------------------------

class TestInventoryScan:
    def test_scan_200(self, client):
        r = client.get("/inventory/scan")
        assert r.status_code == 200

    def test_scan_returns_all_products(self, client):
        body = client.get("/inventory/scan").json()
        assert body["scanned"] == 20

    def test_scan_schema(self, client):
        body = client.get("/inventory/scan").json()
        assert "summary" in body
        assert "counts" in body
        counts = body["counts"]
        assert "CRITICAL" in counts
        assert "WARNING" in counts
        assert "OK" in counts

    def test_scan_count_totals_match_scanned(self, client):
        body = client.get("/inventory/scan").json()
        counts = body["counts"]
        assert counts["CRITICAL"] + counts["WARNING"] + counts["OK"] == body["scanned"]

    def test_scan_sorted_by_risk_then_cover(self, client):
        summary = client.get("/inventory/scan").json()["summary"]
        _order = {"CRITICAL": 0, "WARNING": 1, "OK": 2}
        for a, b in zip(summary, summary[1:]):
            ra, rb = _order[a["risk_level"]], _order[b["risk_level"]]
            if ra == rb:
                assert a["days_of_cover"] <= b["days_of_cover"]
            else:
                assert ra <= rb


class TestInventorySingle:
    def test_known_product_200(self, client):
        r = client.get(f"/inventory/{FIRST_PRODUCT_ID}")
        assert r.status_code == 200

    def test_schema_fields_present(self, client):
        body = client.get(f"/inventory/{FIRST_PRODUCT_ID}").json()
        for field in ("current_stock", "reorder_point", "eoq", "safety_stock",
                      "days_of_cover", "risk_level", "action", "service_level"):
            assert field in body, f"Missing field: {field}"

    def test_risk_level_valid(self, client):
        body = client.get(f"/inventory/{FIRST_PRODUCT_ID}").json()
        assert body["risk_level"] in ("OK", "WARNING", "CRITICAL")

    def test_eoq_and_rop_positive(self, client):
        body = client.get(f"/inventory/{FIRST_PRODUCT_ID}").json()
        assert body["eoq"] > 0
        assert body["reorder_point"] > 0

    def test_unknown_product_404(self, client):
        r = client.get("/inventory/999999")
        assert r.status_code == 404

    @pytest.mark.parametrize("pid", KNOWN_PRODUCT_IDS)
    def test_all_known_products(self, client, pid):
        r = client.get(f"/inventory/{pid}")
        assert r.status_code == 200
