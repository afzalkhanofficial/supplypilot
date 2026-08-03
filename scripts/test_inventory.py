"""
Smoke test for inventory/calculator.py.

Run from project root:
    python scripts/test_inventory.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inventory.calculator import (
    calculate_eoq,
    calculate_reorder_point,
    calculate_safety_stock,
    get_inventory_recommendation,
    z_score,
)

def header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

header("Formula unit tests")
print(f"z_score(0.95)                    = {z_score(0.95):.3f}  (expect 1.645)")
print(f"z_score(0.99)                    = {z_score(0.99):.3f}  (expect 2.326)")
eoq = calculate_eoq(10000, 50, unit_cost=20, holding_cost_rate=0.25)
print(f"EOQ(D=10000, S=50, c=20, h=0.25) = {eoq}     (expect ~448)")
ss  = calculate_safety_stock(500, 7, 0.95)
print(f"Safety stock(std=500, LT=7, 95%)   = {ss}   (expect ~2177)")
rop = calculate_reorder_point(1000, 7, 500, 0.95)
print(f"ROP(avg=1000, LT=7, std=500, 95%)  = {rop}   (expect ~9177)")

header("End-to-end recommendation (product 85)")
rec = get_inventory_recommendation(product_id=85)
print(json.dumps(rec, indent=2))

header("Spot-check 3 more products")
for pid in [262, 733, 863]:
    r = get_inventory_recommendation(product_id=pid)
    print(f"  Product {pid:>4} | stock={r['current_stock']:>8.0f} | "
          f"ROP={r['reorder_point']:>5} | EOQ={r['eoq']:>5} | "
          f"cover={r['days_of_cover']:>5.1f}d | risk={r['risk_level']}")

print("\nAll checks passed.")
