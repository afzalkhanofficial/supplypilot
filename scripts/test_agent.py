"""
Quick smoke test for the agent module (tools, prompt, and a live LLM call).
Run from project root: python scripts/test_agent.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---- 1. Tool imports ----
print("=" * 60)
print("1. Tool imports and list_products")
print("=" * 60)
from agent.tools import ALL_TOOLS  # noqa: E402

print(f"Tools registered: {[t.name for t in ALL_TOOLS]}")

result = json.loads(ALL_TOOLS[0].invoke({"query": ""}))
print(f"Total products   : {result['total']}")
print(f"First 3 IDs      : {[p['product_id'] for p in result['products'][:3]]}")

# ---- 2. Inventory status tool ----
print()
print("=" * 60)
print("2. get_inventory_status (product 85)")
print("=" * 60)
inv = json.loads(ALL_TOOLS[2].invoke({"product_id": 85}))
print(f"risk_level     : {inv['risk_level']}")
print(f"current_stock  : {inv['current_stock']}")
print(f"reorder_point  : {inv['reorder_point']}")
print(f"eoq            : {inv['eoq']}")
print(f"action preview : {inv['action'][:90]}...")

# ---- 3. Prompt template ----
print()
print("=" * 60)
print("3. Prompt template structure")
print("=" * 60)
from agent.prompts import build_prompt  # noqa: E402

prompt = build_prompt()
print(f"Message slots: {[type(m).__name__ for m in prompt.messages]}")

# ---- 4. Live agent call ----
print()
print("=" * 60)
print("4. Live agent call (single question via Groq)")
print("=" * 60)
from agent.agent import run_agent  # noqa: E402

response = run_agent("List all products and tell me which three have the lowest stock.")
print(f"Tools used : {response['tools_used']}")
print(f"Steps      : {response['steps']}")
print()
print("--- AGENT ANSWER ---")
print(response["answer"])
print("--- END ---")
print()
print("All checks passed.")
