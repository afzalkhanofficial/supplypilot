"""
Helper script to start the SupplyPilot API server.

Usage
-----
    python scripts/run_api.py             # default: port 8000, reload on
    python scripts/run_api.py --port 9000 # custom port
    python scripts/run_api.py --no-reload # production-like (no hot reload)
"""

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path so all modules resolve correctly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SupplyPilot FastAPI server.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable auto-reload (use for production-like runs).",
    )
    args = parser.parse_args()

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
