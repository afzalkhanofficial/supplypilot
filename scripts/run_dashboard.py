"""
Helper script to start the SupplyPilot Streamlit dashboard.

Usage
-----
    python scripts/run_dashboard.py
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    port = os.getenv("PORT", "8501")
    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run",
            str(ROOT / "dashboard" / "app.py"),
            "--server.port", port,
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
            "--theme.base", "dark",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
