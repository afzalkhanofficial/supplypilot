"""
Helper script to start the SupplyPilot Streamlit dashboard.

Usage
-----
    python scripts/run_dashboard.py

Equivalent to:
    streamlit run dashboard/app.py --server.port 8501
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run",
            str(ROOT / "dashboard" / "app.py"),
            "--server.port", "8501",
            "--server.headless", "false",
            "--theme.base", "dark",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
