"""Wipe all trades from trading.db for a fresh monitoring baseline.

Usage on NAS (preferred — hits the live container DB via API after deploy):
  curl -X POST 'http://127.0.0.1:8000/api/purge_all_trades?confirm=yes'

Or directly against sqlite:
  python scripts/purge_all_trades.py --db sqlite:///./data/trading.db
  python scripts/purge_all_trades.py --db sqlite:///./trading.db
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description="Delete all SmartTrader trade rows")
    parser.add_argument("--db", default=None, help="SQLAlchemy URL override")
    parser.add_argument("--yes", action="store_true", help="Skip interactive confirm")
    args = parser.parse_args()

    if args.db:
        os.environ["DATABASE_URL"] = args.db

    if not args.yes:
        print("This will DELETE ALL rows in the trades table.")
        ans = input("Type YES to continue: ").strip()
        if ans != "YES":
            print("Aborted.")
            return 1

    from src.data.data_storage import DataStorage

    storage = DataStorage()
    deleted = storage.delete_all_trades()
    print(f"Deleted {deleted} trades.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
