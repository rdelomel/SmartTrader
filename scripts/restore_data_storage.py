#!/usr/bin/env python3
"""One-shot restore of data_storage.py from last known good commit."""
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/rdelomel/SmartTrader/6ae2f57eb882504c7a5bde673e39454d05eb0562/src/data/data_storage.py"
TARGET = Path(__file__).resolve().parent.parent / "src" / "data" / "data_storage.py"

print(f"Downloading good data_storage.py from {URL}")
data = urllib.request.urlopen(URL, timeout=30).read()
TARGET.write_bytes(data)
print(f"Wrote {len(data)} bytes to {TARGET}")
print("Done. Restart the agent.")
