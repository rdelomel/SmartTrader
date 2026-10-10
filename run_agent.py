"""Entry point for running the trading agent.

SMARTTRADER_ENGINE=v2 (default) runs the research-backed core in src/core.
SMARTTRADER_ENGINE=v1 runs the legacy multi-agent bot in src/main.py.
"""

import sys
import os

# Add project root to path so we can import src as a package
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import and run
if __name__ == "__main__":
    if os.getenv('SMARTTRADER_ENGINE', 'v2').lower() == 'v1':
        from src.main import main
    else:
        from src.core.runner import main
    main()
