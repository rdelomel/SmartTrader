"""Entry point for running the trading agent"""

import sys
import os

# Add project root to path so we can import src as a package
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import and run
if __name__ == "__main__":
    from src.main import main
    main()

