"""Install dependencies for SmartTrader"""

import subprocess
import sys

packages = [
    'pyyaml',
    'python-dotenv',
    'pandas',
    'numpy',
    'requests',
    'sqlalchemy',
    'fastapi',
    'uvicorn',
    'scikit-learn',
    'xgboost',
    'structlog',
    'websockets'
]

print("Installing dependencies...")
for package in packages:
    print(f"Installing {package}...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', package])
        print(f"✓ {package} installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package}: {e}")

print("\nChecking installations...")
for package in packages:
    try:
        if package == 'pyyaml':
            __import__('yaml')
        elif package == 'python-dotenv':
            __import__('dotenv')
        else:
            __import__(package)
        print(f"✓ {package} is available")
    except ImportError:
        print(f"✗ {package} is NOT available")

print("\nInstallation complete!")

