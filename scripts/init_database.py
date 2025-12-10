#!/usr/bin/env python3
"""Initialize database tables"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_storage import DataStorage
from sqlalchemy import inspect

def main():
    """Initialize database tables"""
    print("=" * 60)
    print("Initializing database tables...")
    print("=" * 60)
    
    try:
        storage = DataStorage()
        
        # Verify tables exist
        inspector = inspect(storage.engine)
        existing_tables = inspector.get_table_names()
        expected_tables = ['trades', 'ohlcv_data', 'model_predictions']
        
        print("\nDatabase tables status:")
        print("-" * 60)
        for table in expected_tables:
            status = "✓ EXISTS" if table in existing_tables else "✗ MISSING"
            print(f"  {table:20s} {status}")
        
        print("\nAll existing tables:")
        for table in existing_tables:
            print(f"  - {table}")
        
        if all(t in existing_tables for t in expected_tables):
            print("\n✓ All required tables exist!")
            return 0
        else:
            print("\n✗ Some tables are missing. Attempting to create...")
            storage.create_tables()
            
            # Re-check
            inspector = inspect(storage.engine)
            existing_tables = inspector.get_table_names()
            if all(t in existing_tables for t in expected_tables):
                print("\n✓ Tables created successfully!")
                return 0
            else:
                print("\n✗ Failed to create all tables")
                return 1
                
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

