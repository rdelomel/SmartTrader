import os
import sys
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_storage import OHLCVData, DataStorage

def remove_duplicates():
    print("Connecting to database to remove duplicates...")
    storage = DataStorage()
    session = storage.get_session()
    
    try:
        # Get count before
        total_before = session.query(func.count(OHLCVData.id)).scalar()
        print(f"Total rows before deduplication: {total_before}")
        
        # Find duplicates by keeping only the minimum ID for each symbol, timeframe, timestamp
        subq = session.query(
            func.min(OHLCVData.id).label('min_id')
        ).group_by(
            OHLCVData.symbol,
            OHLCVData.timeframe,
            OHLCVData.timestamp
        ).subquery()
        
        # Delete rows where ID is not in the subquery
        deleted = session.query(OHLCVData).filter(OHLCVData.id.notin_(subq)).delete(synchronize_session=False)
        session.commit()
        
        # Get count after
        total_after = session.query(func.count(OHLCVData.id)).scalar()
        print(f"Total rows after deduplication: {total_after}")
        print(f"Removed {deleted} duplicate rows.")
        
        # Calculate integrity score
        if total_before > 0:
            integrity = (total_after / total_before) * 100
            print(f"Data integrity improved from {integrity:.1f}% to 100.0%.")
            
    except Exception as e:
        session.rollback()
        print(f"Error removing duplicates: {e}")
    finally:
        session.close()

if __name__ == '__main__':
    remove_duplicates()
