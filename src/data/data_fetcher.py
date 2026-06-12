"""Data fetcher for acquiring market data from brokers"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta, timezone
from .brokers.base_broker import BaseBroker
from .data_storage import DataStorage
from .data_preprocessor import DataPreprocessor


class DataFetcher:
    """Fetches and stores market data from brokers"""
    
    def __init__(self, broker: BaseBroker, storage: DataStorage):
        """
        Initialize data fetcher
        
        Args:
            broker: Broker instance to fetch data from
            storage: Data storage instance to store data
        """
        self.broker = broker
        self.storage = storage
        self.preprocessor = DataPreprocessor()
    
    def fetch_and_store(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        update: bool = True
    ) -> int:
        """
        Fetch historical data from broker and store in database
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            start_date: Start date for data
            end_date: End date for data (None for latest)
            update: Whether to update existing data
        
        Returns:
            Number of records stored
        """
        try:
            # Clamp dates to current time to avoid future date requests
            now_utc = datetime.now(timezone.utc)
            
            # Ensure start_date is not in the future
            if start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=timezone.utc)
            if start_date > now_utc:
                print(f"Warning: start_date {start_date} is in the future. Clamping to {now_utc}")
                start_date = now_utc - timedelta(days=30)  # Default to 30 days ago
            
            # Ensure end_date is not in the future
            if end_date:
                if end_date.tzinfo is None:
                    end_date = end_date.replace(tzinfo=timezone.utc)
                if end_date > now_utc:
                    print(f"Warning: end_date {end_date} is in the future. Clamping to {now_utc}")
                    end_date = now_utc
            else:
                end_date = now_utc
            
            # Ensure start_date is before end_date
            if start_date >= end_date:
                print(f"Warning: start_date {start_date} is not before end_date {end_date}. Adjusting...")
                start_date = end_date - timedelta(days=30)
            
            # Fetch data from broker
            data = self.broker.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if data is None or (hasattr(data, "empty") and data.empty) or (not hasattr(data, "empty") and not data):
                print(f"No data fetched for {symbol} {timeframe}")
                return 0
            
            # Clean data
            df = self.preprocessor.clean_ohlcv_data(data)
            
            if df.empty:
                print(f"No valid data after cleaning for {symbol} {timeframe}")
                return 0
            
            # Convert back to list of dicts for storage
            data_list = []
            for timestamp, row in df.iterrows():
                data_list.append({
                    'timestamp': timestamp,
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume']
                })
            
            # Store in database
            count = self.storage.store_ohlcv_data(symbol, timeframe, data_list)
            if count > 0:
                print(f"Stored {count} records for {symbol} {timeframe}")
            else:
                # Check if data already exists
                existing = self.storage.get_ohlcv_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    limit=1
                )
                if existing:
                    print(f"Data for {symbol} {timeframe} already exists in database (0 new records stored)")
                else:
                    print(f"Warning: No data stored for {symbol} {timeframe} - data may be invalid or fetch failed")
            
            return count
        except Exception as e:
            print(f"Error fetching and storing data: {e}")
            return 0
    
    def get_latest_data(
        self,
        symbol: str,
        timeframe: str,
        days: int = 30
    ) -> List[Dict]:
        """
        Get latest data, fetching from broker if needed
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            days: Number of days of data to fetch
        
        Returns:
            List of OHLCV dictionaries
        """
        # Try to get from storage first
        # Use UTC and clamp to "now" to avoid future-dated ranges
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        stored_data = self.storage.get_ohlcv_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        # If we have recent data, return it
        hours_ago = 999  # Default to very old if no stored data
        if stored_data:
            latest_timestamp = max(d['timestamp'] for d in stored_data)
            # Handle timezone-aware and naive timestamps
            if latest_timestamp.tzinfo is None:
                latest_timestamp = latest_timestamp.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            hours_ago = (now - latest_timestamp).total_seconds() / 3600
            
            # If data is less than 1 hour old, return it
            if hours_ago < 1:
                return stored_data
        
        # Otherwise, fetch fresh data with retry logic
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                self.fetch_and_store(symbol, timeframe, start_date, end_date)
                # Return from storage
                fresh_data = self.storage.get_ohlcv_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
                if fresh_data:
                    return fresh_data
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  ⚠️  Data fetch attempt {attempt + 1} failed for {symbol} {timeframe}: {e}")
                    print(f"     Retrying in {retry_delay} seconds...")
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"  ❌ Failed to fetch data for {symbol} {timeframe} after {max_retries} attempts: {e}")
                    # Fallback to cached data even if stale
                    if stored_data:
                        print(f"  ⚠️  Using stale cached data (last updated {hours_ago:.1f} hours ago)")
                        return stored_data
        
        # Final fallback: return cached data if available, even if stale
        if stored_data:
            print(f"  ⚠️  Using cached data (may be stale, {hours_ago:.1f} hours old)")
            return stored_data
        
        # No data available at all
        print(f"  ❌ No data available for {symbol} {timeframe}")
        return []

