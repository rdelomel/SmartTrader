"""Manual data fetching script to speed up model training"""

import os
import sys
from datetime import datetime, timedelta, timezone

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

try:
    from src.data.brokers.oanda_api import OANDABroker
    from src.data.brokers.alpaca_api import AlpacaBroker
    from src.data.data_storage import DataStorage
    from src.data.data_fetcher import DataFetcher
    from src.indicators.technical import TechnicalIndicators
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data.brokers.oanda_api import OANDABroker
    from src.data.brokers.alpaca_api import AlpacaBroker
    from src.data.data_storage import DataStorage
    from src.data.data_fetcher import DataFetcher
    from src.indicators.technical import TechnicalIndicators

import yaml

def fetch_all_data(days=60):
    """
    Manually fetch historical data for all configured symbols
    
    Args:
        days: Number of days of historical data to fetch
    """
    print(f"Starting manual data fetch for {days} days of history...")
    
    # Load configuration
    with open('config/trading_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize brokers
    brokers = {}
    
    # OANDA for forex
    oanda_key = os.getenv('OANDA_API_KEY')
    if oanda_key:
        account_id_env = os.getenv('OANDA_ACCOUNT_ID', '').strip()
        if account_id_env and not account_id_env.startswith('#'):
            account_id = account_id_env
        else:
            account_id = None
        
        try:
            brokers['forex'] = OANDABroker(
                oanda_key,
                api_secret=None,
                account_id=account_id,
                testnet=os.getenv('OANDA_ENVIRONMENT', 'practice') == 'practice'
            )
            print("✓ OANDA broker initialized")
        except Exception as e:
            print(f"✗ Failed to initialize OANDA: {e}")
    
    # Alpaca for stocks and crypto
    alpaca_key = os.getenv('ALPACA_API_KEY')
    alpaca_secret = os.getenv('ALPACA_API_SECRET')
    if alpaca_key and alpaca_secret:
        try:
            brokers['stocks'] = AlpacaBroker(
                alpaca_key,
                alpaca_secret,
                testnet='paper' in os.getenv('ALPACA_BASE_URL', '').lower()
            )
            print("✓ Alpaca broker initialized")
        except Exception as e:
            print(f"✗ Failed to initialize Alpaca: {e}")
    
    if not brokers:
        print("✗ No brokers available. Please check your API keys in .env file.")
        return
    
    # Initialize storage
    storage = DataStorage()
    indicators = TechnicalIndicators()
    
    # Get symbols from config (new structure uses 'assets')
    assets_config = config.get('assets', {})
    forex_config = assets_config.get('forex', {})
    crypto_config = assets_config.get('crypto', {})
    stocks_config = assets_config.get('stocks', {})
    
    forex_symbols = forex_config.get('symbols', []) if forex_config.get('enabled', False) else []
    crypto_symbols = crypto_config.get('symbols', []) if crypto_config.get('enabled', False) else []
    stock_symbols = stocks_config.get('symbols', []) if stocks_config.get('enabled', False) else []
    
    # Get timeframes
    timeframes_config = config.get('timeframes', {})
    if isinstance(timeframes_config, dict):
        timeframes = [timeframes_config.get('primary', '1h')] + timeframes_config.get('analysis', [])
    else:
        timeframes = ['1h']  # Default
    
    total_fetched = 0
    total_errors = 0
    
    # Fetch forex data
    if 'forex' in brokers and forex_symbols:
        print(f"\n📊 Fetching {len(forex_symbols)} forex symbols...")
        for symbol in forex_symbols:
            for timeframe in timeframes:
                try:
                    fetcher = DataFetcher(brokers['forex'], storage)
                    data_list = fetcher.get_latest_data(symbol, timeframe, days=days)
                    
                    if data_list:
                        # Add indicators
                        import pandas as pd
                        df = pd.DataFrame(data_list)
                        if len(df) > 0:
                            df.set_index('timestamp', inplace=True)
                            df = indicators.add_all_indicators(df)
                            
                            # Store with indicators
                            stored_count = 0
                            for _, row in df.iterrows():
                                try:
                                    storage.store_market_data(
                                        symbol=symbol,
                                        timeframe=timeframe,
                                        timestamp=row.name,
                                        data=row.to_dict()
                                    )
                                    stored_count += 1
                                except Exception as store_error:
                                    # Data might already exist, that's okay
                                    pass
                            
                            print(f"  ✓ {symbol} {timeframe}: {stored_count} new records stored (from {len(df)} fetched)")
                            total_fetched += stored_count
                        else:
                            print(f"  ⚠ {symbol} {timeframe}: No data in DataFrame")
                    else:
                        print(f"  ⚠ {symbol} {timeframe}: No data fetched from broker")
                except Exception as e:
                    import traceback
                    print(f"  ✗ {symbol} {timeframe}: {type(e).__name__}: {e}")
                    if '--verbose' in sys.argv or '-v' in sys.argv:
                        traceback.print_exc()
                    total_errors += 1
    
    # Fetch crypto data
    if 'stocks' in brokers and crypto_symbols:
        print(f"\n📊 Fetching {len(crypto_symbols)} crypto symbols...")
        for symbol in crypto_symbols:
            for timeframe in timeframes:
                try:
                    fetcher = DataFetcher(brokers['stocks'], storage)
                    data_list = fetcher.get_latest_data(symbol, timeframe, days=days)
                    
                    if data_list:
                        import pandas as pd
                        df = pd.DataFrame(data_list)
                        if len(df) > 0:
                            df.set_index('timestamp', inplace=True)
                            df = indicators.add_all_indicators(df)
                            
                            stored_count = 0
                            for _, row in df.iterrows():
                                try:
                                    storage.store_market_data(
                                        symbol=symbol,
                                        timeframe=timeframe,
                                        timestamp=row.name,
                                        data=row.to_dict()
                                    )
                                    stored_count += 1
                                except Exception as store_error:
                                    # Data might already exist, that's okay
                                    pass
                            
                            print(f"  ✓ {symbol} {timeframe}: {stored_count} new records stored (from {len(df)} fetched)")
                            total_fetched += stored_count
                        else:
                            print(f"  ⚠ {symbol} {timeframe}: No data in DataFrame")
                    else:
                        print(f"  ⚠ {symbol} {timeframe}: No data fetched from broker")
                except Exception as e:
                    import traceback
                    print(f"  ✗ {symbol} {timeframe}: {type(e).__name__}: {e}")
                    if '--verbose' in sys.argv or '-v' in sys.argv:
                        traceback.print_exc()
                    total_errors += 1
    
    # Fetch stock data
    if 'stocks' in brokers and stock_symbols:
        print(f"\n📊 Fetching {len(stock_symbols)} stock symbols...")
        for symbol in stock_symbols:
            for timeframe in timeframes:
                try:
                    fetcher = DataFetcher(brokers['stocks'], storage)
                    data_list = fetcher.get_latest_data(symbol, timeframe, days=days)
                    
                    if data_list:
                        import pandas as pd
                        df = pd.DataFrame(data_list)
                        if len(df) > 0:
                            df.set_index('timestamp', inplace=True)
                            df = indicators.add_all_indicators(df)
                            
                            for _, row in df.iterrows():
                                storage.store_market_data(
                                    symbol=symbol,
                                    timeframe=timeframe,
                                    timestamp=row.name,
                                    data=row.to_dict()
                                )
                            
                            print(f"  ✓ {symbol} {timeframe}: {len(df)} records")
                            total_fetched += len(df)
                        else:
                            print(f"  ⚠ {symbol} {timeframe}: No data")
                    else:
                        print(f"  ⚠ {symbol} {timeframe}: No data fetched")
                except Exception as e:
                    print(f"  ✗ {symbol} {timeframe}: {e}")
                    total_errors += 1
    
    print(f"\n✅ Data fetch complete!")
    print(f"   Total records fetched: {total_fetched}")
    if total_errors > 0:
        print(f"   Errors: {total_errors}")
    print(f"\n💡 Models will now be able to train with this data.")
    print(f"   Restart the trading agent to trigger model training.")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Manually fetch historical data for model training')
    parser.add_argument('--days', type=int, default=60, 
                       help='Number of days of historical data to fetch (default: 60)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed error messages')
    
    args = parser.parse_args()
    
    fetch_all_data(days=args.days)

