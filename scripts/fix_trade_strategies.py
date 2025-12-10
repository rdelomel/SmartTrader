"""Fix trade strategies for existing trades that are marked as 'unknown'"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_storage import DataStorage
from src.data.brokers.oanda_api import OANDABroker
from src.data.brokers.alpaca_api import AlpacaBroker
from dotenv import load_dotenv

load_dotenv()

def fix_trade_strategies():
    """Fix strategies for trades that are marked as 'unknown' by checking brokers"""
    storage = DataStorage()
    
    # Get all open trades
    open_trades = storage.get_open_trades()
    
    print("\n" + "="*80)
    print("FIXING TRADE STRATEGIES")
    print("="*80)
    
    if not open_trades:
        print("\nNo open trades found.")
        return
    
    # Initialize brokers to check positions
    brokers = {}
    
    # Try to initialize OANDA
    oanda_key = os.getenv('OANDA_API_KEY')
    oanda_account = os.getenv('OANDA_ACCOUNT_ID')
    if oanda_key:
        try:
            oanda = OANDABroker(oanda_key, testnet=True, account_id=oanda_account)
            if oanda.connect():
                brokers['oanda'] = oanda
                print(f"✅ Connected to OANDA")
        except Exception as e:
            print(f"⚠️  Could not connect to OANDA: {e}")
    
    # Try to initialize Alpaca
    alpaca_key = os.getenv('ALPACA_API_KEY')
    alpaca_secret = os.getenv('ALPACA_API_SECRET')
    if alpaca_key and alpaca_secret:
        try:
            alpaca = AlpacaBroker(alpaca_key, alpaca_secret, testnet=True)
            if alpaca.connect():
                brokers['alpaca'] = alpaca
                print(f"✅ Connected to Alpaca")
        except Exception as e:
            print(f"⚠️  Could not connect to Alpaca: {e}")
    
    if not brokers:
        print("\n⚠️  No brokers available. Cannot determine trade sources.")
        return
    
    # Get positions from all brokers
    broker_positions = {}
    for broker_name, broker in brokers.items():
        try:
            positions = broker.get_open_positions()
            broker_positions[broker_name] = positions
            print(f"\nFound {len(positions)} positions in {broker_name}")
        except Exception as e:
            print(f"Error getting positions from {broker_name}: {e}")
            broker_positions[broker_name] = []
    
    # Fix strategies for trades marked as 'unknown'
    fixed_count = 0
    for trade in open_trades:
        strategy = trade.get('strategy')
        symbol = trade.get('symbol', '')
        side = trade.get('side', '')
        entry_price = trade.get('entry_price', 0)
        trade_id = trade.get('trade_id')
        
        # Only fix trades with 'unknown' or None strategy
        if strategy and strategy != 'unknown' and strategy != 'None':
            continue
        
        # Check if this trade exists in any broker
        found_in_broker = None
        for broker_name, positions in broker_positions.items():
            for pos in positions:
                pos_symbol = pos.get('symbol', '').replace('_', '/')
                pos_side = pos.get('side', 'buy')
                pos_entry = pos.get('entry_price', 0)
                
                # Match by symbol, side, and entry price (within 0.1% tolerance)
                if (pos_symbol == symbol and 
                    pos_side == side and
                    entry_price > 0 and
                    abs(pos_entry - entry_price) / entry_price < 0.001):
                    found_in_broker = broker_name
                    break
            
            if found_in_broker:
                break
        
        if found_in_broker:
            # This is an external trade synced from broker
            new_strategy = f'{found_in_broker}_sync'
            storage.update_trade(trade_id, {'strategy': new_strategy})
            print(f"✅ Fixed: {symbol} {side} @ ${entry_price:.2f} -> {new_strategy}")
            fixed_count += 1
        else:
            # Not found in any broker - might be a SmartTrader trade that lost its strategy
            # Or it might have been closed. For now, mark as 'smarttrader' if we can't find it
            # (This is a conservative approach - assume it's ours if not in broker)
            storage.update_trade(trade_id, {'strategy': 'smarttrader'})
            print(f"⚠️  Not found in brokers: {symbol} {side} @ ${entry_price:.2f} -> marked as 'smarttrader'")
            fixed_count += 1
    
    print("\n" + "="*80)
    print(f"SUMMARY:")
    print(f"  Fixed trades: {fixed_count}")
    print("="*80 + "\n")

if __name__ == '__main__':
    fix_trade_strategies()

