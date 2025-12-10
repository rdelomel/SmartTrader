"""Check which trades were created by SmartTrader vs external"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_storage import DataStorage
from datetime import datetime

def check_trades():
    """Check trade sources"""
    storage = DataStorage()
    
    # Get all open trades
    open_trades = storage.get_open_trades()
    
    print("\n" + "="*80)
    print("TRADE SOURCE ANALYSIS")
    print("="*80)
    
    if not open_trades:
        print("\nNo open trades found.")
        return
    
    smarttrader_trades = []
    external_trades = []
    
    for trade in open_trades:
        strategy = trade.get('strategy') or 'unknown'
        symbol = trade.get('symbol', 'N/A')
        side = trade.get('side', 'N/A')
        quantity = trade.get('quantity', 0)
        entry_price = trade.get('entry_price', 0)
        
        # Normalize strategy to string
        strategy_str = str(strategy) if strategy else 'unknown'
        
        if strategy_str == 'smarttrader' or (strategy_str != 'unknown' and '_sync' not in strategy_str):
            smarttrader_trades.append(trade)
            source = "✅ SmartTrader"
        elif '_sync' in strategy_str:
            external_trades.append(trade)
            source = "⚠️  External (synced from broker)"
        else:
            external_trades.append(trade)
            source = "❓ Unknown source"
        
        print(f"\n{source}")
        print(f"  Symbol: {symbol}")
        print(f"  Side: {side}")
        print(f"  Quantity: {quantity}")
        print(f"  Entry Price: ${entry_price:.2f}")
        print(f"  Strategy: {strategy}")
    
    print("\n" + "="*80)
    print(f"SUMMARY:")
    print(f"  SmartTrader trades: {len(smarttrader_trades)}")
    print(f"  External trades: {len(external_trades)}")
    print(f"  Total: {len(open_trades)}")
    print("="*80 + "\n")

if __name__ == '__main__':
    check_trades()

