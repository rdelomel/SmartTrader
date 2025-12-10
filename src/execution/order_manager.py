"""Order management and execution"""

from typing import Dict, List, Optional
from datetime import datetime
from ..data.brokers.base_broker import BaseBroker, OrderSide, OrderType
from ..data.data_storage import DataStorage


class OrderManager:
    """Manage order execution and tracking"""
    
    def __init__(self, broker: BaseBroker, storage: DataStorage, paper_trading: bool = True):
        """
        Initialize order manager
        
        Args:
            broker: Broker instance
            storage: Data storage instance
            paper_trading: Whether to use paper trading
        """
        self.broker = broker
        self.storage = storage
        self.paper_trading = paper_trading
        self.open_orders: Dict[str, Dict] = {}
    
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict:
        """
        Place a trading order
        
        Args:
            symbol: Trading symbol
            side: BUY or SELL
            quantity: Order quantity
            order_type: Order type
            price: Limit price (for LIMIT orders)
            stop_loss: Stop loss price
            take_profit: Take profit price
        
        Returns:
            Order result dictionary
        """
        print(f"\n[OrderManager.place_order]")
        print(f"  Symbol: {symbol}")
        print(f"  Side: {side.value if hasattr(side, 'value') else side}")
        print(f"  Quantity: {quantity}")
        print(f"  Order Type: {order_type.value if hasattr(order_type, 'value') else order_type}")
        print(f"  Paper Trading: {self.paper_trading}")
        print(f"  Broker: {type(self.broker).__name__}")
        print(f"  Broker Connected: {getattr(self.broker, 'connected', False)}")
        
        try:
            # Always use broker API - no local simulation
            # All trades go directly to Alpaca/OANDA/Binance APIs
            if not hasattr(self.broker, 'place_order'):
                error_msg = 'Broker does not support place_order method'
                print(f"  ❌ {error_msg}")
                return {
                    'order_id': None,
                    'status': 'rejected',
                    'error': error_msg
                }
            
            if not getattr(self.broker, 'connected', False):
                error_msg = 'Broker is not connected'
                print(f"  ❌ {error_msg}")
                return {
                    'order_id': None,
                    'status': 'rejected',
                    'error': error_msg
                }
            
            print(f"  📤 Calling broker.place_order()...")
            # Use broker API directly (works for both paper trading API and live trading)
            # Alpaca and OANDA paper trading APIs will show trades in their dashboards
            result = self.broker.place_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            print(f"  Broker returned: {result}")
            
            # Store order
            if result.get('order_id'):
                self.open_orders[result['order_id']] = result
                print(f"  ✅ Order stored with ID: {result['order_id']}")
            else:
                print(f"  ⚠️  No order_id in result")
            
            # Log trade to database
            if result.get('status'):
                status_value = result['status'].value if hasattr(result['status'], 'value') else str(result['status'])
                print(f"  Order status: {status_value}")
                if status_value in ['filled', 'FILLED']:
                    print(f"  📝 Logging filled trade to database...")
                    self._log_trade(result, symbol, side)
            
            if result.get('error'):
                print(f"  ❌ Order error: {result.get('error')}")
            
            return result
        except Exception as e:
            import traceback
            print(f"  ❌❌❌ EXCEPTION in OrderManager.place_order: {e}")
            print(f"  Traceback: {traceback.format_exc()}")
            return {
                'order_id': None,
                'status': 'rejected',
                'error': str(e)
            }
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order"""
        try:
            if order_id in self.open_orders:
                success = self.broker.cancel_order(order_id)
                if success:
                    del self.open_orders[order_id]
                return success
            return False
        except Exception as e:
            print(f"Error cancelling order: {e}")
            return False
    
    def get_open_orders(self) -> List[Dict]:
        """Get all open orders"""
        return list(self.open_orders.values())
    
    def close_position(
        self,
        symbol: str,
        side: Optional[OrderSide] = None
    ) -> bool:
        """Close an open position"""
        try:
            return self.broker.close_position(symbol, side)
        except Exception as e:
            print(f"Error closing position: {e}")
            return False
    
    def _log_trade(self, order_result: Dict, symbol: str, side: OrderSide):
        """Log trade to database"""
        try:
            trade_data = {
                'trade_id': order_result.get('order_id', f"trade_{datetime.now().timestamp()}"),
                'symbol': symbol,
                'side': side.value,
                'quantity': order_result.get('filled_quantity', 0),
                'entry_price': order_result.get('price', 0),
                'entry_time': order_result.get('timestamp', datetime.now()),
                'status': 'open',
                'strategy': 'smarttrader'  # Mark as created by our system
            }
            
            self.storage.store_trade(trade_data)
        except Exception as e:
            print(f"Error logging trade: {e}")

