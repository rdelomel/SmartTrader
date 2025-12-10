"""Binance API implementation for cryptocurrency trading"""

import ccxt
from typing import Dict, List, Optional
from datetime import datetime
import time

from .base_broker import BaseBroker, OrderType, OrderSide, OrderStatus


class BinanceBroker(BaseBroker):
    """Binance broker implementation using CCXT library"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        super().__init__(api_key, api_secret, testnet)
        self.exchange = None
        self._initialize_exchange()
    
    def _initialize_exchange(self):
        """Initialize CCXT exchange instance"""
        exchange_class = getattr(ccxt, 'binance')
        self.exchange = exchange_class({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',  # 'spot', 'future', 'delivery'
            }
        })
        
        if self.testnet:
            self.exchange.set_sandbox_mode(True)
    
    def connect(self) -> bool:
        """Establish connection to Binance"""
        try:
            # Test connection by fetching account balance
            self.exchange.load_markets()
            self.connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to Binance: {e}")
            self.connected = False
            return False
    
    def disconnect(self) -> bool:
        """Close connection"""
        try:
            if self.exchange:
                self.exchange.close()
            self.connected = False
            return True
        except Exception as e:
            print(f"Error disconnecting from Binance: {e}")
            return False
    
    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """Fetch historical OHLCV data from Binance"""
        try:
            # Convert timeframe to CCXT format
            timeframe_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1w'
            }
            tf = timeframe_map.get(timeframe, '1h')
            
            # Convert datetime to milliseconds timestamp
            since = int(start_date.timestamp() * 1000)
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, tf, since, limit)
            
            # Convert to standard format
            data = []
            for candle in ohlcv:
                data.append({
                    'timestamp': datetime.fromtimestamp(candle[0] / 1000),
                    'open': candle[1],
                    'high': candle[2],
                    'low': candle[3],
                    'close': candle[4],
                    'volume': candle[5]
                })
            
            return data
        except Exception as e:
            print(f"Error fetching historical data from Binance: {e}")
            return []
    
    def get_current_price(self, symbol: str) -> float:
        """Get current market price"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker['last'])
        except Exception as e:
            print(f"Error fetching current price from Binance: {e}")
            return 0.0
    
    def get_account_balance(self) -> Dict[str, float]:
        """Get account balance"""
        try:
            balance = self.exchange.fetch_balance()
            
            result = {
                'total': 0.0,
                'available': 0.0,
                'used': 0.0,
                'currencies': {}
            }
            
            for currency, amounts in balance.items():
                if currency not in ['info', 'free', 'used', 'total']:
                    if isinstance(amounts, dict):
                        result['currencies'][currency] = {
                            'free': amounts.get('free', 0.0),
                            'used': amounts.get('used', 0.0),
                            'total': amounts.get('total', 0.0)
                        }
            
            # Calculate totals in base currency (USDT)
            if 'USDT' in result['currencies']:
                result['total'] = result['currencies']['USDT']['total']
                result['available'] = result['currencies']['USDT']['free']
                result['used'] = result['currencies']['USDT']['used']
            
            return result
        except Exception as e:
            print(f"Error fetching account balance from Binance: {e}")
            return {'total': 0.0, 'available': 0.0, 'used': 0.0, 'currencies': {}}
    
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None
    ) -> Dict:
        """Place an order on Binance"""
        try:
            order_side = 'buy' if side == OrderSide.BUY else 'sell'
            
            if order_type == OrderType.MARKET:
                order = self.exchange.create_market_order(symbol, order_side, quantity)
            elif order_type == OrderType.LIMIT:
                if price is None:
                    raise ValueError("Price required for LIMIT orders")
                order = self.exchange.create_limit_order(symbol, order_side, quantity, price)
            else:
                raise ValueError(f"Order type {order_type} not supported")
            
            return {
                'order_id': order['id'],
                'status': OrderStatus.PENDING if order['status'] == 'open' else OrderStatus.FILLED,
                'filled_quantity': order.get('filled', 0.0),
                'symbol': symbol,
                'side': side.value,
                'price': order.get('price', price),
                'timestamp': datetime.now()
            }
        except Exception as e:
            print(f"Error placing order on Binance: {e}")
            return {
                'order_id': None,
                'status': OrderStatus.REJECTED,
                'error': str(e)
            }
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            # Note: CCXT requires symbol to cancel order
            # This is a simplified implementation
            self.exchange.cancel_order(order_id)
            return True
        except Exception as e:
            print(f"Error cancelling order on Binance: {e}")
            return False
    
    def get_open_positions(self) -> List[Dict]:
        """Get open positions (for spot trading, this is balances)"""
        try:
            balance = self.exchange.fetch_balance()
            positions = []
            
            for currency, amounts in balance.items():
                if currency not in ['info', 'free', 'used', 'total']:
                    if isinstance(amounts, dict) and amounts.get('total', 0) > 0:
                        # For spot trading, positions are just balances
                        positions.append({
                            'symbol': currency,
                            'quantity': amounts['total'],
                            'available': amounts['free'],
                            'used': amounts['used']
                        })
            
            return positions
        except Exception as e:
            print(f"Error fetching open positions from Binance: {e}")
            return []
    
    def get_order_status(self, order_id: str) -> Dict:
        """Get order status"""
        try:
            # Note: CCXT requires symbol to fetch order
            # This is a simplified implementation
            order = self.exchange.fetch_order(order_id)
            return {
                'order_id': order['id'],
                'status': OrderStatus.FILLED if order['status'] == 'closed' else OrderStatus.PENDING,
                'filled_quantity': order.get('filled', 0.0),
                'price': order.get('price', 0.0)
            }
        except Exception as e:
            print(f"Error fetching order status from Binance: {e}")
            return {'order_id': order_id, 'status': OrderStatus.REJECTED}
    
    def close_position(self, symbol: str, side: Optional[OrderSide] = None) -> bool:
        """Close position by selling all holdings of symbol"""
        try:
            balance = self.exchange.fetch_balance()
            base_currency = symbol.split('/')[0]
            
            if base_currency in balance:
                quantity = balance[base_currency]['free']
                if quantity > 0:
                    order = self.exchange.create_market_order(symbol, 'sell', quantity)
                    return order['status'] == 'closed'
            return False
        except Exception as e:
            print(f"Error closing position on Binance: {e}")
            return False
    
    def get_supported_assets(self) -> List[str]:
        """Get supported asset classes"""
        return ['crypto']
    
    def is_market_open(self, symbol: str) -> bool:
        """Check if market is open (crypto markets are 24/7)"""
        return True

