"""Abstract base class for broker implementations"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class BaseBroker(ABC):
    """Abstract base class for all broker implementations"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        """
        Initialize broker connection
        
        Args:
            api_key: API key for broker
            api_secret: API secret for broker
            testnet: Whether to use testnet/demo environment
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.connected = False
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to broker
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        Close connection to broker
        
        Returns:
            True if disconnection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Fetch historical price data
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT', 'EUR/USD')
            timeframe: Timeframe string (e.g., '1h', '4h', '1d')
            start_date: Start date for data
            end_date: End date for data (None for latest)
            limit: Maximum number of candles to return
        
        Returns:
            List of dictionaries with OHLCV data
            Format: [{'timestamp': datetime, 'open': float, 'high': float, 
                     'low': float, 'close': float, 'volume': float}, ...]
        """
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """
        Get current market price for symbol
        
        Args:
            symbol: Trading pair symbol
        
        Returns:
            Current price as float
        """
        pass
    
    @abstractmethod
    def get_account_balance(self) -> Dict[str, float]:
        """
        Get account balance information
        
        Returns:
            Dictionary with balance information
            Format: {'total': float, 'available': float, 'used': float, 
                    'currencies': {'USD': float, 'BTC': float, ...}}
        """
        pass
    
    @abstractmethod
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
        """
        Place a trading order
        
        Args:
            symbol: Trading pair symbol
            side: BUY or SELL
            order_type: Type of order (MARKET, LIMIT, etc.)
            quantity: Amount to trade
            price: Limit price (required for LIMIT orders)
            stop_price: Stop price (for STOP_LOSS orders)
            take_profit: Take profit price
            stop_loss: Stop loss price
        
        Returns:
            Dictionary with order information
            Format: {'order_id': str, 'status': OrderStatus, 'filled_quantity': float, ...}
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order
        
        Args:
            order_id: ID of order to cancel
        
        Returns:
            True if cancellation successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_open_positions(self) -> List[Dict]:
        """
        Get all open positions
        
        Returns:
            List of position dictionaries
            Format: [{'symbol': str, 'side': str, 'quantity': float, 
                     'entry_price': float, 'unrealized_pnl': float, ...}, ...]
        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Dict:
        """
        Get status of a specific order
        
        Args:
            order_id: ID of order to check
        
        Returns:
            Dictionary with order status information
        """
        pass
    
    @abstractmethod
    def close_position(self, symbol: str, side: Optional[OrderSide] = None) -> bool:
        """
        Close an open position
        
        Args:
            symbol: Trading pair symbol
            side: Side to close (None to close all positions for symbol)
        
        Returns:
            True if position closed successfully, False otherwise
        """
        pass
    
    def get_supported_assets(self) -> List[str]:
        """
        Get list of supported asset classes
        
        Returns:
            List of asset class strings (e.g., ['crypto', 'forex'])
        """
        return []
    
    def is_market_open(self, symbol: str) -> bool:
        """
        Check if market is currently open for trading
        
        Args:
            symbol: Trading pair symbol
        
        Returns:
            True if market is open, False otherwise
        """
        # Default implementation - can be overridden
        return True
    
    def check_liquidity(self, symbol: str, order_size: float) -> Dict:
        """
        Check market liquidity for an order
        
        Args:
            symbol: Trading pair symbol
            order_size: Size of order to check
        
        Returns:
            Dictionary with liquidity information:
            {
                'sufficient': bool,
                'estimated_slippage_bps': float,
                'order_book_depth': float,
                'reason': str
            }
        """
        # Default implementation - returns sufficient liquidity
        # Should be overridden by broker implementations that support order book data
        return {
            'sufficient': True,
            'estimated_slippage_bps': 5.0,  # Default estimate
            'order_book_depth': order_size * 10,  # Placeholder
            'reason': 'Liquidity check not implemented for this broker'
        }

