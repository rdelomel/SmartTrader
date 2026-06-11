"""Abstract base class for broker implementations — with auto-reconnect"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)


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

    # Reconnect defaults (override in subclass __init__ if needed)
    MAX_RECONNECT_ATTEMPTS: int = 5
    RECONNECT_BASE_DELAY: float = 2.0   # seconds; doubles each attempt
    RECONNECT_MAX_DELAY: float = 60.0   # cap

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
        self._reconnect_count = 0
        self._last_connect_attempt: Optional[datetime] = None

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to broker. Returns True on success."""
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """Close connection to broker. Returns True on success."""
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
        """Fetch historical OHLCV data."""
        pass

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """Get current market price for symbol."""
        pass

    @abstractmethod
    def get_account_balance(self) -> Dict[str, float]:
        """Get account balance information."""
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
        """Place a trading order."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        pass

    @abstractmethod
    def get_open_orders(self) -> List[Dict]:
        """Get list of open orders."""
        pass

    @abstractmethod
    def get_positions(self) -> List[Dict]:
        """Get current open positions."""
        pass

    # ── Auto-reconnect (shared by all subclasses) ─────────────────────────────

    def ensure_connected(self) -> bool:
        """
        Guarantee the broker is connected before an operation.
        Uses exponential backoff up to MAX_RECONNECT_ATTEMPTS.

        Returns:
            True  - connected (either was already, or reconnected successfully)
            False - could not reconnect after all attempts
        """
        if self.connected:
            return True

        broker_name = type(self).__name__
        for attempt in range(1, self.MAX_RECONNECT_ATTEMPTS + 1):
            delay = min(
                self.RECONNECT_BASE_DELAY * (2 ** (attempt - 1)),
                self.RECONNECT_MAX_DELAY
            )
            logger.warning(
                f"[{broker_name}] Not connected. Reconnect attempt {attempt}/"
                f"{self.MAX_RECONNECT_ATTEMPTS} (waiting {delay:.0f}s)..."
            )
            if attempt > 1:
                time.sleep(delay)

            try:
                success = self.connect()
                if success:
                    self._reconnect_count += 1
                    self._last_connect_attempt = datetime.now()
                    logger.info(
                        f"[{broker_name}] Reconnected successfully "
                        f"(total reconnects: {self._reconnect_count})"
                    )
                    return True
            except Exception as e:
                logger.error(f"[{broker_name}] Reconnect attempt {attempt} failed: {e}")

        logger.error(
            f"[{broker_name}] All {self.MAX_RECONNECT_ATTEMPTS} reconnect "
            "attempts exhausted. Broker unavailable."
        )
        return False

    def safe_call(self, method_name: str, *args, **kwargs):
        """
        Wrapper that calls a broker method, auto-reconnecting first if needed.

        Usage:
            result = broker.safe_call("place_order", symbol="BTC/USD", ...)
        """
        if not self.ensure_connected():
            return None
        method = getattr(self, method_name, None)
        if method is None:
            raise AttributeError(f"{type(self).__name__} has no method '{method_name}'")
        return method(*args, **kwargs)

    # ── Order fill polling ────────────────────────────────────────────────────

    def wait_for_fill(
        self,
        order_id: str,
        timeout_seconds: int = 60,
        poll_interval: float = 5.0
    ) -> Dict:
        """
        Poll order status until filled, cancelled, or timeout.

        Args:
            order_id:        The order ID returned by place_order
            timeout_seconds: Give up after this many seconds
            poll_interval:   Seconds between polls

        Returns:
            Final order status dict, or {'status': 'timeout'} if not filled
        """
        broker_name = type(self).__name__
        start = time.time()
        while time.time() - start < timeout_seconds:
            if not self.ensure_connected():
                time.sleep(poll_interval)
                continue
            try:
                status = self.get_order_status(order_id)
                s = status.get("status", "")
                sv = s.value if hasattr(s, "value") else str(s)
                if sv in ("filled", "FILLED", "cancelled", "CANCELLED",
                          "rejected", "REJECTED"):
                    logger.info(f"[{broker_name}] Order {order_id} -> {sv}")
                    return status
            except Exception as e:
                logger.warning(f"[{broker_name}] poll error for {order_id}: {e}")
            time.sleep(poll_interval)

        logger.warning(f"[{broker_name}] Order {order_id} not filled within {timeout_seconds}s")
        return {"order_id": order_id, "status": "timeout"}

    def get_order_status(self, order_id: str) -> Dict:
        """
        Get status of a specific order.
        Subclasses should override with broker-specific implementation.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement get_order_status()"
        )

    # ── Utility helpers ───────────────────────────────────────────────────────

    def is_market_open(self) -> bool:
        """
        Check if market is currently open.
        Default returns True - subclasses should override for exchange-specific hours.
        """
        return True

    def __repr__(self) -> str:
        mode = "testnet" if self.testnet else "live"
        state = "connected" if self.connected else "disconnected"
        return f"<{type(self).__name__} [{mode}] [{state}]>"
