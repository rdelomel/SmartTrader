"""Alpaca API implementation for stocks and crypto trading"""

import requests
from typing import Dict, List, Optional
from datetime import datetime
import logging
from .base_broker import BaseBroker, OrderSide, OrderType, OrderStatus

logger = logging.getLogger(__name__)


class AlpacaBroker(BaseBroker):
    """Alpaca broker implementation for stocks and crypto trading."""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        super().__init__()
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = 'https://paper-api.alpaca.markets' if testnet else 'https://api.alpaca.markets'
        self.headers = {
            'APCA-API-KEY-ID': api_key,
            'APCA-API-SECRET-KEY': api_secret,
            'Content-Type': 'application/json'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def connect(self) -> bool:
        """Connect to Alpaca and verify credentials."""
        try:
            response = self.session.get(f"{self.base_url}/v2/account")
            response.raise_for_status()
            account = response.json()
            if account.get('status') == 'ACTIVE':
                self.connected = True
                logger.info("Connected to Alpaca")
                return True
            logger.error(f"Alpaca account not active: {account.get('status')}")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            self.connected = False
            return False

    def disconnect(self) -> bool:
        self.connected = False
        return True

    def get_historical_data(
        self,
        symbol: str,
        timeframe: str = '1H',
        periods: int = 100,
        start_date=None,
        end_date=None
    ):
        """Fetch OHLCV data from Alpaca."""
        import pandas as pd
        if not self.ensure_connected():
            return pd.DataFrame()
        try:
            tf_map = {
                '1m': '1Min', '5m': '5Min', '15m': '15Min', '30m': '30Min',
                '1h': '1Hour', '1H': '1Hour', '4h': '4Hour', '4H': '4Hour',
                '1d': '1Day', '1D': '1Day', 'D': '1Day'
            }
            alpaca_tf = tf_map.get(timeframe, '1Hour')

            # Determine data feed endpoint
            is_crypto = '/' in symbol or symbol.endswith('USD') and len(symbol) <= 7
            if is_crypto:
                base_data_url = 'https://data.alpaca.markets/v1beta3/crypto/us'
                bars_url = f"{base_data_url}/bars"
            else:
                base_data_url = 'https://data.alpaca.markets/v2/stocks'
                bars_url = f"{base_data_url}/{symbol}/bars"

            # Alpaca crypto only supports *USD pairs — normalize USDT/USDC → USD
            alpaca_symbol = symbol.replace('/USDT', '/USD').replace('/USDC', '/USD') if is_crypto else symbol

            params = {'timeframe': alpaca_tf, 'limit': periods}
            if is_crypto:
                params['symbols'] = alpaca_symbol
            def _fmt_ts(dt):
                """Format datetime as RFC3339Z — strips tz to avoid +00:00Z double-suffix."""
                from datetime import timezone as _tz
                if dt.tzinfo is not None:
                    dt = dt.astimezone(_tz.utc).replace(tzinfo=None)
                return dt.strftime('%Y-%m-%dT%H:%M:%S') + 'Z'

            if start_date:
                params['start'] = _fmt_ts(start_date)
            if end_date:
                params['end'] = _fmt_ts(end_date)

            response = self.session.get(bars_url, params=params)
            response.raise_for_status()
            data = response.json()

            if is_crypto:
                bars = data.get('bars', {}).get(alpaca_symbol, [])
            else:
                bars = data.get('bars', [])

            if not bars:
                return pd.DataFrame()

            rows = []
            for b in bars:
                rows.append({
                    'timestamp': pd.to_datetime(b['t']),
                    'open': float(b['o']),
                    'high': float(b['h']),
                    'low': float(b['l']),
                    'close': float(b['c']),
                    'volume': float(b.get('v', 0))
                })
            df = pd.DataFrame(rows)
            if not df.empty:
                df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching Alpaca historical data for {symbol}: {e}")
            import pandas as pd
            return pd.DataFrame()

    def get_current_price(self, symbol: str) -> float:
        """Get latest trade price for a symbol."""
        if not self.ensure_connected():
            return 0.0
        try:
            is_crypto = '/' in symbol or (symbol.endswith('USD') and len(symbol) <= 7)
            if is_crypto:
                # Alpaca only supports *USD pairs — normalize USDT/USDC → USD
                alpaca_symbol = symbol.replace('/USDT', '/USD').replace('/USDC', '/USD')
                url = f"https://data.alpaca.markets/v1beta3/crypto/us/latest/trades"
                response = self.session.get(url, params={'symbols': alpaca_symbol})
                response.raise_for_status()
                data = response.json()
                trade = data.get('trades', {}).get(alpaca_symbol, {})
                return float(trade.get('p', 0))
            else:
                url = f"https://data.alpaca.markets/v2/stocks/{symbol}/trades/latest"
                response = self.session.get(url)
                response.raise_for_status()
                data = response.json()
                return float(data.get('trade', {}).get('p', 0))
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            return 0.0

    def get_account_balance(self) -> Dict:
        """Get account balance and equity.

        NOTE: For Alpaca paper trading crypto accounts, the 'equity' field returns 0.
        'portfolio_value' is the reliable total-account-value field; 'equity' is a
        fallback. Using 'equity' directly causes the leverage-detection code in main.py
        to fire (available_balance > total_balance → set available to $0) which blocks
        all trading.
        """
        if not self.ensure_connected():
            return {'balance': 0.0, 'equity': 0.0}
        try:
            response = self.session.get(f"{self.base_url}/v2/account")
            response.raise_for_status()
            account = response.json()
            # Use portfolio_value as authoritative total (works for both stocks and crypto paper accounts).
            # equity can be 0 for crypto paper accounts even when the account has funds.
            _portfolio_value = float(account.get('portfolio_value', 0))
            _equity = float(account.get('equity', 0))
            _total = _portfolio_value or _equity  # prefer portfolio_value; fall back to equity
            return {
                'balance': _total,
                'equity': _total,
                'buying_power': float(account.get('buying_power', 0)),
                'portfolio_value': _portfolio_value,
                'currency': 'USD'
            }
        except Exception as e:
            logger.error(f"Error fetching Alpaca account balance: {e}")
            return {'balance': 0.0, 'equity': 0.0}

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        **kwargs
    ) -> Dict:
        """Place an order on Alpaca."""
        if not self.ensure_connected():
            return {'success': False, 'error': 'Not connected'}
        try:
            order_body: Dict = {
                'symbol': symbol,
                'qty': str(quantity),
                'side': 'buy' if side == OrderSide.BUY else 'sell',
                'type': 'market' if order_type == OrderType.MARKET else 'limit',
                'time_in_force': 'gtc' if order_type == OrderType.LIMIT else 'day'
            }
            if order_type == OrderType.LIMIT and price is not None:
                order_body['limit_price'] = str(price)

            # Bracket order for stop_loss / take_profit
            if stop_loss is not None or take_profit is not None:
                order_body['order_class'] = 'bracket'
                if stop_loss is not None:
                    order_body['stop_loss'] = {'stop_price': str(stop_loss)}
                if take_profit is not None:
                    order_body['take_profit'] = {'limit_price': str(take_profit)}

            response = self.session.post(f"{self.base_url}/v2/orders", json=order_body)
            response.raise_for_status()
            order = response.json()

            filled_qty = float(order.get('filled_qty', 0) or 0)
            filled_avg = order.get('filled_avg_price')
            fill_price = float(filled_avg) if filled_avg and filled_avg != 'None' else (price or 0.0)

            status_map = {
                'filled': OrderStatus.FILLED,
                'partially_filled': OrderStatus.PARTIALLY_FILLED,
                'canceled': OrderStatus.CANCELLED,
                'rejected': OrderStatus.REJECTED,
            }
            order_status = status_map.get(order.get('status', ''), OrderStatus.PENDING)

            return {
                'success': True,
                'order_id': order.get('id'),
                'status': order_status,
                'filled_quantity': filled_qty,
                'price': fill_price,
                'symbol': symbol
            }
        except requests.exceptions.HTTPError as e:
            error_body = {}
            try:
                error_body = e.response.json()
            except Exception:
                pass
            logger.error(f"Alpaca order HTTP error: {e}, body: {error_body}")
            return {'success': False, 'error': str(e), 'details': error_body}
        except Exception as e:
            logger.error(f"Error placing Alpaca order: {e}")
            return {'success': False, 'error': str(e)}

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if not self.ensure_connected():
            return False
        try:
            response = self.session.delete(f"{self.base_url}/v2/orders/{order_id}")
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Error cancelling Alpaca order {order_id}: {e}")
            return False

    def get_open_positions(self) -> List[Dict]:
        """Get open positions."""
        try:
            response = self.session.get(f"{self.base_url}/v2/positions")
            response.raise_for_status()
            positions_data = response.json()
            positions = []
            for pos in positions_data:
                qty = float(pos['qty'])
                quantity = abs(qty)
                avg_entry_price = float(pos['avg_entry_price'])
                unrealized_pl = float(pos['unrealized_pl'])
                market_value = float(pos.get('market_value', 0))
                current_price = market_value / quantity if quantity > 0 else avg_entry_price
                positions.append({
                    'symbol': pos['symbol'],
                    'side': 'buy' if qty > 0 else 'sell',
                    'quantity': quantity,
                    'entry_price': avg_entry_price,
                    'current_price': current_price,
                    'unrealized_pnl': unrealized_pl,
                    'market_value': market_value
                })
            return positions
        except Exception as e:
            logger.error(f"Error fetching open positions from Alpaca: {e}")
            return []

    def get_orders(self, status: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Get orders from Alpaca (filled, pending, open, etc.)."""
        try:
            url = f"{self.base_url}/v2/orders"
            params = {'limit': limit}
            if status:
                params['status'] = status
            response = self.session.get(url, params=params)
            response.raise_for_status()
            orders_data = response.json()
            orders = []
            for order in orders_data:
                limit_price = order.get('limit_price')
                filled_avg_price = order.get('filled_avg_price')
                if filled_avg_price and filled_avg_price != 'None':
                    try:
                        order_price = float(filled_avg_price)
                    except (ValueError, TypeError):
                        order_price = 0.0
                elif limit_price and limit_price != 'None':
                    try:
                        order_price = float(limit_price)
                    except (ValueError, TypeError):
                        order_price = 0.0
                else:
                    order_price = 0.0
                orders.append({
                    'order_id': order.get('id'),
                    'symbol': order.get('symbol'),
                    'side': order.get('side'),
                    'quantity': float(order.get('qty', 0) or 0),
                    'filled_quantity': float(order.get('filled_qty', 0) or 0),
                    'price': order_price,
                    'status': order.get('status', 'unknown'),
                    'order_type': order.get('type', 'market'),
                    'created_at': order.get('created_at'),
                    'filled_at': order.get('filled_at'),
                    'updated_at': order.get('updated_at')
                })
            return orders
        except Exception as e:
            logger.error(f"Error fetching orders from Alpaca: {e}")
            return []

    def get_order_status(self, order_id: str) -> Dict:
        """Get order status."""
        try:
            response = self.session.get(f"{self.base_url}/v2/orders/{order_id}")
            response.raise_for_status()
            order = response.json()
            filled_avg = order.get('filled_avg_price')
            fill_price = 0.0
            if filled_avg and filled_avg != 'None':
                try:
                    fill_price = float(filled_avg)
                except (ValueError, TypeError):
                    fill_price = 0.0
            status_map = {
                'filled': OrderStatus.FILLED,
                'partially_filled': OrderStatus.PARTIALLY_FILLED,
                'canceled': OrderStatus.CANCELLED,
                'rejected': OrderStatus.REJECTED,
            }
            return {
                'order_id': order.get('id'),
                'status': status_map.get(order.get('status', ''), OrderStatus.PENDING),
                'filled_quantity': float(order.get('filled_qty', 0) or 0),
                'price': fill_price
            }
        except Exception as e:
            logger.error(f"Error getting Alpaca order status for {order_id}: {e}")
            return {'order_id': order_id, 'status': OrderStatus.UNKNOWN, 'filled_quantity': 0, 'price': 0.0}

    def close_position(self, symbol: str, side: Optional[OrderSide] = None) -> bool:
        """Close an open position."""
        if not self.ensure_connected():
            return False
        try:
            response = self.session.delete(f"{self.base_url}/v2/positions/{symbol}")
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Error closing Alpaca position for {symbol}: {e}")
            return False

    def get_supported_assets(self) -> List[str]:
        """Return common US stocks and crypto symbols."""
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'SPY', 'QQQ',
            'BTC/USD', 'ETH/USD', 'SOL/USD', 'DOGE/USD'
        ]

    def is_market_open(self, symbol: str) -> bool:
        """Check if the market is currently open.
        Crypto markets are 24/7 — always return True for crypto symbols.
        """
        # Crypto symbols contain '/' (e.g. BTC/USD) — always open
        if '/' in symbol:
            return True
        try:
            response = self.session.get(f"{self.base_url}/v2/clock")
            response.raise_for_status()
            clock = response.json()
            return bool(clock.get('is_open', False))
        except Exception as e:
            logger.error(f"Error checking market open status: {e}")
            return True  # Default to open on error

    def check_liquidity(self, symbol: str, order_size: float) -> Dict:
        """Estimate liquidity for an Alpaca symbol."""
        try:
            is_crypto = '/' in symbol
            if is_crypto:
                estimated_slippage = 2.0
                return {
                    'sufficient': True,
                    'estimated_slippage_bps': estimated_slippage,
                    'order_book_depth': order_size * 50,
                    'reason': 'Crypto liquidity estimate'
                }
            else:
                estimated_slippage = 3.0
                return {
                    'sufficient': True,
                    'estimated_slippage_bps': estimated_slippage,
                    'order_book_depth': order_size * 20,
                    'reason': 'Stock liquidity check'
                }
        except Exception as e:
            return {
                'sufficient': True,
                'estimated_slippage_bps': 5.0,
                'order_book_depth': order_size * 10,
                'reason': f'Liquidity check error: {str(e)}'
            }

    # ── Abstract method implementations ─────────────────────────────────────

    def get_open_orders(self) -> List[Dict]:
        """Get all open/pending orders."""
        return self.get_orders(status='open')

    def get_positions(self) -> List[Dict]:
        """Get open positions — delegates to get_open_positions()."""
        return self.get_open_positions()
