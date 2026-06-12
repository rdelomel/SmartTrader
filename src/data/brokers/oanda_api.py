"""OANDA API implementation for forex trading"""

import requests
import time
from typing import Dict, List, Optional
from datetime import datetime
import logging
from .base_broker import BaseBroker, OrderSide, OrderType, OrderStatus

logger = logging.getLogger(__name__)


class OANDABroker(BaseBroker):
    """OANDA broker implementation for forex trading."""

    def __init__(self, api_key: str, api_secret: str = None, account_id: str = None, testnet: bool = True):
        super().__init__()
        self.api_key = api_key
        self.account_id = account_id
        environment = 'practice' if testnet else 'live'
        self.base_url = 'https://api-fxpractice.oanda.com' if environment == 'practice' else 'https://api-fxtrade.oanda.com'
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'Accept-Datetime-Format': 'RFC3339'
        })

    def connect(self) -> bool:
        """Connect to OANDA and verify credentials."""
        try:
            response = self.session.get(f"{self.base_url}/v3/accounts")
            response.raise_for_status()
            data = response.json()
            accounts = data.get('accounts', [])
            if not accounts:
                logger.error("No OANDA accounts found")
                return False
            if not self.account_id:
                self.account_id = accounts[0]['id']
                logger.info(f"Auto-detected OANDA account ID: {self.account_id}")
            self.connected = True
            logger.info("Connected to OANDA")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to OANDA: {e}")
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
        """Fetch OHLCV data from OANDA."""
        import pandas as pd
        if not self.ensure_connected():
            return pd.DataFrame()
        try:
            gran_map = {
                '1m': 'M1', '5m': 'M5', '15m': 'M15', '30m': 'M30',
                '1h': 'H1', '1H': 'H1', '4h': 'H4', '4H': 'H4',
                '1d': 'D', '1D': 'D', 'D': 'D'
            }
            granularity = gran_map.get(timeframe, 'H1')
            params = {'granularity': granularity, 'count': periods, 'price': 'M'}
            if start_date:
                params['from'] = start_date.isoformat() + 'Z'
                params.pop('count', None)
            if end_date:
                params['to'] = end_date.isoformat() + 'Z'
            # OANDA instrument IDs use underscores (EUR_USD not EUR/USD)
            oanda_symbol = symbol.replace('/', '_')
            url = f"{self.base_url}/v3/instruments/{oanda_symbol}/candles"
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            candles = data.get('candles', [])
            if not candles:
                return pd.DataFrame()
            rows = []
            for c in candles:
                if c.get('complete', True):
                    mid = c.get('mid', {})
                    rows.append({
                        'timestamp': pd.to_datetime(c['time']),
                        'open': float(mid.get('o', 0)),
                        'high': float(mid.get('h', 0)),
                        'low': float(mid.get('l', 0)),
                        'close': float(mid.get('c', 0)),
                        'volume': int(c.get('volume', 0))
                    })
            df = pd.DataFrame(rows)
            if not df.empty:
                df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching OANDA historical data for {symbol}: {e}")
            import pandas as pd
            return pd.DataFrame()

    def get_current_price(self, symbol: str) -> float:
        """Get current bid/ask midpoint for a symbol."""
        if not self.ensure_connected():
            return 0.0
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}/pricing"
            response = self.session.get(url, params={'instruments': symbol})
            response.raise_for_status()
            data = response.json()
            prices = data.get('prices', [])
            if not prices:
                return 0.0
            price = prices[0]
            bid = float(price.get('bids', [{}])[0].get('price', 0))
            ask = float(price.get('asks', [{}])[0].get('price', 0))
            return (bid + ask) / 2.0
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            return 0.0

    def _get_account_details(self) -> Optional[Dict]:
        """Fetch full account details from OANDA."""
        try:
            response = self.session.get(f"{self.base_url}/v3/accounts/{self.account_id}")
            response.raise_for_status()
            return response.json().get('account', {})
        except Exception as e:
            logger.error(f"Error fetching OANDA account details: {e}")
            return None

    def get_account_balance(self) -> Dict[str, float]:
        """Get account balance and margin info."""
        if not self.ensure_connected():
            return {'balance': 0.0, 'equity': 0.0, 'margin_used': 0.0, 'margin_available': 0.0}
        try:
            account = self._get_account_details()
            if not account:
                return {'balance': 0.0, 'equity': 0.0, 'margin_used': 0.0, 'margin_available': 0.0}
            return {
                'balance': float(account.get('balance', 0)),
                'equity': float(account.get('NAV', 0)),
                'margin_used': float(account.get('marginUsed', 0)),
                'margin_available': float(account.get('marginAvailable', 0)),
                'unrealized_pnl': float(account.get('unrealizedPL', 0)),
                'currency': account.get('currency', 'USD')
            }
        except Exception as e:
            logger.error(f"Error fetching OANDA account balance: {e}")
            return {'balance': 0.0, 'equity': 0.0, 'margin_used': 0.0, 'margin_available': 0.0}

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
        """Place an order on OANDA."""
        if not self.ensure_connected():
            return {'success': False, 'error': 'Not connected'}
        try:
            units = quantity if side == OrderSide.BUY else -quantity

            def get_price_precision(instrument: str) -> int:
                major_pairs = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF',
                               'USD_CAD', 'AUD_USD', 'NZD_USD']
                jpy_pairs = ['USD_JPY', 'EUR_JPY', 'GBP_JPY', 'AUD_JPY',
                             'NZD_JPY', 'CAD_JPY', 'CHF_JPY']
                if any(jpy in instrument for jpy in jpy_pairs):
                    return 3
                return 5

            order_body: Dict = {}
            if order_type == OrderType.MARKET:
                order_body = {
                    'order': {
                        'type': 'MARKET',
                        'instrument': symbol,
                        'units': str(int(units)) if abs(units) >= 1 else str(units),
                        'timeInForce': 'FOK',
                        'positionFill': 'DEFAULT'
                    }
                }
            elif order_type == OrderType.LIMIT and price is not None:
                precision = get_price_precision(symbol)
                order_body = {
                    'order': {
                        'type': 'LIMIT',
                        'instrument': symbol,
                        'units': str(int(units)) if abs(units) >= 1 else str(units),
                        'price': str(round(price, precision)),
                        'timeInForce': 'GTC'
                    }
                }
            else:
                return {'success': False, 'error': f'Unsupported order type: {order_type}'}

            if stop_loss is not None:
                precision = get_price_precision(symbol)
                order_body['order']['stopLossOnFill'] = {
                    'price': str(round(stop_loss, precision)),
                    'timeInForce': 'GTC'
                }
            if take_profit is not None:
                precision = get_price_precision(symbol)
                order_body['order']['takeProfitOnFill'] = {
                    'price': str(round(take_profit, precision))
                }

            url = f"{self.base_url}/v3/accounts/{self.account_id}/orders"
            response = self.session.post(url, json=order_body)
            response.raise_for_status()
            data = response.json()

            if 'orderFillTransaction' in data:
                fill = data['orderFillTransaction']
                return {
                    'success': True,
                    'order_id': fill.get('id'),
                    'status': OrderStatus.FILLED,
                    'filled_quantity': abs(float(fill.get('units', 0))),
                    'price': float(fill.get('price', 0)),
                    'symbol': symbol
                }
            elif 'orderCreateTransaction' in data:
                create = data['orderCreateTransaction']
                return {
                    'success': True,
                    'order_id': create.get('id'),
                    'status': OrderStatus.PENDING,
                    'filled_quantity': 0,
                    'price': price or 0,
                    'symbol': symbol
                }
            else:
                return {'success': False, 'error': f'Unexpected response: {data}'}

        except requests.exceptions.HTTPError as e:
            error_body = {}
            try:
                error_body = e.response.json()
            except Exception:
                pass
            logger.error(f"OANDA order placement HTTP error: {e}, body: {error_body}")
            return {'success': False, 'error': str(e), 'details': error_body}
        except Exception as e:
            logger.error(f"Error placing OANDA order: {e}")
            return {'success': False, 'error': str(e)}

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        if not self.ensure_connected():
            return False
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}/orders/{order_id}/cancel"
            response = self.session.put(url)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Error cancelling OANDA order {order_id}: {e}")
            return False

    def get_open_positions(self) -> List[Dict]:
        """
        Get open positions from account details.

        According to OANDA API docs: https://developer.oanda.com/rest-live-v20/account-ep/
        The account endpoint already includes positions, so we extract them from there
        instead of making a separate API call.
        """
        account = self._get_account_details()
        if not account:
            return []

        positions = []
        for pos in account.get('positions', []):
            long_units = float(pos.get('long', {}).get('units', 0))
            short_units = float(pos.get('short', {}).get('units', 0))

            if long_units != 0:
                entry_price = float(pos.get('long', {}).get('averagePrice', 0))
                unrealized_pnl = float(pos.get('long', {}).get('unrealizedPL', 0))
                quantity = abs(long_units)
                current_price = entry_price + (unrealized_pnl / quantity) if quantity > 0 else entry_price
                positions.append({
                    'symbol': pos['instrument'],
                    'side': 'buy',
                    'quantity': quantity,
                    'entry_price': entry_price,
                    'current_price': round(current_price, 6),
                    'market_price': round(current_price, 6),
                    'unrealized_pnl': unrealized_pnl
                })

            if short_units != 0:
                entry_price = float(pos.get('short', {}).get('averagePrice', 0))
                unrealized_pnl = float(pos.get('short', {}).get('unrealizedPL', 0))
                quantity = abs(short_units)
                current_price = entry_price - (unrealized_pnl / quantity) if quantity > 0 else entry_price
                positions.append({
                    'symbol': pos['instrument'],
                    'side': 'sell',
                    'quantity': quantity,
                    'entry_price': entry_price,
                    'current_price': round(current_price, 6),
                    'market_price': round(current_price, 6),
                    'unrealized_pnl': unrealized_pnl
                })

        return positions

    def get_order_status(self, order_id: str) -> Dict:
        """Get order status."""
        try:
            response = self.session.get(f"{self.base_url}/v3/accounts/{self.account_id}/orders/{order_id}")
            response.raise_for_status()
            data = response.json()
            order = data['order']
            return {
                'order_id': order['id'],
                'status': OrderStatus.FILLED if order['state'] == 'FILLED' else OrderStatus.PENDING,
                'filled_quantity': float(order.get('units', 0)),
                'price': float(order.get('price', 0.0))
            }
        except Exception as e:
            logger.error(f"Error getting OANDA order status for {order_id}: {e}")
            return {'order_id': order_id, 'status': OrderStatus.UNKNOWN, 'filled_quantity': 0, 'price': 0.0}

    def close_position(self, symbol: str, side: Optional[OrderSide] = None) -> bool:
        """Close an open position."""
        if not self.ensure_connected():
            return False
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}/positions/{symbol}/close"
            body = {}
            if side == OrderSide.BUY or side is None:
                body['longUnits'] = 'ALL'
            if side == OrderSide.SELL or side is None:
                body['shortUnits'] = 'ALL'
            response = self.session.put(url, json=body)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Error closing OANDA position for {symbol}: {e}")
            return False

    def get_supported_assets(self) -> List[str]:
        """Return a sample list of supported OANDA instruments."""
        return [
            'EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'AUD_USD',
            'USD_CAD', 'NZD_USD', 'EUR_GBP', 'EUR_JPY', 'GBP_JPY',
            'XAU_USD', 'XAG_USD', 'BCO_USD', 'WTICO_USD'
        ]

    def is_market_open(self, symbol: str) -> bool:
        """Check whether the forex market is currently open for a symbol."""
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}/pricing"
            response = self.session.get(url, params={'instruments': symbol})
            response.raise_for_status()
            data = response.json()
            prices = data.get('prices', [])
            if not prices:
                return False
            status = prices[0].get('status', 'tradeable')
            tradeable = prices[0].get('tradeable', True)
            return tradeable and status == 'tradeable'
        except Exception as e:
            logger.error(f"Error checking market open status for {symbol}: {e}")
            return True  # Default to open on error

    def check_liquidity(self, symbol: str, order_size: float) -> Dict:
        """Estimate liquidity for a forex symbol."""
        try:
            is_major_pair = symbol in [
                'EUR_USD', 'USD_JPY', 'GBP_USD', 'AUD_USD',
                'USD_CAD', 'USD_CHF', 'NZD_USD'
            ]
            estimated_slippage = 1.0 if is_major_pair else 3.0
            return {
                'sufficient': True,
                'estimated_slippage_bps': estimated_slippage,
                'order_book_depth': order_size * 100,
                'reason': 'Forex market liquidity (major pair)' if is_major_pair else 'Forex market liquidity'
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
        """Get all pending orders from OANDA."""
        if not self.ensure_connected():
            return []
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}/orders"
            response = self.session.get(url, params={'state': 'PENDING'})
            response.raise_for_status()
            data = response.json()
            orders = []
            for order in data.get('orders', []):
                orders.append({
                    'order_id': order.get('id'),
                    'symbol': order.get('instrument'),
                    'side': 'buy' if float(order.get('units', 0)) > 0 else 'sell',
                    'quantity': abs(float(order.get('units', 0))),
                    'price': float(order.get('price', 0.0)),
                    'status': order.get('state', 'PENDING').lower(),
                    'order_type': order.get('type', 'MARKET').lower(),
                    'created_at': order.get('createTime'),
                })
            return orders
        except Exception as e:
            logger.error(f"Error fetching open orders from OANDA: {e}")
            return []

    def get_positions(self) -> List[Dict]:
        """Get open positions — delegates to get_open_positions()."""
        return self.get_open_positions()
