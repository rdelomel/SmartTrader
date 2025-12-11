"""OANDA API implementation for forex trading"""

import requests
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta, timezone
import os

from .base_broker import BaseBroker, OrderType, OrderSide, OrderStatus


class OANDABroker(BaseBroker):
    """OANDA broker implementation for forex trading"""
    
    def __init__(self, api_key: str, api_secret: str = None, account_id: str = None, testnet: bool = True):
        # OANDA only uses API key as Bearer token, api_secret is not used but kept for compatibility
        super().__init__(api_key, api_secret or "", testnet)
        # Validate account_id - strip whitespace and check if it's empty or a comment
        if account_id:
            account_id = account_id.strip()
            if not account_id or account_id.startswith('#'):
                account_id = None
        self.account_id = account_id
        
        if testnet:
            self.base_url = "https://api-fxpractice.oanda.com"
        else:
            self.base_url = "https://api-fxtrade.oanda.com"
        
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        # Use session for connection pooling (low latency optimization)
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def connect(self) -> bool:
        """Establish connection to OANDA"""
        try:
            # If account_id not provided or invalid, fetch it from accounts endpoint
            if not self.account_id or not self.account_id.strip() or self.account_id.strip().startswith('#'):
                response = self.session.get(f"{self.base_url}/v3/accounts")
                response.raise_for_status()
                accounts_data = response.json()
                
                # Get the first account ID
                if accounts_data.get('accounts'):
                    self.account_id = accounts_data['accounts'][0]['id']
                    print(f"Auto-detected OANDA account ID: {self.account_id}")
                else:
                    raise ValueError("No accounts found")
            
            # Verify connection by fetching account details
            response = self.session.get(f"{self.base_url}/v3/accounts/{self.account_id}")
            response.raise_for_status()
            self.connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to OANDA: {e}")
            self.connected = False
            return False
    
    def disconnect(self) -> bool:
        """Close connection"""
        self.connected = False
        return True
    
    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """Fetch historical OHLCV data from OANDA"""
        try:
            # Clamp dates to current time to avoid future date requests
            now_utc = datetime.now(timezone.utc)
            
            # Safety check: if system clock seems wrong (year > current year + 1), use a reasonable max date
            # This handles cases where system clock is set incorrectly, but allow current year
            current_year = datetime.now().year
            max_reasonable_year = current_year + 1  # Allow up to next year
            if now_utc.year > max_reasonable_year:
                print(f"Warning: System clock shows year {now_utc.year} (expected <= {max_reasonable_year}). Using {current_year}-12-31 as maximum date.")
                max_reasonable_date = datetime(current_year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
                now_utc = max_reasonable_date
            
            # Ensure dates are timezone-aware
            if start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=timezone.utc)
            if end_date and end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=timezone.utc)
            
            # Clamp to current time (or reasonable max)
            if start_date > now_utc:
                print(f"Warning: start_date {start_date} is in the future. Clamping to 30 days before {now_utc}.")
                start_date = now_utc - timedelta(days=30)
            if end_date and end_date > now_utc:
                print(f"Warning: end_date {end_date} is in the future. Clamping to {now_utc}.")
                end_date = now_utc
            elif not end_date:
                end_date = now_utc
            
            # Additional safety: ensure dates aren't unreasonably far in the future
            current_year = datetime.now().year
            if start_date.year > current_year + 1:
                print(f"Warning: start_date year {start_date.year} seems incorrect. Using {current_year}-11-01.")
                start_date = datetime(current_year, 11, 1, tzinfo=timezone.utc)
            if end_date and end_date.year > current_year + 1:
                print(f"Warning: end_date year {end_date.year} seems incorrect. Using {current_year}-12-31.")
                end_date = datetime(current_year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            
            # Ensure start_date is before end_date
            if start_date >= end_date:
                print(f"Warning: start_date {start_date} >= end_date {end_date}. Adjusting...")
                start_date = end_date - timedelta(days=30)
            
            # Final validation: ensure we're not requesting data more than 1 year old
            # OANDA typically has data going back several years, but let's be conservative
            min_date = now_utc - timedelta(days=365)
            if start_date < min_date:
                print(f"Warning: start_date {start_date} is more than 1 year ago. Adjusting to {min_date}.")
                start_date = min_date
            
            # Convert symbol format (EUR/USD -> EUR_USD)
            oanda_symbol = symbol.replace('/', '_')
            
            # Convert timeframe to OANDA format
            timeframe_map = {
                '1m': 'M1', '5m': 'M5', '15m': 'M15', '30m': 'M30',
                '1h': 'H1', '4h': 'H4', '1d': 'D', '1w': 'W'
            }
            tf = timeframe_map.get(timeframe, 'H1')
            
            # Convert datetime to RFC3339 format
            from_date = start_date.strftime('%Y-%m-%dT%H:%M:%S.000000000Z')
            to_date = end_date.strftime('%Y-%m-%dT%H:%M:%S.000000000Z') if end_date else None
            
            url = f"{self.base_url}/v3/instruments/{oanda_symbol}/candles"
            params = {
                'granularity': tf,
                'from': from_date
            }
            
            if to_date:
                params['to'] = to_date
            if limit:
                params['count'] = limit
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Convert to standard format
            result = []
            for candle in data.get('candles', []):
                if candle['complete']:
                    result.append({
                        'timestamp': datetime.fromisoformat(candle['time'].replace('Z', '+00:00')),
                        'open': float(candle['mid']['o']),
                        'high': float(candle['mid']['h']),
                        'low': float(candle['mid']['l']),
                        'close': float(candle['mid']['c']),
                        'volume': int(candle['volume'])
                    })
            
            return result
        except Exception as e:
            print(f"Error fetching historical data from OANDA: {e}")
            return []
    
    def get_current_price(self, symbol: str) -> float:
        """Get current market price - OANDA supports forex pairs and commodities (metals)"""
        try:
            # OANDA supports forex and commodities (precious metals), not crypto or stocks
            # Check if this is a crypto symbol (contains common crypto bases)
            crypto_bases = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'LINK', 'MATIC', 'AVAX', 'UNI', 'ATOM']
            symbol_base = symbol.split('/')[0].upper() if '/' in symbol else symbol[:3].upper()
            
            # Check if it's a commodity (precious metals)
            commodities = ['XAU', 'XAG']  # Gold, Silver
            is_commodity = symbol_base in commodities
            
            # Check if it's crypto (but not commodity)
            if symbol_base in crypto_bases and not is_commodity:
                # This is crypto - OANDA doesn't support it
                return 0.0
            
            # OANDA supports forex and commodities - proceed with API call
            oanda_symbol = symbol.replace('/', '_')
            url = f"{self.base_url}/v3/instruments/{oanda_symbol}/candles"
            params = {'granularity': 'M1', 'count': 1}
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get('candles'):
                return float(data['candles'][0]['mid']['c'])
            return 0.0
        except Exception as e:
            # Don't log errors for unsupported symbols (crypto/stocks)
            error_str = str(e)
            if '400' not in error_str and 'Bad Request' not in error_str:
                print(f"Error fetching current price from OANDA: {e}")
            return 0.0
    
    def _get_account_details(self) -> Optional[Dict]:
        """
        Get full account details from OANDA API (includes balance and positions)
        
        According to OANDA API docs: https://developer.oanda.com/rest-live-v20/account-ep/
        The /v3/accounts/{accountID} endpoint returns full account details including positions.
        This method caches the response to avoid multiple API calls.
        """
        max_retries = 3
        backoff_seconds = 2
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                response = self.session.get(
                    f"{self.base_url}/v3/accounts/{self.account_id}",
                    timeout=10
                )
                # If 5xx, retry
                if response.status_code >= 500:
                    last_error = f"Server error {response.status_code}"
                    if attempt < max_retries:
                        time.sleep(backoff_seconds * attempt)
                        continue
                response.raise_for_status()
                data = response.json()
                return data.get('account')
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    time.sleep(backoff_seconds * attempt)
                    continue
                # Only print error on final attempt to reduce noise
                if attempt == max_retries:
                    print(f"Error fetching account details from OANDA after {max_retries} attempts: {last_error}")
                break

        return None
    
    def get_account_balance(self) -> Dict[str, float]:
        """Get account balance

        OANDA v3 account payload does not expose an 'available' field.
        It typically has:
          - balance
          - NAV
          - marginAvailable
          - marginUsed
        We approximate:
          total     -> balance
          available -> marginAvailable (or balance if missing)
          used      -> total - available
        """
        account = self._get_account_details()
        if not account:
            # Fallback empty balance on failure (graceful degradation)
            return {'total': 0.0, 'available': 0.0, 'used': 0.0, 'currencies': {}}

        balance = float(account.get('balance', 0.0))
        # Prefer marginAvailable, fall back to NAV, then balance
        available = float(
            account.get('marginAvailable', account.get('NAV', balance))
        )
        used = balance - available

        return {
            'total': balance,
            'available': available,
            'used': used,
            'currencies': {
                account.get('currency', 'USD'): {
                    'total': balance,
                    'available': available,
                    'used': used
                }
            }
        }
    
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
        """Place an order on OANDA"""
        try:
            oanda_symbol = symbol.replace('/', '_')
            url = f"{self.base_url}/v3/accounts/{self.account_id}/orders"
            
            # OANDA uses units of the base currency (not lots). The trading
            # system already calculates a position quantity based on account
            # value, so we pass that through directly and only convert to int.
            units = int(round(abs(quantity)))
            
            # OANDA minimum and maximum order sizes:
            # - Forex pairs: typically 1 unit minimum, 100,000 units maximum
            # - XAU/USD (Gold): 1 oz minimum, 100,000 oz maximum
            # - XAG/USD (Silver): 1 oz minimum, 100,000 oz maximum
            # - Other commodities: check OANDA documentation
            min_units = 1
            max_units = 100000  # OANDA's typical maximum
            
            if 'XAU' in oanda_symbol or 'XAG' in oanda_symbol:
                # Precious metals - ensure at least 1 oz
                min_units = 1
                max_units = 100000
            elif 'USD' in oanda_symbol and '/' in oanda_symbol:
                # Forex pairs - typically 1 unit minimum
                min_units = 1
                max_units = 100000
            
            if units < min_units:
                print(f"  ⚠️  Warning: Order size {units} is below minimum {min_units} for {oanda_symbol}. Adjusting to minimum.")
                units = min_units
            
            if units > max_units:
                print(f"  ⚠️  Warning: Order size {units} exceeds maximum {max_units} for {oanda_symbol}. Adjusting to maximum.")
                units = max_units
            
            if units == 0:
                units = min_units  # ensure we send at least minimum units
            
            if side == OrderSide.SELL:
                units = -units
            
            order_data = {
                'order': {
                    'instrument': oanda_symbol,
                    'units': str(units),
                    'type': 'MARKET' if order_type == OrderType.MARKET else 'LIMIT',
                    # OANDA doesn't support FOK for all instruments. Use GTC (Good Till Cancelled) for market orders
                    # or remove timeInForce entirely for market orders (it's optional)
                    'timeInForce': 'GTC' if order_type == OrderType.LIMIT else None
                }
            }
            # Remove timeInForce if None (OANDA doesn't require it for market orders)
            if order_data['order']['timeInForce'] is None:
                del order_data['order']['timeInForce']
            
            # Get price precision for this instrument (needed for limit orders and stop/take profit)
            # OANDA requires different precision for different instruments:
            # - Forex pairs (EUR/USD, GBP/USD, etc.): 5 decimal places
            # - Precious metals (XAU/USD, XAG/USD): 2-3 decimal places
            # - JPY pairs (USD/JPY): 3 decimal places
            def get_price_precision(instrument: str) -> int:
                """Get the number of decimal places required for price formatting"""
                instrument_upper = instrument.upper()
                if 'XAU' in instrument_upper or 'XAG' in instrument_upper:
                    # Precious metals: 2-3 decimal places (use 3 for safety)
                    return 3
                elif 'JPY' in instrument_upper:
                    # JPY pairs: 3 decimal places
                    return 3
                else:
                    # Forex pairs: 5 decimal places
                    return 5
            
            price_precision = get_price_precision(oanda_symbol)
            
            if order_type == OrderType.LIMIT:
                if price is None:
                    raise ValueError("Price required for LIMIT orders")
                # Format limit price with correct precision
                price_rounded = round(price, price_precision)
                order_data['order']['price'] = f"{price_rounded:.{price_precision}f}".rstrip('0').rstrip('.')
            
            # Format stop loss and take profit with instrument-specific precision
            if stop_loss:
                # Round to appropriate precision and format
                stop_loss_rounded = round(stop_loss, price_precision)
                stop_loss_price = f"{stop_loss_rounded:.{price_precision}f}".rstrip('0').rstrip('.')
                order_data['order']['stopLossOnFill'] = {'price': stop_loss_price}
            if take_profit:
                # Round to appropriate precision and format
                take_profit_rounded = round(take_profit, price_precision)
                take_profit_price = f"{take_profit_rounded:.{price_precision}f}".rstrip('0').rstrip('.')
                order_data['order']['takeProfitOnFill'] = {'price': take_profit_price}
            
            # Log order data for debugging
            print(f"  📤 OANDA Order Data: {order_data}")
            print(f"     Instrument: {oanda_symbol}, Units: {units}, Side: {side.value}")
            if stop_loss:
                print(f"     Stop Loss: {stop_loss_price}")
            if take_profit:
                print(f"     Take Profit: {take_profit_price}")
            
            response = self.session.post(url, json=order_data)
            
            # OANDA returns 200 (OK) or 201 (Created) for successful orders
            # 201 means the order was successfully created and may be filled immediately
            is_success = response.status_code in [200, 201]
            
            # Parse response data
            try:
                data = response.json()
            except:
                data = {}
            
            # Check if response contains an error (even with 200/201 status)
            error_data = data if not is_success else {}
            error_msg = data.get('errorMessage', '') if not is_success else None
            error_code = data.get('errorCode', '') if not is_success else None
            
            # Also check for orderRejectTransaction which indicates rejection
            if 'orderRejectTransaction' in data:
                is_success = False
                error_data = data
                error_msg = data.get('errorMessage', 'Order was rejected')
                error_code = data.get('errorCode', 'ORDER_REJECTED')
            
            if not is_success:
                # Log detailed error information
                print(f"  ❌ OANDA API Error ({response.status_code}): {error_code}")
                print(f"     Error Message: {error_msg}")
                print(f"     Full error response: {error_data}")
                print(f"     Order that was rejected: {order_data}")
                
                # Check for common rejection reasons
                if error_msg:
                    if 'insufficient' in error_msg.lower() or 'margin' in error_msg.lower():
                        print(f"     💡 Suggestion: Check account balance and margin requirements")
                    elif 'size' in error_msg.lower() or 'units' in error_msg.lower():
                        print(f"     💡 Suggestion: Order size may be too large or too small for this instrument")
                    elif 'instrument' in error_msg.lower():
                        print(f"     💡 Suggestion: Check if instrument code '{oanda_symbol}' is correct")
                    elif 'market' in error_msg.lower() and 'closed' in error_msg.lower():
                        print(f"     💡 Suggestion: Market may be closed for this instrument")
                
                return {
                    'order_id': None,
                    'status': OrderStatus.REJECTED,
                    'error': f"{response.status_code} {error_code}: {error_msg}" if error_msg else f"HTTP {response.status_code}",
                    'error_details': error_data
                }
            
            # Success - order was created/filled
            if response.status_code == 201:
                print(f"  ✅ Order created successfully (HTTP 201)")
            
            # Extract order information - OANDA may return orderFillTransaction (filled immediately)
            # or orderCreateTransaction (pending) in the response
            order_fill = data.get('orderFillTransaction')
            order_create = data.get('orderCreateTransaction')
            
            # Prefer fill transaction if available (order was filled immediately)
            if order_fill:
                order_info = order_fill
                order_id = order_fill.get('orderID') or order_fill.get('id')
                filled_units = abs(float(order_fill.get('units', 0)))
                fill_price = float(order_fill.get('price', 0.0))
                print(f"  ✅ Order filled immediately: ID={order_id}, Units={filled_units}, Price={fill_price}")
            elif order_create:
                order_info = order_create
                order_id = order_create.get('id')
                filled_units = 0  # Not filled yet
                fill_price = price or 0.0
                print(f"  ✅ Order created (pending): ID={order_id}")
            else:
                # Fallback - try to find any transaction with order info
                order_info = data.get('orderFillTransaction', data.get('orderCreateTransaction', {}))
                order_id = order_info.get('id') or order_info.get('orderID')
                filled_units = abs(float(order_info.get('units', 0)))
                fill_price = float(order_info.get('price', price or 0.0))
            
            return {
                'order_id': str(order_id) if order_id else None,
                'status': OrderStatus.FILLED if order_fill else OrderStatus.PENDING,
                'filled_quantity': filled_units,
                'symbol': symbol,
                'side': side.value,
                'price': fill_price,
                'timestamp': datetime.now()
            }
        except requests.exceptions.HTTPError as e:
            # Handle HTTP errors with better error messages
            error_msg = str(e)
            try:
                if hasattr(e.response, 'json'):
                    error_data = e.response.json()
                    error_msg = error_data.get('errorMessage', error_data.get('error', error_msg))
                    print(f"  ❌ OANDA HTTP Error: {error_msg}")
                    print(f"     Full error response: {error_data}")
            except:
                pass
            print(f"Error placing order on OANDA: {error_msg}")
            return {
                'order_id': None,
                'status': OrderStatus.REJECTED,
                'error': error_msg
            }
        except Exception as e:
            print(f"Error placing order on OANDA: {e}")
            return {
                'order_id': None,
                'status': OrderStatus.REJECTED,
                'error': str(e)
            }
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}/orders/{order_id}/cancel"
            response = self.session.put(url)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Error cancelling order on OANDA: {e}")
            return False
    
    def get_open_positions(self) -> List[Dict]:
        """
        Get open positions from account details
        
        According to OANDA API docs: https://developer.oanda.com/rest-live-v20/account-ep/
        The account endpoint already includes positions, so we extract them from there
        instead of making a separate API call.
        """
        account = self._get_account_details()
        if not account:
            # Return empty list on failure (graceful degradation)
            return []
        
        positions = []
        # Extract positions from account response
        # Positions are in account['positions'] array
        for pos in account.get('positions', []):
            long_units = float(pos.get('long', {}).get('units', 0))
            short_units = float(pos.get('short', {}).get('units', 0))
            
            if long_units != 0:
                positions.append({
                    'symbol': pos['instrument'],
                    'side': 'buy',
                    'quantity': abs(long_units),
                    'entry_price': float(pos.get('long', {}).get('averagePrice', 0)),
                    'unrealized_pnl': float(pos.get('long', {}).get('unrealizedPL', 0))
                })
            
            if short_units != 0:
                positions.append({
                    'symbol': pos['instrument'],
                    'side': 'sell',
                    'quantity': abs(short_units),
                    'entry_price': float(pos.get('short', {}).get('averagePrice', 0)),
                    'unrealized_pnl': float(pos.get('short', {}).get('unrealizedPL', 0))
                })
        
        return positions
    
    def get_order_status(self, order_id: str) -> Dict:
        """Get order status"""
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
            print(f"Error fetching order status from OANDA: {e}")
            return {'order_id': order_id, 'status': OrderStatus.REJECTED}
    
    def close_position(self, symbol: str, side: Optional[OrderSide] = None) -> bool:
        """Close position"""
        try:
            oanda_symbol = symbol.replace('/', '_')
            url = f"{self.base_url}/v3/accounts/{self.account_id}/positions/{oanda_symbol}/close"
            
            data = {}
            if side:
                data['longUnits'] = 'ALL' if side == OrderSide.BUY else 'NONE'
                data['shortUnits'] = 'ALL' if side == OrderSide.SELL else 'NONE'
            else:
                data['longUnits'] = 'ALL'
                data['shortUnits'] = 'ALL'
            
            response = self.session.put(url, json=data)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Error closing position on OANDA: {e}")
            return False
    
    def get_supported_assets(self) -> List[str]:
        """Get supported asset classes"""
        return ['forex', 'commodities']  # OANDA supports forex and precious metals
    
    def is_market_open(self, symbol: str) -> bool:
        """
        Check if market is open for forex or commodities
        
        Forex markets:
        - Open: Sunday 5:00 PM EST (22:00 UTC) to Friday 5:00 PM EST (22:00 UTC)
        - Closed: Friday 5:00 PM EST to Sunday 5:00 PM EST
        
        Commodities (precious metals):
        - Open: Sunday 5:00 PM EST (22:00 UTC) to Friday 5:00 PM EST (22:00 UTC)
        - Closed: Friday 5:00 PM EST to Sunday 5:00 PM EST
        - Similar to forex trading hours
        
        Note: This is simplified - actual close times vary by broker
        """
        now = datetime.now()
        weekday = now.weekday()  # Monday=0, Sunday=6
        hour = now.hour
        
        # Check if it's a commodity
        symbol_base = symbol.split('/')[0].upper() if '/' in symbol else symbol[:3].upper()
        commodities = ['XAU', 'XAG']  # Gold, Silver
        is_commodity = symbol_base in commodities
        
        # Forex and commodities markets are closed on weekends (Saturday and most of Sunday)
        if weekday == 5:  # Saturday - always closed
            return False
        elif weekday == 6:  # Sunday - closed until 5 PM EST (22:00 UTC)
            # Market opens Sunday at 22:00 UTC (5 PM EST)
            if hour < 22:
                return False
            return True
        elif weekday == 4:  # Friday - closes at 5 PM EST (22:00 UTC)
            # Market closes Friday at 22:00 UTC (5 PM EST)
            if hour >= 22:
                return False
            return True
        else:
            # Monday-Thursday: Market is open 24 hours
            return True
    
    def check_liquidity(self, symbol: str, order_size: float) -> Dict:
        """Check liquidity for OANDA (forex markets are generally very liquid)"""
        try:
            # OANDA forex markets are highly liquid
            # For major pairs, liquidity is typically excellent
            oanda_symbol = symbol.replace('/', '_')
            major_pairs = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'AUD_USD', 'USD_CAD']
            
            is_major_pair = oanda_symbol in major_pairs
            
            # Estimate slippage (very low for major pairs)
            if is_major_pair:
                estimated_slippage = 1.0  # 1 basis point for major pairs
            else:
                estimated_slippage = 3.0  # 3 basis points for minor pairs
            
            return {
                'sufficient': True,
                'estimated_slippage_bps': estimated_slippage,
                'order_book_depth': order_size * 100,  # Forex is very liquid
                'reason': 'Forex market liquidity (major pair)' if is_major_pair else 'Forex market liquidity'
            }
        except Exception as e:
            return {
                'sufficient': True,  # Default to allowing trade
                'estimated_slippage_bps': 5.0,
                'order_book_depth': order_size * 10,
                'reason': f'Liquidity check error: {str(e)}'
            }

