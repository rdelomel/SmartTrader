"""Alpaca API implementation for stocks and crypto trading"""

import requests
from typing import Dict, List, Optional
from datetime import datetime
import os

from .base_broker import BaseBroker, OrderType, OrderSide, OrderStatus


class AlpacaBroker(BaseBroker):
    """Alpaca broker implementation for stocks and crypto"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        super().__init__(api_key, api_secret, testnet)
        
        # Trading base URL (orders, account, positions, etc.)
        if testnet:
            self.base_url = "https://paper-api.alpaca.markets"
        else:
            self.base_url = "https://api.alpaca.markets"

        # Market data base URL for crypto (per Alpaca docs:
        # https://docs.alpaca.markets/docs/crypto-trading)
        # Example: https://data.alpaca.markets/v1beta3/crypto/us/bars
        self.data_base_url = "https://data.alpaca.markets"
        
        self.headers = {
            'APCA-API-KEY-ID': api_key,
            'APCA-API-SECRET-KEY': api_secret
        }
        
        # Use session for connection pooling (low latency optimization)
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def connect(self) -> bool:
        """Establish connection to Alpaca"""
        try:
            response = requests.get(
                f"{self.base_url}/v2/account",
                headers=self.headers
            )
            response.raise_for_status()
            self.connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to Alpaca: {e}")
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
        """Fetch historical OHLCV data from Alpaca"""
        try:
            # Determine if this is a crypto pair (contains '/')
            is_crypto = '/' in symbol
            alpaca_symbol = symbol
            
            # Convert timeframe to Alpaca format
            timeframe_map = {
                '1m': '1Min', '5m': '5Min', '15m': '15Min', '30m': '30Min',
                '1h': '1Hour', '4h': '4Hour', '1d': '1Day', '1w': '1Week'
            }
            tf = timeframe_map.get(timeframe, '1Hour')

            # Use Market Data API for both crypto and stocks.
            # For crypto, per docs: /v1beta3/crypto/us/bars with ?symbols=BTC/USD
            # For stocks, use /v2/stocks/{symbol}/bars as before.
            if is_crypto:
                url = f"{self.data_base_url}/v1beta3/crypto/us/bars"
                params = {
                    'symbols': alpaca_symbol,
                    'timeframe': tf,
                    'start': start_date.strftime('%Y-%m-%dT%H:%M:%S-00:00'),
                    'limit': limit or 1000
                }
                if end_date:
                    params['end'] = end_date.strftime('%Y-%m-%dT%H:%M:%S-00:00')
                
                response = self.session.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                # v1beta3 returns { "bars": { "BTC/USD": [ ... ] } }
                bars = []
                if data.get('bars'):
                    # Use the first symbol key (should match alpaca_symbol)
                    key = alpaca_symbol if alpaca_symbol in data['bars'] else next(iter(data['bars']))
                    bars = data['bars'].get(key, [])
            else:
                url = f"{self.base_url}/v2/stocks/{alpaca_symbol}/bars"
                params = {
                    'timeframe': tf,
                    'start': start_date.strftime('%Y-%m-%dT%H:%M:%S-00:00'),
                    'limit': limit or 1000
                }
                if end_date:
                    params['end'] = end_date.strftime('%Y-%m-%dT%H:%M:%S-00:00')
                
                response = self.session.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                bars = data.get('bars', [])

            # Convert to standard format
            result = []
            for bar in bars:
                result.append({
                    'timestamp': datetime.fromisoformat(bar['t'].replace('Z', '+00:00')),
                    'open': float(bar['o']),
                    'high': float(bar['h']),
                    'low': float(bar['l']),
                    'close': float(bar['c']),
                    'volume': int(bar['v'])
                })
            
            return result
        except Exception as e:
            print(f"Error fetching historical data from Alpaca: {e}")
            return []
    
    def get_current_price(self, symbol: str) -> float:
        """Get current market price"""
        try:
            # Normalize symbol: detect crypto even without slash
            is_crypto = '/' in symbol
            alpaca_symbol = symbol
            
            # If no slash, check if it's a known crypto symbol (e.g., ETHUSD, BTCUSD)
            if not is_crypto:
                symbol_upper = symbol.upper()
                # Common crypto patterns: ends with USD/EUR and starts with crypto base
                crypto_bases_3char = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'LTC', 'BCH', 'XRP', 'DOGE']
                crypto_bases_4char = ['MATIC', 'AVAX', 'LINK', 'UNI', 'ATOM', 'ALGO']
                
                # Check 3-character bases first
                for base in crypto_bases_3char:
                    if symbol_upper.startswith(base) and (symbol_upper.endswith('USD') or symbol_upper.endswith('EUR')):
                        is_crypto = True
                        quote = symbol_upper[len(base):]
                        alpaca_symbol = f"{base}/{quote}"
                        break
                
                # If not found, check 4-character bases
                if not is_crypto:
                    for base in crypto_bases_4char:
                        if symbol_upper.startswith(base) and (symbol_upper.endswith('USD') or symbol_upper.endswith('EUR')):
                            is_crypto = True
                            quote = symbol_upper[len(base):]
                            alpaca_symbol = f"{base}/{quote}"
                            break
            
            # Use Market Data API for latest trade for crypto:
            # GET /v1beta3/crypto/us/latest/trades?symbols=BTC/USD
            if is_crypto:
                url = f"{self.data_base_url}/v1beta3/crypto/us/latest/trades"
                params = {'symbols': alpaca_symbol}
                response = self.session.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                trades = data.get('trades', {})
                trade = trades.get(alpaca_symbol)
                if not trade:
                    # Fallback: try any trade value
                    if trades:
                        trade = next(iter(trades.values()))
                if not trade:
                    return 0.0
                return float(trade['p'])
            else:
                url = f"{self.base_url}/v2/stocks/{alpaca_symbol}/trades/latest"
                response = self.session.get(url)
                response.raise_for_status()
                data = response.json()
                return float(data['trade']['p'])
        except Exception as e:
            print(f"Error fetching current price from Alpaca: {e}")
            return 0.0
    
    def get_account_balance(self) -> Dict[str, float]:
        """Get account balance"""
        try:
            response = requests.get(
                f"{self.base_url}/v2/account",
                headers=self.headers
            )
            response.raise_for_status()
            account = response.json()
            
            portfolio_value = float(account['portfolio_value'])
            buying_power = float(account['buying_power'])
            cash = float(account.get('cash', account.get('non_marginable_buying_power', buying_power)))
            
            # Use cash (actual money) instead of buying_power (which includes margin/leverage)
            # If cash is not available, use non_marginable_buying_power as fallback
            # Only use buying_power if neither cash nor non_marginable_buying_power is available
            if 'cash' in account:
                available = cash
            elif 'non_marginable_buying_power' in account:
                available = float(account['non_marginable_buying_power'])
            else:
                # Fallback to buying_power, but warn if it's higher than portfolio_value (leverage)
                available = buying_power
                if available > portfolio_value * 1.1:
                    print(f"  ⚠️  WARNING: Buying power (${available:.2f}) > Portfolio value (${portfolio_value:.2f}) - likely leverage")
                    print(f"     Using conservative available balance: ${portfolio_value * 0.95:.2f}")
                    available = portfolio_value * 0.95  # Use 95% of portfolio value as conservative estimate
            
            used = portfolio_value - available
            
            return {
                'total': portfolio_value,
                'available': available,
                'used': used,
                'currencies': {
                    'USD': {
                        'total': portfolio_value,
                        'available': available,
                        'used': used
                    }
                }
            }
        except Exception as e:
            print(f"Error fetching account balance from Alpaca: {e}")
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
        """
        Place an order on Alpaca
        
        Bracket Orders (Take Profit / Stop Loss):
        - Stocks: ✅ Supported - Can include stop_loss and take_profit in order
        - Crypto: ❌ NOT Supported - Must be managed separately by the trading system
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL' for stocks, 'BTC/USD' for crypto)
            side: Order side (BUY or SELL)
            order_type: Order type (MARKET or LIMIT)
            quantity: Order quantity
            price: Limit price (required for LIMIT orders)
            stop_price: Stop price (for stop orders)
            take_profit: Take profit price (bracket order - stocks only)
            stop_loss: Stop loss price (bracket order - stocks only)
        
        Returns:
            Dictionary with order_id, status, and error (if any)
        """
        print(f"\n[AlpacaBroker.place_order]")
        print(f"  Symbol: {symbol}")
        print(f"  Side: {side.value if hasattr(side, 'value') else side}")
        print(f"  Quantity: {quantity}")
        print(f"  Order Type: {order_type.value if hasattr(order_type, 'value') else order_type}")
        print(f"  Base URL: {self.base_url}")
        print(f"  Connected: {self.connected}")
        
        try:
            if not self.connected:
                error_msg = "Broker not connected"
                print(f"  ❌ {error_msg}")
                return {
                    'order_id': None,
                    'status': OrderStatus.REJECTED,
                    'error': error_msg
                }
            
            # Detect crypto vs stocks/forex
            # Crypto symbols: BTC/USD, ETH/USD, SOL/USD (crypto base + USD/EUR)
            # Forex symbols: EUR/USD, GBP/USD, USD/JPY (currency pairs)
            is_crypto = False
            if '/' in symbol:
                base = symbol.split('/')[0].upper()
                quote = symbol.split('/')[1].upper()
                # Crypto bases: BTC, ETH, SOL, etc.
                crypto_bases = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'MATIC', 'AVAX', 'LINK', 'UNI', 'ATOM', 'ALGO', 'XRP', 'DOGE', 'LTC', 'BCH']
                # Crypto quotes: USD, EUR (not JPY, GBP, etc.)
                crypto_quotes = ['USD', 'EUR']
                if base in crypto_bases and quote in crypto_quotes:
                    is_crypto = True
                    alpaca_symbol = symbol  # Keep original format: BTC/USD
                    print(f"  Crypto symbol detected: {symbol} (keeping format with slash)")
                else:
                    # Forex or other - Alpaca doesn't support forex trading
                    error_msg = f"Alpaca does not support forex trading for {symbol}"
                    print(f"  ❌ {error_msg}")
                    return {
                        'order_id': None,
                        'status': OrderStatus.REJECTED,
                        'error': error_msg
                    }
            else:
                alpaca_symbol = symbol
                print(f"  Stock symbol: {alpaca_symbol}")
            
            # Alpaca paper trading uses /v2/orders for both stocks and crypto
            # The /v2/crypto/orders endpoint doesn't exist in paper trading
            url = f"{self.base_url}/v2/orders"
            
            print(f"  Order URL: {url}")
            
            order_data = {
                'symbol': alpaca_symbol,  # Use SOL/USD format for crypto
                'qty': str(quantity) if is_crypto else str(int(quantity)),  # Crypto can use fractional quantities
                'side': side.value,
                'type': 'market' if order_type == OrderType.MARKET else 'limit',
                'time_in_force': 'gtc' if is_crypto else 'day'  # Good till cancelled for crypto, day for stocks
            }
            
            if order_type == OrderType.LIMIT:
                if price is None:
                    raise ValueError("Price required for LIMIT orders")
                order_data['limit_price'] = str(price)
            
            sl_submitted = False
            tp_submitted = False

            # Alpaca doesn't support bracket orders (stop-loss/take-profit) for crypto
            # These need to be handled separately by the trading system
            if not is_crypto:
                # Only add stop_loss/take_profit for stocks
                if stop_loss:
                    order_data['stop_loss'] = {'stop_price': str(stop_loss)}
                    sl_submitted = True
                if take_profit:
                    order_data['take_profit'] = {'limit_price': str(take_profit)}
                    tp_submitted = True
            else:
                # For crypto, log that stop_loss/take_profit will be handled separately
                if stop_loss or take_profit:
                    print(f"  ⚠️  Note: Alpaca doesn't support bracket orders for crypto.")
                    print(f"     Stop-loss ({stop_loss}) and take-profit ({take_profit}) will need to be managed separately.")
            
            print(f"  Order data: {order_data}")
            print(f"  📤 Sending POST request to Alpaca...")
            
            response = self.session.post(url, json=order_data)
            
            print(f"  Response status: {response.status_code}")
            print(f"  Response headers: {dict(response.headers)}")
            
            if response.status_code != 200:
                print(f"  ❌ Non-200 response: {response.status_code}")
                print(f"  Response text: {response.text}")
                
                # Try to extract error message from response
                error_msg = f"{response.status_code} Client Error"
                try:
                    error_data = response.json()
                    if 'message' in error_data:
                        error_msg = error_data['message']
                    elif 'error' in error_data:
                        error_msg = error_data['error']
                except:
                    error_msg = response.text or f"{response.status_code} Client Error"
                
                # Return detailed error for 403 (insufficient balance) and other errors
                return {
                    'order_id': None,
                    'status': OrderStatus.REJECTED,
                    'error': error_msg,
                    'error_code': response.status_code,
                    'response_text': response.text
                }
            
            response.raise_for_status()
            order = response.json()
            
            print(f"  ✅ Order response from Alpaca: {order}")
            
            order_id = order.get('id')
            order_status = order.get('status', 'unknown')
            
            print(f"  Order ID: {order_id}")
            print(f"  Order Status: {order_status}")
            
            # Handle price - for market orders, limit_price is None, so use entry price or current price
            limit_price = order.get('limit_price')
            if limit_price is not None:
                order_price = float(limit_price)
            elif price is not None:
                order_price = float(price)
            else:
                # For market orders, we don't have a limit price, use 0.0 as placeholder
                # The actual fill price will be in filled_avg_price when order is filled
                order_price = 0.0
            
            # Get filled average price if available (for market orders that have filled)
            filled_avg_price = order.get('filled_avg_price')
            if filled_avg_price is not None:
                final_price = float(filled_avg_price)
            else:
                final_price = order_price
            
            result = {
                'order_id': order_id,
                'status': OrderStatus.PENDING if order_status in ['new', 'pending_new'] else OrderStatus.FILLED,
                'filled_quantity': float(order.get('filled_qty', 0)),
                'symbol': symbol,
                'side': side.value,
                'price': final_price,
                'timestamp': datetime.now(),
                'stop_loss_requested': bool(stop_loss),
                'take_profit_requested': bool(take_profit),
                'stop_loss_submitted': sl_submitted,
                'take_profit_submitted': tp_submitted,
                'bracket_supported': not is_crypto
            }
            
            print(f"  ✅ Order placed successfully: {result}")
            return result
            
        except Exception as e:
            import traceback
            print(f"  ❌❌❌ EXCEPTION in AlpacaBroker.place_order: {e}")
            print(f"  Traceback: {traceback.format_exc()}")
            return {
                'order_id': None,
                'status': OrderStatus.REJECTED,
                'error': str(e)
            }
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            url = f"{self.base_url}/v2/orders/{order_id}"
            response = self.session.delete(url)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Error cancelling order on Alpaca: {e}")
            return False
    
    def get_open_positions(self) -> List[Dict]:
        """Get open positions"""
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
                
                # Calculate current price from market value and quantity
                # This is more accurate than fetching separately
                current_price = market_value / quantity if quantity > 0 else avg_entry_price
                
                positions.append({
                    'symbol': pos['symbol'],
                    'side': 'buy' if qty > 0 else 'sell',
                    'quantity': quantity,
                    'entry_price': avg_entry_price,
                    'current_price': current_price,  # Add current price from Alpaca
                    'unrealized_pnl': unrealized_pl,  # Use Alpaca's calculated P&L
                    'market_value': market_value
                })
            
            return positions
        except Exception as e:
            print(f"Error fetching open positions from Alpaca: {e}")
            return []
    
    def get_orders(self, status: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Get orders from Alpaca (filled, pending, etc.)"""
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
                # Handle price safely
                limit_price = order.get('limit_price')
                filled_avg_price = order.get('filled_avg_price')
                
                if filled_avg_price is not None and filled_avg_price != 'None':
                    try:
                        order_price = float(filled_avg_price)
                    except (ValueError, TypeError):
                        order_price = 0.0
                elif limit_price is not None and limit_price != 'None':
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
            print(f"Error fetching orders from Alpaca: {e}")
            return []
    
    def get_order_status(self, order_id: str) -> Dict:
        """Get order status"""
        try:
            response = requests.get(
                f"{self.base_url}/v2/orders/{order_id}",
                headers=self.headers
            )
            response.raise_for_status()
            order = response.json()
            
            status_map = {
                'new': OrderStatus.PENDING,
                'filled': OrderStatus.FILLED,
                'partially_filled': OrderStatus.PARTIALLY_FILLED,
                'canceled': OrderStatus.CANCELLED,
                'rejected': OrderStatus.REJECTED
            }
            
            # Handle price safely - limit_price can be None for market orders
            limit_price = order.get('limit_price')
            if limit_price is not None and limit_price != 'None':
                try:
                    order_price = float(limit_price)
                except (ValueError, TypeError):
                    order_price = 0.0
            else:
                order_price = 0.0
            
            return {
                'order_id': order['id'],
                'status': status_map.get(order['status'], OrderStatus.PENDING),
                'filled_quantity': float(order.get('filled_qty', 0) or 0),
                'price': order_price
            }
        except Exception as e:
            print(f"Error fetching order status from Alpaca: {e}")
            return {'order_id': order_id, 'status': OrderStatus.REJECTED}
    
    def close_position(self, symbol: str, side: Optional[OrderSide] = None) -> bool:
        """
        Close position on Alpaca
        
        Note:
        - For stocks: Uses DELETE /v2/positions/{symbol}
        - For crypto: Places an opposite market order (Alpaca doesn't support DELETE for crypto positions)
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL' or 'BTC/USD')
            side: Optional side hint (not used for stocks, but helps for crypto)
        
        Returns:
            True if position closed successfully, False otherwise
        """
        try:
            if not self.connected:
                print(f"Error closing position: Broker not connected")
                return False
            
            is_crypto = '/' in symbol
            
            # For crypto, we need to place an opposite order instead of DELETE
            if is_crypto:
                # Get current position to determine quantity and side
                try:
                    positions = self.get_open_positions()
                    
                    # Normalize the input symbol for comparison
                    # Input might be "ETH/USD" or "ETHUSD", Alpaca returns "ETHUSD"
                    normalized_input = symbol.replace('/', '').replace('_', '').upper()
                    
                    position = None
                    for p in positions:
                        pos_symbol = p.get('symbol', '')
                        # Normalize position symbol (remove slashes/underscores, uppercase)
                        normalized_pos = pos_symbol.replace('/', '').replace('_', '').upper()
                        
                        # Direct match after normalization
                        if normalized_pos == normalized_input:
                            position = p
                            break
                        
                        # Try format conversion: if input is ETH/USD and pos is ETHUSD, they match
                        if '/' in symbol and len(normalized_pos) >= 6:
                            # Try to reconstruct ETH/USD from ETHUSD
                            for base_len in [3, 4]:
                                if len(normalized_pos) > base_len:
                                    base = normalized_pos[:base_len]
                                    quote = normalized_pos[base_len:]
                                    if quote in ['USD', 'EUR', 'GBP', 'JPY']:
                                        reconstructed = f"{base}/{quote}"
                                        if reconstructed.upper() == symbol.upper():
                                            position = p
                                            break
                            if position:
                                break
                        
                        # Also try the reverse: if pos is ETH/USD and input is ETHUSD
                        if '/' in pos_symbol and len(normalized_input) >= 6:
                            for base_len in [3, 4]:
                                if len(normalized_input) > base_len:
                                    base = normalized_input[:base_len]
                                    quote = normalized_input[base_len:]
                                    if quote in ['USD', 'EUR', 'GBP', 'JPY']:
                                        reconstructed = f"{base}/{quote}"
                                        if reconstructed.upper() == pos_symbol.upper():
                                            position = p
                                            break
                            if position:
                                break
                    
                    if not position:
                        # Last resort: try exact match (case-insensitive) and also try without slash
                        symbol_variants = [
                            symbol,
                            symbol.replace('/', ''),
                            symbol.replace('/', '_'),
                            normalized_input
                        ]
                        for variant in symbol_variants:
                            for p in positions:
                                pos_symbol = p.get('symbol', '')
                                if (pos_symbol.upper() == variant.upper() or 
                                    pos_symbol.replace('/', '').replace('_', '').upper() == variant.replace('/', '').replace('_', '').upper()):
                                    position = p
                                    print(f"  Found position using variant matching: {variant} -> {pos_symbol}")
                                    break
                            if position:
                                break
                    
                    if not position:
                        print(f"Error closing position: No open position found for {symbol}")
                        # Build position list for display
                        position_list = []
                        for p in positions:
                            pos_symbol = p.get('symbol', '')
                            pos_side = p.get('side', '')
                            pos_qty = p.get('quantity', 0)
                            position_list.append(f"{pos_symbol} {pos_side} {pos_qty} units")
                        print(f"  Available positions: {position_list}")
                        print(f"  Looking for symbol: {symbol} (normalized: {normalized_input})")
                        symbol_variants = [symbol, symbol.replace('/', ''), symbol.replace('/', '_'), normalized_input]
                        print(f"  Tried variants: {symbol_variants}")
                        return False
                    
                    position_qty = position.get('quantity', 0)
                    position_side = position.get('side', 'buy')
                    
                    if position_qty <= 0:
                        print(f"Error closing position: Invalid quantity {position_qty} for {symbol}")
                        return False
                    
                    # Determine opposite side
                    if position_side.lower() == 'buy':
                        close_side = OrderSide.SELL
                    else:
                        close_side = OrderSide.BUY
                    
                    # Place market order to close position
                    print(f"Closing crypto position {symbol}: placing {close_side.value} order for {position_qty} units")
                    result = self.place_order(
                        symbol=symbol,
                        side=close_side,
                        quantity=position_qty,
                        order_type=OrderType.MARKET
                    )
                    
                    if result.get('order_id'):
                        print(f"Successfully placed close order for {symbol}: {result.get('order_id')}")
                        return True
                    else:
                        error = result.get('error', 'Unknown error')
                        print(f"Failed to place close order for {symbol}: {error}")
                        return False
                        
                except Exception as e:
                    print(f"Error closing crypto position {symbol}: {e}")
                    return False
            else:
                # For stocks, use DELETE endpoint
                alpaca_symbol = symbol
                url = f"{self.base_url}/v2/positions/{alpaca_symbol}"
            
                print(f"Closing stock position {symbol} via DELETE {url}")
                response = self.session.delete(url)
                response.raise_for_status()
                print(f"Successfully closed stock position {symbol}")
                return True
                
        except requests.exceptions.HTTPError as e:
            error_msg = str(e)
            # Try to extract more details from response
            try:
                if hasattr(e.response, 'text'):
                    error_detail = e.response.text
                    print(f"Error closing position on Alpaca: {error_msg}")
                    print(f"  Response: {error_detail[:200]}")
            except:
                pass
            return False
        except Exception as e:
            print(f"Error closing position on Alpaca: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_supported_assets(self) -> List[str]:
        """Get supported asset classes"""
        return ['stocks', 'crypto']
    
    def is_market_open(self, symbol: str) -> bool:
        """
        Check if market is open.

        Stocks (NYSE/NASDAQ):
        - Open: Monday-Friday, 9:30 AM - 4:00 PM EST/EDT
        - Closed: Weekends and market holidays
        - UTC: 13:30-20:00 UTC (EST) or 14:30-21:00 UTC (EDT)

        Crypto:
        - Open: 24/7

        Note: This uses approximate UTC conversion. For precise times,
        consider using pytz for timezone-aware datetime.
        """
        now = datetime.now()
        weekday = now.weekday()  # Monday=0, Sunday=6
        hour = now.hour
        minute = now.minute
        
        # Check if it's a crypto symbol (contains '/')
        if '/' in symbol:
            # Crypto markets are open 24/7
            return True

        # Stocks: Check if it's a weekend
        if weekday >= 5:  # Saturday (5) or Sunday (6)
            return False
        
        # Stocks: Check trading hours
        # NYSE/NASDAQ: 9:30 AM - 4:00 PM EST/EDT
        # EST = UTC-5, EDT = UTC-4
        # Approximate: 13:30-20:00 UTC (EST) or 14:30-21:00 UTC (EDT)
        # Using 13:30-20:00 UTC as conservative estimate (covers both)
        current_time_minutes = hour * 60 + minute
        market_open_minutes = 13 * 60 + 30  # 13:30 UTC
        market_close_minutes = 20 * 60  # 20:00 UTC

        if market_open_minutes <= current_time_minutes < market_close_minutes:
            return True

        return False
    
    def check_liquidity(self, symbol: str, order_size: float) -> Dict:
        """Check liquidity for Alpaca (stocks and crypto)"""
        try:
            is_crypto = '/' in symbol
            if is_crypto:
                base, quote = symbol.split('/')
                alpaca_symbol = f"{base}{quote}"
            else:
                alpaca_symbol = symbol
            
            # For crypto, check volume
            if is_crypto:
                # Major cryptos are liquid
                major_cryptos = ['BTCUSD', 'ETHUSD', 'SOLUSD']
                is_major = alpaca_symbol in major_cryptos
                
                if is_major:
                    estimated_slippage = 2.0  # 2 bps for major cryptos
                else:
                    estimated_slippage = 5.0  # 5 bps for minor cryptos
                
                return {
                    'sufficient': True,
                    'estimated_slippage_bps': estimated_slippage,
                    'order_book_depth': order_size * 50 if is_major else order_size * 10,
                    'reason': 'Crypto liquidity check'
                }
            else:
                # For stocks, estimate based on typical market depth
                # In production, would fetch order book data
                estimated_slippage = 3.0  # 3 bps for stocks
                
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

