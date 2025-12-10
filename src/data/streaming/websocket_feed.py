"""WebSocket-based real-time market data feed"""

import asyncio
import json
from typing import Dict, Optional, Callable, List
from datetime import datetime
import websocket
import threading
from queue import Queue

try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    print("Warning: websocket-client not available. Install with: pip install websocket-client")


class WebSocketFeed:
    """WebSocket feed for real-time market data"""
    
    def __init__(self, broker_name: str, config: Optional[Dict] = None):
        """
        Initialize WebSocket feed
        
        Args:
            broker_name: Name of broker ('oanda', 'alpaca')
            config: Configuration dictionary
        """
        self.broker_name = broker_name.lower()
        self.config = config or {}
        self.ws = None
        self.connected = False
        self.data_queue = Queue()
        self.callbacks: List[Callable] = []
        self.running = False
        self.thread = None
    
    def connect(self, symbols: List[str], on_message: Optional[Callable] = None):
        """
        Connect to WebSocket feed
        
        Args:
            symbols: List of symbols to subscribe to
            on_message: Callback function for received messages
        """
        if not WEBSOCKET_AVAILABLE:
            print("Warning: websocket-client not available. WebSocket feed disabled.")
            return False
        
        if self.broker_name == 'oanda':
            return self._connect_oanda(symbols, on_message)
        elif self.broker_name == 'alpaca':
            return self._connect_alpaca(symbols, on_message)
        else:
            print(f"WebSocket not supported for broker: {self.broker_name}")
            return False
    
    def _connect_oanda(self, symbols: List[str], on_message: Optional[Callable]):
        """Connect to OANDA streaming API"""
        try:
            import os
            api_key = os.getenv('OANDA_API_KEY')
            account_id = os.getenv('OANDA_ACCOUNT_ID', '').strip()
            if not account_id or account_id.startswith('#'):
                account_id = None
            
            environment = os.getenv('OANDA_ENVIRONMENT', 'practice')
            base_url = 'https://api-fxpractice.oanda.com' if environment == 'practice' else 'https://api-fxtrade.oanda.com'
            
            # OANDA streaming URL
            stream_url = f"{base_url.replace('https://', 'wss://')}/v3/accounts/{account_id or 'default'}/pricing/stream"
            
            def on_open(ws):
                self.connected = True
                print(f"OANDA WebSocket connected")
                # Subscribe to instruments
                for symbol in symbols:
                    oanda_symbol = symbol.replace('/', '_')
                    # OANDA auto-subscribes when you connect with account
            
            def on_message_handler(ws, message):
                try:
                    data = json.loads(message)
                    if on_message:
                        on_message(data)
                    self.data_queue.put(data)
                except Exception as e:
                    print(f"Error processing WebSocket message: {e}")
            
            def on_error(ws, error):
                print(f"WebSocket error: {error}")
                self.connected = False
            
            def on_close(ws, close_status_code, close_msg):
                self.connected = False
                print("OANDA WebSocket closed")
            
            self.ws = websocket.WebSocketApp(
                stream_url,
                on_open=on_open,
                on_message=on_message_handler,
                on_error=on_error,
                on_close=on_close,
                header={'Authorization': f'Bearer {api_key}'}
            )
            
            # Run in separate thread
            self.running = True
            self.thread = threading.Thread(target=self.ws.run_forever, daemon=True)
            self.thread.start()
            
            return True
        except Exception as e:
            print(f"Error connecting to OANDA WebSocket: {e}")
            return False
    
    def _connect_alpaca(self, symbols: List[str], on_message: Optional[Callable]):
        """Connect to Alpaca WebSocket"""
        try:
            import os
            api_key = os.getenv('ALPACA_API_KEY')
            api_secret = os.getenv('ALPACA_API_SECRET')
            base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
            
            # Alpaca WebSocket URL
            ws_url = base_url.replace('https://', 'wss://').replace('/v2', '') + '/stream'
            
            def on_open(ws):
                self.connected = True
                print(f"Alpaca WebSocket connected")
                # Authenticate
                auth_msg = {
                    "action": "authenticate",
                    "data": {
                        "key_id": api_key,
                        "secret_key": api_secret
                    }
                }
                ws.send(json.dumps(auth_msg))
                
                # Subscribe to symbols
                subscribe_msg = {
                    "action": "subscribe",
                    "trades": symbols,
                    "quotes": symbols
                }
                ws.send(json.dumps(subscribe_msg))
            
            def on_message_handler(ws, message):
                try:
                    data = json.loads(message)
                    if on_message:
                        on_message(data)
                    self.data_queue.put(data)
                except Exception as e:
                    print(f"Error processing WebSocket message: {e}")
            
            def on_error(ws, error):
                print(f"WebSocket error: {error}")
                self.connected = False
            
            def on_close(ws, close_status_code, close_msg):
                self.connected = False
                print("Alpaca WebSocket closed")
            
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_open=on_open,
                on_message=on_message_handler,
                on_error=on_error,
                on_close=on_close
            )
            
            # Run in separate thread
            self.running = True
            self.thread = threading.Thread(target=self.ws.run_forever, daemon=True)
            self.thread.start()
            
            return True
        except Exception as e:
            print(f"Error connecting to Alpaca WebSocket: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from WebSocket"""
        self.running = False
        if self.ws:
            self.ws.close()
        self.connected = False
    
    def get_latest_data(self, timeout: float = 1.0) -> Optional[Dict]:
        """
        Get latest data from queue (non-blocking)
        
        Args:
            timeout: Timeout in seconds
        
        Returns:
            Latest data dictionary or None
        """
        try:
            return self.data_queue.get(timeout=timeout)
        except:
            return None
    
    def is_connected(self) -> bool:
        """Check if WebSocket is connected"""
        return self.connected

