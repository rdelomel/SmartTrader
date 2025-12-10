"""Paper trading simulator"""

from typing import Dict, List, Optional
from datetime import datetime
import random
from ..data.brokers.base_broker import BaseBroker, OrderSide, OrderType, OrderStatus
from ..data.data_storage import DataStorage


class PaperTrader:
    """Paper trading simulator with realistic slippage and fees"""
    
    def __init__(self, broker: BaseBroker, storage: DataStorage):
        """
        Initialize paper trader
        
        Args:
            broker: Broker instance (for getting current prices)
            storage: Data storage instance
        """
        self.broker = broker
        self.storage = storage
        self.positions: Dict[str, Dict] = {}
        self.balance = 10000.0  # Starting balance
        self.equity_history = []
        self.commission_rate = 0.001  # 0.1% commission
        self.slippage_bps = 5  # 5 basis points slippage
    
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
        Place a paper trade order
        
        Args:
            symbol: Trading symbol
            side: BUY or SELL
            quantity: Order quantity
            order_type: Order type
            price: Limit price
            stop_loss: Stop loss price
            take_profit: Take profit price
        
        Returns:
            Order result dictionary
        """
        try:
            # Get current market price
            current_price = self.broker.get_current_price(symbol)
            
            if current_price == 0:
                return {
                    'order_id': None,
                    'status': OrderStatus.REJECTED,
                    'error': 'Unable to get current price'
                }
            
            # Calculate execution price with slippage
            if order_type == OrderType.MARKET:
                if side == OrderSide.BUY:
                    execution_price = current_price * (1 + self.slippage_bps / 10000)
                else:
                    execution_price = current_price * (1 - self.slippage_bps / 10000)
            else:  # LIMIT order
                execution_price = price if price else current_price
            
            # Calculate order value and commission
            order_value = quantity * execution_price
            commission = order_value * self.commission_rate
            
            # Check if we have enough balance
            if side == OrderSide.BUY:
                required = order_value + commission
                if required > self.balance:
                    return {
                        'order_id': None,
                        'status': OrderStatus.REJECTED,
                        'error': 'Insufficient balance'
                    }
                self.balance -= required
            else:  # SELL
                # Check if we have the position
                position_key = f"{symbol}_{OrderSide.BUY.value}"
                if position_key not in self.positions:
                    return {
                        'order_id': None,
                        'status': OrderStatus.REJECTED,
                        'error': 'No position to sell'
                    }
                
                # Close position
                position = self.positions[position_key]
                pnl = (execution_price - position['entry_price']) * quantity
                self.balance += order_value - commission + pnl
                del self.positions[position_key]
            
            # Generate order ID
            order_id = f"paper_{datetime.now().timestamp()}_{random.randint(1000, 9999)}"
            
            # Store position if buying
            if side == OrderSide.BUY:
                position_key = f"{symbol}_{side.value}"
                self.positions[position_key] = {
                    'symbol': symbol,
                    'side': side.value,
                    'quantity': quantity,
                    'entry_price': execution_price,
                    'entry_time': datetime.now(),
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'order_id': order_id
                }
            
            # Update equity history
            self._update_equity()
            
            # Store trade
            trade_data = {
                'trade_id': order_id,
                'symbol': symbol,
                'side': side.value,
                'quantity': quantity,
                'entry_price': execution_price,
                'entry_time': datetime.now(),
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'status': 'open' if side == OrderSide.BUY else 'closed'
            }
            self.storage.store_trade(trade_data)
            
            return {
                'order_id': order_id,
                'status': OrderStatus.FILLED,
                'filled_quantity': quantity,
                'symbol': symbol,
                'side': side.value,
                'price': execution_price,
                'commission': commission,
                'timestamp': datetime.now()
            }
        except Exception as e:
            print(f"Error in paper trading: {e}")
            return {
                'order_id': None,
                'status': OrderStatus.REJECTED,
                'error': str(e)
            }
    
    def get_balance(self) -> float:
        """Get current paper trading balance"""
        return self.balance
    
    def get_positions(self) -> List[Dict]:
        """Get all open positions"""
        return list(self.positions.values())
    
    def get_equity(self) -> float:
        """Get current equity (balance + unrealized P&L)"""
        equity = self.balance
        
        for position in self.positions.values():
            try:
                current_price = self.broker.get_current_price(position['symbol'])
                if position['side'] == 'buy':
                    unrealized_pnl = (current_price - position['entry_price']) * position['quantity']
                else:
                    unrealized_pnl = (position['entry_price'] - current_price) * position['quantity']
                
                equity += unrealized_pnl
            except:
                pass
        
        return equity
    
    def _update_equity(self):
        """Update equity history"""
        equity = self.get_equity()
        self.equity_history.append({
            'timestamp': datetime.now(),
            'equity': equity,
            'balance': self.balance
        })
    
    def get_equity_history(self) -> list:
        """Get equity history"""
        return self.equity_history
    
    def check_stops(self) -> List[Dict]:
        """
        Check stop-loss and take-profit levels
        
        Returns:
            List of closed positions
        """
        closed_positions = []
        
        for position_key, position in list(self.positions.items()):
            try:
                current_price = self.broker.get_current_price(position['symbol'])
                
                # Check stop loss
                if position.get('stop_loss'):
                    if position['side'] == 'buy' and current_price <= position['stop_loss']:
                        closed_positions.append(self._close_position(position, current_price, 'stop_loss'))
                        del self.positions[position_key]
                        continue
                    elif position['side'] == 'sell' and current_price >= position['stop_loss']:
                        closed_positions.append(self._close_position(position, current_price, 'stop_loss'))
                        del self.positions[position_key]
                        continue
                
                # Check take profit
                if position.get('take_profit'):
                    if position['side'] == 'buy' and current_price >= position['take_profit']:
                        closed_positions.append(self._close_position(position, current_price, 'take_profit'))
                        del self.positions[position_key]
                        continue
                    elif position['side'] == 'sell' and current_price <= position['take_profit']:
                        closed_positions.append(self._close_position(position, current_price, 'take_profit'))
                        del self.positions[position_key]
                        continue
            except Exception as e:
                print(f"Error checking stops for {position['symbol']}: {e}")
        
        return closed_positions
    
    def _close_position(self, position: Dict, exit_price: float, reason: str) -> Dict:
        """Close a position"""
        quantity = position['quantity']
        entry_price = position['entry_price']
        
        # Calculate P&L
        if position['side'] == 'buy':
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity
        
        # Calculate commission
        exit_value = quantity * exit_price
        commission = exit_value * self.commission_rate
        
        # Update balance
        self.balance += exit_value - commission + pnl
        
        # Update trade in database
        trade_data = {
            'trade_id': position['order_id'],
            'exit_price': exit_price,
            'exit_time': datetime.now(),
            'pnl': pnl,
            'pnl_percent': (pnl / (entry_price * quantity)) * 100,
            'status': 'closed'
        }
        self.storage.update_trade(position['order_id'], trade_data)
        
        return {
            'symbol': position['symbol'],
            'exit_price': exit_price,
            'pnl': pnl,
            'reason': reason
        }

