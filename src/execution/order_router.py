"""Smart Order Routing for optimal execution"""

from typing import Dict, Optional, List
from ..data.brokers.base_broker import BaseBroker, OrderSide, OrderType


class OrderRouter:
    """Smart order router for optimal execution"""
    
    def __init__(self, brokers: Dict[str, BaseBroker], config: Optional[Dict] = None):
        """
        Initialize order router
        
        Args:
            brokers: Dictionary of broker instances
            config: Router configuration
        """
        self.brokers = brokers
        self.config = config or {}
        self.min_price_improvement_bps = self.config.get('min_price_improvement_bps', 1.0)  # 1 basis point
    
    def route_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET
    ) -> Dict:
        """
        Route order to best available broker
        
        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Order quantity
            order_type: Order type
        
        Returns:
            Dictionary with routing decision
        """
        if len(self.brokers) == 0:
            return {
                'broker': None,
                'reason': 'No brokers available'
            }
        
        if len(self.brokers) == 1:
            # Only one broker, use it
            broker_name = list(self.brokers.keys())[0]
            return {
                'broker': self.brokers[broker_name],
                'broker_name': broker_name,
                'reason': 'Single broker available'
            }
        
        # Compare prices across brokers
        best_broker = None
        best_price = None
        best_slippage = float('inf')
        
        for broker_name, broker in self.brokers.items():
            try:
                current_price = broker.get_current_price(symbol)
                
                # Check liquidity
                liquidity = broker.check_liquidity(symbol, quantity * current_price)
                
                if not liquidity.get('sufficient', True):
                    continue
                
                estimated_slippage = liquidity.get('estimated_slippage_bps', 5.0)
                
                # For buy orders, prefer lower price; for sell, prefer higher
                if side == OrderSide.BUY:
                    effective_price = current_price * (1 + estimated_slippage / 10000)
                else:
                    effective_price = current_price * (1 - estimated_slippage / 10000)
                
                if best_broker is None or (side == OrderSide.BUY and effective_price < best_price) or \
                   (side == OrderSide.SELL and effective_price > best_price):
                    best_broker = broker
                    best_price = effective_price
                    best_slippage = estimated_slippage
                    
            except Exception as e:
                print(f"Error checking broker {broker_name}: {e}")
                continue
        
        if best_broker is None:
            # Fallback to first available broker
            broker_name = list(self.brokers.keys())[0]
            return {
                'broker': self.brokers[broker_name],
                'broker_name': broker_name,
                'reason': 'Fallback: no brokers with sufficient liquidity'
            }
        
        return {
            'broker': best_broker,
            'broker_name': [name for name, b in self.brokers.items() if b == best_broker][0],
            'estimated_price': best_price,
            'estimated_slippage_bps': best_slippage,
            'reason': f'Best price with {best_slippage:.1f} bps slippage'
        }

