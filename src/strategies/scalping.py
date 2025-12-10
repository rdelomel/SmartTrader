"""Scalping trading strategy - Very short-term trades with small price movements"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
try:
    from .base_strategy import BaseStrategy, Signal
    from ..indicators.technical import TechnicalIndicators
except ImportError:
    from base_strategy import BaseStrategy, Signal
    from indicators.technical import TechnicalIndicators


class ScalpingStrategy(BaseStrategy):
    """
    Scalping trading strategy based on CMC Markets guide
    - Very short-term trades with small price movements
    - Aims to 'scalp' small profits that accumulate
    - Requires disciplined exit strategy
    - Risk/reward ratio around 1:1
    - Works in high volatility and high volume markets
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize scalping strategy
        
        Config parameters:
            min_volatility: Minimum volatility to scalp (default: 0.01 = 1%)
            min_volume_multiplier: Volume must be X times average (default: 2.0)
            price_tick_size: Minimum price movement to trade (default: 0.001 = 0.1%)
            use_order_flow: Use order flow indicators (default: True)
            max_hold_time_minutes: Maximum time to hold position (default: 15)
        """
        super().__init__("Scalping", config)
        self.min_volatility = self.config.get('min_volatility', 0.01)
        self.min_volume_multiplier = self.config.get('min_volume_multiplier', 2.0)
        self.price_tick_size = self.config.get('price_tick_size', 0.001)
        self.use_order_flow = self.config.get('use_order_flow', True)
        self.max_hold_time_minutes = self.config.get('max_hold_time_minutes', 15)
        self.indicators = TechnicalIndicators()
    
    def _calculate_micro_momentum(self, data: pd.DataFrame) -> Dict:
        """Calculate very short-term momentum for scalping"""
        if len(data) < 5:
            return {'momentum_1': 0, 'momentum_3': 0, 'momentum_5': 0}
        
        current_price = data['close'].iloc[-1]
        
        # 1-bar momentum
        momentum_1 = (current_price - data['close'].iloc[-2]) / data['close'].iloc[-2] if len(data) >= 2 else 0
        
        # 3-bar momentum
        momentum_3 = (current_price - data['close'].iloc[-4]) / data['close'].iloc[-4] if len(data) >= 4 else 0
        
        # 5-bar momentum
        momentum_5 = (current_price - data['close'].iloc[-6]) / data['close'].iloc[-6] if len(data) >= 6 else 0
        
        return {
            'momentum_1': momentum_1,
            'momentum_3': momentum_3,
            'momentum_5': momentum_5
        }
    
    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """
        Generate signal for scalping
        
        Args:
            data: Market data DataFrame
        
        Returns:
            Signal dictionary
        """
        if len(data) < 20:
            return {
                'signal': Signal.HOLD,
                'confidence': 0.0,
                'entry_price': data['close'].iloc[-1],
                'stop_loss': None,
                'take_profit': None,
                'reason': 'Insufficient data'
            }
        
        current_price = data['close'].iloc[-1]
        
        # Calculate volatility (scalping needs volatility)
        returns = data['close'].pct_change().dropna()
        recent_returns = returns.iloc[-10:] if len(returns) >= 10 else returns
        volatility = recent_returns.std()
        
        # Check if volatility is sufficient
        if volatility < self.min_volatility:
            return {
                'signal': Signal.HOLD,
                'confidence': 0.0,
                'entry_price': current_price,
                'stop_loss': None,
                'take_profit': None,
                'reason': f'Insufficient volatility ({volatility*100:.2f}% < {self.min_volatility*100:.2f}%)'
            }
        
        # Calculate volume (scalping needs high volume)
        if 'volume' in data.columns:
            current_volume = data['volume'].iloc[-1]
            avg_volume = data['volume'].rolling(20).mean().iloc[-1] if len(data) >= 20 else current_volume
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            if volume_ratio < self.min_volume_multiplier:
                return {
                    'signal': Signal.HOLD,
                    'confidence': 0.0,
                    'entry_price': current_price,
                    'stop_loss': None,
                    'take_profit': None,
                    'reason': f'Insufficient volume ({volume_ratio:.1f}x < {self.min_volume_multiplier}x)'
                }
        else:
            volume_ratio = 1
        
        # Calculate micro momentum
        momentum = self._calculate_micro_momentum(data)
        
        # Calculate very short-term RSI (faster period for scalping)
        rsi = self.indicators.rsi(data, period=7)  # Shorter period
        rsi_current = rsi.iloc[-1] if not rsi.empty else 50
        
        # Calculate very short-term moving averages
        ema_fast = self.indicators.ema(data, period=5)
        ema_slow = self.indicators.ema(data, period=10)
        
        ema_fast_current = ema_fast.iloc[-1] if not ema_fast.empty else current_price
        ema_slow_current = ema_slow.iloc[-1] if not ema_slow.empty else current_price
        
        # Calculate bid-ask spread proxy (using high-low range)
        recent_range = (data['high'].iloc[-1] - data['low'].iloc[-1]) / data['low'].iloc[-1]
        
        signal = Signal.HOLD
        confidence = 0.0
        reason = "No scalping signal"
        
        # Scalping Strategy: Trade very small price movements
        # Look for quick momentum shifts with high volume
        
        # Buy signal: Quick upward momentum with volume
        if (momentum['momentum_1'] > self.price_tick_size and 
            momentum['momentum_3'] > 0 and 
            ema_fast_current > ema_slow_current and
            rsi_current < 70 and
            volume_ratio >= self.min_volume_multiplier):
            
            signal = Signal.BUY
            confidence = min(0.4 + abs(momentum['momentum_1']) * 50 + (volume_ratio - 1) * 0.1, 0.65)
            reason = f"Scalp buy - micro momentum (+{momentum['momentum_1']*100:.3f}%, Vol: {volume_ratio:.1f}x)"
        
        # Sell signal: Quick downward momentum with volume
        elif (momentum['momentum_1'] < -self.price_tick_size and 
              momentum['momentum_3'] < 0 and 
              ema_fast_current < ema_slow_current and
              rsi_current > 30 and
              volume_ratio >= self.min_volume_multiplier):
            
            signal = Signal.SELL
            confidence = min(0.4 + abs(momentum['momentum_1']) * 50 + (volume_ratio - 1) * 0.1, 0.65)
            reason = f"Scalp sell - micro momentum ({momentum['momentum_1']*100:.3f}%, Vol: {volume_ratio:.1f}x)"
        
        # Alternative: Trade bounces off very short-term support/resistance
        if signal == Signal.HOLD:
            # Quick bounce from recent low
            recent_low = data['low'].rolling(5).min().iloc[-1]
            if current_price <= recent_low * 1.002 and momentum['momentum_1'] > 0:
                signal = Signal.BUY
                confidence = 0.35
                reason = f"Scalp buy - bounce from micro low"
            
            # Quick rejection from recent high
            recent_high = data['high'].rolling(5).max().iloc[-1]
            if current_price >= recent_high * 0.998 and momentum['momentum_1'] < 0:
                signal = Signal.SELL
                confidence = 0.35
                reason = f"Scalp sell - rejection from micro high"
        
        # Calculate stop-loss and take-profit (tight for scalping, 1:1 risk/reward)
        stop_loss = None
        take_profit = None
        
        if signal != Signal.HOLD:
            # Very tight stops for scalping
            atr = self.indicators.atr(data, period=14)
            atr_value = atr.iloc[-1] if not atr.empty else current_price * 0.005  # 0.5% default
            
            if signal == Signal.BUY:
                # 1:1 risk/reward for scalping
                stop_loss = current_price - (atr_value * 0.8)  # Very tight
                take_profit = current_price + (atr_value * 0.8)  # Equal to stop
            else:  # SELL
                stop_loss = current_price + (atr_value * 0.8)
                take_profit = current_price - (atr_value * 0.8)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'reason': reason,
            'volatility': volatility,
            'volume_ratio': volume_ratio,
            'micro_momentum': momentum['momentum_1']
        }
    
    def should_exit(self, data: pd.DataFrame, position: Dict) -> bool:
        """Exit quickly - scalping doesn't hold positions long"""
        if len(data) < 2:
            return False
        
        current_price = data['close'].iloc[-1]
        entry_price = position.get('entry_price', current_price)
        
        # Exit if moved even slightly (scalping takes small profits)
        price_change = (current_price - entry_price) / entry_price
        
        # Exit if moved 0.3% or more (scalping targets are small)
        if abs(price_change) > 0.003:
            return True
        
        # Exit if momentum reverses
        momentum = self._calculate_micro_momentum(data)
        if position['side'] == 'buy' and momentum['momentum_1'] < -0.0005:
            return True
        if position['side'] == 'sell' and momentum['momentum_1'] > 0.0005:
            return True
        
        return False

