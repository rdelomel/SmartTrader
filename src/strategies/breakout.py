"""Breakout trading strategy"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
try:
    from .base_strategy import BaseStrategy, Signal
    from ..indicators.technical import TechnicalIndicators
except ImportError:
    from base_strategy import BaseStrategy, Signal
    from indicators.technical import TechnicalIndicators


class BreakoutStrategy(BaseStrategy):
    """Breakout strategy based on support/resistance levels"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize breakout strategy
        
        Config parameters:
            lookback_period: Period for calculating support/resistance (default: 20)
            breakout_threshold: Percentage above/below level to confirm breakout (default: 0.5%)
            volume_threshold: Minimum volume increase for breakout (default: 1.5x)
            use_atr: Use ATR for dynamic breakout levels (default: True)
        """
        super().__init__("Breakout", config)
        self.lookback_period = self.config.get('lookback_period', 20)
        self.breakout_threshold = self.config.get('breakout_threshold', 0.005)  # 0.5%
        self.volume_threshold = self.config.get('volume_threshold', 1.5)
        self.use_atr = self.config.get('use_atr', True)
        self.indicators = TechnicalIndicators()
    
    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """Generate signal based on breakout patterns"""
        if len(data) < self.lookback_period + 5:
            return {
                'signal': Signal.HOLD,
                'confidence': 0.0,
                'entry_price': data['close'].iloc[-1],
                'stop_loss': None,
                'take_profit': None,
                'reason': 'Insufficient data'
            }
        
        current_price = data['close'].iloc[-1]
        current_volume = data['volume'].iloc[-1]
        avg_volume = data['volume'].rolling(window=self.lookback_period).mean().iloc[-1]
        
        # Calculate support and resistance levels
        resistance = data['high'].rolling(window=self.lookback_period).max().iloc[-1]
        support = data['low'].rolling(window=self.lookback_period).min().iloc[-1]
        
        # Use ATR for dynamic levels if enabled
        if self.use_atr:
            atr = self.indicators.atr(data)
            atr_value = atr.iloc[-1] if not atr.empty else current_price * 0.02
            resistance = resistance + atr_value * 0.5
            support = support - atr_value * 0.5
        
        # Calculate breakout levels
        resistance_breakout = resistance * (1 + self.breakout_threshold)
        support_breakout = support * (1 - self.breakout_threshold)
        
        # Check volume confirmation
        volume_confirmed = current_volume >= (avg_volume * self.volume_threshold)
        
        signal = Signal.HOLD
        confidence = 0.0
        reason = "No breakout detected"
        
        # Bullish breakout: price breaks above resistance with volume
        if current_price > resistance_breakout and volume_confirmed:
            signal = Signal.BUY
            # Confidence based on how far above resistance and volume strength
            breakout_strength = (current_price - resistance) / resistance
            volume_strength = min(current_volume / avg_volume / self.volume_threshold, 2.0)
            confidence = min(breakout_strength * 20 + volume_strength * 0.3, 0.85)
            reason = f"Bullish breakout above resistance {resistance:.2f} with volume confirmation"
        
        # Bearish breakout: price breaks below support with volume
        elif current_price < support_breakout and volume_confirmed:
            signal = Signal.SELL
            breakout_strength = (support - current_price) / support
            volume_strength = min(current_volume / avg_volume / self.volume_threshold, 2.0)
            confidence = min(breakout_strength * 20 + volume_strength * 0.3, 0.85)
            reason = f"Bearish breakout below support {support:.2f} with volume confirmation"
        
        # Calculate stop-loss and take-profit
        stop_loss = None
        take_profit = None
        
        if signal != Signal.HOLD:
            atr = self.indicators.atr(data)
            atr_value = atr.iloc[-1] if not atr.empty else current_price * 0.02
            
            if signal == Signal.BUY:
                # Stop loss below support, take profit at 2x ATR
                stop_loss = support - (atr_value * 0.5)
                take_profit = current_price + (atr_value * 2.5)
            else:  # SELL
                # Stop loss above resistance, take profit at 2x ATR
                stop_loss = resistance + (atr_value * 0.5)
                take_profit = current_price - (atr_value * 2.5)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'reason': reason,
            'resistance': resistance,
            'support': support
        }
    
    def should_exit(self, data: pd.DataFrame, position: Dict) -> bool:
        """Check if position should be exited (breakout failed)"""
        if len(data) < self.lookback_period:
            return False
        
        current_price = data['close'].iloc[-1]
        entry_price = position.get('entry_price', current_price)
        
        # Exit long if price falls back below original resistance
        if position['side'] == 'buy':
            resistance = data['high'].rolling(window=self.lookback_period).max().iloc[-1]
            if current_price < resistance * 0.98:  # 2% below resistance
                return True
        
        # Exit short if price rises back above original support
        if position['side'] == 'sell':
            support = data['low'].rolling(window=self.lookback_period).min().iloc[-1]
            if current_price > support * 1.02:  # 2% above support
                return True
        
        return False

