"""Volatility-based trading strategy"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
try:
    from .base_strategy import BaseStrategy, Signal
    from ..indicators.technical import TechnicalIndicators
except ImportError:
    from base_strategy import BaseStrategy, Signal
    from indicators.technical import TechnicalIndicators


class VolatilityStrategy(BaseStrategy):
    """Volatility strategy using Bollinger Bands and ATR"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize volatility strategy
        
        Config parameters:
            bb_period: Bollinger Bands period (default: 20)
            bb_std: Bollinger Bands standard deviation (default: 2)
            atr_period: ATR period (default: 14)
            low_volatility_threshold: Threshold for low volatility (default: 0.5)
            high_volatility_threshold: Threshold for high volatility (default: 2.0)
            volatility_expansion_factor: Factor for volatility expansion detection (default: 1.5)
        """
        super().__init__("Volatility", config)
        self.bb_period = self.config.get('bb_period', 20)
        self.bb_std = self.config.get('bb_std', 2)
        self.atr_period = self.config.get('atr_period', 14)
        self.low_vol_threshold = self.config.get('low_volatility_threshold', 0.5)
        self.high_vol_threshold = self.config.get('high_volatility_threshold', 2.0)
        self.vol_expansion_factor = self.config.get('volatility_expansion_factor', 1.5)
        self.indicators = TechnicalIndicators()
    
    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """Generate signal based on volatility conditions"""
        if len(data) < max(self.bb_period, self.atr_period) + 10:
            return {
                'signal': Signal.HOLD,
                'confidence': 0.0,
                'entry_price': data['close'].iloc[-1],
                'stop_loss': None,
                'take_profit': None,
                'reason': 'Insufficient data'
            }
        
        current_price = data['close'].iloc[-1]
        
        # Calculate Bollinger Bands
        bb = self.indicators.bollinger_bands(data, period=self.bb_period, std_dev=self.bb_std)
        bb_upper = bb['upper'].iloc[-1] if not bb.empty else current_price * 1.02
        bb_lower = bb['lower'].iloc[-1] if not bb.empty else current_price * 0.98
        bb_middle = bb['middle'].iloc[-1] if not bb.empty else current_price
        bb_width = (bb_upper - bb_lower) / bb_middle  # Normalized bandwidth
        
        # Calculate ATR
        atr = self.indicators.atr(data)
        atr_current = atr.iloc[-1] if not atr.empty else current_price * 0.02
        atr_avg = atr.rolling(window=self.atr_period).mean().iloc[-1] if len(atr) >= self.atr_period else atr_current
        
        # Calculate historical volatility
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=self.atr_period).std().iloc[-1]
        avg_volatility = returns.rolling(window=self.atr_period * 2).std().iloc[-1] if len(returns) >= self.atr_period * 2 else volatility
        
        # Volatility ratios
        atr_ratio = atr_current / atr_avg if atr_avg > 0 else 1.0
        vol_ratio = volatility / avg_volatility if avg_volatility > 0 else 1.0
        
        # Detect volatility regime
        is_low_vol = (atr_ratio < self.low_vol_threshold) and (vol_ratio < self.low_vol_threshold)
        is_high_vol = (atr_ratio > self.high_vol_threshold) or (vol_ratio > self.high_vol_threshold)
        is_expanding = (atr_ratio > self.vol_expansion_factor) or (vol_ratio > self.vol_expansion_factor)
        
        signal = Signal.HOLD
        confidence = 0.0
        reason = "No volatility signal"
        
        # Strategy 1: Low volatility squeeze - expect breakout
        if is_low_vol and bb_width < 0.02:  # Tight Bollinger Bands
            # Wait for price to break out of the squeeze
            if current_price > bb_upper:
                signal = Signal.BUY
                confidence = min((current_price - bb_upper) / bb_middle * 50, 0.75)
                reason = f"Bullish breakout from low volatility squeeze (BB width: {bb_width*100:.2f}%)"
            elif current_price < bb_lower:
                signal = Signal.SELL
                confidence = min((bb_lower - current_price) / bb_middle * 50, 0.75)
                reason = f"Bearish breakout from low volatility squeeze (BB width: {bb_width*100:.2f}%)"
        
        # Strategy 2: High volatility - mean reversion
        elif is_high_vol:
            if current_price > bb_upper:
                signal = Signal.SELL  # Overbought in high volatility
                confidence = min((current_price - bb_upper) / bb_middle * 30, 0.7)
                reason = f"Mean reversion: Overbought in high volatility (ATR ratio: {atr_ratio:.2f})"
            elif current_price < bb_lower:
                signal = Signal.BUY  # Oversold in high volatility
                confidence = min((bb_lower - current_price) / bb_middle * 30, 0.7)
                reason = f"Mean reversion: Oversold in high volatility (ATR ratio: {atr_ratio:.2f})"
        
        # Strategy 3: Volatility expansion - trend following
        elif is_expanding:
            # Follow the direction of volatility expansion
            price_change = (current_price - data['close'].iloc[-5]) / data['close'].iloc[-5]
            if price_change > 0.01:  # 1% move up
                signal = Signal.BUY
                confidence = min(abs(price_change) * 20 + atr_ratio * 0.2, 0.8)
                reason = f"Volatility expansion with upward momentum (ATR ratio: {atr_ratio:.2f})"
            elif price_change < -0.01:  # 1% move down
                signal = Signal.SELL
                confidence = min(abs(price_change) * 20 + atr_ratio * 0.2, 0.8)
                reason = f"Volatility expansion with downward momentum (ATR ratio: {atr_ratio:.2f})"
        
        # Calculate stop-loss and take-profit based on volatility
        stop_loss = None
        take_profit = None
        
        if signal != Signal.HOLD:
            # Use ATR-based stops
            if signal == Signal.BUY:
                stop_loss = current_price - (atr_current * 2)
                take_profit = current_price + (atr_current * 2.5)
            else:  # SELL
                stop_loss = current_price + (atr_current * 2)
                take_profit = current_price - (atr_current * 2.5)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'reason': reason,
            'atr_ratio': atr_ratio,
            'volatility_ratio': vol_ratio,
            'bb_width': bb_width
        }
    
    def should_exit(self, data: pd.DataFrame, position: Dict) -> bool:
        """Check if position should be exited (volatility regime change)"""
        if len(data) < max(self.bb_period, self.atr_period):
            return False
        
        current_price = data['close'].iloc[-1]
        
        # Calculate Bollinger Bands
        bb = self.indicators.bollinger_bands(data, period=self.bb_period, std_dev=self.bb_std)
        bb_upper = bb['upper'].iloc[-1] if not bb.empty else current_price * 1.02
        bb_lower = bb['lower'].iloc[-1] if not bb.empty else current_price * 0.98
        
        # Exit long if price returns to middle band (volatility strategy target)
        if position['side'] == 'buy':
            if current_price <= bb['middle'].iloc[-1] if not bb.empty else current_price:
                return True
        
        # Exit short if price returns to middle band
        if position['side'] == 'sell':
            if current_price >= bb['middle'].iloc[-1] if not bb.empty else current_price:
                return True
        
        return False

