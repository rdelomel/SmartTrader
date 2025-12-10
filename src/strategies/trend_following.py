"""Trend following strategies"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
try:
    from .base_strategy import BaseStrategy, Signal
    from ..indicators.technical import TechnicalIndicators
except ImportError:
    from base_strategy import BaseStrategy, Signal
    from indicators.technical import TechnicalIndicators


class TrendFollowingStrategy(BaseStrategy):
    """Moving average crossover trend following strategy"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize trend following strategy
        
        Config parameters:
            fast_ma_period: Fast moving average period (default: 20)
            slow_ma_period: Slow moving average period (default: 50)
            use_ema: Whether to use EMA instead of SMA (default: False)
        """
        super().__init__("TrendFollowing", config)
        self.fast_period = self.config.get('fast_ma_period', 20)
        self.slow_period = self.config.get('slow_ma_period', 50)
        self.use_ema = self.config.get('use_ema', False)
        self.indicators = TechnicalIndicators()
    
    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """Generate signal based on MA crossover and trend strength"""
        if len(data) < self.slow_period:
            return {
                'signal': Signal.HOLD,
                'confidence': 0.0,
                'entry_price': data['close'].iloc[-1],
                'stop_loss': None,
                'take_profit': None,
                'reason': 'Insufficient data'
            }
        
        # Calculate moving averages
        if self.use_ema:
            fast_ma = self.indicators.ema(data, self.fast_period)
            slow_ma = self.indicators.ema(data, self.slow_period)
        else:
            fast_ma = self.indicators.sma(data, self.fast_period)
            slow_ma = self.indicators.sma(data, self.slow_period)
        
        current_price = data['close'].iloc[-1]
        fast_current = fast_ma.iloc[-1]
        slow_current = slow_ma.iloc[-1]
        fast_prev = fast_ma.iloc[-2] if len(fast_ma) > 1 else fast_current
        slow_prev = slow_ma.iloc[-2] if len(slow_ma) > 1 else slow_current
        
        # Calculate ATR for stop-loss and take-profit
        atr = self.indicators.atr(data)
        atr_value = atr.iloc[-1] if not atr.empty else current_price * 0.02
        
        # Calculate ADX for trend strength
        adx = self.indicators.adx(data)
        adx_value = adx.iloc[-1] if not adx.empty else 0
        
        # Calculate MACD for momentum confirmation
        macd = self.indicators.macd(data)
        macd_value = macd['macd'].iloc[-1] if not macd.empty else 0
        macd_signal = macd['signal'].iloc[-1] if not macd.empty else 0
        macd_hist = macd['histogram'].iloc[-1] if not macd.empty else 0
        
        # Check for crossover
        bullish_cross = (fast_prev <= slow_prev) and (fast_current > slow_current)
        bearish_cross = (fast_prev >= slow_prev) and (fast_current < slow_current)
        
        # Calculate trend strength (normalized percentage difference)
        trend_strength_raw = abs(fast_current - slow_current) / slow_current
        
        # Calculate MA slope (momentum) - look at last 5 periods
        if len(fast_ma) >= 5:
            fast_slope = (fast_ma.iloc[-1] - fast_ma.iloc[-5]) / fast_ma.iloc[-5]
            slow_slope = (slow_ma.iloc[-1] - slow_ma.iloc[-5]) / slow_ma.iloc[-5]
        else:
            fast_slope = 0
            slow_slope = 0
        
        # Price position relative to MAs
        price_above_fast = (current_price - fast_current) / fast_current
        price_above_slow = (current_price - slow_current) / slow_current
        
        # Determine signal and calculate multi-factor confidence
        signal = Signal.HOLD
        confidence = 0.0
        reason = "No clear trend"
        
        if bullish_cross:
            # Strong bullish signal on crossover
            signal = Signal.BUY
            
            # Multi-factor confidence calculation
            conf_factors = []
            conf_factors.append(min(trend_strength_raw * 20, 0.3))  # Trend strength
            conf_factors.append(min(adx_value / 50, 0.25))  # ADX strength
            conf_factors.append(0.2 if macd_hist > 0 else 0)  # MACD confirmation
            conf_factors.append(min(abs(fast_slope) * 20, 0.15))  # MA momentum
            conf_factors.append(0.1)  # Base confidence for crossover
            
            confidence = min(sum(conf_factors), 0.9)
            reason = f"Bullish MA crossover (Fast: {fast_current:.2f}, Slow: {slow_current:.2f}, ADX: {adx_value:.1f})"
            
        elif bearish_cross:
            # Strong bearish signal on crossover
            signal = Signal.SELL
            
            # Multi-factor confidence calculation
            conf_factors = []
            conf_factors.append(min(trend_strength_raw * 20, 0.3))
            conf_factors.append(min(adx_value / 50, 0.25))
            conf_factors.append(0.2 if macd_hist < 0 else 0)
            conf_factors.append(min(abs(fast_slope) * 20, 0.15))
            conf_factors.append(0.1)  # Base confidence for crossover
            
            confidence = min(sum(conf_factors), 0.9)
            reason = f"Bearish MA crossover (Fast: {fast_current:.2f}, Slow: {slow_current:.2f}, ADX: {adx_value:.1f})"
            
        elif fast_current > slow_current:
            # Uptrend - generate signal with graduated confidence
            signal = Signal.BUY
            
            # Multi-factor confidence for established uptrend
            conf_factors = []
            
            # 1. Trend strength (up to 0.25)
            conf_factors.append(min(trend_strength_raw * 15, 0.25))
            
            # 2. ADX strength (up to 0.2)
            if adx_value > 25:  # Strong trend
                conf_factors.append(min(adx_value / 60, 0.2))
            elif adx_value > 20:  # Moderate trend
                conf_factors.append(0.1)
            else:  # Weak trend
                conf_factors.append(0.05)
            
            # 3. MACD confirmation (up to 0.15)
            if macd_value > macd_signal and macd_hist > 0:
                conf_factors.append(0.15)
            elif macd_value > macd_signal:
                conf_factors.append(0.08)
            
            # 4. MA slope/momentum (up to 0.15)
            if fast_slope > 0 and slow_slope > 0:
                conf_factors.append(min(abs(fast_slope) * 10, 0.15))
            elif fast_slope > 0:
                conf_factors.append(0.08)
            
            # 5. Price position (up to 0.1)
            if price_above_fast > 0:
                conf_factors.append(min(price_above_fast * 5, 0.1))
            
            # 6. Base confidence for being in uptrend
            conf_factors.append(0.15)
            
            confidence = min(sum(conf_factors), 0.75)
            reason = f"Uptrend (Fast: {fast_current:.2f} > Slow: {slow_current:.2f}, ADX: {adx_value:.1f}, Slope: {fast_slope:.4f})"
            
        elif fast_current < slow_current:
            # Downtrend - generate signal with graduated confidence
            signal = Signal.SELL
            
            # Multi-factor confidence for established downtrend
            conf_factors = []
            
            # 1. Trend strength
            conf_factors.append(min(trend_strength_raw * 15, 0.25))
            
            # 2. ADX strength
            if adx_value > 25:
                conf_factors.append(min(adx_value / 60, 0.2))
            elif adx_value > 20:
                conf_factors.append(0.1)
            else:
                conf_factors.append(0.05)
            
            # 3. MACD confirmation
            if macd_value < macd_signal and macd_hist < 0:
                conf_factors.append(0.15)
            elif macd_value < macd_signal:
                conf_factors.append(0.08)
            
            # 4. MA slope/momentum
            if fast_slope < 0 and slow_slope < 0:
                conf_factors.append(min(abs(fast_slope) * 10, 0.15))
            elif fast_slope < 0:
                conf_factors.append(0.08)
            
            # 5. Price position
            if price_above_fast < 0:
                conf_factors.append(min(abs(price_above_fast) * 5, 0.1))
            
            # 6. Base confidence for being in downtrend
            conf_factors.append(0.15)
            
            confidence = min(sum(conf_factors), 0.75)
            reason = f"Downtrend (Fast: {fast_current:.2f} < Slow: {slow_current:.2f}, ADX: {adx_value:.1f}, Slope: {fast_slope:.4f})"
        
        # Calculate stop-loss and take-profit
        stop_loss = None
        take_profit = None
        
        if signal != Signal.HOLD:
            if signal == Signal.BUY:
                stop_loss = current_price - (atr_value * 2)
                take_profit = current_price + (atr_value * 3)
            else:  # SELL
                stop_loss = current_price + (atr_value * 2)
                take_profit = current_price - (atr_value * 3)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'reason': reason
        }
    
    def should_exit(self, data: pd.DataFrame, position: Dict) -> bool:
        """Check if position should be exited based on trend reversal"""
        if len(data) < self.slow_period:
            return False
        
        # Calculate moving averages
        if self.use_ema:
            fast_ma = self.indicators.ema(data, self.fast_period)
            slow_ma = self.indicators.ema(data, self.slow_period)
        else:
            fast_ma = self.indicators.sma(data, self.fast_period)
            slow_ma = self.indicators.sma(data, self.slow_period)
        
        fast_current = fast_ma.iloc[-1]
        slow_current = slow_ma.iloc[-1]
        
        # Exit long position if trend reverses
        if position['side'] == 'buy' and fast_current < slow_current:
            return True
        
        # Exit short position if trend reverses
        if position['side'] == 'sell' and fast_current > slow_current:
            return True
        
        return False

