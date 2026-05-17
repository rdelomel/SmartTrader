"""Momentum trading strategy"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
try:
    from .base_strategy import BaseStrategy, Signal
    from ..indicators.technical import TechnicalIndicators
except ImportError:
    from base_strategy import BaseStrategy, Signal
    from indicators.technical import TechnicalIndicators


class MomentumStrategy(BaseStrategy):
    """Momentum strategy using RSI, MACD, and price momentum"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize momentum strategy
        
        Config parameters:
            rsi_period: RSI period (default: 14)
            rsi_oversold: RSI oversold level (default: 30)
            rsi_overbought: RSI overbought level (default: 70)
            macd_fast: MACD fast period (default: 12)
            macd_slow: MACD slow period (default: 26)
            macd_signal: MACD signal period (default: 9)
            momentum_period: Period for momentum calculation (default: 10)
            min_momentum: Minimum momentum threshold (default: 0.02)
        """
        super().__init__("Momentum", config)
        self.rsi_period = self.config.get('rsi_period', 14)
        self.rsi_oversold = self.config.get('rsi_oversold', 30)
        self.rsi_overbought = self.config.get('rsi_overbought', 70)
        self.macd_fast = self.config.get('macd_fast', 12)
        self.macd_slow = self.config.get('macd_slow', 26)
        self.macd_signal = self.config.get('macd_signal', 9)
        self.momentum_period = self.config.get('momentum_period', 10)
        self.min_momentum = self.config.get('min_momentum', 0.02)  # 2%
        self.indicators = TechnicalIndicators()
    
    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """Generate signal based on momentum indicators"""
        if len(data) < max(self.macd_slow, self.rsi_period, self.momentum_period) + 5:
            return {
                'signal': Signal.HOLD,
                'confidence': 0.0,
                'entry_price': data['close'].iloc[-1],
                'stop_loss': None,
                'take_profit': None,
                'reason': 'Insufficient data'
            }
        
        current_price = data['close'].iloc[-1]
        
        # Calculate RSI
        rsi = self.indicators.rsi(data, period=self.rsi_period)
        rsi_current = rsi.iloc[-1] if not rsi.empty else 50
        rsi_prev = rsi.iloc[-2] if len(rsi) > 1 else rsi_current
        
        # Calculate MACD
        macd_data = self.indicators.macd(data, fast_period=self.macd_fast, slow_period=self.macd_slow, signal_period=self.macd_signal)
        macd_line = macd_data['macd'].iloc[-1] if 'macd' in macd_data.columns else 0
        macd_signal_line = macd_data['signal'].iloc[-1] if 'signal' in macd_data.columns else 0
        macd_histogram = macd_data['histogram'].iloc[-1] if 'histogram' in macd_data.columns else 0
        macd_prev_hist = macd_data['histogram'].iloc[-2] if len(macd_data) > 1 and 'histogram' in macd_data.columns else macd_histogram
        
        # Calculate price momentum
        momentum = (current_price - data['close'].iloc[-self.momentum_period]) / data['close'].iloc[-self.momentum_period]
        
        # Calculate ATR for stop-loss
        atr = self.indicators.atr(data)
        atr_value = atr.iloc[-1] if not atr.empty else current_price * 0.02
        
        signal = Signal.HOLD
        confidence = 0.0
        reason = "No momentum signal"
        
        # Bullish momentum: RSI rising, MACD bullish crossover, positive momentum
        bullish_conditions = [
            rsi_current > 50 and rsi_current > rsi_prev,  # RSI rising above neutral
            macd_line > macd_signal_line and macd_histogram > macd_prev_hist,  # MACD bullish crossover
            momentum > self.min_momentum,  # Positive momentum
            rsi_current < self.rsi_overbought  # Not overbought yet
        ]
        
        # Bearish momentum: RSI falling, MACD bearish crossover, negative momentum
        bearish_conditions = [
            rsi_current < 50 and rsi_current < rsi_prev,  # RSI falling below neutral
            macd_line < macd_signal_line and macd_histogram < macd_prev_hist,  # MACD bearish crossover
            momentum < -self.min_momentum,  # Negative momentum
            rsi_current > self.rsi_oversold  # Not oversold yet
        ]
        
        bullish_score = sum(bullish_conditions)
        bearish_score = sum(bearish_conditions)
        
        if bullish_score >= 3:
            signal = Signal.BUY
            confidence = min(bullish_score / 4.0 * 0.8 + abs(momentum) * 5, 0.85)
            reason = f"Bullish momentum: RSI={rsi_current:.1f}, MACD={macd_histogram:.4f}, Momentum={momentum*100:.2f}%"
        
        elif bearish_score >= 3:
            signal = Signal.SELL
            confidence = min(bearish_score / 4.0 * 0.8 + abs(momentum) * 5, 0.85)
            reason = f"Bearish momentum: RSI={rsi_current:.1f}, MACD={macd_histogram:.4f}, Momentum={momentum*100:.2f}%"
        
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
            'reason': reason,
            'rsi': rsi_current,
            'macd_histogram': macd_histogram,
            'momentum': momentum
        }
    
    def should_exit(self, data: pd.DataFrame, position: Dict) -> bool:
        """Check if position should be exited (momentum reversal)"""
        if len(data) < max(self.macd_slow, self.rsi_period):
            return False
        
        # Calculate RSI
        rsi = self.indicators.rsi(data, period=self.rsi_period)
        rsi_current = rsi.iloc[-1] if not rsi.empty else 50
        
        # Calculate MACD
        macd_data = self.indicators.macd(data, fast_period=self.macd_fast, slow_period=self.macd_slow, signal_period=self.macd_signal)
        macd_histogram = macd_data['histogram'].iloc[-1] if 'histogram' in macd_data.columns else 0
        
        # Exit long if momentum reverses
        if position['side'] == 'buy':
            if rsi_current > self.rsi_overbought or macd_histogram < 0:
                return True
        
        # Exit short if momentum reverses
        if position['side'] == 'sell':
            if rsi_current < self.rsi_oversold or macd_histogram > 0:
                return True
        
        return False

