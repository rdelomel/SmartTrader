"""End-of-day trading strategy - Trading near market close"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, time
try:
    from .base_strategy import BaseStrategy, Signal
    from ..indicators.technical import TechnicalIndicators
except ImportError:
    from base_strategy import BaseStrategy, Signal
    from indicators.technical import TechnicalIndicators


class EndOfDayStrategy(BaseStrategy):
    """
    End-of-day trading strategy based on CMC Markets guide
    - Trades near market close when price is 'settling'
    - Compares price action to previous day's movements
    - Requires risk management orders (stop-loss, take-profit)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize end-of-day strategy
        
        Config parameters:
            close_hour: Hour to consider as market close (default: 15 = 3 PM)
            close_minute: Minute for market close (default: 45)
            min_price_change: Minimum price change % to trade (default: 0.003 = 0.3%)
            compare_previous_day: Compare to previous day's close (default: True)
        """
        super().__init__("EndOfDay", config)
        self.close_hour = self.config.get('close_hour', 15)
        self.close_minute = self.config.get('close_minute', 45)
        self.min_price_change = self.config.get('min_price_change', 0.003)
        self.compare_previous_day = self.config.get('compare_previous_day', True)
        self.indicators = TechnicalIndicators()
    
    def _is_near_close(self, current_time: Optional[datetime] = None) -> bool:
        """Check if current time is near market close"""
        if current_time is None:
            current_time = datetime.now()
        
        # Check if within 30 minutes of close
        close_time = time(self.close_hour, self.close_minute)
        current_time_only = current_time.time()
        
        # Simple check: if hour matches and minute is close
        if current_time_only.hour == self.close_hour:
            return abs(current_time_only.minute - self.close_minute) <= 30
        elif current_time_only.hour == self.close_hour - 1:
            return current_time_only.minute >= 30
        
        return False
    
    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """
        Generate signal based on end-of-day price action
        
        Args:
            data: Market data DataFrame with datetime index
        
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
        current_time = data.index[-1] if hasattr(data.index[-1], 'hour') else datetime.now()
        
        # Check if we're near market close
        if not self._is_near_close(current_time):
            return {
                'signal': Signal.HOLD,
                'confidence': 0.0,
                'entry_price': current_price,
                'stop_loss': None,
                'take_profit': None,
                'reason': 'Not near market close'
            }
        
        # Calculate today's price action
        if len(data) >= 2:
            open_price = data['open'].iloc[0] if hasattr(data.index, 'date') else data['open'].iloc[-len(data)]
            day_change = (current_price - open_price) / open_price
        else:
            day_change = 0
        
        # Calculate recent price action (last hour equivalent - last 60 bars if available)
        lookback = min(60, len(data) - 1)
        price_change_recent = (current_price - data['close'].iloc[-lookback]) / data['close'].iloc[-lookback]
        
        # Calculate ATR for stop-loss
        atr = self.indicators.atr(data)
        atr_value = atr.iloc[-1] if not atr.empty else current_price * 0.015
        
        # Calculate RSI
        rsi = self.indicators.rsi(data, period=14)
        rsi_current = rsi.iloc[-1] if not rsi.empty else 50
        
        # Calculate volume trend (decreasing volume near close suggests settling)
        if 'volume' in data.columns and len(data) >= 20:
            recent_volume = data['volume'].iloc[-5:].mean()
            earlier_volume = data['volume'].iloc[-20:-5].mean()
            volume_decreasing = recent_volume < earlier_volume * 0.8 if earlier_volume > 0 else False
        else:
            volume_decreasing = False
        
        signal = Signal.HOLD
        confidence = 0.0
        reason = "No end-of-day signal"
        
        # Strategy: Trade when price is 'settling' near close
        # Look for price that has moved during the day and is now stabilizing
        
        # Bullish: Price moved up during day and is settling near high
        if day_change > self.min_price_change and price_change_recent > -0.001:
            # Price moved up and is holding near high
            if rsi_current < 70 and volume_decreasing:
                signal = Signal.BUY
                confidence = min(0.4 + abs(day_change) * 10, 0.65)
                reason = f"EOD bullish settlement (Day: +{day_change*100:.2f}%, RSI: {rsi_current:.1f})"
        
        # Bearish: Price moved down during day and is settling near low
        elif day_change < -self.min_price_change and price_change_recent < 0.001:
            # Price moved down and is holding near low
            if rsi_current > 30 and volume_decreasing:
                signal = Signal.SELL
                confidence = min(0.4 + abs(day_change) * 10, 0.65)
                reason = f"EOD bearish settlement (Day: {day_change*100:.2f}%, RSI: {rsi_current:.1f})"
        
        # Alternative: Fade the day's move if it's extreme and reversing
        if signal == Signal.HOLD:
            if day_change > 0.02 and price_change_recent < -0.003 and rsi_current > 65:
                # Strong up day but reversing near close - potential fade
                signal = Signal.SELL
                confidence = 0.35
                reason = f"EOD fade - strong up day reversing (Day: +{day_change*100:.2f}%)"
            elif day_change < -0.02 and price_change_recent > 0.003 and rsi_current < 35:
                # Strong down day but reversing near close - potential fade
                signal = Signal.BUY
                confidence = 0.35
                reason = f"EOD fade - strong down day reversing (Day: {day_change*100:.2f}%)"
        
        # Calculate stop-loss and take-profit
        stop_loss = None
        take_profit = None
        
        if signal != Signal.HOLD:
            # Wider stops for overnight positions (EOD trades often held overnight)
            if signal == Signal.BUY:
                stop_loss = current_price - (atr_value * 2.5)  # Wider for overnight
                take_profit = current_price + (atr_value * 3.5)  # 1:1.4 risk/reward
            else:  # SELL
                stop_loss = current_price + (atr_value * 2.5)
                take_profit = current_price - (atr_value * 3.5)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'reason': reason,
            'day_change': day_change,
            'volume_decreasing': volume_decreasing
        }
    
    def should_exit(self, data: pd.DataFrame, position: Dict) -> bool:
        """Exit at next day's open or if position moves significantly"""
        if len(data) < 2:
            return False
        
        current_price = data['close'].iloc[-1]
        entry_price = position.get('entry_price', current_price)
        
        # Exit if position has moved significantly (take profit or stop loss)
        price_change = (current_price - entry_price) / entry_price
        
        if position['side'] == 'buy':
            # Exit if moved 2% or more
            if abs(price_change) > 0.02:
                return True
        else:  # sell
            if abs(price_change) > 0.02:
                return True
        
        return False

