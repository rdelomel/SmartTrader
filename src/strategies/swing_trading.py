"""Swing trading strategy - Trading market oscillations"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
try:
    from .base_strategy import BaseStrategy, Signal
    from ..indicators.technical import TechnicalIndicators
except ImportError:
    from base_strategy import BaseStrategy, Signal
    from indicators.technical import TechnicalIndicators


class SwingTradingStrategy(BaseStrategy):
    """
    Swing trading strategy based on CMC Markets guide
    - Trades both sides of market movements
    - Buys when market will rise, sells when market will fall
    - Takes advantage of oscillations from overbought to oversold
    - Uses support/resistance levels and swing length/duration
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize swing trading strategy
        
        Config parameters:
            swing_period: Period to identify swings (default: 14)
            rsi_period: RSI period for overbought/oversold (default: 14)
            rsi_oversold: RSI oversold level (default: 30)
            rsi_overbought: RSI overbought level (default: 70)
            min_swing_size: Minimum swing size % to trade (default: 0.02 = 2%)
            use_support_resistance: Use support/resistance levels (default: True)
        """
        super().__init__("SwingTrading", config)
        self.swing_period = self.config.get('swing_period', 14)
        self.rsi_period = self.config.get('rsi_period', 14)
        self.rsi_oversold = self.config.get('rsi_oversold', 30)
        self.rsi_overbought = self.config.get('rsi_overbought', 70)
        self.min_swing_size = self.config.get('min_swing_size', 0.02)
        self.use_support_resistance = self.config.get('use_support_resistance', True)
        self.indicators = TechnicalIndicators()
    
    def _identify_swings(self, data: pd.DataFrame) -> Dict:
        """Identify swing highs and lows"""
        if len(data) < self.swing_period * 2:
            return {'swing_high': None, 'swing_low': None, 'swing_range': 0}
        
        highs = data['high'].rolling(self.swing_period, center=True).max()
        lows = data['low'].rolling(self.swing_period, center=True).min()
        
        # Find recent swing high and low
        recent_highs = highs.iloc[-self.swing_period:]
        recent_lows = lows.iloc[-self.swing_period:]
        
        swing_high = recent_highs.max()
        swing_low = recent_lows.min()
        swing_range = (swing_high - swing_low) / swing_low if swing_low > 0 else 0
        
        return {
            'swing_high': swing_high,
            'swing_low': swing_low,
            'swing_range': swing_range
        }
    
    def _find_support_resistance(self, data: pd.DataFrame) -> Dict:
        """Find support and resistance levels"""
        if len(data) < 20:
            return {'support': None, 'resistance': None}
        
        # Use recent highs and lows
        lookback = min(50, len(data))
        recent_data = data.iloc[-lookback:]
        
        # Resistance: recent highs
        resistance = recent_data['high'].quantile(0.9)
        
        # Support: recent lows
        support = recent_data['low'].quantile(0.1)
        
        return {'support': support, 'resistance': resistance}
    
    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """
        Generate signal based on swing trading principles
        
        Args:
            data: Market data DataFrame
        
        Returns:
            Signal dictionary
        """
        if len(data) < max(self.swing_period * 2, self.rsi_period + 5):
            return {
                'signal': Signal.HOLD,
                'confidence': 0.0,
                'entry_price': data['close'].iloc[-1],
                'stop_loss': None,
                'take_profit': None,
                'reason': 'Insufficient data'
            }
        
        current_price = data['close'].iloc[-1]
        
        # Calculate RSI for overbought/oversold
        rsi = self.indicators.rsi(data, period=self.rsi_period)
        rsi_current = rsi.iloc[-1] if not rsi.empty else 50
        rsi_prev = rsi.iloc[-2] if len(rsi) > 1 else rsi_current
        
        # Identify swings
        swings = self._identify_swings(data)
        swing_high = swings['swing_high']
        swing_low = swings['swing_low']
        swing_range = swings['swing_range']
        
        # Find support/resistance
        levels = self._find_support_resistance(data) if self.use_support_resistance else {}
        support = levels.get('support')
        resistance = levels.get('resistance')
        
        # Calculate ATR for stop-loss
        atr = self.indicators.atr(data)
        atr_value = atr.iloc[-1] if not atr.empty else current_price * 0.02
        
        # Calculate MACD for momentum
        macd = self.indicators.macd(data)
        macd_hist = macd['histogram'].iloc[-1] if not macd.empty and 'histogram' in macd.columns else 0
        
        signal = Signal.HOLD
        confidence = 0.0
        reason = "No swing signal"
        
        # Strategy 1: Buy at oversold (swing low), sell at overbought (swing high)
        # Buy when RSI is oversold and starting to rise
        if rsi_current < self.rsi_oversold and rsi_current > rsi_prev:
            # Price near swing low or support
            near_support = False
            if swing_low and current_price <= swing_low * 1.02:
                near_support = True
            elif support and current_price <= support * 1.02:
                near_support = True
            
            if near_support or swing_range > self.min_swing_size:
                signal = Signal.BUY
                confidence = min(0.5 + (self.rsi_oversold - rsi_current) / 50, 0.75)
                reason = f"Swing buy - oversold bounce (RSI: {rsi_current:.1f}, Near support: {near_support})"
        
        # Sell when RSI is overbought and starting to fall
        elif rsi_current > self.rsi_overbought and rsi_current < rsi_prev:
            # Price near swing high or resistance
            near_resistance = False
            if swing_high and current_price >= swing_high * 0.98:
                near_resistance = True
            elif resistance and current_price >= resistance * 0.98:
                near_resistance = True
            
            if near_resistance or swing_range > self.min_swing_size:
                signal = Signal.SELL
                confidence = min(0.5 + (rsi_current - self.rsi_overbought) / 50, 0.75)
                reason = f"Swing sell - overbought rejection (RSI: {rsi_current:.1f}, Near resistance: {near_resistance})"
        
        # Strategy 2: Trade retracements in strong trends
        # If in uptrend, buy on pullbacks (swing retracements)
        if signal == Signal.HOLD and swing_high and swing_low:
            price_position = (current_price - swing_low) / (swing_high - swing_low) if swing_high > swing_low else 0.5
            
            # Uptrend: buy on pullback to 30-40% of swing range
            if price_position > 0.5 and price_position < 0.7 and rsi_current > 40 and rsi_current < 60:
                if macd_hist > 0:  # Confirming uptrend
                    signal = Signal.BUY
                    confidence = 0.45
                    reason = f"Swing retracement buy - uptrend pullback (Position: {price_position*100:.1f}%)"
            
            # Downtrend: sell on bounce to 60-70% of swing range
            elif price_position < 0.5 and price_position > 0.3 and rsi_current < 60 and rsi_current > 40:
                if macd_hist < 0:  # Confirming downtrend
                    signal = Signal.SELL
                    confidence = 0.45
                    reason = f"Swing retracement sell - downtrend bounce (Position: {price_position*100:.1f}%)"
        
        # Calculate stop-loss and take-profit
        stop_loss = None
        take_profit = None
        
        if signal != Signal.HOLD:
            # Stop loss at opposite swing level
            if signal == Signal.BUY:
                if swing_low:
                    stop_loss = swing_low * 0.995  # Slightly below swing low
                else:
                    stop_loss = current_price - (atr_value * 2)
                
                # Take profit at swing high or resistance
                if swing_high:
                    take_profit = swing_high * 0.995
                elif resistance:
                    take_profit = resistance * 0.995
                else:
                    take_profit = current_price + (atr_value * 3)
            else:  # SELL
                if swing_high:
                    stop_loss = swing_high * 1.005  # Slightly above swing high
                else:
                    stop_loss = current_price + (atr_value * 2)
                
                # Take profit at swing low or support
                if swing_low:
                    take_profit = swing_low * 1.005
                elif support:
                    take_profit = support * 1.005
                else:
                    take_profit = current_price - (atr_value * 3)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'reason': reason,
            'rsi': rsi_current,
            'swing_range': swing_range
        }
    
    def should_exit(self, data: pd.DataFrame, position: Dict) -> bool:
        """Exit when swing completes (reaches opposite extreme)"""
        if len(data) < self.swing_period:
            return False
        
        current_price = data['close'].iloc[-1]
        
        # Calculate RSI
        rsi = self.indicators.rsi(data, period=self.rsi_period)
        rsi_current = rsi.iloc[-1] if not rsi.empty else 50
        
        # Exit long if RSI becomes overbought
        if position['side'] == 'buy' and rsi_current > self.rsi_overbought:
            return True
        
        # Exit short if RSI becomes oversold
        if position['side'] == 'sell' and rsi_current < self.rsi_oversold:
            return True
        
        return False

