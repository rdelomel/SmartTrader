"""Day trading strategy - Opening and closing positions within the same day"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, time
try:
    from .base_strategy import BaseStrategy, Signal
    from ..indicators.technical import TechnicalIndicators
    from .trend_following import TrendFollowingStrategy
except ImportError:
    from base_strategy import BaseStrategy, Signal
    from indicators.technical import TechnicalIndicators
    from trend_following import TrendFollowingStrategy


class DayTradingStrategy(BaseStrategy):
    """
    Day trading strategy based on CMC Markets guide
    - Opens and closes positions within the same day
    - No overnight positions
    - Requires discipline and pre-determined entry/exit levels
    - Focuses on intraday price movements
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize day trading strategy
        
        Config parameters:
            open_hour: Market open hour (default: 9)
            close_hour: Market close hour (default: 16)
            min_intraday_move: Minimum intraday move % to trade (default: 0.005 = 0.5%)
            use_breakouts: Trade breakouts (default: True)
            use_reversals: Trade reversals (default: True)
            max_trades_per_day: Maximum trades per day (default: 5)
        """
        super().__init__("DayTrading", config)
        self.open_hour = self.config.get('open_hour', 9)
        self.close_hour = self.config.get('close_hour', 16)
        self.min_intraday_move = self.config.get('min_intraday_move', 0.005)
        self.use_breakouts = self.config.get('use_breakouts', True)
        self.use_reversals = self.config.get('use_reversals', True)
        self.max_trades_per_day = self.config.get('max_trades_per_day', 5)
        self.indicators = TechnicalIndicators()
    
    def _is_trading_hours(self, current_time: Optional[datetime] = None, asset_class: str = 'stocks') -> bool:
        """Check if current time is within trading hours.

        Crypto trades 24/7 and forex trades ~24/5 (no reliable single-timezone
        session) so the fixed open/close hour gate — designed for stock market
        hours — only applies to stocks/commodities.
        """
        if asset_class == 'crypto':
            return True
        if asset_class == 'forex':
            # Forex trades continuously Mon-Fri; only exclude weekend close.
            if current_time is None:
                current_time = datetime.now()
            weekday = current_time.weekday() if hasattr(current_time, 'weekday') else datetime.now().weekday()
            return weekday < 5

        if current_time is None:
            current_time = datetime.now()
        
        # Handle both datetime objects and timestamps
        if hasattr(current_time, 'time'):
            current_time_only = current_time.time()
        elif hasattr(current_time, 'hour'):
            # It's already a time-like object
            current_time_only = current_time
        else:
            # Try to extract from index or use current time
            current_time = datetime.now()
            current_time_only = current_time.time()
        
        # Stocks/commodities: check fixed trading hours
        return self.open_hour <= current_time_only.hour < self.close_hour
    
    def _get_intraday_data(self, data: pd.DataFrame) -> Dict:
        """Get intraday price action"""
        if len(data) < 2:
            return {'open': None, 'high': None, 'low': None, 'current': None, 'range': 0}
        
        # For crypto/24h markets, use recent data (last 24 hours equivalent)
        # Use last 24 bars if available, or all data if less
        lookback = min(24, len(data))
        recent_data = data.iloc[-lookback:]
        
        # Get session open (first price in lookback period)
        session_open = recent_data['open'].iloc[0] if 'open' in recent_data.columns else recent_data['close'].iloc[0]
        
        # Get session high and low from recent period
        session_high = recent_data['high'].max() if 'high' in recent_data.columns else recent_data['close'].max()
        session_low = recent_data['low'].min() if 'low' in recent_data.columns else recent_data['close'].min()
        current_price = data['close'].iloc[-1]
        
        # Calculate intraday range
        intraday_range = (session_high - session_low) / session_low if session_low > 0 else 0
        
        return {
            'open': session_open,
            'high': session_high,
            'low': session_low,
            'current': current_price,
            'range': intraday_range
        }
    
    def generate_signal(self, data: pd.DataFrame, symbol: str = "") -> Dict:
        """
        Generate signal for day trading
        
        Args:
            data: Market data DataFrame
            symbol: Trading symbol — used to select per-asset-class trading-hours rules.
        
        Returns:
            Signal dictionary
        """
        asset_class = TrendFollowingStrategy._detect_asset_class(symbol)
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
        
        # Check if within trading hours (asset-class aware)
        if not self._is_trading_hours(current_time, asset_class):
            return {
                'signal': Signal.HOLD,
                'confidence': 0.0,
                'entry_price': current_price,
                'stop_loss': None,
                'take_profit': None,
                'reason': 'Outside trading hours'
            }
        
        # Get intraday data
        intraday = self._get_intraday_data(data)
        session_open = intraday['open']
        session_high = intraday['high']
        session_low = intraday['low']
        intraday_range = intraday['range']
        
        if session_open is None:
            return {
                'signal': Signal.HOLD,
                'confidence': 0.0,
                'entry_price': current_price,
                'stop_loss': None,
                'take_profit': None,
                'reason': 'No session data'
            }
        
        # Calculate intraday price change
        intraday_change = (current_price - session_open) / session_open
        
        # Calculate ATR for stop-loss (use shorter period for day trading)
        atr = self.indicators.atr(data, period=14)
        atr_value = atr.iloc[-1] if not atr.empty else current_price * 0.01
        
        # Calculate RSI
        rsi = self.indicators.rsi(data, period=14)
        rsi_current = rsi.iloc[-1] if not rsi.empty else 50
        
        # Calculate volume (day traders need volume)
        if 'volume' in data.columns:
            recent_volume = data['volume'].iloc[-5:].mean()
            avg_volume = data['volume'].rolling(20).mean().iloc[-1] if len(data) >= 20 else recent_volume
            volume_ok = recent_volume >= avg_volume * 0.8
        else:
            volume_ok = True
        
        # Calculate MACD for momentum
        macd = self.indicators.macd(data)
        macd_hist = macd['histogram'].iloc[-1] if not macd.empty and 'histogram' in macd.columns else 0
        
        signal = Signal.HOLD
        confidence = 0.0
        reason = "No day trading signal"
        
        # Strategy 1: Breakout trading
        if self.use_breakouts and intraday_range > self.min_intraday_move:
            # Breakout above session high
            if current_price >= session_high * 0.999 and rsi_current < 70 and volume_ok:
                signal = Signal.BUY
                confidence = min(0.5 + intraday_range * 10, 0.7)
                reason = f"Intraday breakout [{asset_class}] - above high (Range: {intraday_range*100:.2f}%)"
            
            # Breakdown below session low
            elif current_price <= session_low * 1.001 and rsi_current > 30 and volume_ok:
                signal = Signal.SELL
                confidence = min(0.5 + intraday_range * 10, 0.7)
                reason = f"Intraday breakdown [{asset_class}] - below low (Range: {intraday_range*100:.2f}%)"
        
        # Strategy 2: Reversal trading
        if self.use_reversals and signal == Signal.HOLD:
            # Price moved up but reversing (potential short)
            if intraday_change > 0.01 and current_price < session_high * 0.995 and rsi_current > 60:
                if macd_hist < 0:  # Momentum turning negative
                    signal = Signal.SELL
                    confidence = 0.4
                    reason = f"Intraday reversal - fading up move (Change: +{intraday_change*100:.2f}%)"
            
            # Price moved down but reversing (potential long)
            elif intraday_change < -0.01 and current_price > session_low * 1.005 and rsi_current < 40:
                if macd_hist > 0:  # Momentum turning positive
                    signal = Signal.BUY
                    confidence = 0.4
                    reason = f"Intraday reversal - fading down move (Change: {intraday_change*100:.2f}%)"
        
        # Strategy 3: Range trading (buy low, sell high of the day)
        if signal == Signal.HOLD and intraday_range > self.min_intraday_move:
            # Buy near session low
            if current_price <= session_low * 1.005 and rsi_current < 50:
                signal = Signal.BUY
                confidence = 0.35
                reason = f"Range trade - buy near session low (RSI: {rsi_current:.1f})"
            
            # Sell near session high
            elif current_price >= session_high * 0.995 and rsi_current > 50:
                signal = Signal.SELL
                confidence = 0.35
                reason = f"Range trade - sell near session high (RSI: {rsi_current:.1f})"
        
        # Calculate stop-loss and take-profit
        stop_loss = None
        take_profit = None
        
        if signal != Signal.HOLD:
            # Tighter stops for day trading (no overnight risk)
            if signal == Signal.BUY:
                stop_loss = current_price - (atr_value * 1.5)  # Tighter stop
                take_profit = current_price + (atr_value * 2.0)  # 1:1.33 risk/reward
            else:  # SELL
                stop_loss = current_price + (atr_value * 1.5)
                take_profit = current_price - (atr_value * 2.0)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'reason': reason,
            'intraday_change': intraday_change,
            'intraday_range': intraday_range
        }
    
    def should_exit(self, data: pd.DataFrame, position: Dict) -> bool:
        """Exit before market close (no overnight positions)"""
        if len(data) < 1:
            return False
        
        current_time = data.index[-1] if hasattr(data.index[-1], 'hour') else datetime.now()
        
        # Exit if approaching market close (within 30 minutes)
        current_time_only = current_time.time()
        if current_time_only.hour >= self.close_hour - 1 and current_time_only.minute >= 30:
            return True
        
        # Exit if position has moved significantly (take profit or stop loss)
        current_price = data['close'].iloc[-1]
        entry_price = position.get('entry_price', current_price)
        price_change = (current_price - entry_price) / entry_price
        
        # Exit if moved 1% or more (day trading targets smaller moves)
        if abs(price_change) > 0.01:
            return True
        
        return False

