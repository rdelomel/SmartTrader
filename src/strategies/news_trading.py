"""News trading strategy - Trading based on news and market expectations"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime, time
try:
    from .base_strategy import BaseStrategy, Signal
    from ..indicators.technical import TechnicalIndicators
except ImportError:
    from base_strategy import BaseStrategy, Signal
    from indicators.technical import TechnicalIndicators


class NewsTradingStrategy(BaseStrategy):
    """
    News trading strategy based on CMC Markets guide
    - Trades based on news and market expectations
    - Assesses if news is already factored into price
    - Can trade before or after news releases
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize news trading strategy
        
        Config parameters:
            volatility_threshold: Minimum volatility spike to trade (default: 0.015 = 1.5%)
            volume_spike_multiplier: Volume must be X times average (default: 1.5)
            price_movement_threshold: Minimum price movement % to trade (default: 0.005 = 0.5%)
            use_pre_news: Trade before news based on anticipation (default: True)
            use_post_news: Trade after news based on reaction (default: True)
        """
        super().__init__("NewsTrading", config)
        self.volatility_threshold = self.config.get('volatility_threshold', 0.015)
        self.volume_spike_multiplier = self.config.get('volume_spike_multiplier', 1.5)
        self.price_movement_threshold = self.config.get('price_movement_threshold', 0.005)
        self.use_pre_news = self.config.get('use_pre_news', True)
        self.use_post_news = self.config.get('use_post_news', True)
        self.indicators = TechnicalIndicators()
    
    def generate_signal(self, data: pd.DataFrame, news_items: Optional[List[Dict]] = None) -> Dict:
        """
        Generate signal based on news-driven price action
        
        Args:
            data: Market data DataFrame
            news_items: Optional list of news items with 'time', 'impact', 'sentiment' fields
        
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
        current_volume = data['volume'].iloc[-1] if 'volume' in data.columns else 0
        
        # Calculate recent volatility (standard deviation of returns)
        returns = data['close'].pct_change().dropna()
        recent_returns = returns.iloc[-10:] if len(returns) >= 10 else returns
        volatility = recent_returns.std()
        
        # Calculate average volume
        avg_volume = data['volume'].rolling(20).mean().iloc[-1] if 'volume' in data.columns and len(data) >= 20 else current_volume
        
        # Detect volume spike (indicator of news-driven activity)
        volume_spike = False
        if avg_volume > 0:
            volume_ratio = current_volume / avg_volume
            volume_spike = volume_ratio >= self.volume_spike_multiplier
        
        # Calculate recent price movement (fix: compare to previous bars, not current)
        price_change_1m = (current_price - data['close'].iloc[-2]) / data['close'].iloc[-2] if len(data) >= 2 else 0
        price_change_5m = (current_price - data['close'].iloc[-6]) / data['close'].iloc[-6] if len(data) >= 6 else 0
        price_change_10m = (current_price - data['close'].iloc[-11]) / data['close'].iloc[-11] if len(data) >= 11 else 0
        
        # Calculate ATR for stop-loss
        atr = self.indicators.atr(data)
        atr_value = atr.iloc[-1] if not atr.empty else current_price * 0.02
        
        # Calculate RSI to avoid overbought/oversold extremes
        rsi = self.indicators.rsi(data, period=14)
        rsi_current = rsi.iloc[-1] if not rsi.empty else 50
        
        signal = Signal.HOLD
        confidence = 0.0
        reason = "No news-driven signal"
        
        # Check for news-driven price action
        # Strategy 1: Pre-news anticipation (price moving before news)
        if self.use_pre_news:
            # Look for unusual price movement with volume spike OR just significant movement
            # Lower threshold if volume is elevated (even if not 1.5x spike)
            volume_elevated = volume_ratio >= 1.2 if avg_volume > 0 else False
            movement_threshold = self.price_movement_threshold * 0.7 if volume_elevated else self.price_movement_threshold
            
            if (volume_spike or volume_elevated) and abs(price_change_5m) > movement_threshold:
                if price_change_5m > 0 and rsi_current < 70:
                    # Bullish pre-news move
                    signal = Signal.BUY
                    confidence = min(0.3 + abs(price_change_5m) * 10 + (volume_ratio - 1) * 0.1, 0.7)
                    reason = f"Pre-news bullish anticipation (Price: +{price_change_5m*100:.2f}%, Volume: {volume_ratio:.1f}x)"
                elif price_change_5m < 0 and rsi_current > 30:
                    # Bearish pre-news move
                    signal = Signal.SELL
                    confidence = min(0.3 + abs(price_change_5m) * 10 + (volume_ratio - 1) * 0.1, 0.7)
                    reason = f"Pre-news bearish anticipation (Price: {price_change_5m*100:.2f}%, Volume: {volume_ratio:.1f}x)"
        
        # Strategy 2: Post-news reaction (strong price movement after news)
        if self.use_post_news and signal == Signal.HOLD:
            # Look for strong price movement with high volatility
            # Lower volatility threshold slightly and allow elevated volume
            effective_vol_threshold = self.volatility_threshold * 0.8
            if volatility > effective_vol_threshold and (volume_spike or volume_elevated):
                if price_change_1m > self.price_movement_threshold and rsi_current < 75:
                    # Strong bullish reaction
                    signal = Signal.BUY
                    confidence = min(0.4 + abs(price_change_1m) * 20 + volatility * 10, 0.75)
                    reason = f"Post-news bullish reaction (Move: +{price_change_1m*100:.2f}%, Vol: {volatility*100:.2f}%)"
                elif price_change_1m < -self.price_movement_threshold and rsi_current > 25:
                    # Strong bearish reaction
                    signal = Signal.SELL
                    confidence = min(0.4 + abs(price_change_1m) * 20 + volatility * 10, 0.75)
                    reason = f"Post-news bearish reaction (Move: {price_change_1m*100:.2f}%, Vol: {volatility*100:.2f}%)"
        
        # Strategy 3: News already factored in (fade the move)
        # If price moved significantly but is now reversing, trade the reversal
        if signal == Signal.HOLD and volume_spike:
            if price_change_10m > 0.01 and price_change_1m < -0.002 and rsi_current > 60:
                # Bullish move fading - potential sell
                signal = Signal.SELL
                confidence = 0.35
                reason = f"News factored in - fading bullish move (RSI: {rsi_current:.1f})"
            elif price_change_10m < -0.01 and price_change_1m > 0.002 and rsi_current < 40:
                # Bearish move fading - potential buy
                signal = Signal.BUY
                confidence = 0.35
                reason = f"News factored in - fading bearish move (RSI: {rsi_current:.1f})"
        
        # Calculate stop-loss and take-profit
        stop_loss = None
        take_profit = None
        
        if signal != Signal.HOLD:
            # Tighter stops for news trading (news moves can reverse quickly)
            if signal == Signal.BUY:
                stop_loss = current_price - (atr_value * 1.5)  # Tighter stop
                take_profit = current_price + (atr_value * 2.5)  # 1:1.67 risk/reward
            else:  # SELL
                stop_loss = current_price + (atr_value * 1.5)
                take_profit = current_price - (atr_value * 2.5)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'reason': reason,
            'volatility': volatility,
            'volume_spike': volume_spike
        }
    
    def should_exit(self, data: pd.DataFrame, position: Dict) -> bool:
        """Exit if news-driven move reverses"""
        if len(data) < 5:
            return False
        
        current_price = data['close'].iloc[-1]
        entry_price = position.get('entry_price', current_price)
        
        # Exit if price reverses significantly (news move may be over)
        price_change = (current_price - entry_price) / entry_price
        
        if position['side'] == 'buy':
            # Exit long if price drops significantly
            if price_change < -0.01:  # 1% reversal
                return True
        else:  # sell
            # Exit short if price rises significantly
            if price_change > 0.01:  # 1% reversal
                return True
        
        return False

