"""Mean reversion strategies"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
try:
    from .base_strategy import BaseStrategy, Signal
    from ..indicators.technical import TechnicalIndicators
except ImportError:
    from base_strategy import BaseStrategy, Signal
    from indicators.technical import TechnicalIndicators


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy using Bollinger Bands and RSI"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize mean reversion strategy
        
        Config parameters:
            lookback_period: Period for mean calculation (default: 14)
            z_score_threshold: Z-score threshold for entry (default: 2.0)
            rsi_oversold: RSI oversold level (default: 30)
            rsi_overbought: RSI overbought level (default: 70)
        """
        super().__init__("MeanReversion", config)
        self.lookback_period = self.config.get('lookback_period', 14)
        self.z_score_threshold = self.config.get('z_score_threshold', 2.0)
        self.rsi_oversold = self.config.get('rsi_oversold', 30)
        self.rsi_overbought = self.config.get('rsi_overbought', 70)
        self.indicators = TechnicalIndicators()
    
    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """Generate signal based on mean reversion with graduated confidence"""
        if len(data) < self.lookback_period:
            return {
                'signal': Signal.HOLD,
                'confidence': 0.0,
                'entry_price': data['close'].iloc[-1],
                'stop_loss': None,
                'take_profit': None,
                'reason': 'Insufficient data'
            }
        
        current_price = data['close'].iloc[-1]
        
        # Calculate Z-score
        mean = data['close'].rolling(window=self.lookback_period).mean().iloc[-1]
        std = data['close'].rolling(window=self.lookback_period).std().iloc[-1]
        
        if std == 0:
            z_score = 0
        else:
            z_score = (current_price - mean) / std
        
        # Calculate RSI
        rsi = self.indicators.rsi(data, period=self.lookback_period)
        rsi_current = rsi.iloc[-1] if not rsi.empty else 50
        
        # Calculate ATR for stop-loss
        atr = self.indicators.atr(data)
        atr_value = atr.iloc[-1] if not atr.empty else current_price * 0.02
        
        # Bollinger Bands
        bb = self.indicators.bollinger_bands(data, period=20)
        bb_upper = bb['upper'].iloc[-1] if not bb.empty else current_price * 1.02
        bb_lower = bb['lower'].iloc[-1] if not bb.empty else current_price * 0.98
        bb_middle = bb['middle'].iloc[-1] if not bb.empty else mean
        
        # Calculate BB width for volatility context
        bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0
        
        # Calculate price position in BB range
        if bb_upper > bb_lower:
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)  # 0 = lower band, 1 = upper band
        else:
            bb_position = 0.5
        
        # Calculate momentum (rate of change)
        if len(data) >= 5:
            momentum = (current_price - data['close'].iloc[-5]) / data['close'].iloc[-5]
        else:
            momentum = 0
        
        signal = Signal.HOLD
        confidence = 0.0
        reason = "No signal"
        
        # LOWERED THRESHOLD: Use 1.5 instead of 2.0 for z-score
        lower_z_threshold = 1.5
        
        # === BUY SIGNALS (Oversold) ===
        # Strong buy: Extreme oversold with confirmation
        if (z_score < -self.z_score_threshold or current_price < bb_lower) and rsi_current < self.rsi_oversold:
            signal = Signal.BUY
            
            # Multi-factor confidence
            conf_factors = []
            conf_factors.append(min(abs(z_score) / self.z_score_threshold * 0.35, 0.35))  # Z-score strength
            conf_factors.append((self.rsi_oversold - rsi_current) / self.rsi_oversold * 0.25)  # RSI oversold
            conf_factors.append(0.15)  # Base confidence for extreme condition
            if momentum < 0:  # Falling into oversold
                conf_factors.append(min(abs(momentum) * 5, 0.15))
            
            confidence = min(sum(conf_factors), 0.85)
            reason = f"Strong oversold (Z: {z_score:.2f}, RSI: {rsi_current:.1f}, BB: {bb_position:.2f})"
        
        # Moderate buy: Moderately oversold
        elif z_score < -lower_z_threshold or current_price < bb_lower or rsi_current < 35:
            signal = Signal.BUY
            
            # Graduated confidence based on multiple factors
            conf_factors = []
            
            # Z-score contribution
            if z_score < -lower_z_threshold:
                conf_factors.append(min(abs(z_score) / lower_z_threshold * 0.25, 0.25))
            
            # BB position contribution
            if bb_position < 0.2:  # Near lower band
                conf_factors.append((0.2 - bb_position) * 1.0)  # Up to 0.2
            
            # RSI contribution
            if rsi_current < 40:
                conf_factors.append((40 - rsi_current) / 40 * 0.2)
            
            # Momentum check (avoid catching falling knives)
            if momentum > -0.01:  # Not falling too fast
                conf_factors.append(0.1)
            elif momentum > -0.02:
                conf_factors.append(0.05)
            
            # Base confidence
            conf_factors.append(0.15)
            
            confidence = min(sum(conf_factors), 0.65)
            reason = f"Moderate oversold (Z: {z_score:.2f}, RSI: {rsi_current:.1f}, BB: {bb_position:.2f})"
        
        # Weak buy: Approaching mean from below
        elif z_score < -0.5 and z_score > -lower_z_threshold and rsi_current < 45:
            signal = Signal.BUY
            
            conf_factors = []
            conf_factors.append(abs(z_score) * 0.15)  # Z-score factor
            conf_factors.append((45 - rsi_current) / 45 * 0.15)  # RSI factor
            conf_factors.append(0.1)  # Base confidence
            
            confidence = min(sum(conf_factors), 0.4)
            reason = f"Weak oversold signal (Z: {z_score:.2f}, RSI: {rsi_current:.1f})"
        
        # === SELL SIGNALS (Overbought) ===
        # Strong sell: Extreme overbought with confirmation
        elif (z_score > self.z_score_threshold or current_price > bb_upper) and rsi_current > self.rsi_overbought:
            signal = Signal.SELL
            
            conf_factors = []
            conf_factors.append(min(z_score / self.z_score_threshold * 0.35, 0.35))
            conf_factors.append((rsi_current - self.rsi_overbought) / (100 - self.rsi_overbought) * 0.25)
            conf_factors.append(0.15)  # Base confidence
            if momentum > 0:  # Rising into overbought
                conf_factors.append(min(momentum * 5, 0.15))
            
            confidence = min(sum(conf_factors), 0.85)
            reason = f"Strong overbought (Z: {z_score:.2f}, RSI: {rsi_current:.1f}, BB: {bb_position:.2f})"
        
        # Moderate sell: Moderately overbought
        elif z_score > lower_z_threshold or current_price > bb_upper or rsi_current > 65:
            signal = Signal.SELL
            
            conf_factors = []
            
            # Z-score contribution
            if z_score > lower_z_threshold:
                conf_factors.append(min(z_score / lower_z_threshold * 0.25, 0.25))
            
            # BB position contribution
            if bb_position > 0.8:  # Near upper band
                conf_factors.append((bb_position - 0.8) * 1.0)  # Up to 0.2
            
            # RSI contribution
            if rsi_current > 60:
                conf_factors.append((rsi_current - 60) / 40 * 0.2)
            
            # Momentum check
            if momentum < 0.01:  # Not rising too fast
                conf_factors.append(0.1)
            elif momentum < 0.02:
                conf_factors.append(0.05)
            
            # Base confidence
            conf_factors.append(0.15)
            
            confidence = min(sum(conf_factors), 0.65)
            reason = f"Moderate overbought (Z: {z_score:.2f}, RSI: {rsi_current:.1f}, BB: {bb_position:.2f})"
        
        # Weak sell: Approaching mean from above
        elif z_score > 0.5 and z_score < lower_z_threshold and rsi_current > 55:
            signal = Signal.SELL
            
            conf_factors = []
            conf_factors.append(z_score * 0.15)
            conf_factors.append((rsi_current - 55) / 45 * 0.15)
            conf_factors.append(0.1)  # Base confidence
            
            confidence = min(sum(conf_factors), 0.4)
            reason = f"Weak overbought signal (Z: {z_score:.2f}, RSI: {rsi_current:.1f})"
        
        # Calculate stop-loss and take-profit
        stop_loss = None
        take_profit = None
        
        if signal != Signal.HOLD:
            if signal == Signal.BUY:
                stop_loss = current_price - (atr_value * 1.5)
                take_profit = bb_middle  # Target the mean
            else:  # SELL
                stop_loss = current_price + (atr_value * 1.5)
                take_profit = bb_middle
        
        return {
            'signal': signal,
            'confidence': confidence,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'reason': reason
        }
    
    def should_exit(self, data: pd.DataFrame, position: Dict) -> bool:
        """Check if position should be exited (mean reversion target reached)"""
        if len(data) < self.lookback_period:
            return False
        
        current_price = data['close'].iloc[-1]
        entry_price = position.get('entry_price', current_price)
        
        # Calculate mean
        mean = data['close'].rolling(window=self.lookback_period).mean().iloc[-1]
        
        # Calculate RSI
        rsi = self.indicators.rsi(data, period=self.lookback_period)
        rsi_current = rsi.iloc[-1] if not rsi.empty else 50
        
        # Exit long position if price returns to mean or RSI becomes neutral
        if position['side'] == 'buy':
            if current_price >= mean or rsi_current > 50:
                return True
        
        # Exit short position if price returns to mean or RSI becomes neutral
        if position['side'] == 'sell':
            if current_price <= mean or rsi_current < 50:
                return True
        
        return False

