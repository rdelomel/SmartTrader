"""Breakout trading strategy"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
try:
    from .base_strategy import BaseStrategy, Signal
    from ..indicators.technical import TechnicalIndicators
    from .trend_following import TrendFollowingStrategy
except ImportError:
    from base_strategy import BaseStrategy, Signal
    from indicators.technical import TechnicalIndicators
    from trend_following import TrendFollowingStrategy


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
        self.stop_atr_mult = self.config.get('stop_loss_atr_multiple', 0.5)
        self.tp_atr_mult = self.config.get('take_profit_atr_multiple', 2.5)
        self.indicators = TechnicalIndicators()

    def _get_ac_params(self, symbol: str) -> Dict:
        """Return per-asset-class parameter overrides for this symbol."""
        asset_class = TrendFollowingStrategy._detect_asset_class(symbol)
        ac = self.config.get('asset_class_overrides', {}).get(asset_class, {})
        # Commodities/crypto need wider breakout confirmation and volume thresholds
        _bo_def = 0.008 if asset_class == 'crypto' else (0.006 if asset_class == 'commodities' else self.breakout_threshold)
        _vol_def = self.volume_threshold
        _stop_def = ac.get('stop_loss_atr_multiple', self.stop_atr_mult)
        _tp_def = ac.get('take_profit_rr_ratio', self.tp_atr_mult)
        return {
            'asset_class': asset_class,
            'breakout_threshold': ac.get('breakout_threshold', _bo_def),
            'volume_threshold': ac.get('volume_threshold', _vol_def),
            'stop_atr_mult': _stop_def,
            'tp_atr_mult': _tp_def,
        }

    def generate_signal(self, data: pd.DataFrame, symbol: str = "") -> Dict:
        """Generate signal based on breakout patterns

        Args:
            data:   OHLCV DataFrame.
            symbol: Trading symbol — used to select per-asset-class parameters.
        """
        if len(data) < self.lookback_period + 5:
            return {
                'signal': Signal.HOLD,
                'confidence': 0.0,
                'entry_price': data['close'].iloc[-1],
                'stop_loss': None,
                'take_profit': None,
                'reason': 'Insufficient data'
            }
        
        p = self._get_ac_params(symbol)
        breakout_threshold = p['breakout_threshold']
        volume_threshold = p['volume_threshold']
        stop_atr_mult = p['stop_atr_mult']
        tp_atr_mult = p['tp_atr_mult']
        asset_class = p['asset_class']

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
        resistance_breakout = resistance * (1 + breakout_threshold)
        support_breakout = support * (1 - breakout_threshold)
        
        # Check volume confirmation
        volume_confirmed = current_volume >= (avg_volume * volume_threshold)
        
        signal = Signal.HOLD
        confidence = 0.0
        reason = "No breakout detected"
        
        # Bullish breakout: price breaks above resistance with volume
        if current_price > resistance_breakout and volume_confirmed:
            signal = Signal.BUY
            # Confidence based on how far above resistance and volume strength
            breakout_strength = (current_price - resistance) / resistance
            volume_strength = min(current_volume / avg_volume / volume_threshold, 2.0)
            confidence = min(breakout_strength * 20 + volume_strength * 0.3, 0.85)
            reason = f"Bullish breakout [{asset_class}] above resistance {resistance:.2f} with volume confirmation"
        
        # Bearish breakout: price breaks below support with volume
        elif current_price < support_breakout and volume_confirmed:
            signal = Signal.SELL
            breakout_strength = (support - current_price) / support
            volume_strength = min(current_volume / avg_volume / volume_threshold, 2.0)
            confidence = min(breakout_strength * 20 + volume_strength * 0.3, 0.85)
            reason = f"Bearish breakout [{asset_class}] below support {support:.2f} with volume confirmation"
        
        # Calculate stop-loss and take-profit
        stop_loss = None
        take_profit = None
        
        if signal != Signal.HOLD:
            atr = self.indicators.atr(data)
            atr_value = atr.iloc[-1] if not atr.empty else current_price * 0.02

            if signal == Signal.BUY:
                # Stop loss below support, take profit at calibrated R:R
                stop_loss = support - (atr_value * stop_atr_mult)
                take_profit = current_price + (atr_value * stop_atr_mult * tp_atr_mult)
            else:  # SELL
                # Stop loss above resistance, take profit at calibrated R:R
                stop_loss = resistance + (atr_value * stop_atr_mult)
                take_profit = current_price - (atr_value * stop_atr_mult * tp_atr_mult)
        
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

