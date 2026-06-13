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
    """Moving average crossover trend following strategy with per-asset-class calibration."""

    @staticmethod
    def _detect_asset_class(symbol: str) -> str:
        """Classify a trading symbol into 'crypto', 'forex', 'commodities', or 'stocks'."""
        if not symbol:
            return 'stocks'
        raw = symbol.upper()
        # Commodities: OANDA metals format XAU_USD / XAG_USD or slash XAU/USD
        metal_bases = {'XAU', 'XAG', 'BCO', 'WTICO'}  # gold, silver, Brent, WTI
        stripped_base = raw.replace('_', '').replace('/', '').replace('-', '')[:3]
        if stripped_base in metal_bases:
            return 'commodities'
        # Crypto: slash notation BTC/USD, ETH/USD, AVAX/USD etc.
        if '/' in raw:
            base = raw.split('/')[0]
            crypto_bases = {'BTC', 'ETH', 'SOL', 'BNB', 'ADA', 'DOT', 'LINK',
                            'MATIC', 'AVAX', 'UNI', 'ATOM', 'XRP', 'DOGE', 'LTC'}
            if base in crypto_bases:
                return 'crypto'
        # Forex: underscore OANDA (EUR_USD) or 6-char code (EURUSD)
        forex_ccys = {'EUR', 'GBP', 'USD', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD'}
        stripped = raw.replace('_', '').replace('/', '').replace('-', '')
        if len(stripped) == 6 and stripped[:3] in forex_ccys and stripped[3:] in forex_ccys:
            return 'forex'
        return 'stocks'

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize trend following strategy.

        Config parameters (global — can be overridden per-asset-class at signal time):
            fast_ma_period: Fast MA period (default: 20)
            slow_ma_period: Slow MA period (default: 50)
            use_ema: Use EMA instead of SMA (default: False)
            stop_atr_multiple: ATR multiple for stop distance (default: 2.0)
            tp_rr_ratio: Take-profit R:R ratio (default: 3.0)
            adx_min_trend: Minimum ADX for crypto entries (default: 20)
            asset_class_overrides: Per-class overrides dict (see trading_config.yaml)
        """
        super().__init__("TrendFollowing", config)
        # Global defaults (may be overridden per-symbol inside generate_signal)
        self.fast_period    = self.config.get('fast_ma_period', 20)
        self.slow_period    = self.config.get('slow_ma_period', 50)
        self.use_ema        = self.config.get('use_ema', False)
        self.stop_atr_mult  = self.config.get('stop_atr_multiple', 2.0)
        self.tp_rr_ratio    = self.config.get('tp_rr_ratio', 3.0)
        self.adx_min_trend  = self.config.get('adx_min_trend', 20)
        self.indicators     = TechnicalIndicators()

    def _get_ac_params(self, symbol: str) -> Dict:
        """Return per-asset-class parameter overrides for this symbol."""
        asset_class = self._detect_asset_class(symbol)
        ac_overrides = self.config.get('asset_class_overrides', {}).get(asset_class, {})
        # Commodities fallback defaults (wider stops, slower MAs)
        _stop_default = 2.0 if asset_class == 'commodities' else self.stop_atr_mult
        _tp_default   = 2.5 if asset_class == 'commodities' else self.tp_rr_ratio
        _fast_default = 20  if asset_class == 'commodities' else self.fast_period
        _slow_default = 50  if asset_class == 'commodities' else self.slow_period
        _adx_default  = 22  if asset_class == 'commodities' else self.adx_min_trend
        return {
            'asset_class':   asset_class,
            'fast_period':   ac_overrides.get('fast_ma_period',  _fast_default),
            'slow_period':   ac_overrides.get('slow_ma_period',  _slow_default),
            'use_ema':       ac_overrides.get('use_ema',         True if asset_class == 'commodities' else self.use_ema),
            'stop_atr_mult': ac_overrides.get('stop_atr_multiple', _stop_default),
            'tp_rr_ratio':   ac_overrides.get('tp_rr_ratio',    _tp_default),
            'adx_min_trend': ac_overrides.get('adx_min_trend',  _adx_default),
        }

    def generate_signal(self, data: pd.DataFrame, symbol: str = "") -> Dict:
        """Generate signal based on MA crossover and trend strength.

        Args:
            data:   OHLCV DataFrame with at least slow_period rows.
            symbol: Trading symbol (e.g. 'BTC/USD', 'EUR_USD').
                    Used to select per-asset-class parameters.
        """
        # Resolve per-asset-class parameters for this symbol
        p = self._get_ac_params(symbol)
        fast_period   = p['fast_period']
        slow_period   = p['slow_period']
        use_ema       = p['use_ema']
        stop_atr_mult = p['stop_atr_mult']
        tp_rr_ratio   = p['tp_rr_ratio']
        adx_min_trend = p['adx_min_trend']
        asset_class   = p['asset_class']

        if len(data) < slow_period:
            return {
                'signal': Signal.HOLD, 'confidence': 0.0,
                'entry_price': data['close'].iloc[-1],
                'stop_loss': None, 'take_profit': None,
                'reason': 'Insufficient data'
            }

        # Calculate moving averages
        if use_ema:
            fast_ma = self.indicators.ema(data, fast_period)
            slow_ma = self.indicators.ema(data, slow_period)
        else:
            fast_ma = self.indicators.sma(data, fast_period)
            slow_ma = self.indicators.sma(data, slow_period)

        current_price = data['close'].iloc[-1]
        fast_current  = fast_ma.iloc[-1]
        slow_current  = slow_ma.iloc[-1]
        fast_prev     = fast_ma.iloc[-2] if len(fast_ma) > 1 else fast_current
        slow_prev     = slow_ma.iloc[-2] if len(slow_ma) > 1 else slow_current

        # ATR for stop/TP
        atr       = self.indicators.atr(data)
        atr_value = atr.iloc[-1] if not atr.empty else current_price * 0.02

        # ADX for trend strength
        adx       = self.indicators.adx(data)
        adx_value = adx.iloc[-1] if not adx.empty else 0

        # MACD for momentum confirmation
        macd        = self.indicators.macd(data)
        macd_value  = macd['macd'].iloc[-1]      if not macd.empty else 0
        macd_signal = macd['signal'].iloc[-1]    if not macd.empty else 0
        macd_hist   = macd['histogram'].iloc[-1] if not macd.empty else 0

        # For crypto and commodities: require a minimum trend strength before entering
        # Crypto needs strong trends (ADX >= 30) to avoid whipsaws in volatile markets
        # Commodities need some trend (ADX >= 22) to avoid news-spike false signals
        if asset_class in ('crypto', 'commodities') and adx_value < adx_min_trend:
            return {
                'signal': Signal.HOLD, 'confidence': 0.0,
                'entry_price': current_price,
                'stop_loss': None, 'take_profit': None,
                'reason': (f'ADX {adx_value:.1f} < {adx_min_trend} — '
                           f'trend too weak for {asset_class} entry '
                           f'(using EMA {fast_period}/{slow_period})')
            }

        # Crossover detection
        bullish_cross = (fast_prev <= slow_prev) and (fast_current > slow_current)
        bearish_cross = (fast_prev >= slow_prev) and (fast_current < slow_current)

        # Trend strength and slope
        trend_strength_raw = abs(fast_current - slow_current) / slow_current
        if len(fast_ma) >= 5:
            fast_slope = (fast_ma.iloc[-1] - fast_ma.iloc[-5]) / fast_ma.iloc[-5]
            slow_slope = (slow_ma.iloc[-1] - slow_ma.iloc[-5]) / slow_ma.iloc[-5]
        else:
            fast_slope = slow_slope = 0

        price_above_fast = (current_price - fast_current) / fast_current
        price_above_slow = (current_price - slow_current) / slow_current

        signal     = Signal.HOLD
        confidence = 0.0
        reason     = "No clear trend"

        if bullish_cross:
            signal = Signal.BUY
            conf_factors = [
                min(trend_strength_raw * 20, 0.3),
                min(adx_value / 50, 0.25),
                0.2 if macd_hist > 0 else 0,
                min(abs(fast_slope) * 20, 0.15),
                0.1,
            ]
            confidence = min(sum(conf_factors), 0.9)
            reason = (f"Bullish MA crossover [{asset_class}] "
                      f"(EMA{fast_period if use_ema else 'SMA'+str(fast_period)}: {fast_current:.2f} "
                      f"> {slow_current:.2f}, ADX: {adx_value:.1f})")

        elif bearish_cross:
            signal = Signal.SELL
            conf_factors = [
                min(trend_strength_raw * 20, 0.3),
                min(adx_value / 50, 0.25),
                0.2 if macd_hist < 0 else 0,
                min(abs(fast_slope) * 20, 0.15),
                0.1,
            ]
            confidence = min(sum(conf_factors), 0.9)
            reason = (f"Bearish MA crossover [{asset_class}] "
                      f"({fast_current:.2f} < {slow_current:.2f}, ADX: {adx_value:.1f})")

        elif fast_current > slow_current:
            signal = Signal.BUY
            conf_factors = [
                min(trend_strength_raw * 15, 0.25),
                min(adx_value / 60, 0.2) if adx_value > 25 else (0.1 if adx_value > 20 else 0.05),
                0.15 if (macd_value > macd_signal and macd_hist > 0) else (0.08 if macd_value > macd_signal else 0),
                min(abs(fast_slope) * 10, 0.15) if fast_slope > 0 and slow_slope > 0 else (0.08 if fast_slope > 0 else 0),
                min(price_above_fast * 5, 0.1) if price_above_fast > 0 else 0,
                0.15,
            ]
            confidence = min(sum(conf_factors), 0.75)
            reason = (f"Uptrend [{asset_class}] "
                      f"(Fast: {fast_current:.2f} > Slow: {slow_current:.2f}, ADX: {adx_value:.1f})")

        elif fast_current < slow_current:
            signal = Signal.SELL
            conf_factors = [
                min(trend_strength_raw * 15, 0.25),
                min(adx_value / 60, 0.2) if adx_value > 25 else (0.1 if adx_value > 20 else 0.05),
                0.15 if (macd_value < macd_signal and macd_hist < 0) else (0.08 if macd_value < macd_signal else 0),
                min(abs(fast_slope) * 10, 0.15) if fast_slope < 0 and slow_slope < 0 else (0.08 if fast_slope < 0 else 0),
                min(abs(price_above_fast) * 5, 0.1) if price_above_fast < 0 else 0,
                0.15,
            ]
            confidence = min(sum(conf_factors), 0.75)
            reason = (f"Downtrend [{asset_class}] "
                      f"(Fast: {fast_current:.2f} < Slow: {slow_current:.2f}, ADX: {adx_value:.1f})")

        # Calculate stop-loss and take-profit using per-class ATR multiples
        stop_loss = take_profit = None
        if signal != Signal.HOLD:
            if signal == Signal.BUY:
                stop_loss   = current_price - (atr_value * stop_atr_mult)
                take_profit = current_price + (atr_value * stop_atr_mult * tp_rr_ratio)
            else:
                stop_loss   = current_price + (atr_value * stop_atr_mult)
                take_profit = current_price - (atr_value * stop_atr_mult * tp_rr_ratio)

        return {
            'signal': signal,
            'confidence': confidence,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'reason': reason
        }

    def should_exit(self, data: pd.DataFrame, position: Dict, symbol: str = "") -> bool:
        """Check if position should be exited based on trend reversal."""
        p = self._get_ac_params(symbol)
        fast_period = p['fast_period']
        slow_period = p['slow_period']
        use_ema     = p['use_ema']

        if len(data) < slow_period:
            return False

        if use_ema:
            fast_ma = self.indicators.ema(data, fast_period)
            slow_ma = self.indicators.ema(data, slow_period)
        else:
            fast_ma = self.indicators.sma(data, fast_period)
            slow_ma = self.indicators.sma(data, slow_period)

        fast_current = fast_ma.iloc[-1]
        slow_current = slow_ma.iloc[-1]

        if position['side'] == 'buy' and fast_current < slow_current:
            return True
        if position['side'] == 'sell' and fast_current > slow_current:
            return True
        return False
