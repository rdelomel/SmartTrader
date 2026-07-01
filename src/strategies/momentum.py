"""Momentum trading strategy with per-asset-class calibration"""

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


class MomentumStrategy(BaseStrategy):
    """Momentum strategy using RSI, MACD, and price momentum — per-asset-class calibrated."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Config parameters (global defaults — overridden per-symbol at signal time):
            rsi_period, rsi_oversold, rsi_overbought,
            macd_fast, macd_slow, macd_signal,
            momentum_period, min_momentum, stop_atr_multiple, tp_rr_ratio.
            asset_class_overrides: Per-class override dict (see trading_config.yaml).
        """
        super().__init__("Momentum", config)
        self.rsi_period     = self.config.get('rsi_period',      14)
        self.rsi_oversold   = self.config.get('rsi_oversold',    30)
        self.rsi_overbought = self.config.get('rsi_overbought',  70)
        self.macd_fast      = self.config.get('macd_fast',       12)
        self.macd_slow      = self.config.get('macd_slow',       26)
        self.macd_signal_p  = self.config.get('macd_signal',      9)
        self.momentum_period = self.config.get('momentum_period', 10)
        self.min_momentum   = self.config.get('min_momentum',   0.02)
        self.stop_atr_mult  = self.config.get('stop_atr_multiple', 2.0)
        self.tp_rr_ratio    = self.config.get('tp_rr_ratio',     3.0)
        self.indicators     = TechnicalIndicators()

    def _get_ac_params(self, symbol: str) -> Dict:
        """Return per-asset-class parameter overrides for this symbol."""
        asset_class = TrendFollowingStrategy._detect_asset_class(symbol)
        ac = self.config.get('asset_class_overrides', {}).get(asset_class, {})
        # Commodities fallback defaults — metals behave like slow-moving forex
        _rsi_os_def  = 32   if asset_class == 'commodities' else self.rsi_oversold
        _rsi_ob_def  = 68   if asset_class == 'commodities' else self.rsi_overbought
        _mom_def     = 0.008 if asset_class == 'commodities' else self.min_momentum
        _stop_def    = 2.0  if asset_class == 'commodities' else self.stop_atr_mult
        _tp_def      = 2.5  if asset_class == 'commodities' else self.tp_rr_ratio
        return {
            'asset_class':    asset_class,
            'rsi_oversold':   ac.get('rsi_oversold',    _rsi_os_def),
            'rsi_overbought': ac.get('rsi_overbought',  _rsi_ob_def),
            'min_momentum':   ac.get('min_momentum',    _mom_def),
            'stop_atr_mult':  ac.get('stop_loss_atr_multiple', _stop_def),
            'tp_rr_ratio':    ac.get('take_profit_rr_ratio', _tp_def),
        }

    def generate_signal(self, data: pd.DataFrame, symbol: str = "") -> Dict:
        """Generate signal based on RSI, MACD and price momentum.

        Args:
            data:   OHLCV DataFrame.
            symbol: Trading symbol — used to select per-asset-class parameters.
        """
        p = self._get_ac_params(symbol)
        rsi_oversold   = p['rsi_oversold']
        rsi_overbought = p['rsi_overbought']
        min_momentum   = p['min_momentum']
        stop_atr_mult  = p['stop_atr_mult']
        tp_rr_ratio    = p['tp_rr_ratio']
        asset_class    = p['asset_class']

        min_bars = max(self.macd_slow, self.rsi_period, self.momentum_period) + 5
        if len(data) < min_bars:
            return {
                'signal': Signal.HOLD, 'confidence': 0.0,
                'entry_price': data['close'].iloc[-1],
                'stop_loss': None, 'take_profit': None,
                'reason': 'Insufficient data'
            }

        current_price = data['close'].iloc[-1]

        # RSI
        rsi      = self.indicators.rsi(data, period=self.rsi_period)
        rsi_cur  = rsi.iloc[-1] if not rsi.empty else 50
        rsi_prev = rsi.iloc[-2] if len(rsi) > 1 else rsi_cur

        # MACD
        macd_data  = self.indicators.macd(data, fast_period=self.macd_fast,
                                          slow_period=self.macd_slow,
                                          signal_period=self.macd_signal_p)
        macd_line  = macd_data['macd'].iloc[-1]      if 'macd'      in macd_data.columns else 0
        macd_sig   = macd_data['signal'].iloc[-1]    if 'signal'    in macd_data.columns else 0
        macd_hist  = macd_data['histogram'].iloc[-1] if 'histogram' in macd_data.columns else 0
        macd_prev_h = macd_data['histogram'].iloc[-2] if (len(macd_data) > 1 and
                                                          'histogram' in macd_data.columns) else macd_hist

        # Price momentum
        momentum = ((current_price - data['close'].iloc[-self.momentum_period])
                    / data['close'].iloc[-self.momentum_period])

        # ATR for stops
        atr       = self.indicators.atr(data)
        atr_value = atr.iloc[-1] if not atr.empty else current_price * 0.02

        signal     = Signal.HOLD
        confidence = 0.0
        reason     = "No momentum signal"

        bullish_conditions = [
            rsi_cur > 50 and rsi_cur > rsi_prev,                           # RSI rising above neutral
            macd_line > macd_sig and macd_hist > macd_prev_h,              # MACD bullish momentum
            momentum > min_momentum,                                        # Price momentum threshold
            rsi_cur < rsi_overbought,                                      # Not yet overbought
        ]
        bearish_conditions = [
            rsi_cur < 50 and rsi_cur < rsi_prev,                           # RSI falling below neutral
            macd_line < macd_sig and macd_hist < macd_prev_h,              # MACD bearish momentum
            momentum < -min_momentum,                                       # Negative momentum threshold
            rsi_cur > rsi_oversold,                                        # Not yet oversold
        ]

        bullish_score = sum(bullish_conditions)
        bearish_score = sum(bearish_conditions)

        if bullish_score >= 3:
            signal     = Signal.BUY
            confidence = min(bullish_score / 4.0 * 0.8 + abs(momentum) * 5, 0.85)
            reason     = (f"Bullish momentum [{asset_class}]: "
                          f"RSI={rsi_cur:.1f}, MACD={macd_hist:.4f}, Momentum={momentum*100:.2f}%")
        elif bearish_score >= 3:
            signal     = Signal.SELL
            confidence = min(bearish_score / 4.0 * 0.8 + abs(momentum) * 5, 0.85)
            reason     = (f"Bearish momentum [{asset_class}]: "
                          f"RSI={rsi_cur:.1f}, MACD={macd_hist:.4f}, Momentum={momentum*100:.2f}%")

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
            'reason': reason,
            'rsi': rsi_cur,
            'macd_histogram': macd_hist,
            'momentum': momentum
        }

    def should_exit(self, data: pd.DataFrame, position: Dict, symbol: str = "") -> bool:
        """Check if position should be exited (momentum reversal)."""
        if len(data) < max(self.macd_slow, self.rsi_period):
            return False

        p = self._get_ac_params(symbol)
        rsi_oversold   = p['rsi_oversold']
        rsi_overbought = p['rsi_overbought']

        rsi      = self.indicators.rsi(data, period=self.rsi_period)
        rsi_cur  = rsi.iloc[-1] if not rsi.empty else 50
        macd_data = self.indicators.macd(data, fast_period=self.macd_fast,
                                         slow_period=self.macd_slow,
                                         signal_period=self.macd_signal_p)
        macd_hist = macd_data['histogram'].iloc[-1] if 'histogram' in macd_data.columns else 0

        if position['side'] == 'buy' and (rsi_cur > rsi_overbought or macd_hist < 0):
            return True
        if position['side'] == 'sell' and (rsi_cur < rsi_oversold or macd_hist > 0):
            return True
        return False
