"""Rule-based strategies.

Each strategy turns OHLCV plus a parameter dict into a SignalFrame. Signals are
decided on a bar's close; the backtester and live engine act on the next open.
"""

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
import pandas as pd

from . import indicators as ind


@dataclass
class SignalFrame:
    entry: np.ndarray        # int8: +1 long, -1 short, 0 none
    exit_long: np.ndarray    # bool
    exit_short: np.ndarray   # bool
    stop_dist: np.ndarray    # price distance from entry to initial stop
    atr: np.ndarray
    warmup: int
    trail_atr: float = 0.0
    take_profit_r: float = 0.0
    max_bars: int = 0


def _frame(entry_long, entry_short, exit_long, exit_short, stop_dist, atr_s, warmup, **kw) -> SignalFrame:
    entry = np.where(entry_long.fillna(False).to_numpy(bool), 1, 0) - np.where(
        entry_short.fillna(False).to_numpy(bool), 1, 0
    )
    return SignalFrame(
        entry=entry.astype(np.int8),
        exit_long=exit_long.fillna(False).to_numpy(bool),
        exit_short=exit_short.fillna(False).to_numpy(bool),
        stop_dist=stop_dist.to_numpy(float),
        atr=atr_s.to_numpy(float),
        warmup=int(warmup),
        **kw,
    )


def donchian_trend(df: pd.DataFrame, p: Dict) -> SignalFrame:
    entry_n, exit_n = int(p['entry_n']), int(p['exit_n'])
    trend_n = int(p.get('trend_ema', 0))
    a = ind.atr(df, 20)
    c = df['close']
    long_e = c > ind.donchian_high(df, entry_n)
    short_e = c < ind.donchian_low(df, entry_n)
    if trend_n >= 50:
        t = ind.ema(c, trend_n)
        long_e &= c > t
        short_e &= c < t
    return _frame(
        long_e, short_e,
        exit_long=c < ind.donchian_low(df, exit_n),
        exit_short=c > ind.donchian_high(df, exit_n),
        stop_dist=float(p['stop_atr']) * a,
        atr_s=a,
        warmup=max(entry_n, trend_n, 20) + 1,
        trail_atr=float(p.get('trail_atr', 0.0)),
    )


def ema_trend(df: pd.DataFrame, p: Dict) -> SignalFrame:
    fast_n, slow_n = int(p['fast']), int(p['slow'])
    a = ind.atr(df, 20)
    c = df['close']
    fast, slow = ind.ema(c, fast_n), ind.ema(c, slow_n)
    above = fast > slow
    prev_above = above.shift(1).fillna(False).astype(bool)
    valid = fast.notna() & slow.notna()
    return _frame(
        valid & above & ~prev_above,
        valid & ~above & prev_above,
        exit_long=valid & ~above,
        exit_short=valid & above,
        stop_dist=float(p['stop_atr']) * a,
        atr_s=a,
        warmup=slow_n + 1,
        trail_atr=float(p.get('trail_atr', 0.0)),
    )


def rsi_reversion(df: pd.DataFrame, p: Dict) -> SignalFrame:
    a = ind.atr(df, 20)
    c = df['close']
    r = ind.rsi(c, int(p['rsi_n']))
    trend = ind.ema(c, int(p['trend_ema']))
    exit_ma = ind.sma(c, int(p['exit_ma']))
    return _frame(
        (r < float(p['lower'])) & (c > trend),
        (r > float(p['upper'])) & (c < trend),
        exit_long=c > exit_ma,
        exit_short=c < exit_ma,
        stop_dist=float(p['stop_atr']) * a,
        atr_s=a,
        warmup=int(p['trend_ema']) + 1,
        max_bars=int(p['max_bars']),
    )


def normalize_params(name: str, p: Dict) -> Dict:
    """Enforce cross-parameter constraints the bounds alone cannot express."""
    p = dict(p)
    if name == 'donchian_trend':
        p['exit_n'] = int(min(p['exit_n'], p['entry_n'] - 1))
    elif name == 'ema_trend':
        if p['fast'] >= p['slow']:
            p['fast'] = max(2, int(p['slow'] // 3))
    return p


STRATEGIES: Dict[str, Callable[[pd.DataFrame, Dict], SignalFrame]] = {
    'donchian_trend': donchian_trend,
    'ema_trend': ema_trend,
    'rsi_reversion': rsi_reversion,
}


def generate(name: str, df: pd.DataFrame, params: Dict) -> SignalFrame:
    return STRATEGIES[name](df, normalize_params(name, params))
