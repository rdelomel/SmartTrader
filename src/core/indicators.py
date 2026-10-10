"""Causal indicators. A value at bar t only uses data up to and including bar t."""

import numpy as np
import pandas as pd


def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False, min_periods=n).mean()


def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df['close'].shift(1)
    ranges = pd.concat(
        [df['high'] - df['low'], (df['high'] - prev_close).abs(), (df['low'] - prev_close).abs()],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, n: int = 20) -> pd.Series:
    return true_range(df).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain / loss
        out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(loss != 0, 100.0)
    return out.where(gain.notna())


def donchian_high(df: pd.DataFrame, n: int) -> pd.Series:
    """Highest high of the previous n bars (excludes the current bar)."""
    return df['high'].rolling(n, min_periods=n).max().shift(1)


def donchian_low(df: pd.DataFrame, n: int) -> pd.Series:
    """Lowest low of the previous n bars (excludes the current bar)."""
    return df['low'].rolling(n, min_periods=n).min().shift(1)
