"""Historical OHLCV loading with an on-disk cache.

Sources:
- OANDA (forex/metals) when a connected OANDABroker is supplied; spot mid prices.
- Yahoo Finance otherwise. Metals fall back to COMEX futures (GC=F / SI=F) as a proxy.
"""

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

CACHE_DIR = os.path.join('data', 'history')
_METAL_PROXY = {'XAU_USD': 'GC=F', 'XAG_USD': 'SI=F'}


def yahoo_ticker(symbol: str, asset_class: str) -> str:
    s = symbol.upper().replace('/', '_')
    if asset_class == 'forex':
        base, quote = s.split('_')
        return f'{base}{quote}=X'
    if asset_class == 'metals':
        return _METAL_PROXY[s]
    if asset_class == 'crypto':
        return s.replace('_', '-')
    return s


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=str.lower)[['open', 'high', 'low', 'close', 'volume']].copy()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert('UTC').tz_localize(None)
    df.index = idx
    df = df[~df.index.duplicated(keep='last')].sort_index()
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    return df[(df['close'] > 0) & (df['high'] >= df['low'])]


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    return df.resample(rule).agg(agg).dropna(subset=['open', 'close'])


class HistoryProvider:
    def __init__(self, cache_dir: str = CACHE_DIR, max_age_hours: float = 20.0, oanda_broker=None):
        self.cache_dir = cache_dir
        self.max_age_hours = max_age_hours
        self.oanda = oanda_broker
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, symbol: str, timeframe: str) -> str:
        safe = symbol.upper().replace('/', '_')
        return os.path.join(self.cache_dir, f'{safe}_{timeframe}.csv')

    def load(self, symbol: str, asset_class: str, timeframe: str, years: float) -> pd.DataFrame:
        path = self._cache_path(symbol, timeframe)
        if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < self.max_age_hours * 3600:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            df = self._fetch(symbol, asset_class, timeframe, years)
            if not df.empty:
                df.to_csv(path)
        if df.empty:
            return df
        cutoff = df.index[-1] - pd.Timedelta(days=int(years * 365.25))
        return df[df.index >= cutoff]

    def _fetch(self, symbol: str, asset_class: str, timeframe: str, years: float) -> pd.DataFrame:
        if self.oanda is not None and asset_class in ('forex', 'metals'):
            df = self._fetch_oanda(symbol, timeframe, years)
            if not df.empty:
                return df
        return self._fetch_yahoo(symbol, asset_class, timeframe, years)

    def _fetch_yahoo(self, symbol: str, asset_class: str, timeframe: str, years: float) -> pd.DataFrame:
        import yfinance as yf

        ticker = yf.Ticker(yahoo_ticker(symbol, asset_class))
        if timeframe == '1d':
            raw = ticker.history(period=f'{max(1, int(years + 1))}y', interval='1d', auto_adjust=True)
            return _clean(raw) if not raw.empty else pd.DataFrame()
        if timeframe in ('1h', '4h'):
            raw = ticker.history(period='729d', interval='1h', auto_adjust=True)
            if raw.empty:
                return pd.DataFrame()
            df = _clean(raw)
            return _resample(df, '4h') if timeframe == '4h' else df
        raise ValueError(f'Unsupported timeframe {timeframe}')

    def _fetch_oanda(self, symbol: str, timeframe: str, years: float) -> pd.DataFrame:
        """Pages OANDA candles in windows of < 5000 bars."""
        step = {'1d': timedelta(days=4000), '4h': timedelta(days=600), '1h': timedelta(days=150)}[timeframe]
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=int(years * 365.25))
        frames = []
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + step, end)
            df = self.oanda.get_historical_data(symbol, timeframe, start_date=cursor, end_date=chunk_end)
            if df is not None and not df.empty:
                frames.append(df)
            cursor = chunk_end
        if not frames:
            return pd.DataFrame()
        return _clean(pd.concat(frames))


def latest_closed_bars(df: pd.DataFrame, timeframe: str, now: Optional[datetime] = None) -> pd.DataFrame:
    """Drops a trailing bar that has not closed yet (live use)."""
    if df.empty:
        return df
    now = (now or datetime.now(timezone.utc)).replace(tzinfo=None)
    span = {'1d': pd.Timedelta(days=1), '4h': pd.Timedelta(hours=4), '1h': pd.Timedelta(hours=1)}[timeframe]
    return df[df.index + span <= now]
