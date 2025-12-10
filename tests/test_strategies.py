"""Tests for trading strategies"""

import pytest
import pandas as pd
import numpy as np
from src.strategies.trend_following import TrendFollowingStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.base_strategy import Signal


def create_sample_data(length=100):
    """Create sample OHLCV data"""
    dates = pd.date_range(start='2023-01-01', periods=length, freq='1H')
    prices = 100 + np.cumsum(np.random.randn(length) * 0.5)
    
    data = pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.randint(1000, 10000, length)
    }, index=dates)
    
    return data


def test_trend_following_strategy():
    """Test trend following strategy"""
    strategy = TrendFollowingStrategy({'fast_ma_period': 10, 'slow_ma_period': 20})
    data = create_sample_data(50)
    
    signal = strategy.generate_signal(data)
    
    assert 'signal' in signal
    assert 'confidence' in signal
    assert signal['signal'] in [Signal.BUY, Signal.SELL, Signal.HOLD]


def test_mean_reversion_strategy():
    """Test mean reversion strategy"""
    strategy = MeanReversionStrategy({'lookback_period': 14})
    data = create_sample_data(50)
    
    signal = strategy.generate_signal(data)
    
    assert 'signal' in signal
    assert 'confidence' in signal
    assert signal['signal'] in [Signal.BUY, Signal.SELL, Signal.HOLD]

