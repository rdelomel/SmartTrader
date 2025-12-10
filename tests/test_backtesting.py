"""Tests for backtesting"""

import pytest
import pandas as pd
import numpy as np
from src.backtesting.engine import BacktestEngine
from src.strategies.trend_following import TrendFollowingStrategy


def create_backtest_data(length=200):
    """Create sample data for backtesting"""
    dates = pd.date_range(start='2023-01-01', periods=length, freq='1H')
    trend = np.linspace(100, 110, length)
    noise = np.random.randn(length) * 0.5
    prices = trend + noise
    
    data = pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.randint(1000, 10000, length)
    }, index=dates)
    
    return data


def test_backtest_engine():
    """Test backtesting engine"""
    engine = BacktestEngine({'initial_capital': 10000})
    strategy = TrendFollowingStrategy({'fast_ma_period': 10, 'slow_ma_period': 20})
    data = create_backtest_data(200)
    
    results = engine.run_backtest(data, strategy)
    
    assert 'equity_curve' in results
    assert 'trades' in results
    assert 'metrics' in results
    assert results['final_capital'] > 0

