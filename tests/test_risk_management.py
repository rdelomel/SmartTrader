"""Tests for risk management"""

import pytest
import pandas as pd
from src.risk.position_sizer import PositionSizer
from src.risk.stop_loss import StopLossManager
from src.risk.drawdown_manager import DrawdownManager


def test_position_sizing():
    """Test position sizing"""
    sizer = PositionSizer({'position_size_percent': 1.5})
    
    result = sizer.calculate_position_size(
        account_balance=10000,
        entry_price=100,
        stop_loss=98
    )
    
    assert 'quantity' in result
    assert result['quantity'] > 0
    assert result['risk_amount'] > 0


def test_stop_loss_calculation():
    """Test stop loss calculation"""
    manager = StopLossManager({'stop_loss_atr_multiplier': 2.0})
    
    data = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [101, 102, 103],
        'low': [99, 100, 101],
        'close': [100, 101, 102],
        'volume': [1000, 1100, 1200]
    })
    
    stop_loss = manager.calculate_stop_loss(
        entry_price=100,
        side='buy',
        data=data
    )
    
    assert stop_loss < 100  # Stop loss should be below entry for long


def test_drawdown_manager():
    """Test drawdown manager"""
    manager = DrawdownManager({'max_drawdown_percent': 10.0})
    
    manager.update_equity(10000)
    manager.update_equity(9500)  # 5% drawdown
    
    drawdown = manager.calculate_drawdown()
    assert drawdown == 5.0
    
    manager.update_equity(9000)  # 10% drawdown
    assert manager.check_kill_switch() == True
    assert manager.is_trading_allowed() == False

