"""Stop loss management"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta
from ..indicators.technical import TechnicalIndicators


class StopLossManager:
    """Manage stop-loss levels"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize stop loss manager
        
        Config parameters:
            stop_loss_atr_multiplier: Stop loss distance in ATR multiples (default: 2.0)
            trailing_stop_enabled: Enable trailing stop (default: True)
            trailing_stop_atr_multiplier: Trailing stop distance (default: 1.5)
            time_based_stop_hours: Exit if no movement after X hours (default: None)
        """
        self.config = config or {}
        # Prefer risk.* ATR multiple keys used in trading_config.yaml
        self.stop_loss_atr_multiplier = self.config.get(
            'stop_loss_atr_multiplier',
            self.config.get('default_stop_loss_atr_multiple', 2.0),
        )
        self.trailing_stop_enabled = self.config.get('trailing_stop_enabled', True)
        self.trailing_stop_atr_multiplier = self.config.get('trailing_stop_atr_multiplier', 1.5)
        self.trailing_activation_rr = float(self.config.get('trailing_activation_rr', 1.0))
        self.trailing_min_lock_rr = float(self.config.get('trailing_min_lock_rr', 0.5))
        self.time_based_stop_hours = self.config.get('time_based_stop_hours', None)
        self.indicators = TechnicalIndicators()
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        side: str,
        data: pd.DataFrame
    ) -> float:
        """
        Calculate stop loss level
        
        Args:
            entry_price: Entry price
            side: 'buy' or 'sell'
            data: Market data for ATR calculation
        
        Returns:
            Stop loss price
        """
        # Calculate ATR (with robust fallback when ATR is NaN due to short history)
        atr = self.indicators.atr(data)
        atr_value = atr.iloc[-1] if not atr.empty else np.nan

        # ATR may be NaN when there are not enough bars; fall back to 2% of price
        if pd.isna(atr_value) or atr_value <= 0:
            atr_value = entry_price * 0.02
        
        # Calculate stop loss distance
        stop_distance = atr_value * self.stop_loss_atr_multiplier
        
        # Set stop loss based on side
        if side == 'buy':
            stop_loss = entry_price - stop_distance
        else:  # sell
            stop_loss = entry_price + stop_distance
        
        return stop_loss
    
    def update_trailing_stop(
        self,
        current_price: float,
        entry_price: float,
        current_stop_loss: float,
        side: str,
        data: pd.DataFrame
    ) -> float:
        """
        Update trailing stop loss.

        Trail only activates after unrealized profit >= trailing_activation_rr * initial risk.
        Once trailing, the stop never locks in less than trailing_min_lock_rr * initial risk
        (avoids truncating winners into ~1:1 outcomes).
        """
        if not self.trailing_stop_enabled:
            return current_stop_loss

        initial_risk = abs(entry_price - current_stop_loss)
        # Prefer risk from original stop distance when stop already trailed past entry
        # Recompute risk from entry vs the farther of current stop and a synthetic stop
        if side == 'buy':
            risk_from_stop = max(entry_price - current_stop_loss, 0.0)
        else:
            risk_from_stop = max(current_stop_loss - entry_price, 0.0)

        # If stop already moved to/through breakeven, recover initial risk from ATR
        atr = self.indicators.atr(data)
        atr_value = atr.iloc[-1] if not atr.empty else np.nan
        if pd.isna(atr_value) or atr_value <= 0:
            atr_value = current_price * 0.02
        initial_risk = risk_from_stop if risk_from_stop > 0 else (atr_value * self.stop_loss_atr_multiplier)
        if initial_risk <= 0:
            return current_stop_loss

        if side == 'buy':
            unrealized_r = (current_price - entry_price) / initial_risk
        else:
            unrealized_r = (entry_price - current_price) / initial_risk

        if unrealized_r < self.trailing_activation_rr:
            return current_stop_loss

        trailing_distance = atr_value * self.trailing_stop_atr_multiplier
        min_lock = entry_price + (self.trailing_min_lock_rr * initial_risk) if side == 'buy' \
            else entry_price - (self.trailing_min_lock_rr * initial_risk)

        if side == 'buy':
            new_stop_loss = current_price - trailing_distance
            # Never lock below min_lock once activated
            new_stop_loss = max(new_stop_loss, min_lock)
            if new_stop_loss > current_stop_loss:
                return new_stop_loss
            return current_stop_loss
        else:
            new_stop_loss = current_price + trailing_distance
            new_stop_loss = min(new_stop_loss, min_lock)
            if new_stop_loss < current_stop_loss:
                return new_stop_loss
            return current_stop_loss
    
    def check_stop_loss(self, current_price: float, stop_loss: float, side: str) -> bool:
        """
        Check if stop loss should be triggered
        
        Args:
            current_price: Current market price
            stop_loss: Stop loss level
            side: 'buy' or 'sell'
        
        Returns:
            True if stop loss should be triggered
        """
        if side == 'buy':
            return current_price <= stop_loss
        else:  # sell
            return current_price >= stop_loss
    
    def check_time_based_exit(
        self,
        entry_time: datetime,
        current_price: float,
        entry_price: float
    ) -> bool:
        """
        Check if position should be exited based on time
        
        Args:
            entry_time: Time when position was opened
            current_price: Current market price
            entry_price: Entry price
        
        Returns:
            True if position should be exited
        """
        if self.time_based_stop_hours is None:
            return False
        
        time_elapsed = datetime.now() - entry_time
        hours_elapsed = time_elapsed.total_seconds() / 3600
        
        if hours_elapsed >= self.time_based_stop_hours:
            # Check if price has moved significantly
            price_change = abs(current_price - entry_price) / entry_price
            
            # Exit if price hasn't moved much (e.g., < 1%)
            if price_change < 0.01:
                return True
        
        return False
    
    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        side: str,
        risk_reward_ratio: float = 1.5
    ) -> float:
        """
        Calculate take profit level based on risk/reward ratio
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss level
            side: 'buy' or 'sell'
            risk_reward_ratio: Desired risk/reward ratio (default: 1.5)
        
        Returns:
            Take profit price
        """
        # Calculate risk
        risk = abs(entry_price - stop_loss)
        
        # Calculate reward
        reward = risk * risk_reward_ratio
        
        # Set take profit
        if side == 'buy':
            take_profit = entry_price + reward
        else:  # sell
            take_profit = entry_price - reward
        
        return take_profit
