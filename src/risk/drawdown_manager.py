"""Drawdown monitoring and kill switch"""

from typing import Dict, Optional
from datetime import datetime
import pandas as pd
import numpy as np


class DrawdownManager:
    """Monitor drawdown and implement kill switch"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize drawdown manager
        
        Config parameters:
            max_drawdown_percent: Maximum drawdown before kill switch (default: 10.0)
            peak_equity: Peak equity value
            current_equity: Current equity value
        """
        self.config = config or {}
        self.max_drawdown_percent = self.config.get('max_drawdown_percent', 10.0)
        self.peak_equity = self.config.get('peak_equity', None)
        self.current_equity = self.config.get('current_equity', None)
        self.kill_switch_active = False
        self.kill_switch_triggered_at = None
        self.equity_history = []
        self.position_reduction_factor = 1.0  # For progressive position reduction
    
    def update_equity(self, equity: float, timestamp: Optional[datetime] = None):
        """
        Update current equity and track peak
        
        Args:
            equity: Current equity value
            timestamp: Timestamp of update
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.current_equity = equity
        
        # Update peak equity
        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity
        
        # Record in history
        self.equity_history.append({
            'timestamp': timestamp,
            'equity': equity,
            'peak': self.peak_equity
        })
        
        # Enhanced progressive position reduction with exponential curve
        # For aggressive mode (4.5% max drawdown), reduce positions more aggressively
        drawdown = self.calculate_drawdown()
        
        if self.max_drawdown_percent <= 4.5:
            # Exponential reduction curve for smoother transitions
            # Formula: reduction = e^(-k * (1 - drawdown_ratio))
            # where k controls the steepness of the curve
            drawdown_ratio = drawdown / self.max_drawdown_percent if self.max_drawdown_percent > 0 else 0.0
            
            if drawdown_ratio >= 0.5:  # Start reducing at 50% of max drawdown
                # Exponential curve: more aggressive reduction as drawdown increases
                # At 50%: ~0.6x, at 75%: ~0.3x, at 90%: ~0.1x, at 100%: ~0.0x
                k = 3.0  # Steepness parameter
                self.position_reduction_factor = max(0.0, min(1.0, np.exp(-k * (drawdown_ratio - 0.5))))
            else:
                # No reduction when drawdown < 50% of limit
                self.position_reduction_factor = 1.0
            
            # Recovery mode: Gradually increase position size as drawdown recovers
            # Check if we're recovering from a drawdown
            if len(self.equity_history) >= 2:
                prev_equity = self.equity_history[-2]['equity']
                if equity > prev_equity and drawdown < self.max_drawdown_percent * 0.8:
                    # Gradually increase position size as we recover
                    # Recovery factor: 1.0 when drawdown < 80% of limit
                    recovery_ratio = 1.0 - (drawdown / (self.max_drawdown_percent * 0.8))
                    recovery_factor = min(1.2, 1.0 + (recovery_ratio * 0.2))  # Up to 20% bonus
                    self.position_reduction_factor = min(1.0, self.position_reduction_factor * recovery_factor)
        else:
            self.position_reduction_factor = 1.0
    
    def calculate_drawdown(self) -> float:
        """
        Calculate current drawdown percentage
        
        Returns:
            Drawdown percentage (positive value)
        """
        if self.peak_equity is None or self.current_equity is None:
            return 0.0
        
        if self.peak_equity == 0:
            return 0.0
        
        drawdown = ((self.peak_equity - self.current_equity) / self.peak_equity) * 100
        
        return max(0.0, drawdown)
    
    def check_kill_switch(self) -> bool:
        """
        Check if kill switch should be activated
        
        Returns:
            True if kill switch should be active
        """
        drawdown = self.calculate_drawdown()
        
        if drawdown >= self.max_drawdown_percent and not self.kill_switch_active:
            self.kill_switch_active = True
            self.kill_switch_triggered_at = datetime.now()
            return True
        
        return self.kill_switch_active
    
    def is_trading_allowed(self) -> bool:
        """
        Check if trading is currently allowed
        
        Returns:
            True if trading is allowed, False if kill switch is active
        """
        return not self.kill_switch_active
    
    def reset_kill_switch(self):
        """Reset kill switch (manual intervention required)"""
        self.kill_switch_active = False
        self.kill_switch_triggered_at = None
        # Optionally reset peak equity to current equity
        if self.current_equity:
            self.peak_equity = self.current_equity
    
    def check_and_reset_stale_state(self, current_equity: Optional[float] = None):
        """
        Check for stale kill switch state and reset if needed
        
        This helps unblock trading when the kill switch is stuck due to stale data.
        
        Args:
            current_equity: Current equity value (if None, uses self.current_equity)
        
        Returns:
            True if state was reset, False otherwise
        """
        if current_equity is None:
            current_equity = self.current_equity
        
        if current_equity is None:
            return False
        
        # If kill switch is active, check if it's stale
        if self.kill_switch_active:
            drawdown = self.calculate_drawdown()
            
            # If drawdown is below threshold, reset kill switch
            if drawdown < self.max_drawdown_percent * 0.8:  # Reset when drawdown drops below 80% of limit
                print(f"DrawdownManager: Auto-resetting kill switch (drawdown {drawdown:.2f}% < {self.max_drawdown_percent * 0.8:.2f}%)")
                self.reset_kill_switch()
                return True
            
            # If peak equity is stale (more than 1 day old), reset it
            if self.peak_equity and len(self.equity_history) > 0:
                peak_entry = next((e for e in reversed(self.equity_history) if e['equity'] == self.peak_equity), None)
                if peak_entry:
                    days_since_peak = (datetime.now() - peak_entry['timestamp']).total_seconds() / 86400
                    if days_since_peak > 1.0:
                        print(f"DrawdownManager: Resetting stale peak equity (peak from {days_since_peak:.1f} days ago)")
                        self.peak_equity = current_equity
                        # Recalculate drawdown - if it's now below threshold, reset kill switch
                        drawdown = self.calculate_drawdown()
                        if drawdown < self.max_drawdown_percent:
                            self.reset_kill_switch()
                            return True
        
        return False
    
    def get_status(self) -> Dict:
        """
        Get current drawdown status
        
        Returns:
            Dictionary with status information
        """
        drawdown = self.calculate_drawdown()
        
        return {
            'current_equity': self.current_equity,
            'peak_equity': self.peak_equity,
            'drawdown_percent': drawdown,
            'max_drawdown_percent': self.max_drawdown_percent,
            'kill_switch_active': self.kill_switch_active,
            'kill_switch_triggered_at': self.kill_switch_triggered_at,
            'trading_allowed': self.is_trading_allowed()
        }
    
    def get_equity_curve(self) -> pd.DataFrame:
        """
        Get equity curve as DataFrame
        
        Returns:
            DataFrame with equity history
        """
        if not self.equity_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.equity_history)
        df.set_index('timestamp', inplace=True)
        
        return df

