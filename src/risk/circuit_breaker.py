"""Real-time circuit breakers for aggressive trading"""

from typing import Dict, Optional
from datetime import datetime, timedelta
from collections import deque


class CircuitBreaker:
    """Real-time circuit breakers with multi-timeframe loss limits"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize circuit breaker
        
        Config parameters:
            max_daily_loss_percent: Max loss in 24-hour period (default: 1.5)
            max_hourly_loss_percent: Max loss in 1-hour period (default: 0.5)
            max_drawdown_percent: Max absolute drawdown (default: 4.5)
            cooling_period_hours: Hours to wait after trigger (default: 4)
        """
        self.config = config or {}
        self.max_daily_loss = self.config.get('max_daily_loss_percent', 1.5)
        self.max_hourly_loss = self.config.get('max_hourly_loss_percent', 0.5)
        self.max_drawdown = self.config.get('max_drawdown_percent', 4.5)
        self.cooling_period_hours = self.config.get('cooling_period_hours', 4)
        
        # Track equity history by time period
        self.equity_history = deque(maxlen=1000)  # Store (timestamp, equity) tuples
        self.peak_equity = None
        self.peak_equity_time = None
        self.triggered = False
        self.triggered_at = None
        self.trigger_reason = None
    
    def update_equity(self, equity: float, timestamp: Optional[datetime] = None):
        """
        Update equity and check circuit breakers
        
        Args:
            equity: Current equity value
            timestamp: Timestamp of update (default: now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Add to history
        self.equity_history.append((timestamp, equity))
        
        # Update peak equity
        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity
            self.peak_equity_time = timestamp
        elif not hasattr(self, 'peak_equity_time'):
            self.peak_equity_time = timestamp
        
        # Auto-reset peak equity if it's stale and causing false drawdowns
        if self.peak_equity and hasattr(self, 'peak_equity_time') and self.peak_equity_time:
            days_since_peak = (timestamp - self.peak_equity_time).total_seconds() / 86400
            current_drawdown = ((self.peak_equity - equity) / self.peak_equity) * 100 if self.peak_equity > 0 else 0.0
            
            # Reset if peak is more than 1 day old AND causing significant drawdown (>5%)
            # This prevents stale peaks from blocking trading
            if days_since_peak > 1 and current_drawdown > 5.0:
                print(f"Circuit Breaker: Auto-resetting stale peak in update_equity (peak from {days_since_peak:.1f} days ago, drawdown: {current_drawdown:.2f}%)")
                self.peak_equity = equity
                self.peak_equity_time = timestamp
            # Reset if we've recovered to within 1% of peak (new high)
            elif equity >= self.peak_equity * 0.99:
                self.peak_equity = equity
                self.peak_equity_time = timestamp
        
        # Check if cooling period has passed
        if self.triggered and self.triggered_at:
            hours_since_trigger = (timestamp - self.triggered_at).total_seconds() / 3600
            if hours_since_trigger >= self.cooling_period_hours:
                self.triggered = False
                self.triggered_at = None
                self.trigger_reason = None
                print(f"Circuit breaker cooling period ended. Trading resumed.")
    
    def check_limits(self, current_equity: float, timestamp: Optional[datetime] = None,
                    volatility: Optional[float] = None) -> Dict:
        """
        Check all circuit breaker limits with adaptive adjustments
        
        Args:
            current_equity: Current equity value
            timestamp: Current timestamp (default: now)
            volatility: Current market volatility (optional, for adaptive limits)
        
        Returns:
            Dictionary with limit check results
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # If already triggered and in cooling period, return triggered status
        if self.triggered:
            return {
                'triggered': True,
                'reason': self.trigger_reason,
                'cooling_until': self.triggered_at + timedelta(hours=self.cooling_period_hours) if self.triggered_at else None
            }
        
        # Calculate current drawdown for adaptive limits
        current_drawdown = 0.0
        if self.peak_equity and self.peak_equity > 0:
            current_drawdown = ((self.peak_equity - current_equity) / self.peak_equity) * 100
        
        # Adaptive limits: tighten when drawdown > 3% (approaching 4.5% limit)
        # Relax slightly when performing well (drawdown < 1%)
        adaptive_daily_multiplier = 1.0
        adaptive_hourly_multiplier = 1.0
        
        if current_drawdown > 3.0:
            # Tighten limits as we approach max drawdown
            tightness = (current_drawdown - 3.0) / 1.5  # Scale from 0 to 1 as drawdown goes from 3% to 4.5%
            adaptive_daily_multiplier = 1.0 - (tightness * 0.3)  # Reduce by up to 30%
            adaptive_hourly_multiplier = 1.0 - (tightness * 0.4)  # Reduce by up to 40%
        elif current_drawdown < 1.0:
            # Relax limits slightly when performing well
            adaptive_daily_multiplier = 1.1  # Allow 10% more
            adaptive_hourly_multiplier = 1.15  # Allow 15% more
        
        # Volatility-adjusted limits (tighter in high volatility)
        if volatility is not None and volatility > 0.02:  # High volatility threshold (2%)
            vol_adjustment = min(volatility / 0.05, 1.5)  # Scale adjustment up to 1.5x for very high vol
            adaptive_daily_multiplier *= (1.0 / vol_adjustment)
            adaptive_hourly_multiplier *= (1.0 / vol_adjustment)
        
        # Apply adaptive multipliers
        effective_daily_limit = self.max_daily_loss * adaptive_daily_multiplier
        effective_hourly_limit = self.max_hourly_loss * adaptive_hourly_multiplier
        
        # Check hourly loss with adaptive limit
        # Note: loss_pct is positive when there's a loss, limit may be negative
        # We trigger when loss exceeds the absolute value of the limit
        hourly_loss = self._check_hourly_loss(current_equity, timestamp)
        hourly_loss_pct = hourly_loss['loss_pct']
        hourly_limit_abs = abs(effective_hourly_limit)
        
        # Only trigger if there's an actual loss (positive) that exceeds the limit
        if hourly_loss_pct > 0 and hourly_loss_pct >= hourly_limit_abs:
            self._trigger(f"Hourly loss limit exceeded: {hourly_loss_pct:.2f}% (limit: {hourly_limit_abs:.2f}%)", timestamp)
            return {
                'triggered': True,
                'reason': f"Hourly loss limit: {hourly_loss_pct:.2f}%",
                'limit': hourly_limit_abs,
                'adaptive': True
            }
        
        # Check daily loss with adaptive limit
        # Same logic: only trigger on actual losses that exceed the limit
        daily_loss = self._check_daily_loss(current_equity, timestamp)
        daily_loss_pct = daily_loss['loss_pct']
        daily_limit_abs = abs(effective_daily_limit)
        
        if daily_loss_pct > 0 and daily_loss_pct >= daily_limit_abs:
            self._trigger(f"Daily loss limit exceeded: {daily_loss_pct:.2f}% (limit: {daily_limit_abs:.2f}%)", timestamp)
            return {
                'triggered': True,
                'reason': f"Daily loss limit: {daily_loss_pct:.2f}%",
                'limit': daily_limit_abs,
                'adaptive': True
            }
        
        # Check drawdown
        if self.peak_equity and self.peak_equity > 0:
            # Log current drawdown status for debugging
            print(f"Circuit Breaker: Peak equity: ${self.peak_equity:,.2f}, Current equity: ${current_equity:,.2f}, Drawdown: {current_drawdown:.2f}%")
            
            # Reset peak equity if it's stale or if we've recovered
            # IMPORTANT: Check reset conditions BEFORE checking if drawdown exceeds limit
            should_reset = False
            reset_reason = ""
            
            if hasattr(self, 'peak_equity_time') and self.peak_equity_time:
                days_since_peak = (timestamp - self.peak_equity_time).total_seconds() / 86400
                
                # Aggressive reset: If drawdown exceeds max AND peak is stale (even slightly), reset it
                # This prevents stale peaks from blocking trading
                if current_drawdown >= self.max_drawdown:
                    # If drawdown exceeds limit, check if peak is stale
                    if days_since_peak > 0.5:  # More than 12 hours old
                        should_reset = True
                        reset_reason = f"drawdown ({current_drawdown:.2f}%) exceeds limit ({self.max_drawdown:.2f}%) and peak is stale ({days_since_peak:.1f} days old)"
                    elif days_since_peak > 1 and current_drawdown > 5.0:
                        should_reset = True
                        reset_reason = f"stale peak ({days_since_peak:.1f} days old) causing significant drawdown ({current_drawdown:.2f}%)"
                elif days_since_peak > 1 and current_drawdown > 5.0:
                    should_reset = True
                    reset_reason = f"stale peak ({days_since_peak:.1f} days old) causing significant drawdown ({current_drawdown:.2f}%)"
                elif current_drawdown < 5.0 and current_equity >= self.peak_equity * 0.95:
                    should_reset = True
                    reset_reason = "account recovered to within 5% of peak"
            else:
                # No peak_equity_time set - likely stale peak from before this feature
                # If drawdown is high, reset it to allow trading
                if current_drawdown >= self.max_drawdown:
                    should_reset = True
                    reset_reason = f"drawdown ({current_drawdown:.2f}%) exceeds limit ({self.max_drawdown:.2f}%) and peak has no timestamp (stale)"
            
            if should_reset:
                print(f"Circuit Breaker: Auto-resetting peak equity - {reset_reason}")
                print(f"  Old peak: ${self.peak_equity:,.2f} -> New peak: ${current_equity:,.2f}")
                self.peak_equity = current_equity
                if not hasattr(self, 'peak_equity_time') or not self.peak_equity_time:
                    self.peak_equity_time = timestamp
                else:
                    self.peak_equity_time = timestamp
                current_drawdown = 0.0
                print(f"Circuit Breaker: ✅ Reset complete - trading can resume")
            elif current_drawdown >= self.max_drawdown:
                # Only trigger if we didn't reset
                self._trigger(f"Max drawdown exceeded: {current_drawdown:.2f}%", timestamp)
                return {
                    'triggered': True,
                    'reason': f"Max drawdown: {current_drawdown:.2f}%",
                    'limit': self.max_drawdown
                }
        
        return {
            'triggered': False,
            'hourly_loss': hourly_loss['loss_pct'],
            'daily_loss': daily_loss['loss_pct'],
            'drawdown': current_drawdown,
            'effective_daily_limit': effective_daily_limit,
            'effective_hourly_limit': effective_hourly_limit,
            'adaptive_multipliers': {
                'daily': adaptive_daily_multiplier,
                'hourly': adaptive_hourly_multiplier
            }
        }
    
    def _check_hourly_loss(self, current_equity: float, timestamp: datetime) -> Dict:
        """Check hourly loss limit"""
        one_hour_ago = timestamp - timedelta(hours=1)
        
        # Find equity one hour ago
        equity_one_hour_ago = None
        for ts, equity in self.equity_history:
            if ts <= one_hour_ago:
                equity_one_hour_ago = equity
            else:
                break
        
        if equity_one_hour_ago is None or equity_one_hour_ago == 0:
            return {'exceeded': False, 'loss_pct': 0.0}
        
        loss_pct = ((equity_one_hour_ago - current_equity) / equity_one_hour_ago) * 100
        
        # loss_pct is positive for losses, negative for gains
        # Only trigger on actual losses (positive) that exceed the limit
        max_hourly_loss_abs = abs(self.max_hourly_loss)
        exceeded = loss_pct > 0 and loss_pct >= max_hourly_loss_abs
        
        return {
            'exceeded': exceeded,
            'loss_pct': loss_pct
        }
    
    def _check_daily_loss(self, current_equity: float, timestamp: datetime) -> Dict:
        """Check daily loss limit"""
        one_day_ago = timestamp - timedelta(days=1)
        
        # Find equity one day ago
        equity_one_day_ago = None
        for ts, equity in self.equity_history:
            if ts <= one_day_ago:
                equity_one_day_ago = equity
            else:
                break
        
        if equity_one_day_ago is None or equity_one_day_ago == 0:
            return {'exceeded': False, 'loss_pct': 0.0}
        
        loss_pct = ((equity_one_day_ago - current_equity) / equity_one_day_ago) * 100
        
        # loss_pct is positive for losses, negative for gains
        # Only trigger on actual losses (positive) that exceed the limit
        max_daily_loss_abs = abs(self.max_daily_loss)
        exceeded = loss_pct > 0 and loss_pct >= max_daily_loss_abs
        
        return {
            'exceeded': exceeded,
            'loss_pct': loss_pct
        }
    
    def _trigger(self, reason: str, timestamp: datetime):
        """Trigger circuit breaker"""
        self.triggered = True
        self.triggered_at = timestamp
        self.trigger_reason = reason
        print(f"⚠️ CIRCUIT BREAKER TRIGGERED: {reason}")
        print(f"Trading halted for {self.cooling_period_hours} hours.")
    
    def reset(self):
        """Reset circuit breaker (manual override)"""
        self.triggered = False
        self.triggered_at = None
        self.trigger_reason = None
        print("Circuit breaker manually reset.")
    
    def reset_peak_equity(self, current_equity: Optional[float] = None):
        """
        Reset peak equity to current equity (useful after account recovery or manual reset)
        
        Args:
            current_equity: Current equity value (if None, uses latest from history)
        """
        if current_equity is None:
            if self.equity_history:
                current_equity = self.equity_history[-1][1]
            else:
                return
        
        self.peak_equity = current_equity
        self.peak_equity_time = datetime.now()
        print(f"Circuit Breaker: Peak equity reset to {current_equity:.2f}")
    
    def reset(self):
        """Reset circuit breaker state (clears trigger and resets peak equity)"""
        self.triggered = False
        self.triggered_at = None
        self.trigger_reason = None
        if self.equity_history:
            current_equity = self.equity_history[-1][1]
            self.peak_equity = current_equity
            self.peak_equity_time = datetime.now()
        print("Circuit Breaker: Reset complete")
    
    def get_status(self) -> Dict:
        """Get circuit breaker status"""
        return {
            'triggered': self.triggered,
            'triggered_at': self.triggered_at.isoformat() if self.triggered_at else None,
            'trigger_reason': self.trigger_reason,
            'cooling_period_hours': self.cooling_period_hours,
            'peak_equity': self.peak_equity,
            'peak_equity_time': self.peak_equity_time.isoformat() if hasattr(self, 'peak_equity_time') and self.peak_equity_time else None,
            'limits': {
                'max_hourly_loss': self.max_hourly_loss,
                'max_daily_loss': self.max_daily_loss,
                'max_drawdown': self.max_drawdown
            }
        }

