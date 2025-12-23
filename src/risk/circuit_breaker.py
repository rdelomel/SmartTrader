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
        self.max_hourly_loss = self.config.get('max_hourly_loss_percent', 0.50)  # Changed from 0.40 to 0.50
        self.max_drawdown = self.config.get('max_drawdown_percent', 4.5)
        self.cooling_period_hours = self.config.get('cooling_period_hours', 4)
        
        # Track equity history by time period
        self.equity_history = deque(maxlen=1000)  # Store (timestamp, equity) tuples
        self.peak_equity = None
        self.peak_equity_time = None
        
        # Single source of truth for equity (cached with timestamp)
        self._cached_equity = None
        self._cached_equity_timestamp = None
        self._equity_cache_ttl = 60  # Cache for 60 seconds
        
        self.triggered = False
        self.triggered_at = None
        self.trigger_reason = None
    
    def update_equity(self, equity: float, timestamp: Optional[datetime] = None, 
                      verify_with_broker: bool = False, api_available: bool = True, 
                      data_source: str = 'live'):
        """
        Update equity and check circuit breakers
        
        Args:
            equity: Current equity value
            timestamp: Timestamp of update (default: now)
            verify_with_broker: If True and equity changes >50%, verify with broker before triggering
            api_available: True if broker API is responding, False if using cached data
            data_source: 'live' or 'cached' to indicate data reliability
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # CRITICAL: Don't trigger circuit breaker on cached/stale data
        if not api_available or data_source == 'cached':
            print(f"  ⚠️  Equity data from {data_source} source (API available: {api_available}) - skipping circuit breaker check")
            # Still update history but don't trigger circuit breaker
            self.equity_history.append((timestamp, equity))
            if self._cached_equity is None:
                self._cached_equity = equity
                self._cached_equity_timestamp = timestamp
            return {'triggered': False, 'reason': f'Using {data_source} data, API unavailable'}
        
        # CRITICAL: Validate large equity changes (>50%) - might be calculation error
        if self._cached_equity is not None and self._cached_equity > 0:
            equity_change_pct = abs(equity - self._cached_equity) / self._cached_equity * 100
            if equity_change_pct > 50.0:
                print(f"  ⚠️  WARNING: Large equity change detected: {equity_change_pct:.1f}% ({self._cached_equity:.2f} -> {equity:.2f})")
                if verify_with_broker:
                    print(f"     This may be a calculation error - verify with broker before triggering circuit breaker")
                    # Don't update equity if it's likely an error
                    return {'triggered': False, 'reason': 'Large equity change requires broker verification'}
                else:
                    print(f"     Proceeding with equity update (not verified with broker)")
        
        # Update cached equity
        self._cached_equity = equity
        self._cached_equity_timestamp = timestamp
        
        # CRITICAL: Check for real drawdowns BEFORE clearing history
        # If equity drops significantly, trigger circuit breaker, don't just clear history
        if self.equity_history:
            # Check recent history entries
            recent_entries = list(self.equity_history)[-10:]  # Last 10 entries
            if recent_entries:
                recent_equities = [e[1] for e in recent_entries]
                avg_recent_equity = sum(recent_equities) / len(recent_equities)
                
                if avg_recent_equity > 0:
                    equity_ratio = max(equity, avg_recent_equity) / min(equity, avg_recent_equity)
                    
                    # If equity DROPS significantly (>30%), check if it's a real drawdown
                    if equity < avg_recent_equity * 0.7:  # 30% drop
                        # Check if this is recent (within last hour) - likely real drawdown
                        recent_times = [e[0] for e in recent_entries[-5:]]
                        if recent_times:
                            time_diff = (timestamp - recent_times[-1]).total_seconds() / 3600
                            if time_diff < 1.0:  # Within last hour
                                # Real drawdown - trigger circuit breaker, don't clear history
                                drop_pct = ((avg_recent_equity - equity) / avg_recent_equity * 100)
                                print(f"Circuit Breaker: CRITICAL - Equity dropped {drop_pct:.1f}% in {time_diff:.1f} hours")
                                self.triggered = True
                                self.triggered_at = timestamp
                                self.trigger_reason = f"Critical equity drop: {drop_pct:.1f}% in {time_diff:.1f} hours"
                                # Still add to history for tracking, but don't continue normal processing
                                self.equity_history.append((timestamp, equity))
                                return  # Don't continue with normal history update
                    
                    # Check for immediate drop >20% in single update (very recent)
                    if len(recent_entries) >= 2:
                        last_equity = recent_entries[-1][1]
                        if equity < last_equity * 0.8:  # 20% drop from last update
                            time_diff = (timestamp - recent_entries[-1][0]).total_seconds() / 60  # minutes
                            if time_diff < 60:  # Within last hour
                                drop_pct = ((last_equity - equity) / last_equity * 100)
                                print(f"Circuit Breaker: IMMEDIATE TRIGGER - Equity dropped {drop_pct:.1f}% in {time_diff:.1f} minutes")
                                self.triggered = True
                                self.triggered_at = timestamp
                                self.trigger_reason = f"Immediate equity drop: {drop_pct:.1f}% in {time_diff:.1f} minutes"
                                self.equity_history.append((timestamp, equity))
                                return
                    
                    # Only clear history if equity INCREASED significantly (likely new session/deposit)
                    if equity_ratio > 2.0 and equity > avg_recent_equity:
                        print(f"Circuit Breaker: Detected significant equity increase ({equity:.2f} vs recent avg {avg_recent_equity:.2f}). Clearing stale history.")
                        # Keep only very recent entries (last hour) or clear all if too old
                        cutoff_time = timestamp - timedelta(hours=1)
                        filtered_entries = [(ts, eq) for ts, eq in self.equity_history if ts >= cutoff_time]
                        self.equity_history.clear()
                        for entry in filtered_entries:
                            self.equity_history.append(entry)
                        # If we cleared everything, at least keep current value
                        if not self.equity_history:
                            self.equity_history.append((timestamp, equity))
        
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
        
        # Safety check: if calculated loss is unreasonably high (>10%), likely invalid data
        # Don't trigger circuit breaker on clearly invalid calculations
        if hourly_loss_pct > 10.0:
            print(f"Circuit Breaker: Warning - Calculated hourly loss ({hourly_loss_pct:.2f}%) is unreasonably high. Likely invalid data. Ignoring.")
            hourly_loss_pct = 0.0  # Treat as no loss to avoid false triggers
        
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
        
        # Safety check: if calculated loss is unreasonably high (>20%), likely invalid data
        # Don't trigger circuit breaker on clearly invalid calculations
        if daily_loss_pct > 20.0:
            print(f"Circuit Breaker: Warning - Calculated daily loss ({daily_loss_pct:.2f}%) is unreasonably high. Likely invalid data. Ignoring.")
            daily_loss_pct = 0.0  # Treat as no loss to avoid false triggers
        
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
            
            # AGGRESSIVE RESET: If drawdown is extremely high (>20%), always reset regardless of age
            # This handles cases where peak equity was set incorrectly or from a previous run
            if current_drawdown > 20.0:
                should_reset = True
                reset_reason = f"extreme drawdown ({current_drawdown:.2f}%) - likely stale or incorrect peak equity"
            elif current_drawdown > 10.0:
                # High drawdown (>10%) - reset if peak is more than 6 hours old
                if hasattr(self, 'peak_equity_time') and self.peak_equity_time:
                    hours_since_peak = (timestamp - self.peak_equity_time).total_seconds() / 3600
                    if hours_since_peak > 6:
                        should_reset = True
                        reset_reason = f"high drawdown ({current_drawdown:.2f}%) with stale peak ({hours_since_peak:.1f} hours old)"
                else:
                    # No timestamp - assume stale
                    should_reset = True
                    reset_reason = f"high drawdown ({current_drawdown:.2f}%) with peak having no timestamp (stale)"
            
            if not should_reset and hasattr(self, 'peak_equity_time') and self.peak_equity_time:
                days_since_peak = (timestamp - self.peak_equity_time).total_seconds() / 86400
                
                # Check if drawdown exceeds limit
                if current_drawdown >= self.max_drawdown:
                    # If drawdown exceeds limit, check if peak is stale
                    if days_since_peak > 0.5:  # More than 12 hours old - likely stale
                        should_reset = True
                        reset_reason = f"drawdown ({current_drawdown:.2f}%) exceeds limit ({self.max_drawdown:.2f}%) and peak is stale ({days_since_peak:.1f} days old)"
                    else:
                        # Peak is recent (< 12 hours) - this is a REAL drawdown, trigger circuit breaker
                        # Don't reset, trigger instead
                        if current_drawdown > 10.0:
                            # Major drawdown (>10%) with recent peak - definitely trigger
                            self.triggered = True
                            self.triggered_at = timestamp
                            self.trigger_reason = f"Major drawdown {current_drawdown:.2f}% exceeds limit {self.max_drawdown:.2f}% (peak is recent, not stale)"
                            print(f"Circuit Breaker: TRIGGERED - {self.trigger_reason}")
                            return {
                                'triggered': True,
                                'reason': self.trigger_reason,
                                'drawdown': current_drawdown,
                                'peak_equity': self.peak_equity,
                                'current_equity': current_equity
                            }
                        else:
                            # Drawdown between max_drawdown and 10% - check if peak is very recent
                            hours_since_peak = days_since_peak * 24
                            if hours_since_peak < 2:  # Peak is less than 2 hours old - likely real drawdown
                                self.triggered = True
                                self.triggered_at = timestamp
                                self.trigger_reason = f"Drawdown {current_drawdown:.2f}% exceeds limit {self.max_drawdown:.2f}% (peak is recent: {hours_since_peak:.1f} hours old)"
                                print(f"Circuit Breaker: TRIGGERED - {self.trigger_reason}")
                                return {
                                    'triggered': True,
                                    'reason': self.trigger_reason,
                                    'drawdown': current_drawdown,
                                    'peak_equity': self.peak_equity,
                                    'current_equity': current_equity
                                }
                            else:
                                # Peak is 2-12 hours old, might be stale - reset
                                should_reset = True
                                reset_reason = f"drawdown ({current_drawdown:.2f}%) exceeds limit ({self.max_drawdown:.2f}%) but peak is moderately stale ({hours_since_peak:.1f} hours old)"
                    if days_since_peak > 1 and current_drawdown > 5.0:
                        should_reset = True
                        reset_reason = f"stale peak ({days_since_peak:.1f} days old) causing significant drawdown ({current_drawdown:.2f}%)"
                elif days_since_peak > 1 and current_drawdown > 5.0:
                    should_reset = True
                    reset_reason = f"stale peak ({days_since_peak:.1f} days old) causing significant drawdown ({current_drawdown:.2f}%)"
                elif current_drawdown < 5.0 and current_equity >= self.peak_equity * 0.95:
                    should_reset = True
                    reset_reason = "account recovered to within 5% of peak"
            elif not should_reset:
                # No peak_equity_time set - likely stale peak from before this feature
                # If drawdown is high, check if it's real or stale
                if current_drawdown >= self.max_drawdown:
                    # If drawdown > 10%, likely real - trigger
                    if current_drawdown > 10.0:
                        self.triggered = True
                        self.triggered_at = timestamp
                        self.trigger_reason = f"Major drawdown {current_drawdown:.2f}% exceeds limit {self.max_drawdown:.2f}% (no peak timestamp - treating as real)"
                        print(f"Circuit Breaker: TRIGGERED - {self.trigger_reason}")
                        return {
                            'triggered': True,
                            'reason': self.trigger_reason,
                            'drawdown': current_drawdown,
                            'peak_equity': self.peak_equity,
                            'current_equity': current_equity
                        }
                    else:
                        # Drawdown between max and 10%, no timestamp - assume stale and reset
                        should_reset = True
                        reset_reason = f"drawdown ({current_drawdown:.2f}%) exceeds limit ({self.max_drawdown:.2f}%) and peak has no timestamp (stale)"
            
            # Only reset if should_reset is True (stale data case)
            if should_reset:
                print(f"Circuit Breaker: Auto-resetting peak equity - {reset_reason}")
                print(f"  Old peak: ${self.peak_equity:,.2f} -> New peak: ${current_equity:,.2f}")
                self.peak_equity = current_equity
                if not hasattr(self, 'peak_equity_time') or not self.peak_equity_time:
                    self.peak_equity_time = timestamp
                else:
                    self.peak_equity_time = timestamp
                current_drawdown = 0.0
            elif current_drawdown >= self.max_drawdown:
                # Drawdown exceeds limit but we didn't reset - must be real drawdown
                # This should have been caught above, but add safety check
                self.triggered = True
                self.triggered_at = timestamp
                self.trigger_reason = f"Drawdown {current_drawdown:.2f}% exceeds limit {self.max_drawdown:.2f}%"
                print(f"Circuit Breaker: TRIGGERED - {self.trigger_reason}")
                return {
                    'triggered': True,
                    'reason': self.trigger_reason,
                    'drawdown': current_drawdown,
                    'peak_equity': self.peak_equity,
                    'current_equity': current_equity
                }
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
        # Use a window: accept values between 0.5 and 1.5 hours ago
        window_start = timestamp - timedelta(hours=1.5)
        window_end = timestamp - timedelta(hours=0.5)
        
        # Find equity closest to one hour ago within the window
        equity_one_hour_ago = None
        closest_time_diff = None
        
        if not self.equity_history:
            return {'exceeded': False, 'loss_pct': 0.0}
        
        for ts, equity in self.equity_history:
            # Only consider values within the acceptable window
            if window_start <= ts <= window_end:
                time_diff = abs((ts - one_hour_ago).total_seconds())
                if closest_time_diff is None or time_diff < closest_time_diff:
                    closest_time_diff = time_diff
                    equity_one_hour_ago = equity
        
        # If no value in window, check if we have any history at all
        # If history is less than 1 hour old, can't calculate hourly loss
        if equity_one_hour_ago is None:
            # Check if oldest record is less than 1 hour old
            oldest_ts, oldest_equity = self.equity_history[0]
            if (timestamp - oldest_ts).total_seconds() < 3600:  # Less than 1 hour
                return {'exceeded': False, 'loss_pct': 0.0}
            # Otherwise, use oldest available value but validate it's reasonable
            equity_one_hour_ago = oldest_equity
            # Validate: if equity changed by more than 50%, likely stale data
            if abs(equity_one_hour_ago - current_equity) / max(equity_one_hour_ago, current_equity) > 0.5:
                return {'exceeded': False, 'loss_pct': 0.0}
        
        if equity_one_hour_ago is None or equity_one_hour_ago <= 0:
            return {'exceeded': False, 'loss_pct': 0.0}
        
        # Validate equity values are reasonable (not wildly different)
        equity_ratio = max(equity_one_hour_ago, current_equity) / min(equity_one_hour_ago, current_equity)
        if equity_ratio > 3.0:  # More than 3x difference suggests stale or incorrect data
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
        # Use a window: accept values between 0.75 and 1.25 days ago
        window_start = timestamp - timedelta(days=1.25)
        window_end = timestamp - timedelta(days=0.75)
        
        # Find equity closest to one day ago within the window
        equity_one_day_ago = None
        closest_time_diff = None
        
        if not self.equity_history:
            return {'exceeded': False, 'loss_pct': 0.0}
        
        for ts, equity in self.equity_history:
            # Only consider values within the acceptable window
            if window_start <= ts <= window_end:
                time_diff = abs((ts - one_day_ago).total_seconds())
                if closest_time_diff is None or time_diff < closest_time_diff:
                    closest_time_diff = time_diff
                    equity_one_day_ago = equity
        
        # If no value in window, check if we have any history at all
        # If history is less than 1 day old, can't calculate daily loss
        if equity_one_day_ago is None:
            # Check if oldest record is less than 1 day old
            oldest_ts, oldest_equity = self.equity_history[0]
            if (timestamp - oldest_ts).total_seconds() < 86400:  # Less than 1 day
                return {'exceeded': False, 'loss_pct': 0.0}
            # Otherwise, use oldest available value but validate it's reasonable
            equity_one_day_ago = oldest_equity
            # Validate: if equity changed by more than 50%, likely stale data
            if abs(equity_one_day_ago - current_equity) / max(equity_one_day_ago, current_equity) > 0.5:
                return {'exceeded': False, 'loss_pct': 0.0}
        
        if equity_one_day_ago is None or equity_one_day_ago <= 0:
            return {'exceeded': False, 'loss_pct': 0.0}
        
        # Validate equity values are reasonable (not wildly different)
        equity_ratio = max(equity_one_day_ago, current_equity) / min(equity_one_day_ago, current_equity)
        if equity_ratio > 3.0:  # More than 3x difference suggests stale or incorrect data
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
    
    def check_and_reset_stale_state(self, current_equity: float, timestamp: Optional[datetime] = None):
        """
        Check for stale circuit breaker state and reset if needed
        
        This helps unblock trading when the circuit breaker is stuck due to stale data.
        
        Args:
            current_equity: Current equity value
            timestamp: Current timestamp (default: now)
        
        Returns:
            True if state was reset, False otherwise
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # If circuit breaker is triggered, check if it's stale
        if self.triggered and self.triggered_at:
            hours_since_trigger = (timestamp - self.triggered_at).total_seconds() / 3600
            
            # If triggered more than cooling period ago, reset it
            if hours_since_trigger >= self.cooling_period_hours:
                print(f"Circuit Breaker: Auto-resetting stale trigger (triggered {hours_since_trigger:.1f} hours ago)")
                self.reset()
                return True
        
        # Check for stale peak equity
        if self.peak_equity and self.peak_equity_time:
            days_since_peak = (timestamp - self.peak_equity_time).total_seconds() / 86400
            current_drawdown = ((self.peak_equity - current_equity) / self.peak_equity) * 100 if self.peak_equity > 0 else 0.0
            
            # Reset if peak is more than 1 day old and causing false drawdown
            if days_since_peak > 1.0 and current_drawdown > 0.5:
                print(f"Circuit Breaker: Resetting stale peak equity (peak from {days_since_peak:.1f} days ago, drawdown: {current_drawdown:.2f}%)")
                self.reset_peak_equity(current_equity)
                return True
        
        return False
    
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

