# Circuit Breaker and Date Handling Fixes

## Issues Identified

1. **Circuit Breaker Blocking All Trades**: The system showed 66.61% drawdown, blocking all trading. This was likely due to:
   - Stale peak equity from a previous run
   - Peak equity not being initialized from current account balance on startup
   - Auto-reset logic not being aggressive enough for extreme drawdowns

2. **Date Warnings**: System clock shows year 2025, but code was hardcoded to clamp dates to 2024-12-31, causing warnings.

## Fixes Applied

### 1. Circuit Breaker Auto-Reset Improvements (`src/risk/circuit_breaker.py`)

Added more aggressive auto-reset logic:
- **Extreme drawdown (>20%)**: Always reset regardless of peak age
- **High drawdown (>10%)**: Reset if peak is more than 6 hours old
- **Normal drawdown (>4.5%)**: Reset if peak is more than 12 hours old

This prevents stale peaks from blocking trading indefinitely.

### 2. Circuit Breaker Initialization (`src/main.py`)

Added proper initialization of peak equity:
- On startup, if peak equity is `None`, initialize it to current account balance
- If peak equity differs by more than 50% from current balance, reset it
- This handles cases where peak equity was set incorrectly from a previous run

```python
# Initialize peak equity if not set
if cb.peak_equity is None:
    cb.reset_peak_equity(account_balance)
elif cb.peak_equity > 0 and abs(cb.peak_equity - account_balance) / cb.peak_equity > 0.5:
    # Reset if significantly different
    cb.reset_peak_equity(account_balance)
```

### 3. Date Handling Fix (`src/data/brokers/oanda_api.py`)

Updated date clamping to use current year instead of hardcoded 2024:
- Uses `datetime.now().year` to get current year
- Allows dates up to current year + 1 (handles timezone edge cases)
- Only clamps dates that are unreasonably far in the future (> current year + 1)

## Expected Results

1. **Circuit Breaker**: Should auto-reset when drawdown is extreme or peak is stale, allowing trading to resume
2. **Date Warnings**: Should be reduced or eliminated for dates in 2025
3. **Trading**: Should resume once circuit breaker auto-resets the peak equity

## Testing

After restarting the system:
1. Check logs for "Circuit Breaker: Auto-resetting peak equity" messages
2. Verify that trading resumes after peak equity is reset
3. Confirm date warnings are reduced for 2025 dates

## Manual Reset (if needed)

If circuit breaker is still blocking trading, you can manually reset it:

```python
# In Python console or script
from src.risk.circuit_breaker import CircuitBreaker

# Get current account balance
account_balance = 100000.0  # Replace with actual balance

# Reset circuit breaker
circuit_breaker.reset_peak_equity(account_balance)
circuit_breaker.reset()  # Clear any triggered state
```

## Configuration

Circuit breaker limits are in `config/trading_config.yaml`:

```yaml
agents:
  risk_manager:
    circuit_breaker:
      max_drawdown_percent: 4.5  # Max drawdown before trigger
      max_daily_loss_percent: 1.5
      max_hourly_loss_percent: 0.5
      cooling_period_hours: 4
```

You can adjust these values if needed, but the auto-reset logic should handle most cases automatically.

