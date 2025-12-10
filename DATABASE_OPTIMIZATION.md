# Database Connection Optimization

## Issue
PostgreSQL connection resets were occurring due to long-running queries from the pattern forecaster agent.

## Optimizations Applied

### 1. Reduced Data Fetching
- **Historical lookback**: Reduced from 365 to 180 days
- **Max records**: Limited to 5,000 records per query
- **Max bars to process**: Limited to 2,000 bars to prevent memory issues

### 2. Pattern Extraction Limits
- **Max patterns**: Limited to 500 patterns per analysis
- **Step size**: Uses larger step between patterns for efficiency

### 3. Query Optimization
- **Order optimization**: Fetch most recent data first (DESC), then reverse
- **Error handling**: Better timeout and error catching
- **Connection management**: Uses existing connection pool settings

### 4. Configuration
Added to `config/trading_config.yaml`:
```yaml
pattern_forecaster:
  historical_lookback_days: 180  # Reduced from 365
  max_historical_records: 5000   # Limit database fetch
  max_historical_bars: 2000      # Limit processing
  analysis_timeout_seconds: 30    # Timeout protection
```

## Database Index Recommendations

For better query performance, ensure these indexes exist:

```sql
-- Composite index for common query pattern
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timeframe_timestamp 
ON ohlcv_data(symbol, timeframe, timestamp DESC);

-- Verify indexes
SELECT indexname, indexdef 
FROM pg_indexes 
WHERE tablename = 'ohlcv_data';
```

## Monitoring

Watch for these PostgreSQL log patterns:
- `checkpoint complete` - Normal, indicates database health
- `could not receive data from client` - Connection reset (should be reduced now)
- `unexpected EOF` - Connection closed unexpectedly (should be reduced now)

## Performance Tips

1. **Reduce lookback days** if still experiencing timeouts:
   ```yaml
   historical_lookback_days: 90  # 3 months instead of 6
   ```

2. **Reduce max records** for faster queries:
   ```yaml
   max_historical_records: 3000
   ```

3. **Increase step size** in pattern extraction (in code) for fewer patterns

4. **Monitor query times** in PostgreSQL logs

## Connection Pool Settings

Current settings in `DataStorage`:
- `pool_size=5` - Base connection pool
- `max_overflow=10` - Additional connections
- `pool_recycle=300` - Recycle connections after 5 minutes
- `pool_pre_ping=True` - Verify connections before use

These settings help prevent stale connections and connection leaks.

