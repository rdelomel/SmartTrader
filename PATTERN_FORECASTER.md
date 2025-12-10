# Pattern Forecaster - Historical Pattern Matching

This document describes the Pattern Forecaster feature, which replicates the functionality of forecaster.biz by comparing and correlating patterns from the past to forecast future market movements.

## Overview

The Pattern Forecaster uses historical pattern matching to predict future price movements. It:

1. **Extracts patterns** from current market data
2. **Searches historical data** for similar patterns
3. **Correlates patterns** using similarity and correlation metrics
4. **Forecasts future movements** based on what happened after similar historical patterns

This approach is similar to forecaster.biz, which uses AI-driven projections to identify patterns similar to current market behavior and estimate potential future scenarios.

## Architecture

### Components

1. **PatternExtractor** (`src/indicators/pattern_matcher.py`)
   - Extracts normalized patterns from price data
   - Normalizes prices to percentage changes for comparison
   - Calculates pattern features (volatility, trend, volume, etc.)

2. **PatternMatcher** (`src/indicators/pattern_matcher.py`)
   - Matches current patterns with historical patterns
   - Uses multiple similarity metrics:
     - Pearson correlation
     - Normalized Euclidean distance
     - Feature similarity
   - Filters matches by similarity and correlation thresholds

3. **PatternForecaster** (`src/indicators/pattern_matcher.py`)
   - Aggregates outcomes from matched patterns
   - Creates weighted forecasts based on pattern similarity
   - Calculates confidence based on match quality and consistency

4. **PatternForecasterAgent** (`src/agent/pattern_forecaster_agent.py`)
   - Integrates pattern matching into the agent framework
   - Retrieves historical data from storage
   - Converts forecasts to trading signals

## How It Works

### 1. Pattern Extraction

For each pattern:
- Takes the last N bars (default: 20)
- Normalizes prices to percentage changes from start
- Normalizes volumes relative to mean
- Calculates features:
  - Price range and volatility
  - Trend direction and strength
  - Volume trend
  - High-low ratio

### 2. Historical Pattern Matching

The system:
- Extracts patterns from historical data (up to 365 days)
- For each historical pattern, calculates what happened after it
- Stores outcomes (price change, direction, max gain/loss)

### 3. Pattern Comparison

When analyzing current market:
- Extracts current pattern
- Compares with all historical patterns using:
  - **Correlation**: Pearson correlation of price patterns
  - **Distance**: Normalized Euclidean distance
  - **Features**: Similarity of pattern features
- Filters matches by:
  - Similarity threshold (default: 0.7)
  - Correlation threshold (default: 0.6)

### 4. Forecast Generation

From matched patterns:
- Aggregates outcomes weighted by similarity
- Calculates:
  - Forecasted price change percentage
  - Forecasted price level
  - Confidence score
- Confidence based on:
  - Number of matches
  - Average similarity
  - Direction consistency

### 5. Signal Generation

Converts forecast to trading signal:
- **Bullish** (>1% forecasted gain) → BUY signal
- **Bearish** (<-1% forecasted loss) → SELL signal
- **Neutral** → HOLD signal

## Configuration

Add to `config/trading_config.yaml`:

```yaml
agents:
  pattern_forecaster:
    enabled: true
    pattern_length: 20  # Number of bars in a pattern
    lookahead_periods: 10  # Periods to forecast ahead
    min_matches: 3  # Minimum similar patterns required
    similarity_threshold: 0.7  # Minimum similarity score (0-1)
    correlation_threshold: 0.6  # Minimum correlation coefficient
    historical_lookback_days: 365  # Days of history to search
```

## Integration

The Pattern Forecaster is integrated into the orchestrator with:
- **Weight**: 15% (alongside other agents)
- **Signal**: BUY/SELL/HOLD based on forecast
- **Confidence**: Based on pattern match quality

## Example Output

```
=== PatternForecaster analyzing BTC/USD ===
  Pattern Matches: 12
  Forecast Signal: BUY
  Forecast Change: 3.45%
  Forecast Price: $95,234.56
  Confidence: 0.742
  Avg Similarity: 0.785
  Top Match Similarity: 0.852
=== End PatternForecaster for BTC/USD ===
```

## Benefits

1. **Historical Validation**: Uses actual historical outcomes, not theoretical models
2. **Pattern Recognition**: Identifies recurring market patterns
3. **Correlation Analysis**: Finds truly similar patterns, not just similar prices
4. **Confidence Scoring**: Provides confidence based on match quality
5. **Adaptive**: Works with any timeframe and symbol

## Limitations

1. **Requires Historical Data**: Needs sufficient historical data (at least 365 days recommended)
2. **Pattern Quality**: Forecast quality depends on finding good pattern matches
3. **Market Changes**: Past patterns may not always repeat in changed market conditions
4. **Computational Cost**: Pattern matching can be computationally intensive with large datasets

## Performance Tips

1. **Increase Historical Data**: More historical data = more pattern matches
2. **Adjust Thresholds**: Lower thresholds = more matches but potentially lower quality
3. **Pattern Length**: Longer patterns (30-50 bars) may be more reliable but require more data
4. **Timeframe**: Works best with consistent timeframes (1h, 4h, 1d)

## Comparison with forecaster.biz

| Feature | forecaster.biz | SmartTrader Pattern Forecaster |
|---------|---------------|-------------------------------|
| Pattern Matching | ✅ | ✅ |
| Historical Correlation | ✅ | ✅ |
| AI-Driven Projections | ✅ | ✅ (via ML models) |
| Multiple Timeframes | ✅ | ✅ |
| Confidence Scoring | ✅ | ✅ |
| Real-time Updates | ✅ | ✅ |
| Customizable Thresholds | ❓ | ✅ |
| Open Source | ❌ | ✅ |

## Future Enhancements

1. **Pattern Caching**: Cache extracted patterns for faster matching
2. **Multi-Timeframe Patterns**: Match patterns across different timeframes
3. **Pattern Clustering**: Group similar patterns for better aggregation
4. **Machine Learning**: Use ML to improve pattern matching accuracy
5. **Pattern Database**: Store patterns in database for faster retrieval

## References

- [forecaster.biz](https://forecaster.biz) - Original inspiration
- Pattern matching algorithms based on correlation and similarity metrics
- Historical analog forecasting methodology

