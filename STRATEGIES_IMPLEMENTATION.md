# Trading Strategies Implementation

## Overview

All 6 trading strategies from [CMC Markets Trading Strategies Guide](https://www.cmcmarkets.com/en-ie/trading-guides/trading-strategies) have been implemented in SmartTrader.

## Implemented Strategies

### 1. News Trading Strategy (`NewsTradingStrategy`)
**File**: `src/strategies/news_trading.py`

**Description**: Trades based on news and market expectations, both before and after news releases.

**Key Features**:
- Detects volume spikes indicating news-driven activity
- Trades pre-news anticipation moves
- Trades post-news reactions
- Can fade moves when news is already factored in
- Uses volatility and volume analysis

**Configuration**:
```python
{
    'volatility_threshold': 0.015,  # 1.5% minimum volatility
    'volume_spike_multiplier': 1.5,  # Volume must be 1.5x average
    'price_movement_threshold': 0.005,  # 0.5% minimum move
    'use_pre_news': True,
    'use_post_news': True
}
```

**Usage**:
```python
from src.strategies import NewsTradingStrategy

strategy = NewsTradingStrategy(config={
    'volatility_threshold': 0.02,
    'volume_spike_multiplier': 2.0
})

signal = strategy.generate_signal(data, news_items=news_data)
```

---

### 2. End-of-Day Trading Strategy (`EndOfDayStrategy`)
**File**: `src/strategies/end_of_day.py`

**Description**: Trades near market close when price is 'settling'. Compares price action to previous day's movements.

**Key Features**:
- Only trades near market close (configurable hours)
- Identifies price 'settling' behavior
- Compares to previous day's price action
- Can fade extreme day moves
- Wider stops for overnight positions

**Configuration**:
```python
{
    'close_hour': 15,  # 3 PM
    'close_minute': 45,
    'min_price_change': 0.003,  # 0.3% minimum
    'compare_previous_day': True
}
```

**Usage**:
```python
from src.strategies import EndOfDayStrategy

strategy = EndOfDayStrategy(config={
    'close_hour': 16,  # 4 PM close
    'close_minute': 0
})

signal = strategy.generate_signal(data)
```

---

### 3. Swing Trading Strategy (`SwingTradingStrategy`)
**File**: `src/strategies/swing_trading.py`

**Description**: Trades market oscillations from overbought to oversold states. Buys at swing lows, sells at swing highs.

**Key Features**:
- Identifies swing highs and lows
- Uses support/resistance levels
- Trades retracements in trends
- RSI-based overbought/oversold signals
- Stop loss at opposite swing level

**Configuration**:
```python
{
    'swing_period': 14,
    'rsi_period': 14,
    'rsi_oversold': 30,
    'rsi_overbought': 70,
    'min_swing_size': 0.02,  # 2% minimum swing
    'use_support_resistance': True
}
```

**Usage**:
```python
from src.strategies import SwingTradingStrategy

strategy = SwingTradingStrategy(config={
    'rsi_oversold': 25,
    'rsi_overbought': 75
})

signal = strategy.generate_signal(data)
```

---

### 4. Day Trading Strategy (`DayTradingStrategy`)
**File**: `src/strategies/day_trading.py`

**Description**: Opens and closes positions within the same day. No overnight positions. Focuses on intraday movements.

**Key Features**:
- Only trades during market hours
- Trades breakouts and reversals
- Range trading (buy low, sell high of day)
- Tighter stops (no overnight risk)
- Exits before market close

**Configuration**:
```python
{
    'open_hour': 9,
    'close_hour': 16,
    'min_intraday_move': 0.005,  # 0.5% minimum
    'use_breakouts': True,
    'use_reversals': True,
    'max_trades_per_day': 5
}
```

**Usage**:
```python
from src.strategies import DayTradingStrategy

strategy = DayTradingStrategy(config={
    'open_hour': 9,
    'close_hour': 16,
    'min_intraday_move': 0.01  # 1% minimum
})

signal = strategy.generate_signal(data)
```

---

### 5. Trend Trading Strategy (`TrendFollowingStrategy`)
**File**: `src/strategies/trend_following.py` (already existed, enhanced)

**Description**: Uses technical analysis to define trends and only enters trades in the direction of the trend.

**Key Features**:
- Moving average crossovers
- ADX for trend strength
- MACD confirmation
- Trailing stops
- Multi-factor confidence calculation

**Configuration**:
```python
{
    'fast_ma_period': 20,
    'slow_ma_period': 50,
    'use_ema': False
}
```

**Usage**:
```python
from src.strategies import TrendFollowingStrategy

strategy = TrendFollowingStrategy(config={
    'fast_ma_period': 10,
    'slow_ma_period': 30,
    'use_ema': True
})

signal = strategy.generate_signal(data)
```

---

### 6. Scalping Trading Strategy (`ScalpingStrategy`)
**File**: `src/strategies/scalping.py`

**Description**: Very short-term trades with small price movements. Aims to accumulate small profits. Risk/reward around 1:1.

**Key Features**:
- Requires high volatility and volume
- Very tight stops (0.5-0.8% typically)
- 1:1 risk/reward ratio
- Micro-momentum analysis
- Quick exits (minutes, not hours)

**Configuration**:
```python
{
    'min_volatility': 0.01,  # 1% minimum
    'min_volume_multiplier': 2.0,  # 2x average volume
    'price_tick_size': 0.001,  # 0.1% minimum move
    'use_order_flow': True,
    'max_hold_time_minutes': 15
}
```

**Usage**:
```python
from src.strategies import ScalpingStrategy

strategy = ScalpingStrategy(config={
    'min_volatility': 0.015,  # 1.5% minimum
    'min_volume_multiplier': 2.5
})

signal = strategy.generate_signal(data)
```

---

## Integration with SmartTrader

All strategies follow the `BaseStrategy` interface and can be used with the orchestrator agent:

```python
from src.strategies import (
    NewsTradingStrategy,
    EndOfDayStrategy,
    SwingTradingStrategy,
    DayTradingStrategy,
    TrendFollowingStrategy,
    ScalpingStrategy
)

# Initialize strategies
strategies = {
    'news': NewsTradingStrategy(),
    'eod': EndOfDayStrategy(),
    'swing': SwingTradingStrategy(),
    'day': DayTradingStrategy(),
    'trend': TrendFollowingStrategy(),
    'scalping': ScalpingStrategy()
}

# Use with orchestrator (already integrated)
```

## Strategy Selection Guide

Based on CMC Markets recommendations:

1. **News Trading**: Requires expert skills, quick decision-making
2. **End-of-Day**: Suitable for most traders, less time commitment
3. **Swing Trading**: Technical approach, requires patience
4. **Day Trading**: Requires discipline, no overnight risk
5. **Trend Trading**: Useful hobby, many opportunities
6. **Scalping**: Extremely tense, not for beginners, requires high volatility

## Risk Management

All strategies include:
- Stop loss calculation (ATR-based)
- Take profit targets (risk/reward ratios)
- Position sizing considerations
- Exit logic in `should_exit()` method

## Configuration

Each strategy can be configured via the `config` parameter. Default values are provided but should be adjusted based on:
- Market conditions
- Asset class (crypto, forex, stocks)
- Timeframe
- Risk tolerance

## Testing

Before using in live trading:
1. Backtest each strategy
2. Paper trade for validation
3. Start with small position sizes
4. Monitor performance metrics
5. Adjust configuration as needed

## References

- [CMC Markets Trading Strategies Guide](https://www.cmcmarkets.com/en-ie/trading-guides/trading-strategies)
- Strategy implementations follow CMC Markets best practices
- All strategies include proper risk management

