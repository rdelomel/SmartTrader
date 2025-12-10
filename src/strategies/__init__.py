"""Trading strategies module"""

try:
    from .base_strategy import BaseStrategy, Signal
    from .trend_following import TrendFollowingStrategy
    from .mean_reversion import MeanReversionStrategy
    from .momentum import MomentumStrategy
    from .news_trading import NewsTradingStrategy
    from .end_of_day import EndOfDayStrategy
    from .swing_trading import SwingTradingStrategy
    from .day_trading import DayTradingStrategy
    from .scalping import ScalpingStrategy
except ImportError:
    # Fallback for when not used as a package
    from base_strategy import BaseStrategy, Signal
    from trend_following import TrendFollowingStrategy
    from mean_reversion import MeanReversionStrategy
    from momentum import MomentumStrategy
    from news_trading import NewsTradingStrategy
    from end_of_day import EndOfDayStrategy
    from swing_trading import SwingTradingStrategy
    from day_trading import DayTradingStrategy
    from scalping import ScalpingStrategy

__all__ = [
    'BaseStrategy',
    'Signal',
    'TrendFollowingStrategy',
    'MeanReversionStrategy',
    'MomentumStrategy',
    'NewsTradingStrategy',
    'EndOfDayStrategy',
    'SwingTradingStrategy',
    'DayTradingStrategy',
    'ScalpingStrategy'
]

