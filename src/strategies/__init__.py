"""Trading strategies module"""

try:
    from .base_strategy import BaseStrategy
    from .trend_following import TrendFollowingStrategy
    from .mean_reversion import MeanReversionStrategy
except ImportError:
    # Fallback for when not used as a package
    from base_strategy import BaseStrategy
    from trend_following import TrendFollowingStrategy
    from mean_reversion import MeanReversionStrategy

__all__ = ['BaseStrategy', 'TrendFollowingStrategy', 'MeanReversionStrategy']

