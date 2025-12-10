"""Base strategy interface"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import pandas as pd
from enum import Enum


class Signal(Enum):
    """Trading signal enumeration"""
    BUY = 1
    SELL = -1
    HOLD = 0


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, name: str, config: Optional[Dict] = None):
        """
        Initialize strategy
        
        Args:
            name: Strategy name
            config: Strategy configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        self.weight = self.config.get('weight', 1.0)
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """
        Generate trading signal based on data
        
        Args:
            data: DataFrame with OHLCV and indicator data
        
        Returns:
            Dictionary with signal information:
            {
                'signal': Signal (BUY/SELL/HOLD),
                'confidence': float (0-1),
                'entry_price': float,
                'stop_loss': float,
                'take_profit': float,
                'reason': str
            }
        """
        pass
    
    @abstractmethod
    def should_exit(self, data: pd.DataFrame, position: Dict) -> bool:
        """
        Check if position should be exited
        
        Args:
            data: Current market data
            position: Current position information
        
        Returns:
            True if position should be exited, False otherwise
        """
        pass
    
    def get_weight(self) -> float:
        """Get strategy weight for signal aggregation"""
        return self.weight if self.enabled else 0.0
    
    def is_enabled(self) -> bool:
        """Check if strategy is enabled"""
        return self.enabled

