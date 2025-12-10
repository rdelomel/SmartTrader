"""Base agent class for all trading agents"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from datetime import datetime


class BaseAgent(ABC):
    """Base class for all trading agents"""
    
    def __init__(self, name: str, config: Optional[Dict] = None):
        """
        Initialize base agent
        
        Args:
            name: Agent name/identifier
            config: Agent configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        self.last_update = None
        self.performance_metrics = {}
    
    @abstractmethod
    def analyze(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Perform analysis and return signal/result
        
        Must be implemented by subclasses
        
        Returns:
            Dictionary with analysis results
        """
        pass
    
    def is_enabled(self) -> bool:
        """Check if agent is enabled"""
        return self.enabled
    
    def enable(self):
        """Enable the agent"""
        self.enabled = True
    
    def disable(self):
        """Disable the agent"""
        self.enabled = False
    
    def update_config(self, config: Dict):
        """Update agent configuration"""
        self.config.update(config)
    
    def get_status(self) -> Dict:
        """Get agent status"""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'performance_metrics': self.performance_metrics
        }
    
    def _update_timestamp(self):
        """Update last update timestamp"""
        self.last_update = datetime.now()

