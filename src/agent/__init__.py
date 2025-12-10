"""Agent module for multi-agent orchestration framework"""

from .base_agent import BaseAgent
from .analytical_agents import TechnicalAnalystAgent, SentimentAgent, FundamentalAgent
from .quantitative_agent import QuantitativeAgent
from .pattern_forecaster_agent import PatternForecasterAgent
from .regime_switching_agent import RegimeSwitchingAgent
from .orchestrator_agent import OrchestratorAgent
from .risk_manager_agent import RiskManagerAgent

__all__ = [
    'BaseAgent',
    'TechnicalAnalystAgent',
    'SentimentAgent',
    'FundamentalAgent',
    'QuantitativeAgent',
    'PatternForecasterAgent',
    'RegimeSwitchingAgent',
    'OrchestratorAgent',
    'RiskManagerAgent'
]
