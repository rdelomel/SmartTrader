"""Structured logging for trading system"""

import logging
import structlog
import json
from datetime import datetime
from typing import Dict, Optional
import os
from pathlib import Path


class TradingLogger:
    """Structured logger for trading operations"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize logger
        
        Config parameters:
            log_level: Logging level (default: INFO)
            log_file: Path to log file (default: logs/trading_agent.log)
            json_format: Use JSON format (default: True)
        """
        self.config = config or {}
        log_level = self.config.get('log_level', 'INFO')
        log_file = self.config.get('log_file', 'logs/trading_agent.log')
        json_format = self.config.get('json_format', True)
        
        # Create logs directory
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure structlog
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ]
        
        if json_format:
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())
        
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        # Configure file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level))
        
        # Configure console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level))
        
        # Get logger
        self.logger = structlog.get_logger()
        
        # Add handlers to root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level))
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    
    def log_trade(self, trade_data: Dict):
        """Log a trade"""
        self.logger.info(
            "trade_executed",
            **trade_data
        )
    
    def log_signal(self, signal_data: Dict):
        """Log a trading signal"""
        self.logger.info(
            "signal_generated",
            **signal_data
        )
    
    def log_decision(self, decision_data: Dict, event: str = "decision_made"):
        """Log a trading decision"""
        # Remove 'event' from decision_data if present to avoid conflict with positional argument
        decision_data = {k: v for k, v in decision_data.items() if k != 'event'}
        self.logger.info(
            event,
            **decision_data
        )
    
    def log_error(self, error: Exception, context: Optional[Dict] = None):
        """Log an error"""
        error_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
        }
        if context:
            error_data.update(context)
        
        self.logger.error(
            "error_occurred",
            **error_data,
            exc_info=True
        )
    
    def log_risk_event(self, event_data: Dict):
        """Log a risk management event"""
        # Avoid passing duplicate 'event' parameter
        event_data = {k: v for k, v in event_data.items() if k != 'event'}
        self.logger.warning(
            "risk_event",
            **event_data
        )
    
    def log_performance(self, performance_data: Dict):
        """Log performance metrics"""
        self.logger.info(
            "performance_update",
            **performance_data
        )

