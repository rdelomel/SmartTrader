"""Strategy selector for adaptive strategy switching"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from ..strategies.base_strategy import BaseStrategy
from ..ai.training.evaluator import ModelEvaluator


class StrategySelector:
    """Monitors and selects best performing strategies"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize strategy selector
        
        Config parameters:
            performance_window_days: Days to look back for performance (default: 30)
            min_sharpe_ratio: Minimum Sharpe ratio to continue (default: 0.5)
            min_accuracy: Minimum accuracy for classification models (default: 0.52)
            retrain_threshold: Accuracy threshold to trigger retraining (default: 0.48)
        """
        self.config = config or {}
        self.performance_window_days = self.config.get('performance_window_days', 30)
        self.min_sharpe_ratio = self.config.get('min_sharpe_ratio', 0.5)
        self.min_accuracy = self.config.get('min_accuracy', 0.52)
        self.retrain_threshold = self.config.get('retrain_threshold', 0.48)
        self.evaluator = ModelEvaluator()
        self.strategy_performance: Dict[str, Dict] = {}
    
    def evaluate_strategy(
        self,
        strategy: BaseStrategy,
        trades: List[Dict],
        start_date: datetime
    ) -> Dict:
        """
        Evaluate strategy performance
        
        Args:
            strategy: Strategy to evaluate
            trades: List of trades from this strategy
            start_date: Start date for evaluation period
        
        Returns:
            Performance metrics dictionary
        """
        if not trades:
            return {
                'sharpe_ratio': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'trade_count': 0
            }
        
        # Filter trades in performance window
        cutoff_date = start_date - timedelta(days=self.performance_window_days)
        recent_trades = [t for t in trades if t.get('entry_time', datetime.now()) >= cutoff_date]
        
        if not recent_trades:
            return {
                'sharpe_ratio': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'trade_count': 0
            }
        
        # Calculate returns
        returns = pd.Series([t.get('pnl_percent', 0) / 100 for t in recent_trades])
        
        # Calculate metrics
        sharpe = self.evaluator.calculate_sharpe_ratio(returns)
        win_rate = self.evaluator.calculate_win_rate(recent_trades)
        profit_factor = self.evaluator.calculate_profit_factor(recent_trades)
        total_return = returns.sum()
        
        # Calculate drawdown
        equity_curve = (1 + returns).cumprod()
        max_drawdown = self.evaluator.calculate_max_drawdown(equity_curve)
        
        performance = {
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'trade_count': len(recent_trades)
        }
        
        # Store performance
        self.strategy_performance[strategy.name] = performance
        
        return performance
    
    def should_disable_strategy(self, strategy: BaseStrategy) -> bool:
        """
        Check if strategy should be disabled due to poor performance
        
        Args:
            strategy: Strategy to check
        
        Returns:
            True if strategy should be disabled
        """
        performance = self.strategy_performance.get(strategy.name)
        
        if not performance:
            return False
        
        # Disable if Sharpe ratio too low
        if performance['sharpe_ratio'] < self.min_sharpe_ratio:
            return True
        
        # Disable if win rate too low
        if performance['win_rate'] < 0.4:
            return True
        
        # Disable if too many losses
        if performance['profit_factor'] < 0.5:
            return True
        
        return False
    
    def should_retrain_model(self, accuracy: float) -> bool:
        """
        Check if model should be retrained
        
        Args:
            accuracy: Current model accuracy
        
        Returns:
            True if model should be retrained
        """
        return accuracy < self.retrain_threshold
    
    def get_best_strategies(self, count: int = 3) -> List[str]:
        """
        Get best performing strategies
        
        Args:
            count: Number of strategies to return
        
        Returns:
            List of strategy names
        """
        if not self.strategy_performance:
            return []
        
        # Sort by Sharpe ratio
        sorted_strategies = sorted(
            self.strategy_performance.items(),
            key=lambda x: x[1].get('sharpe_ratio', 0),
            reverse=True
        )
        
        return [name for name, _ in sorted_strategies[:count]]
    
    def adjust_weights(self, strategies: List[BaseStrategy]) -> Dict[str, float]:
        """
        Adjust strategy weights based on performance
        
        Args:
            strategies: List of strategies
        
        Returns:
            Dictionary of adjusted weights
        """
        weights = {}
        
        if not self.strategy_performance:
            # Default weights if no performance data
            for strategy in strategies:
                weights[strategy.name] = strategy.get_weight()
            return weights
        
        # Calculate total performance score
        total_score = 0
        for strategy in strategies:
            perf = self.strategy_performance.get(strategy.name, {})
            score = perf.get('sharpe_ratio', 0) * perf.get('win_rate', 0)
            total_score += max(score, 0.1)  # Minimum weight
        
        # Assign weights proportional to performance
        for strategy in strategies:
            perf = self.strategy_performance.get(strategy.name, {})
            score = perf.get('sharpe_ratio', 0) * perf.get('win_rate', 0)
            
            if total_score > 0:
                weight = max(score / total_score, 0.1)  # Minimum 10% weight
            else:
                weight = 1.0 / len(strategies)  # Equal weights if no data
            
            weights[strategy.name] = weight
        
        return weights

