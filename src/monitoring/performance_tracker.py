"""Performance tracking and monitoring for aggressive trading"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from collections import deque


class PerformanceTracker:
    """Track trading performance metrics"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize performance tracker
        
        Config parameters:
            target_weekly_return: Target weekly return percentage (default: 10.0)
            max_drawdown: Maximum allowed drawdown (default: 4.5)
            lookback_days: Days to look back for metrics (default: 30)
        """
        self.config = config or {}
        self.target_weekly_return = self.config.get('target_weekly_return', 10.0)
        self.max_drawdown = self.config.get('max_drawdown', 4.5)
        self.lookback_days = self.config.get('lookback_days', 30)
        
        # Performance history
        self.equity_history = deque(maxlen=10000)
        self.trade_history = deque(maxlen=1000)
        self.weekly_returns = deque(maxlen=52)  # Track 1 year of weekly returns
    
    def update_equity(self, equity: float, timestamp: Optional[datetime] = None):
        """Update equity value"""
        if timestamp is None:
            timestamp = datetime.now()
        self.equity_history.append((timestamp, equity))
    
    def record_trade(self, trade: Dict):
        """Record a completed trade"""
        self.trade_history.append(trade)
    
    def calculate_metrics(self) -> Dict:
        """Calculate current performance metrics"""
        if len(self.equity_history) < 2:
            return {
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'weekly_return': 0.0,
                'annualized_return': 0.0,
                'total_trades': 0
            }
        
        # Convert to DataFrame
        equity_df = pd.DataFrame(list(self.equity_history), columns=['timestamp', 'equity'])
        equity_df.set_index('timestamp', inplace=True)
        
        # Calculate returns
        returns = equity_df['equity'].pct_change().dropna()
        
        # Sharpe ratio (annualized)
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Max drawdown
        peak = equity_df['equity'].expanding().max()
        drawdown = (equity_df['equity'] - peak) / peak * 100
        max_drawdown = abs(drawdown.min())
        
        # Weekly return
        weekly_return = self._calculate_weekly_return(equity_df)
        
        # Annualized return
        if len(equity_df) > 1:
            total_return = (equity_df['equity'].iloc[-1] - equity_df['equity'].iloc[0]) / equity_df['equity'].iloc[0]
            days = (equity_df.index[-1] - equity_df.index[0]).days
            if days > 0:
                annualized_return = (1 + total_return) ** (365 / days) - 1
            else:
                annualized_return = 0.0
        else:
            annualized_return = 0.0
        
        # Win rate
        if len(self.trade_history) > 0:
            winning_trades = sum(1 for t in self.trade_history if t.get('pnl', 0) > 0)
            win_rate = winning_trades / len(self.trade_history) * 100
        else:
            win_rate = 0.0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'weekly_return': weekly_return,
            'annualized_return': annualized_return * 100,
            'total_trades': len(self.trade_history),
            'current_equity': equity_df['equity'].iloc[-1] if len(equity_df) > 0 else 0.0,
            'peak_equity': equity_df['equity'].max() if len(equity_df) > 0 else 0.0
        }
    
    def _calculate_weekly_return(self, equity_df: pd.DataFrame) -> float:
        """Calculate weekly return"""
        if len(equity_df) < 2:
            return 0.0
        
        # Get equity one week ago
        one_week_ago = equity_df.index[-1] - timedelta(days=7)
        week_ago_equity = None
        
        for timestamp in equity_df.index:
            if timestamp <= one_week_ago:
                week_ago_equity = equity_df.loc[timestamp, 'equity']
            else:
                break
        
        if week_ago_equity is None or week_ago_equity == 0:
            return 0.0
        
        current_equity = equity_df['equity'].iloc[-1]
        weekly_return = ((current_equity - week_ago_equity) / week_ago_equity) * 100
        
        # Store for tracking
        self.weekly_returns.append(weekly_return)
        
        return weekly_return
    
    def check_overfitting(self, backtest_sharpe: float, live_sharpe: float) -> Dict:
        """
        Check for signs of overfitting
        
        Args:
            backtest_sharpe: Sharpe ratio from backtesting
            live_sharpe: Sharpe ratio from live trading
        
        Returns:
            Overfitting analysis
        """
        degradation = ((backtest_sharpe - live_sharpe) / backtest_sharpe * 100) if backtest_sharpe > 0 else 0.0
        
        is_overfit = degradation > 30.0  # More than 30% degradation suggests overfitting
        
        return {
            'is_overfit': is_overfit,
            'degradation_pct': degradation,
            'backtest_sharpe': backtest_sharpe,
            'live_sharpe': live_sharpe,
            'recommendation': 'Retrain model' if is_overfit else 'Model performing as expected'
        }
    
    def check_performance_degradation(self, current_metrics: Dict, baseline_metrics: Dict) -> Dict:
        """
        Check if performance has degraded
        
        Args:
            current_metrics: Current performance metrics
            baseline_metrics: Baseline performance metrics
        
        Returns:
            Degradation analysis
        """
        sharpe_degradation = ((baseline_metrics.get('sharpe_ratio', 0) - current_metrics.get('sharpe_ratio', 0)) / 
                             baseline_metrics.get('sharpe_ratio', 1)) * 100 if baseline_metrics.get('sharpe_ratio', 0) > 0 else 0
        
        dd_increase = current_metrics.get('max_drawdown', 0) - baseline_metrics.get('max_drawdown', 0)
        
        return_degradation = ((baseline_metrics.get('weekly_return', 0) - current_metrics.get('weekly_return', 0)) / 
                             baseline_metrics.get('weekly_return', 1)) * 100 if baseline_metrics.get('weekly_return', 0) > 0 else 0
        
        is_degraded = (sharpe_degradation > 30.0 or 
                      dd_increase > 1.0 or 
                      return_degradation > 50.0)
        
        return {
            'is_degraded': is_degraded,
            'sharpe_degradation_pct': sharpe_degradation,
            'drawdown_increase': dd_increase,
            'return_degradation_pct': return_degradation,
            'alert': is_degraded
        }
    
    def generate_report(self) -> Dict:
        """Generate performance report"""
        metrics = self.calculate_metrics()
        
        # Check against targets
        target_status = {
            'weekly_return_on_track': metrics['weekly_return'] >= self.target_weekly_return * 0.8,  # 80% of target
            'drawdown_within_limit': metrics['max_drawdown'] <= self.max_drawdown,
            'sharpe_acceptable': metrics['sharpe_ratio'] >= 1.0
        }
        
        return {
            'metrics': metrics,
            'targets': {
                'weekly_return_target': self.target_weekly_return,
                'max_drawdown_limit': self.max_drawdown
            },
            'target_status': target_status,
            'timestamp': datetime.now().isoformat()
        }

