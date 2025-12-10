"""Backtesting performance metrics"""

import pandas as pd
import numpy as np
from typing import List, Dict
from ..ai.training.evaluator import ModelEvaluator


class BacktestMetrics:
    """Calculate backtesting performance metrics"""
    
    def __init__(self):
        self.evaluator = ModelEvaluator()
    
    def calculate_all_metrics(
        self,
        equity_curve: pd.Series,
        trades: List[Dict],
        initial_capital: float
    ) -> Dict:
        """
        Calculate all performance metrics
        
        Args:
            equity_curve: Series of equity values over time
            trades: List of completed trades
            initial_capital: Initial capital
        
        Returns:
            Dictionary with all metrics
        """
        if equity_curve.empty:
            return self._empty_metrics()
        
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        # Basic metrics
        total_return = (equity_curve.iloc[-1] - initial_capital) / initial_capital * 100
        annualized_return = self._annualize_return(returns, total_return)
        
        # Risk metrics
        sharpe_ratio = self.evaluator.calculate_sharpe_ratio(returns)
        sortino_ratio = self.evaluator.calculate_sortino_ratio(returns)
        max_drawdown = self.evaluator.calculate_max_drawdown(equity_curve)
        
        # Trade metrics
        win_rate = self.evaluator.calculate_win_rate(trades)
        profit_factor = self.evaluator.calculate_profit_factor(trades)
        
        # Additional metrics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        
        avg_win = np.mean([t.get('pnl', 0) for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.get('pnl', 0) for t in losing_trades]) if losing_trades else 0
        
        # Volatility
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        return {
            'total_return_percent': total_return,
            'annualized_return_percent': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown_percent': max_drawdown * 100,
            'calmar_ratio': calmar_ratio,
            'volatility_percent': volatility,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'average_win': avg_win,
            'average_loss': avg_loss,
            'largest_win': max([t.get('pnl', 0) for t in trades], default=0),
            'largest_loss': min([t.get('pnl', 0) for t in trades], default=0),
            'final_equity': equity_curve.iloc[-1] if not equity_curve.empty else initial_capital
        }
    
    def _annualize_return(self, returns: pd.Series, total_return: float) -> float:
        """Calculate annualized return"""
        if len(returns) == 0:
            return 0.0
        
        # Estimate trading days
        trading_days = len(returns)
        years = trading_days / 252
        
        if years <= 0:
            return 0.0
        
        # Annualized return
        annualized = ((1 + total_return / 100) ** (1 / years) - 1) * 100
        
        return annualized
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics dictionary"""
        return {
            'total_return_percent': 0.0,
            'annualized_return_percent': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown_percent': 0.0,
            'calmar_ratio': 0.0,
            'volatility_percent': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'final_equity': 0.0
        }
    
    def generate_report(self, metrics: Dict) -> str:
        """
        Generate text report from metrics
        
        Args:
            metrics: Dictionary of metrics
        
        Returns:
            Formatted report string
        """
        report = f"""
Backtest Performance Report
===========================

Returns:
  Total Return: {metrics['total_return_percent']:.2f}%
  Annualized Return: {metrics['annualized_return_percent']:.2f}%

Risk Metrics:
  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
  Sortino Ratio: {metrics['sortino_ratio']:.2f}
  Max Drawdown: {metrics['max_drawdown_percent']:.2f}%
  Calmar Ratio: {metrics['calmar_ratio']:.2f}
  Volatility: {metrics['volatility_percent']:.2f}%

Trade Statistics:
  Total Trades: {metrics['total_trades']}
  Winning Trades: {metrics['winning_trades']}
  Losing Trades: {metrics['losing_trades']}
  Win Rate: {metrics['win_rate']:.2%}
  Profit Factor: {metrics['profit_factor']:.2f}

Trade Performance:
  Average Win: ${metrics['average_win']:.2f}
  Average Loss: ${metrics['average_loss']:.2f}
  Largest Win: ${metrics['largest_win']:.2f}
  Largest Loss: ${metrics['largest_loss']:.2f}

Final Equity: ${metrics['final_equity']:.2f}
"""
        return report

