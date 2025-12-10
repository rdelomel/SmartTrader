"""Walk-Forward Optimization (WFO) for DRL models"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from ..ai.drl.drl_agent import DRLAgent
from ..ai.drl.trainer import DRLTrainer
import os


class WalkForwardOptimizer:
    """Walk-Forward Optimization engine for DRL models"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize WFO engine
        
        Config parameters:
            in_sample_days: Days for in-sample training (default: 60)
            out_of_sample_days: Days for out-of-sample testing (default: 30)
            step_days: Days to step forward each iteration (default: 30)
            min_performance_threshold: Minimum Sharpe ratio to retrain (default: 0.5)
            models_dir: Directory to save models (default: 'models')
        """
        self.config = config or {}
        self.in_sample_days = self.config.get('in_sample_days', 60)
        self.out_of_sample_days = self.config.get('out_of_sample_days', 30)
        self.step_days = self.config.get('step_days', 30)
        self.min_performance_threshold = self.config.get('min_performance_threshold', 0.5)
        self.models_dir = self.config.get('models_dir', 'models')
        os.makedirs(self.models_dir, exist_ok=True)
    
    def run_wfo(
        self,
        data: pd.DataFrame,
        drl_agent: DRLAgent,
        drl_trainer: DRLTrainer,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """
        Run walk-forward optimization
        
        Args:
            data: Historical OHLCV data with indicators
            drl_agent: DRL agent to train
            drl_trainer: DRL trainer
            start_date: Start date for WFO (default: earliest in data)
            end_date: End date for WFO (default: latest in data)
        
        Returns:
            WFO results dictionary
        """
        if data.empty:
            return {'error': 'No data provided'}
        
        # Set date range
        if start_date is None:
            start_date = data.index[0]
        if end_date is None:
            end_date = data.index[-1]
        
        # Filter data
        wfo_data = data[(data.index >= start_date) & (data.index <= end_date)]
        
        if len(wfo_data) < self.in_sample_days + self.out_of_sample_days:
            return {'error': 'Insufficient data for WFO'}
        
        results = []
        current_date = start_date + timedelta(days=self.in_sample_days)
        
        iteration = 0
        while current_date + timedelta(days=self.out_of_sample_days) <= end_date:
            iteration += 1
            print(f"WFO Iteration {iteration}: Training on {start_date.date()} to {current_date.date()}, Testing on {current_date.date()} to {(current_date + timedelta(days=self.out_of_sample_days)).date()}")
            
            # Split data
            in_sample_end = current_date
            out_of_sample_start = current_date
            out_of_sample_end = current_date + timedelta(days=self.out_of_sample_days)
            
            in_sample_data = wfo_data[(wfo_data.index >= start_date) & (wfo_data.index < in_sample_end)]
            out_of_sample_data = wfo_data[(wfo_data.index >= out_of_sample_start) & (wfo_data.index < out_of_sample_end)]
            
            if len(in_sample_data) < 30 or len(out_of_sample_data) < 10:
                current_date += timedelta(days=self.step_days)
                continue
            
            # Train on in-sample data
            try:
                training_metrics = drl_trainer.train(drl_agent, in_sample_data)
                
                # Test on out-of-sample data
                test_metrics = self._test_model(drl_agent, out_of_sample_data)
                
                # Calculate performance metrics
                sharpe_ratio = test_metrics.get('sharpe_ratio', 0.0)
                max_drawdown = test_metrics.get('max_drawdown', 0.0)
                return_pct = test_metrics.get('return_pct', 0.0)
                
                results.append({
                    'iteration': iteration,
                    'in_sample_start': start_date,
                    'in_sample_end': in_sample_end,
                    'out_of_sample_start': out_of_sample_start,
                    'out_of_sample_end': out_of_sample_end,
                    'training_metrics': training_metrics,
                    'test_metrics': test_metrics,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'return_pct': return_pct,
                    'passed': sharpe_ratio >= self.min_performance_threshold and max_drawdown <= 4.5
                })
                
                # Save model if performance is good
                if sharpe_ratio >= self.min_performance_threshold and max_drawdown <= 4.5:
                    model_path = os.path.join(self.models_dir, f"drl_wfo_iter_{iteration}.zip")
                    drl_agent.save(model_path)
                    print(f"Model saved: {model_path} (Sharpe: {sharpe_ratio:.2f}, DD: {max_drawdown:.2f}%)")
                
            except Exception as e:
                print(f"Error in WFO iteration {iteration}: {e}")
            
            # Step forward
            current_date += timedelta(days=self.step_days)
        
        # Aggregate results
        if not results:
            return {'error': 'No valid WFO iterations completed'}
        
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
        avg_drawdown = np.mean([r['max_drawdown'] for r in results])
        avg_return = np.mean([r['return_pct'] for r in results])
        pass_rate = sum(1 for r in results if r['passed']) / len(results)
        
        return {
            'iterations': results,
            'summary': {
                'total_iterations': len(results),
                'avg_sharpe_ratio': avg_sharpe,
                'avg_max_drawdown': avg_drawdown,
                'avg_return_pct': avg_return,
                'pass_rate': pass_rate,
                'best_iteration': max(results, key=lambda x: x['sharpe_ratio']) if results else None
            }
        }
    
    def _test_model(self, drl_agent: DRLAgent, test_data: pd.DataFrame) -> Dict:
        """Test model on out-of-sample data"""
        from ..backtesting.engine import BacktestEngine
        
        # Create simple backtest
        initial_capital = 10000.0
        capital = initial_capital
        equity_history = [capital]
        peak_equity = capital
        max_drawdown = 0.0
        returns = []
        
        position = 0.0
        entry_price = None
        
        for i in range(len(test_data) - 1):
            current_price = test_data.iloc[i]['close']
            next_price = test_data.iloc[i + 1]['close']
            
            # Build state (simplified for testing)
            state = np.random.rand(20)  # Placeholder - would use actual state building
            action, _ = drl_agent.predict(state, deterministic=True)
            
            # Update position
            if abs(action - position) > 0.1:
                # Close old position
                if position != 0.0 and entry_price:
                    pnl = position * (next_price - entry_price) / entry_price * capital * abs(position)
                    capital += pnl
                    returns.append(pnl / capital)
                
                # Open new position
                position = np.clip(action, -1.0, 1.0)
                entry_price = next_price
            
            # Update equity
            if position != 0.0 and entry_price:
                unrealized_pnl = position * (next_price - entry_price) / entry_price * capital * abs(position)
                equity = capital + unrealized_pnl
            else:
                equity = capital
            
            equity_history.append(equity)
            
            if equity > peak_equity:
                peak_equity = equity
            
            current_dd = (peak_equity - equity) / peak_equity
            if current_dd > max_drawdown:
                max_drawdown = current_dd
        
        # Calculate metrics
        equity_series = pd.Series(equity_history)
        total_return = (equity_series.iloc[-1] - initial_capital) / initial_capital
        
        if len(returns) > 1:
            returns_series = pd.Series(returns)
            sharpe_ratio = returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        return {
            'final_capital': equity_series.iloc[-1],
            'return_pct': total_return * 100,
            'max_drawdown': max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'equity_curve': equity_series
        }
    
    def should_retrain(self, current_performance: Dict, wfo_results: Dict) -> bool:
        """
        Determine if model should be retrained based on performance
        
        Args:
            current_performance: Current live performance metrics
            wfo_results: Latest WFO results
        
        Returns:
            True if retraining is recommended
        """
        current_sharpe = current_performance.get('sharpe_ratio', 0.0)
        current_dd = current_performance.get('max_drawdown', 0.0)
        
        # Retrain if performance degrades
        if current_sharpe < self.min_performance_threshold:
            return True
        
        if current_dd > 4.5:  # Exceeded max drawdown
            return True
        
        # Compare to WFO average
        if wfo_results and 'summary' in wfo_results:
            avg_sharpe = wfo_results['summary'].get('avg_sharpe_ratio', 0.0)
            if current_sharpe < avg_sharpe * 0.7:  # 30% degradation
                return True
        
        return False

