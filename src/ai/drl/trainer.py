"""DRL training infrastructure"""

# Patch gymnasium as gym BEFORE stable-baselines3 imports it
try:
    import gymnasium as gym
    import sys
    sys.modules['gym'] = gym  # Make gymnasium available as 'gym' for stable-baselines3
except ImportError:
    pass  # If gymnasium not available, let stable-baselines3 use gym

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
import os
from .drl_agent import DRLAgent

try:
    from stable_baselines3 import PPO, A2C
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import BaseCallback
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False


class TradingEnvironment:
    """Trading environment for DRL training"""
    
    def __init__(self, data: pd.DataFrame, initial_capital: float = 10000.0,
                 commission_rate: float = 0.001, slippage_bps: float = 5.0,
                 drl_agent: Optional[DRLAgent] = None):
        """
        Initialize trading environment
        
        Args:
            data: Historical OHLCV data with indicators
            initial_capital: Starting capital
            commission_rate: Commission per trade
            slippage_bps: Slippage in basis points
            drl_agent: DRL agent for reward calculation (optional)
        """
        self.data = data
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps / 10000.0
        self.drl_agent = drl_agent
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        self.capital = self.initial_capital
        self.position = 0.0  # Position size (-1 to 1)
        self.entry_price = None
        self.equity_history = [self.initial_capital]
        self.peak_equity = self.initial_capital
        self.max_drawdown = 0.0
        self.trades = []
        
        # Performance tracking for enhanced rewards
        self.weekly_returns = []  # Track weekly returns
        self.weekly_start_capital = self.initial_capital
        self.weekly_start_step = 0
        self.consecutive_wins = 0  # Track consecutive profitable periods
        self.returns_history = []  # Track period returns for Sharpe calculation
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state vector"""
        if self.current_step >= len(self.data):
            return np.zeros(20)  # Return zero state at end
        
        row = self.data.iloc[self.current_step]
        
        # Build state from available features
        state = [
            row.get('rsi', 50.0) / 100.0,
            row.get('macd', 0.0),
            row.get('macd_signal', 0.0),
            row.get('bb_upper', row['close']) / row['close'] - 1.0,
            row.get('bb_lower', row['close']) / row['close'] - 1.0,
            row.get('adx', 0.0) / 100.0,
            row.get('atr', 0.0) / row['close'],
            row['close'] / row['close'] - 1.0,  # Price change (normalized)
            row['volume'] / row['volume'].mean() if hasattr(row['volume'], 'mean') else 1.0,
            self.position,  # Current position
            self.capital / self.initial_capital - 1.0,  # Capital change
            self.max_drawdown,
        ]
        
        # Pad to 20 dimensions
        while len(state) < 20:
            state.append(0.0)
        
        return np.array(state[:20], dtype=np.float32)
    
    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return next state, reward, done, info
        
        Args:
            action: Position size in [-1.0, 1.0]
        
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if self.current_step >= len(self.data) - 1:
            return self._get_state(), 0.0, True, {}
        
        current_price = self.data.iloc[self.current_step]['close']
        next_price = self.data.iloc[self.current_step + 1]['close']
        
        # Calculate P&L from previous position
        pnl_delta = 0.0
        transaction_costs = 0.0
        
        if self.position != 0.0 and self.entry_price:
            # Close existing position
            price_change = (next_price - self.entry_price) / self.entry_price
            if self.position < 0:
                price_change = -price_change  # Short position
            
            position_value = abs(self.position) * self.capital
            pnl_delta = position_value * price_change
            transaction_costs += position_value * (self.commission_rate + self.slippage_bps)
        
        # Update position
        if abs(action - self.position) > 0.01:  # Only trade if significant change
            # Close old position
            if self.position != 0.0:
                transaction_costs += abs(self.position) * self.capital * (self.commission_rate + self.slippage_bps)
            
            # Open new position
            self.position = np.clip(action, -1.0, 1.0)
            self.entry_price = next_price
            transaction_costs += abs(self.position) * self.capital * (self.commission_rate + self.slippage_bps)
        
        # Update capital
        self.capital += pnl_delta - transaction_costs
        self.equity_history.append(self.capital)
        
        # Update peak equity and drawdown
        if self.capital > self.peak_equity:
            self.peak_equity = self.capital
        
        current_drawdown = (self.peak_equity - self.capital) / self.peak_equity
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        # Calculate reward
        reward = (1.0 * pnl_delta - 
                 10.0 * self.max_drawdown - 
                 0.1 * transaction_costs)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        info = {
            'capital': self.capital,
            'drawdown': self.max_drawdown,
            'pnl': pnl_delta
        }
        
        return self._get_state(), reward, done, info


class DRLTrainer:
    """Trainer for DRL agent"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize DRL trainer
        
        Config parameters:
            models_dir: Directory to save models (default: 'models')
            total_timesteps: Total training timesteps (default: 100000)
            learning_rate: Learning rate (default: 3e-4)
        """
        self.config = config or {}
        self.models_dir = self.config.get('models_dir', 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        self.total_timesteps = self.config.get('total_timesteps', 100000)
    
    def train(self, drl_agent: DRLAgent, data: pd.DataFrame, 
             initial_capital: float = 10000.0) -> Dict:
        """
        Train DRL agent on historical data
        
        Args:
            drl_agent: DRL agent to train
            data: Historical OHLCV data with indicators
            initial_capital: Starting capital for simulation
        
        Returns:
            Training metrics dictionary
        """
        if not STABLE_BASELINES_AVAILABLE or not drl_agent.use_stable_baselines:
            print("Warning: stable-baselines3 not available. Using simplified training.")
            return self._train_pytorch(drl_agent, data, initial_capital)
        
        # Create environment with DRL agent for enhanced rewards
        env = TradingEnvironment(data, initial_capital, drl_agent=drl_agent)
        
        # Create model
        if drl_agent.algorithm == 'ppo':
            model = PPO('MlpPolicy', env, learning_rate=drl_agent.learning_rate, 
                       verbose=1, tensorboard_log=f"{self.models_dir}/tensorboard/")
        else:
            model = A2C('MlpPolicy', env, learning_rate=drl_agent.learning_rate,
                       verbose=1, tensorboard_log=f"{self.models_dir}/tensorboard/")
        
        # Train
        model.learn(total_timesteps=self.total_timesteps)
        
        # Save model
        model_path = os.path.join(self.models_dir, f"drl_{drl_agent.algorithm}.zip")
        model.save(model_path)
        
        drl_agent.model = model
        drl_agent.is_trained = True
        
        # Evaluate
        metrics = self._evaluate_model(model, env)
        
        return metrics
    
    def _train_pytorch(self, drl_agent: DRLAgent, data: pd.DataFrame,
                      initial_capital: float) -> Dict:
        """Simplified PyTorch training (placeholder)"""
        print("PyTorch training not fully implemented. Install stable-baselines3 for full DRL training.")
        return {'status': 'incomplete', 'message': 'PyTorch training requires stable-baselines3'}
    
    def _evaluate_model(self, model, env: TradingEnvironment) -> Dict:
        """Evaluate trained model"""
        obs = env.reset()
        total_reward = 0.0
        steps = 0
        
        while steps < len(env.data) - 1:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            if done:
                break
        
        final_capital = env.capital
        return_on_capital = (final_capital - env.initial_capital) / env.initial_capital
        
        return {
            'total_reward': total_reward,
            'final_capital': final_capital,
            'return_pct': return_on_capital * 100,
            'max_drawdown': env.max_drawdown * 100,
            'steps': steps
        }

