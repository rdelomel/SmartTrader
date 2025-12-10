"""Deep Reinforcement Learning agent for trading decisions"""

# Patch gymnasium as gym BEFORE stable-baselines3 imports it
try:
    import gymnasium as gym
    import sys
    sys.modules['gym'] = gym  # Make gymnasium available as 'gym' for stable-baselines3
except ImportError:
    pass  # If gymnasium not available, let stable-baselines3 use gym

import numpy as np
from typing import Dict, Optional, Tuple, List
import os
import pickle
from datetime import datetime

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

try:
    from stable_baselines3 import PPO, A2C
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import BaseCallback
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False
    print("Warning: stable-baselines3 not available. DRL will use PyTorch implementation.")


class DRLAgent:
    """Deep Reinforcement Learning agent for trading decisions"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize DRL agent
        
        Config parameters:
            algorithm: 'ppo' or 'a2c' (default: 'ppo')
            state_dim: Dimension of state space (default: 20)
            action_dim: Dimension of action space (default: 1, continuous)
            learning_rate: Learning rate (default: 3e-4)
            use_stable_baselines: Use stable-baselines3 if available (default: True)
            reward_alpha: Coefficient for PNL in reward function (default: 1.0)
            reward_beta: Coefficient for drawdown penalty (default: 10.0)
            reward_gamma: Coefficient for transaction costs (default: 0.1)
        """
        self.config = config or {}
        self.algorithm = self.config.get('algorithm', 'ppo').lower()
        self.state_dim = self.config.get('state_dim', 20)
        self.action_dim = self.config.get('action_dim', 1)
        self.learning_rate = self.config.get('learning_rate', 3e-4)
        self.use_stable_baselines = self.config.get('use_stable_baselines', True) and STABLE_BASELINES_AVAILABLE
        
        # Reward function coefficients
        self.reward_alpha = self.config.get('reward_alpha', 2.0)  # PNL weight (increased for aggressive mode)
        self.reward_beta = self.config.get('reward_beta', 8.0)  # Drawdown penalty (slightly reduced, still strict)
        self.reward_gamma = self.config.get('reward_gamma', 0.05)  # Transaction costs (reduced to encourage trades)
        self.reward_weekly_bonus = self.config.get('reward_weekly_bonus', 5.0)  # Weekly target bonus
        self.reward_sharpe_bonus = self.config.get('reward_sharpe_bonus', 3.0)  # Sharpe ratio bonus
        self.reward_momentum_bonus = self.config.get('reward_momentum_bonus', 2.0)  # Momentum bonus
        
        self.model = None
        self.is_trained = False
        self.state_normalizer = None  # For normalizing state inputs
        
        # PyTorch implementation (fallback)
        if not self.use_stable_baselines:
            self._create_pytorch_model()
    
    def _create_pytorch_model(self):
        """Create PyTorch-based DRL model (Actor-Critic)"""
        if not TORCH_AVAILABLE:
            print("Warning: PyTorch not available. PyTorch DRL model cannot be created.")
            return
        
        class ActorCritic(nn.Module):
            def __init__(self, state_dim, action_dim):
                super().__init__()
                # Shared feature extractor
                self.feature_extractor = nn.Sequential(
                    nn.Linear(state_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU()
                )
                
                # Actor (policy network)
                self.actor = nn.Sequential(
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, action_dim),
                    nn.Tanh()  # Output in [-1, 1] range
                )
                
                # Critic (value network)
                self.critic = nn.Sequential(
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1)
                )
            
            def forward(self, state):
                features = self.feature_extractor(state)
                action = self.actor(features)
                value = self.critic(features)
                return action, value
        
        self.pytorch_model = ActorCritic(self.state_dim, self.action_dim)
        self.optimizer = torch.optim.Adam(self.pytorch_model.parameters(), lr=self.learning_rate)
    
    def predict(self, state: np.ndarray, deterministic: bool = True) -> Tuple[float, float]:
        """
        Predict action from state
        
        Args:
            state: State vector from analytical agents
            deterministic: Whether to use deterministic policy
        
        Returns:
            Tuple of (action, confidence)
            action: Position sizing in [-1.0, 1.0] range
            confidence: Confidence score [0.0, 1.0]
        """
        if not self.is_trained:
            # Return neutral action if not trained
            return 0.0, 0.0
        
        # Normalize state
        if self.state_normalizer:
            state = self.state_normalizer.normalize(state)
        
        # Ensure state is correct shape
        if isinstance(state, list):
            state = np.array(state)
        if state.ndim == 1:
            state = state.reshape(1, -1)
        
        if self.use_stable_baselines and self.model:
            # Use stable-baselines3 model
            action, _ = self.model.predict(state, deterministic=deterministic)
            action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)
            # Confidence based on action magnitude
            confidence = min(abs(action_value), 1.0)
            return action_value, confidence
        else:
            # Use PyTorch model
            if not TORCH_AVAILABLE or not hasattr(self, 'pytorch_model'):
                # Return neutral action if PyTorch not available
                return 0.0, 0.0
            self.pytorch_model.eval()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                action, value = self.pytorch_model(state_tensor)
                action_value = float(action[0][0])
                confidence = min(abs(action_value), 1.0)
            return action_value, confidence
    
    def build_state_vector(self, agent_signals: Dict, regime_info: Dict, 
                          market_data: Dict, performance_metrics: Optional[Dict] = None,
                          position_info: Optional[Dict] = None) -> np.ndarray:
        """
        Build state vector from analytical agent outputs with expanded features
        
        Args:
            agent_signals: Dictionary with signals from Technical, Sentiment, Fundamental agents
            regime_info: Regime switching agent output
            market_data: Additional market data (volatility, volume, etc.)
            performance_metrics: Optional recent performance metrics (Sharpe, weekly return, etc.)
            position_info: Optional current position information
        
        Returns:
            Normalized state vector
        """
        state_features = []
        
        # Technical agent signal
        tech_signal = agent_signals.get('technical', {})
        state_features.append(tech_signal.get('score', 0.0))
        state_features.append(tech_signal.get('confidence', 0.0))
        
        # Sentiment agent signal
        sent_signal = agent_signals.get('sentiment', {})
        state_features.append(sent_signal.get('score', 0.0))
        state_features.append(sent_signal.get('confidence', 0.0))
        
        # Fundamental agent signal
        fund_signal = agent_signals.get('fundamental', {})
        state_features.append(fund_signal.get('score', 0.0))
        state_features.append(fund_signal.get('confidence', 0.0))
        
        # Regime information
        state_features.append(1.0 if regime_info.get('regime') == 'Bullish Trend' else 0.0)
        state_features.append(1.0 if regime_info.get('regime') == 'Bearish Trend' else 0.0)
        state_features.append(1.0 if regime_info.get('regime') == 'Consolidation' else 0.0)
        state_features.append(1.0 if regime_info.get('regime') == 'High Volatility' else 0.0)
        state_features.append(regime_info.get('confidence', 0.0))
        
        # Market data features
        state_features.append(market_data.get('volatility', 0.0))
        state_features.append(market_data.get('volume_ratio', 1.0))
        state_features.append(market_data.get('price_momentum', 0.0))
        state_features.append(market_data.get('rsi', 50.0) / 100.0)  # Normalize RSI
        state_features.append(market_data.get('adx', 0.0) / 100.0)  # Normalize ADX
        
        # Recent performance metrics (NEW)
        if performance_metrics:
            # Last 7 days return (normalized)
            state_features.append(min(performance_metrics.get('last_7d_return', 0.0) / 10.0, 1.0))
            # Sharpe ratio (normalized, assuming max Sharpe of 3.0)
            state_features.append(min(performance_metrics.get('sharpe_ratio', 0.0) / 3.0, 1.0))
            # Win rate (already 0-1)
            state_features.append(performance_metrics.get('win_rate', 0.5) / 100.0)
        else:
            state_features.extend([0.0, 0.0, 0.5])  # Default values
        
        # Position-level features (NEW)
        if position_info:
            # Current exposure as % of capital (normalized)
            state_features.append(min(position_info.get('exposure_pct', 0.0) / 10.0, 1.0))
            # Number of open positions (normalized, assuming max 10)
            state_features.append(min(position_info.get('num_positions', 0) / 10.0, 1.0))
        else:
            state_features.extend([0.0, 0.0])
        
        # Time-based features (NEW)
        now = datetime.now()
        # Hour of day (normalized to 0-1)
        state_features.append(now.hour / 24.0)
        # Day of week (normalized to 0-1, Monday=0, Sunday=6)
        state_features.append(now.weekday() / 6.0)
        # Is weekend (0 or 1)
        state_features.append(1.0 if now.weekday() >= 5 else 0.0)
        
        # Pad or truncate to state_dim
        while len(state_features) < self.state_dim:
            state_features.append(0.0)
        state_features = state_features[:self.state_dim]
        
        return np.array(state_features, dtype=np.float32)
    
    def calculate_reward(self, pnl_delta: float, max_drawdown: float, 
                        transaction_costs: float, weekly_return: float = 0.0,
                        sharpe_ratio: float = 0.0, consecutive_wins: int = 0) -> float:
        """
        Calculate reward using enhanced reward function
        
        R_t = α·ΔPNL_t + weekly_bonus + sharpe_bonus + momentum_bonus - β·MaxDrawdown_t - γ·TransactionCosts_t
        
        Args:
            pnl_delta: Change in profit/loss
            max_drawdown: Maximum drawdown (as positive value)
            transaction_costs: Transaction costs incurred
            weekly_return: Weekly return percentage (for bonus calculation)
            sharpe_ratio: Current Sharpe ratio (for bonus calculation)
            consecutive_wins: Number of consecutive profitable periods
        
        Returns:
            Reward value
        """
        # Base reward components
        base_reward = (self.reward_alpha * pnl_delta - 
                      self.reward_beta * max_drawdown - 
                      self.reward_gamma * transaction_costs)
        
        # Weekly return bonus (if target is met or exceeded)
        weekly_bonus = 0.0
        target_weekly_return = 3.0  # 3% weekly target
        if weekly_return >= target_weekly_return:
            # Bonus proportional to how much target is exceeded
            weekly_bonus = self.reward_weekly_bonus * (weekly_return / target_weekly_return)
        
        # Sharpe ratio bonus (for risk-adjusted returns)
        sharpe_bonus = 0.0
        if sharpe_ratio > 1.0:
            # Bonus increases with higher Sharpe ratio
            sharpe_bonus = self.reward_sharpe_bonus * min(sharpe_ratio / 2.0, 1.0)
        
        # Momentum bonus (for consecutive profitable periods)
        momentum_bonus = 0.0
        if consecutive_wins > 0:
            # Bonus increases with win streak (capped at 5 consecutive wins)
            momentum_bonus = self.reward_momentum_bonus * min(consecutive_wins / 5.0, 1.0)
        
        total_reward = base_reward + weekly_bonus + sharpe_bonus + momentum_bonus
        
        return total_reward
    
    def save(self, filepath: str):
        """Save trained model"""
        if self.use_stable_baselines and self.model:
            self.model.save(filepath)
        else:
            # Save PyTorch model
            if not TORCH_AVAILABLE or not hasattr(self, 'pytorch_model'):
                raise ValueError("PyTorch model not available. Cannot save.")
            torch.save({
                'model_state_dict': self.pytorch_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config,
                'is_trained': self.is_trained
            }, filepath)
    
    def load(self, filepath: str):
        """Load trained model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        if self.use_stable_baselines:
            if self.algorithm == 'ppo':
                self.model = PPO.load(filepath)
            else:
                self.model = A2C.load(filepath)
            self.is_trained = True
        else:
            # Load PyTorch model
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch not available. Cannot load PyTorch model. Install with: pip install torch")
            checkpoint = torch.load(filepath)
            if not hasattr(self, 'pytorch_model'):
                self._create_pytorch_model()
            self.pytorch_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.is_trained = checkpoint.get('is_trained', True)

