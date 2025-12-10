"""Deep Reinforcement Learning module for trading"""

# Patch gymnasium as gym before stable-baselines3 imports it
try:
    import gymnasium as gym
    import sys
    sys.modules['gym'] = gym  # Make gymnasium available as 'gym' for stable-baselines3
except ImportError:
    pass  # If gymnasium not available, let stable-baselines3 use gym

from .drl_agent import DRLAgent
from .trainer import DRLTrainer

__all__ = ['DRLAgent', 'DRLTrainer']

