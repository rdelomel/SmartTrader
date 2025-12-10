"""Regime-switching agent using Hidden Markov Model"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime
from .base_agent import BaseAgent
from ..indicators.technical import TechnicalIndicators

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("Warning: hmmlearn not available. Using statistical regime detection.")


class RegimeSwitchingAgent(BaseAgent):
    """Market regime detection agent using HMM"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize regime-switching agent
        
        Config parameters:
            n_regimes: Number of regimes to detect (default: 4)
            use_hmm: Whether to use HMM (default: True, falls back if unavailable)
            lookback_period: Period for regime detection (default: 100)
        """
        super().__init__("RegimeSwitchingAgent", config)
        self.n_regimes = self.config.get('n_regimes', 4)
        self.use_hmm = self.config.get('use_hmm', True) and HMM_AVAILABLE
        self.lookback_period = self.config.get('lookback_period', 100)
        self.indicators = TechnicalIndicators()
        
        # HMM model
        self.hmm_model = None
        self.is_trained = False
        self.regime_names = ['Bullish Trend', 'Bearish Trend', 'Consolidation', 'High Volatility']
        
        # Regime-to-weight mappings
        self.regime_weights = {
            'Bullish Trend': {
                'trend_following': 1.5,
                'mean_reversion': 0.5,
                'sentiment': 1.0,
                'fundamental': 1.0,
                'momentum': 1.2,
                'breakout': 1.3
            },
            'Bearish Trend': {
                'trend_following': 1.5,
                'mean_reversion': 0.5,
                'sentiment': 1.2,
                'fundamental': 1.0,
                'momentum': 1.0,
                'breakout': 0.8
            },
            'Consolidation': {
                'trend_following': 0.5,
                'mean_reversion': 1.5,
                'sentiment': 0.8,
                'fundamental': 0.8,
                'momentum': 0.7,
                'breakout': 1.2
            },
            'High Volatility': {
                'trend_following': 0.7,
                'mean_reversion': 1.2,
                'sentiment': 1.5,
                'fundamental': 0.5,
                'momentum': 0.8,
                'breakout': 0.9
            },
            'Neutral': {
                'trend_following': 1.0,
                'mean_reversion': 1.0,
                'sentiment': 1.0,
                'fundamental': 1.0,
                'momentum': 1.0,
                'breakout': 1.0
            }
        }
    
    def analyze(self, data: pd.DataFrame) -> Dict:
        """
        Detect current market regime
        
        Args:
            data: Market data DataFrame
        
        Returns:
            Dictionary with regime information and agent weights
        """
        if not self.enabled or len(data) < 50:
            return {
                'regime': 'Neutral',
                'confidence': 0.0,
                'probabilities': {},
                'agent_weights': self.regime_weights.get('Neutral', {}),
                'source': self.name
            }
        
        try:
            if self.use_hmm and self.is_trained:
                regime_result = self._detect_regime_hmm(data)
            else:
                regime_result = self._detect_regime_statistical(data)
            
            # Get agent weights for detected regime
            regime_name = regime_result['regime']
            agent_weights = self.regime_weights.get(regime_name, self.regime_weights['Neutral'])
            
            self._update_timestamp()
            
            return {
                'regime': regime_name,
                'confidence': regime_result.get('confidence', 0.5),
                'probabilities': regime_result.get('probabilities', {}),
                'agent_weights': agent_weights,
                'source': self.name,
                'method': 'HMM' if (self.use_hmm and self.is_trained) else 'Statistical'
            }
        
        except Exception as e:
            print(f"Error in regime detection: {e}")
            return {
                'regime': 'Neutral',
                'confidence': 0.0,
                'probabilities': {},
                'agent_weights': self.regime_weights.get('Neutral', {}),
                'source': self.name,
                'error': str(e)
            }
    
    def train(self, data: pd.DataFrame):
        """Train HMM model on historical data"""
        if not self.use_hmm or len(data) < self.lookback_period:
            return False
        
        try:
            # Extract features for regime detection
            features = self._extract_regime_features(data)
            
            if len(features) < self.lookback_period:
                return False
            
            # Create HMM model
            self.hmm_model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=100,
                random_state=42
            )
            
            # Train model
            self.hmm_model.fit(features)
            self.is_trained = True
            
            return True
        
        except Exception as e:
            print(f"Error training HMM: {e}")
            self.use_hmm = False
            return False
    
    def _extract_regime_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features for regime detection"""
        features_list = []
        
        # ADX (trend strength)
        adx = self.indicators.adx(data)
        if not adx.empty:
            features_list.append(adx.values.reshape(-1, 1))
        
        # ATR (volatility)
        atr = self.indicators.atr(data)
        if not atr.empty:
            # Normalize ATR
            atr_normalized = (atr / data['close']).values.reshape(-1, 1)
            features_list.append(atr_normalized)
        
        # Price momentum
        returns = data['close'].pct_change().dropna()
        if len(returns) > 0:
            momentum = returns.rolling(window=10).mean().values.reshape(-1, 1)
            features_list.append(momentum)
        
        # Volatility (rolling std)
        volatility = returns.rolling(window=20).std().values.reshape(-1, 1)
        features_list.append(volatility)
        
        # Volume pattern
        volume_ma = data['volume'].rolling(window=20).mean()
        volume_ratio = (data['volume'] / volume_ma).values.reshape(-1, 1)
        features_list.append(volume_ratio)
        
        # Combine features
        if features_list:
            # Align lengths
            min_len = min(len(f) for f in features_list)
            features_array = np.hstack([f[:min_len] for f in features_list])
            
            # Remove NaN rows
            features_array = features_array[~np.isnan(features_array).any(axis=1)]
            
            return features_array
        
        return np.array([])
    
    def _detect_regime_hmm(self, data: pd.DataFrame) -> Dict:
        """Detect regime using trained HMM"""
        try:
            features = self._extract_regime_features(data)
            
            if len(features) == 0:
                return self._detect_regime_statistical(data)
            
            # Get recent features
            recent_features = features[-self.lookback_period:] if len(features) > self.lookback_period else features
            
            # Predict regime
            regime_idx = self.hmm_model.predict(recent_features[-1:])[0]
            
            # Get probabilities
            probabilities = self.hmm_model.predict_proba(recent_features[-1:])[0]
            
            # Map to regime name
            if regime_idx < len(self.regime_names):
                regime_name = self.regime_names[regime_idx]
            else:
                regime_name = 'Neutral'
            
            # Confidence is max probability
            confidence = float(np.max(probabilities))
            
            prob_dict = {self.regime_names[i]: float(p) for i, p in enumerate(probabilities) 
                        if i < len(self.regime_names)}
            
            return {
                'regime': regime_name,
                'confidence': confidence,
                'probabilities': prob_dict
            }
        
        except Exception as e:
            print(f"Error in HMM regime detection: {e}")
            return self._detect_regime_statistical(data)
    
    def _detect_regime_statistical(self, data: pd.DataFrame) -> Dict:
        """Detect regime using statistical methods (fallback)"""
        try:
            # Calculate indicators
            sma_20 = self.indicators.sma(data, 20)
            sma_50 = self.indicators.sma(data, 50)
            
            if len(sma_20) < 2 or len(sma_50) < 2:
                return {
                    'regime': 'Neutral',
                    'confidence': 0.3,
                    'probabilities': {}
                }
            
            trend_strength = abs(sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1]
            trend_direction = 1 if sma_20.iloc[-1] > sma_50.iloc[-1] else -1
            
            # Calculate volatility
            returns = data['close'].pct_change()
            volatility = returns.rolling(window=20).std().iloc[-1]
            avg_volatility = returns.std()
            
            # Calculate ADX
            adx = self.indicators.adx(data)
            adx_value = adx.iloc[-1] if not adx.empty else 0
            
            # Determine regime
            if adx_value > 25 and trend_strength > 0.02:
                if trend_direction > 0:
                    regime = 'Bullish Trend'
                else:
                    regime = 'Bearish Trend'
                confidence = min(adx_value / 50.0, 1.0)
            elif volatility > avg_volatility * 1.5:
                regime = 'High Volatility'
                confidence = min((volatility / avg_volatility - 1.0) / 1.0, 1.0)
            elif adx_value < 20 and trend_strength < 0.01:
                regime = 'Consolidation'
                confidence = 0.7
            else:
                regime = 'Neutral'
                confidence = 0.5
            
            return {
                'regime': regime,
                'confidence': confidence,
                'probabilities': {regime: confidence}
            }
        
        except Exception as e:
            print(f"Error in statistical regime detection: {e}")
            return {
                'regime': 'Neutral',
                'confidence': 0.3,
                'probabilities': {}
            }

