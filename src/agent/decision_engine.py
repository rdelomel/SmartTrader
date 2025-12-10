"""Agentic decision engine that combines multiple models and strategies"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from ..strategies.base_strategy import BaseStrategy, Signal
from ..ai.models.ml_models import MLModel
try:
    from ..ai.models.lstm_model import LSTMModel
except ImportError:
    LSTMModel = None  # LSTM model optional if TensorFlow not available
from ..ai.models.sentiment_analyzer import SentimentAnalyzer
from ..indicators.technical import TechnicalIndicators
from ..indicators.feature_engineering import FeatureEngineer
from .signal_aggregator import SignalAggregator


class DecisionEngine:
    """Core decision engine that combines multiple signals"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize decision engine
        
        Config parameters:
            strategies: List of strategy configurations
            models: Model configurations
            market_regime_detection: Enable market regime detection
        """
        self.config = config or {}
        self.strategies: List[BaseStrategy] = []
        self.ml_model: Optional[MLModel] = None
        self.lstm_model: Optional[LSTMModel] = None
        self.sentiment_analyzer: Optional[SentimentAnalyzer] = None
        self.signal_aggregator = SignalAggregator()
        self.indicators = TechnicalIndicators()
        self.feature_engineer = FeatureEngineer()
        self.market_regime = 'neutral'  # 'trending', 'ranging', 'volatile', 'neutral'
    
    def add_strategy(self, strategy: BaseStrategy):
        """Add a trading strategy"""
        self.strategies.append(strategy)
    
    def set_ml_model(self, model: MLModel):
        """Set ML model"""
        self.ml_model = model
    
    def set_lstm_model(self, model):
        """Set LSTM model"""
        if LSTMModel is None:
            print("Warning: LSTM model not available (TensorFlow not installed)")
            return
        self.lstm_model = model
    
    def set_sentiment_analyzer(self, analyzer: SentimentAnalyzer):
        """Set sentiment analyzer"""
        self.sentiment_analyzer = analyzer
    
    def detect_market_regime(self, data: pd.DataFrame) -> str:
        """
        Detect current market regime
        
        Args:
            data: Market data with indicators
        
        Returns:
            Market regime string
        """
        if len(data) < 50:
            return 'neutral'
        
        # Calculate trend strength
        sma_20 = self.indicators.sma(data, 20)
        sma_50 = self.indicators.sma(data, 50)
        
        if len(sma_20) < 2 or len(sma_50) < 2:
            return 'neutral'
        
        trend_strength = abs(sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1]
        
        # Calculate volatility
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=20).std().iloc[-1]
        
        # Calculate ADX for trend strength
        adx = self.indicators.adx(data)
        adx_value = adx.iloc[-1] if not adx.empty else 0
        
        # Determine regime
        if adx_value > 25 and trend_strength > 0.02:
            regime = 'trending'
        elif volatility > returns.std() * 1.5:
            regime = 'volatile'
        elif adx_value < 20 and trend_strength < 0.01:
            regime = 'ranging'
        else:
            regime = 'neutral'
        
        self.market_regime = regime
        return regime
    
    def adjust_strategy_weights(self, regime: str) -> Dict[str, float]:
        """
        Adjust strategy weights based on market regime
        
        Args:
            regime: Current market regime
        
        Returns:
            Dictionary of strategy weights
        """
        weights = {}
        
        for strategy in self.strategies:
            base_weight = strategy.get_weight()
            
            # Adjust weights based on regime
            if regime == 'trending':
                if 'trend' in strategy.name.lower():
                    weights[strategy.name] = base_weight * 1.5
                elif 'mean' in strategy.name.lower():
                    weights[strategy.name] = base_weight * 0.5
                else:
                    weights[strategy.name] = base_weight
            elif regime == 'ranging':
                if 'mean' in strategy.name.lower():
                    weights[strategy.name] = base_weight * 1.5
                elif 'trend' in strategy.name.lower():
                    weights[strategy.name] = base_weight * 0.5
                else:
                    weights[strategy.name] = base_weight
            else:
                weights[strategy.name] = base_weight
        
        return weights
    
    def generate_decision(
        self,
        data: pd.DataFrame,
        sentiment_data: Optional[Dict] = None
    ) -> Dict:
        """
        Generate final trading decision
        
        Args:
            data: Market data with indicators
            sentiment_data: Optional sentiment analysis results
        
        Returns:
            Final decision dictionary
        """
        signals = []
        
        # Detect market regime
        regime = self.detect_market_regime(data)
        
        # Adjust strategy weights
        strategy_weights = self.adjust_strategy_weights(regime)
        
        # Get signals from strategies
        for strategy in self.strategies:
            if not strategy.is_enabled():
                continue
            
            try:
                signal = strategy.generate_signal(data)
                signal['source'] = strategy.name
                signal['weight'] = strategy_weights.get(strategy.name, strategy.get_weight())
                signal['enabled'] = True
                signals.append(signal)
            except Exception as e:
                print(f"Error generating signal from {strategy.name}: {e}")
        
        # Get signal from ML model
        if self.ml_model and self.ml_model.is_trained:
            try:
                ml_signal = self._get_ml_signal(data)
                if ml_signal:
                    signals.append(ml_signal)
            except Exception as e:
                print(f"Error getting ML signal: {e}")
        
        # Get signal from LSTM model
        if self.lstm_model and self.lstm_model.is_trained:
            try:
                lstm_signal = self._get_lstm_signal(data)
                if lstm_signal:
                    signals.append(lstm_signal)
            except Exception as e:
                print(f"Error getting LSTM signal: {e}")
        
        # Get signal from sentiment analysis
        if self.sentiment_analyzer and sentiment_data:
            try:
                sentiment_signal = self._get_sentiment_signal(sentiment_data)
                if sentiment_signal:
                    signals.append(sentiment_signal)
            except Exception as e:
                print(f"Error getting sentiment signal: {e}")
        
        # Aggregate signals
        final_decision = self.signal_aggregator.aggregate(signals)
        final_decision['market_regime'] = regime
        final_decision['signal_count'] = len(signals)
        final_decision['timestamp'] = datetime.now()
        
        return final_decision
    
    def _get_ml_signal(self, data: pd.DataFrame) -> Optional[Dict]:
        """Get signal from ML model"""
        try:
            # Prepare features
            df_features = self.feature_engineer.create_features(data)
            feature_columns = self.feature_engineer.get_feature_columns(df_features)
            
            if not feature_columns or len(df_features) == 0:
                return None
            
            # Get latest features
            X = df_features[feature_columns].iloc[-1:].fillna(0)
            
            # Predict
            if self.ml_model.model_type == 'classification':
                prediction = self.ml_model.predict(X)[0]
                proba = self.ml_model.predict_proba(X)[0]
                confidence = max(proba)
                
                signal = Signal.BUY if prediction == 1 else Signal.SELL
            else:
                prediction = self.ml_model.predict(X)[0]
                signal = Signal.BUY if prediction > 0 else Signal.SELL
                confidence = min(abs(prediction) * 10, 1.0)
            
            return {
                'signal': signal,
                'confidence': confidence,
                'source': 'ML_Model',
                'weight': 0.3,
                'enabled': True
            }
        except Exception as e:
            print(f"Error in ML signal generation: {e}")
            return None
    
    def _get_lstm_signal(self, data: pd.DataFrame) -> Optional[Dict]:
        """Get signal from LSTM model"""
        try:
            # Prepare features
            df_features = self.feature_engineer.create_features(data)
            feature_columns = self.feature_engineer.get_feature_columns(df_features)
            
            if not feature_columns or len(df_features) < 60:
                return None
            
            # Get recent features (need sequence length)
            X = df_features[feature_columns].tail(60).copy()
            
            # Clean data: replace infinity and NaN
            X = X.replace([np.inf, -np.inf], np.nan)
            # Fill NaN with median, then 0
            for col in X.columns:
                if X[col].isnull().any():
                    median_val = X[col].median()
                    if pd.isna(median_val):
                        X[col] = X[col].fillna(0)
                    else:
                        X[col] = X[col].fillna(median_val)
                        X[col] = X[col].fillna(0)  # Final fallback
            
            # Predict
            prediction = self.lstm_model.predict(X)[0]
            
            signal = Signal.BUY if prediction > 0 else Signal.SELL
            confidence = min(abs(prediction) * 10, 1.0)
            
            return {
                'signal': signal,
                'confidence': confidence,
                'source': 'LSTM_Model',
                'weight': 0.15,
                'enabled': True
            }
        except Exception as e:
            print(f"Error in LSTM signal generation: {e}")
            return None
    
    def _get_sentiment_signal(self, sentiment_data: Dict) -> Optional[Dict]:
        """Get signal from sentiment analysis"""
        try:
            sentiment_score = sentiment_data.get('sentiment', 0.0)
            confidence = sentiment_data.get('confidence', 0.0)
            
            if abs(sentiment_score) < 0.3:  # Threshold
                signal = Signal.HOLD
            elif sentiment_score > 0:
                signal = Signal.BUY
            else:
                signal = Signal.SELL
            
            return {
                'signal': signal,
                'confidence': min(confidence * abs(sentiment_score), 1.0),
                'source': 'Sentiment',
                'weight': 0.05,
                'enabled': True
            }
        except Exception as e:
            print(f"Error in sentiment signal generation: {e}")
            return None

