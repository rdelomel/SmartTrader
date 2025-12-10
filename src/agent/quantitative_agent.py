"""Quantitative Analysis Agent - Mathematical and statistical modeling"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime
from .base_agent import BaseAgent
from ..strategies.base_strategy import Signal
from ..indicators.technical import TechnicalIndicators
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class QuantitativeAgent(BaseAgent):
    """
    Quantitative Analysis Agent based on Investopedia principles
    - Uses mathematical and statistical modeling
    - Analyzes numerical data to identify patterns
    - Develops algorithms for trading opportunities
    - Manages risk through statistical models
    - Optimizes portfolio allocation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize quantitative analysis agent
        
        Args:
            config: Agent configuration
        """
        super().__init__("QuantitativeAgent", config)
        self.indicators = TechnicalIndicators()
        self.scaler = StandardScaler()
        
        # Configuration
        self.lookback_period = self.config.get('lookback_period', 100)
        self.min_confidence = self.config.get('min_confidence', 0.4)
        self.use_statistical_models = self.config.get('use_statistical_models', True)
        self.use_momentum_models = self.config.get('use_momentum_models', True)
        self.use_mean_reversion_models = self.config.get('use_mean_reversion_models', True)
    
    def analyze(self, data: pd.DataFrame, symbol: str = None) -> Dict:
        """
        Perform quantitative analysis using statistical and mathematical models
        
        Args:
            data: Market data DataFrame
            symbol: Trading symbol
        
        Returns:
            Dictionary with quantitative signal and score
        """
        if not self.enabled or len(data) < self.lookback_period:
            return {
                'signal': Signal.HOLD,
                'score': 0.0,
                'confidence': 0.0,
                'source': self.name,
                'reason': 'Insufficient data or disabled'
            }
        
        print(f"\n=== QuantitativeAgent analyzing {symbol if symbol else 'symbol'} ===")
        
        # Use recent data for analysis
        recent_data = data.iloc[-self.lookback_period:].copy()
        
        signals = []
        confidences = []
        models_used = []
        
        # === MODEL 1: Statistical Mean Reversion (Z-Score) ===
        if self.use_mean_reversion_models:
            mean_reversion_result = self._mean_reversion_model(recent_data)
            if mean_reversion_result:
                signals.append(mean_reversion_result['signal'])
                confidences.append(mean_reversion_result['confidence'])
                models_used.append('MeanReversion')
                print(f"  Mean Reversion Model: {mean_reversion_result['signal'].name}, "
                      f"confidence={mean_reversion_result['confidence']:.3f}, "
                      f"z-score={mean_reversion_result.get('z_score', 0):.2f}")
        
        # === MODEL 2: Momentum Model (Rate of Change) ===
        if self.use_momentum_models:
            momentum_result = self._momentum_model(recent_data)
            if momentum_result:
                signals.append(momentum_result['signal'])
                confidences.append(momentum_result['confidence'])
                models_used.append('Momentum')
                print(f"  Momentum Model: {momentum_result['signal'].name}, "
                      f"confidence={momentum_result['confidence']:.3f}, "
                      f"roc={momentum_result.get('rate_of_change', 0):.3f}")
        
        # === MODEL 3: Linear Regression Trend ===
        if self.use_statistical_models:
            regression_result = self._regression_trend_model(recent_data)
            if regression_result:
                signals.append(regression_result['signal'])
                confidences.append(regression_result['confidence'])
                models_used.append('Regression')
                print(f"  Regression Model: {regression_result['signal'].name}, "
                      f"confidence={regression_result['confidence']:.3f}, "
                      f"slope={regression_result.get('slope', 0):.4f}")
        
        # === MODEL 4: Volatility Clustering Model ===
        volatility_result = self._volatility_clustering_model(recent_data)
        if volatility_result:
            signals.append(volatility_result['signal'])
            confidences.append(volatility_result['confidence'])
            models_used.append('Volatility')
            print(f"  Volatility Model: {volatility_result['signal'].name}, "
                  f"confidence={volatility_result['confidence']:.3f}")
        
        # === MODEL 5: Statistical Arbitrage (Pairs Trading Signal) ===
        # Note: Full pairs trading requires multiple symbols, simplified here
        arbitrage_result = self._statistical_arbitrage_model(recent_data)
        if arbitrage_result:
            signals.append(arbitrage_result['signal'])
            confidences.append(arbitrage_result['confidence'])
            models_used.append('Arbitrage')
            print(f"  Arbitrage Model: {arbitrage_result['signal'].name}, "
                  f"confidence={arbitrage_result['confidence']:.3f}")
        
        if not signals:
            print(f"  No quantitative signals generated")
            return {
                'signal': Signal.HOLD,
                'score': 0.0,
                'confidence': 0.0,
                'source': self.name,
                'reason': 'No quantitative models generated signals'
            }
        
        # Aggregate signals using weighted average
        signal_values = [1 if s == Signal.BUY else (-1 if s == Signal.SELL else 0) for s in signals]
        weighted_score = np.average(signal_values, weights=confidences)
        avg_confidence = np.mean(confidences)
        
        # Determine final signal
        if weighted_score > 0.2:
            final_signal = Signal.BUY
        elif weighted_score < -0.2:
            final_signal = Signal.SELL
        else:
            final_signal = Signal.HOLD
        
        print(f"  AGGREGATED: {final_signal.name}, weighted_score={weighted_score:.3f}, "
              f"confidence={avg_confidence:.3f}, models={len(models_used)}")
        print(f"=== End QuantitativeAgent for {symbol if symbol else 'symbol'} ===\n")
        
        self._update_timestamp()
        
        return {
            'signal': final_signal,
            'score': weighted_score,
            'confidence': avg_confidence,
            'source': self.name,
            'reason': f'Quantitative analysis: {len(models_used)} models, score={weighted_score:.2f}',
            'models_used': models_used,
            'model_count': len(models_used)
        }
    
    def _mean_reversion_model(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Mean reversion model using Z-score
        Based on statistical principle that prices revert to mean
        """
        if len(data) < 30:
            return None
        
        prices = data['close'].values
        mean = np.mean(prices)
        std = np.std(prices)
        
        if std == 0:
            return None
        
        current_price = prices[-1]
        z_score = (current_price - mean) / std
        
        # Mean reversion signals
        if z_score > 2.0:  # 2 standard deviations above mean
            return {
                'signal': Signal.SELL,
                'confidence': min(0.7, 0.4 + abs(z_score - 2.0) * 0.1),
                'z_score': z_score,
                'model': 'mean_reversion'
            }
        elif z_score < -2.0:  # 2 standard deviations below mean
            return {
                'signal': Signal.BUY,
                'confidence': min(0.7, 0.4 + abs(z_score + 2.0) * 0.1),
                'z_score': z_score,
                'model': 'mean_reversion'
            }
        
        return None
    
    def _momentum_model(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Momentum model using rate of change and acceleration
        Statistical analysis of price momentum
        """
        if len(data) < 20:
            return None
        
        prices = data['close'].values
        
        # Calculate rate of change over multiple periods
        roc_5 = (prices[-1] - prices[-6]) / prices[-6] if len(prices) >= 6 else 0
        roc_10 = (prices[-1] - prices[-11]) / prices[-11] if len(prices) >= 11 else 0
        roc_20 = (prices[-1] - prices[-21]) / prices[-21] if len(prices) >= 21 else 0
        
        # Calculate momentum acceleration (rate of change of rate of change)
        if len(prices) >= 11:
            roc_prev = (prices[-6] - prices[-11]) / prices[-11]
            acceleration = roc_5 - roc_prev
        else:
            acceleration = 0
        
        # Momentum scoring
        momentum_score = (roc_5 * 0.5 + roc_10 * 0.3 + roc_20 * 0.2)
        
        # Strong momentum with acceleration
        if momentum_score > 0.03 and acceleration > 0:
            return {
                'signal': Signal.BUY,
                'confidence': min(0.75, 0.5 + abs(momentum_score) * 5),
                'rate_of_change': momentum_score,
                'acceleration': acceleration,
                'model': 'momentum'
            }
        elif momentum_score < -0.03 and acceleration < 0:
            return {
                'signal': Signal.SELL,
                'confidence': min(0.75, 0.5 + abs(momentum_score) * 5),
                'rate_of_change': momentum_score,
                'acceleration': acceleration,
                'model': 'momentum'
            }
        
        return None
    
    def _regression_trend_model(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Linear regression model to identify trends
        Uses statistical regression to predict price direction
        """
        if len(data) < 30:
            return None
        
        prices = data['close'].values
        X = np.arange(len(prices)).reshape(-1, 1)
        y = prices
        
        # Fit linear regression
        try:
            model = LinearRegression()
            model.fit(X, y)
            slope = model.coef_[0]
            r_squared = model.score(X, y)
            
            # Predict next price
            next_x = np.array([[len(prices)]])
            predicted_price = model.predict(next_x)[0]
            current_price = prices[-1]
            
            # Trend strength based on R-squared and slope
            trend_strength = abs(slope) / current_price  # Normalized slope
            confidence = min(0.7, r_squared * 0.8 + trend_strength * 10)
            
            if slope > 0 and predicted_price > current_price * 1.01:
                return {
                    'signal': Signal.BUY,
                    'confidence': confidence,
                    'slope': slope,
                    'r_squared': r_squared,
                    'predicted_price': predicted_price,
                    'model': 'regression'
                }
            elif slope < 0 and predicted_price < current_price * 0.99:
                return {
                    'signal': Signal.SELL,
                    'confidence': confidence,
                    'slope': slope,
                    'r_squared': r_squared,
                    'predicted_price': predicted_price,
                    'model': 'regression'
                }
        except Exception as e:
            print(f"  Regression model error: {e}")
        
        return None
    
    def _volatility_clustering_model(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Volatility clustering model
        Based on statistical observation that volatility clusters
        """
        if len(data) < 30:
            return None
        
        returns = data['close'].pct_change().dropna()
        
        if len(returns) < 20:
            return None
        
        # Calculate rolling volatility
        short_vol = returns.iloc[-10:].std()
        long_vol = returns.iloc[-30:].std() if len(returns) >= 30 else returns.std()
        
        # Volatility ratio
        vol_ratio = short_vol / long_vol if long_vol > 0 else 1.0
        
        # Low volatility often precedes high volatility moves
        if vol_ratio < 0.7:  # Current volatility is low
            # Expect volatility expansion - could go either way
            # Use price trend to determine direction
            price_trend = (data['close'].iloc[-1] - data['close'].iloc[-10]) / data['close'].iloc[-10]
            
            if price_trend > 0.02:
                return {
                    'signal': Signal.BUY,
                    'confidence': 0.5,
                    'volatility_ratio': vol_ratio,
                    'model': 'volatility_clustering'
                }
            elif price_trend < -0.02:
                return {
                    'signal': Signal.SELL,
                    'confidence': 0.5,
                    'volatility_ratio': vol_ratio,
                    'model': 'volatility_clustering'
                }
        
        return None
    
    def _statistical_arbitrage_model(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Simplified statistical arbitrage model
        Looks for price deviations from statistical norms
        """
        if len(data) < 50:
            return None
        
        prices = data['close'].values
        
        # Calculate rolling mean and standard deviation
        window = min(30, len(prices) // 2)
        rolling_mean = pd.Series(prices).rolling(window).mean().iloc[-1]
        rolling_std = pd.Series(prices).rolling(window).std().iloc[-1]
        
        if rolling_std == 0 or pd.isna(rolling_mean) or pd.isna(rolling_std):
            return None
        
        current_price = prices[-1]
        deviation = (current_price - rolling_mean) / rolling_std
        
        # Statistical arbitrage: trade when price deviates significantly
        if deviation > 1.5:  # Price significantly above mean
            return {
                'signal': Signal.SELL,
                'confidence': min(0.65, 0.4 + abs(deviation - 1.5) * 0.1),
                'deviation': deviation,
                'model': 'statistical_arbitrage'
            }
        elif deviation < -1.5:  # Price significantly below mean
            return {
                'signal': Signal.BUY,
                'confidence': min(0.65, 0.4 + abs(deviation + 1.5) * 0.1),
                'deviation': deviation,
                'model': 'statistical_arbitrage'
            }
        
        return None
    
    def calculate_risk_metrics(self, data: pd.DataFrame, position_size: float) -> Dict:
        """
        Calculate quantitative risk metrics
        
        Args:
            data: Market data
            position_size: Position size in dollars
        
        Returns:
            Dictionary with risk metrics
        """
        if len(data) < 30:
            return {}
        
        returns = data['close'].pct_change().dropna()
        
        # Value at Risk (VaR) - 95% confidence
        var_95 = np.percentile(returns, 5) * position_size
        
        # Expected Shortfall (Conditional VaR)
        es_95 = returns[returns <= np.percentile(returns, 5)].mean() * position_size
        
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Sharpe Ratio (risk-adjusted return)
        if returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized
        else:
            sharpe = 0
        
        return {
            'var_95': var_95,
            'expected_shortfall': es_95,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'volatility': returns.std() * np.sqrt(252)  # Annualized
        }

