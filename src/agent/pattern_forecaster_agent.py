"""Pattern Forecaster Agent - Historical pattern matching for market forecasting"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from .base_agent import BaseAgent
from ..strategies.base_strategy import Signal
from ..indicators.pattern_matcher import PatternForecaster, PatternExtractor
from ..data.data_storage import DataStorage


class PatternForecasterAgent(BaseAgent):
    """
    Pattern Forecaster Agent - Similar to forecaster.biz
    Forecasts market movements by comparing and correlating patterns from the past
    """
    
    def __init__(self, config: Optional[Dict] = None, storage: Optional[DataStorage] = None):
        """
        Initialize pattern forecaster agent
        
        Args:
            config: Agent configuration
            storage: Data storage instance for historical data
        """
        super().__init__("PatternForecaster", config)
        self.storage = storage
        self.forecaster = PatternForecaster(
            lookahead_periods=self.config.get('lookahead_periods', 10)
        )
        self.pattern_length = self.config.get('pattern_length', 20)
        self.min_matches = self.config.get('min_matches', 3)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.7)
        self.correlation_threshold = self.config.get('correlation_threshold', 0.6)
        self.historical_lookback_days = self.config.get('historical_lookback_days', 365)
    
    def analyze(
        self,
        data: pd.DataFrame,
        symbol: str,
        timeframe: str = '1h'
    ) -> Dict:
        """
        Analyze current pattern and forecast future movement
        
        Args:
            data: Current market data DataFrame
            symbol: Trading symbol
            timeframe: Timeframe string
        
        Returns:
            Dictionary with forecast signal and confidence
        """
        if not self.enabled or len(data) < self.pattern_length:
            return {
                'signal': Signal.HOLD,
                'score': 0.0,
                'confidence': 0.0,
                'source': self.name,
                'reason': 'Insufficient data or disabled'
            }
        
        print(f"\n=== PatternForecaster analyzing {symbol} ===")
        
        try:
            # Get historical data for pattern matching with error handling
            try:
                historical_data = self._get_historical_data(symbol, timeframe)
            except Exception as e:
                print(f"  Error fetching historical data: {e}")
                return {
                    'signal': Signal.HOLD,
                    'score': 0.0,
                    'confidence': 0.0,
                    'source': self.name,
                    'reason': f'Error fetching historical data: {str(e)}'
                }
            
            if historical_data is None or len(historical_data) < self.pattern_length * 2:
                print(f"  Insufficient historical data for pattern matching")
                return {
                    'signal': Signal.HOLD,
                    'score': 0.0,
                    'confidence': 0.0,
                    'source': self.name,
                    'reason': 'Insufficient historical data'
                }
            
            # Limit historical data size to prevent memory issues
            max_historical_bars = self.config.get('max_historical_bars', 2000)
            if len(historical_data) > max_historical_bars:
                print(f"  Limiting historical data from {len(historical_data)} to {max_historical_bars} bars")
                historical_data = historical_data.iloc[-max_historical_bars:]
            
            # Forecast using pattern matching with error handling
            try:
                forecast = self.forecaster.forecast(
                    current_data=data,
                    historical_data=historical_data,
                    min_matches=self.min_matches
                )
            except Exception as e:
                print(f"  Error in pattern forecasting: {e}")
                return {
                    'signal': Signal.HOLD,
                    'score': 0.0,
                    'confidence': 0.0,
                    'source': self.name,
                    'reason': f'Pattern matching error: {str(e)}'
                }
            
            if forecast.get('signal') is None:
                print(f"  No forecast generated: {forecast.get('reason', 'Unknown')}")
                return {
                    'signal': Signal.HOLD,
                    'score': 0.0,
                    'confidence': 0.0,
                    'source': self.name,
                    'reason': forecast.get('reason', 'No pattern matches found')
                }
            
            # Convert forecast to trading signal
            signal, score, confidence = self._forecast_to_signal(forecast)
            
            print(f"  Pattern Matches: {forecast.get('matches_found', 0)}")
            print(f"  Forecast Signal: {signal.name}")
            print(f"  Forecast Change: {forecast.get('forecast_change_pct', 0):.2f}%")
            print(f"  Forecast Price: ${forecast.get('forecast_price', 0):.2f}")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Avg Similarity: {forecast.get('avg_similarity', 0):.3f}")
            
            if forecast.get('top_matches'):
                print(f"  Top Match Similarity: {forecast['top_matches'][0]['similarity']:.3f}")
            
            print(f"=== End PatternForecaster for {symbol} ===\n")
            
            self._update_timestamp()
            
            return {
                'signal': signal,
                'score': score,
                'confidence': confidence,
                'source': self.name,
                'reason': f'Pattern forecast: {forecast.get("forecast_change_pct", 0):.2f}% change, '
                          f'{forecast.get("matches_found", 0)} matches',
                'forecast': forecast,
                'forecast_price': forecast.get('forecast_price'),
                'forecast_change_pct': forecast.get('forecast_change_pct'),
                'matches_found': forecast.get('matches_found', 0)
            }
        
        except Exception as e:
            print(f"  Error in pattern forecasting: {e}")
            import traceback
            traceback.print_exc()
            return {
                'signal': Signal.HOLD,
                'score': 0.0,
                'confidence': 0.0,
                'source': self.name,
                'reason': f'Error: {str(e)}'
            }
    
    def _get_historical_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Get historical data for pattern matching
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
        
        Returns:
            Historical data DataFrame or None
        """
        if not self.storage:
            return None
        
        try:
            # Calculate date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=self.historical_lookback_days)
            
            # Get historical data from storage with optimized limit
            # Reduce limit to avoid long-running queries and connection timeouts
            max_records = self.config.get('max_historical_records', 5000)
            historical_records = self.storage.get_ohlcv_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                limit=max_records  # Configurable limit (default: 5k records)
            )
            
            if not historical_records:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(historical_records)
            
            # Ensure timestamp is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            # Ensure required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                return None
            
            # Sort by timestamp
            df.sort_index(inplace=True)
            
            return df
        
        except Exception as e:
            print(f"  Error fetching historical data: {e}")
            return None
    
    def _forecast_to_signal(self, forecast: Dict) -> tuple:
        """
        Convert forecast to trading signal
        
        Args:
            forecast: Forecast dictionary
        
        Returns:
            Tuple of (signal, score, confidence)
        """
        signal_str = forecast.get('signal', 'neutral')
        forecast_change = forecast.get('forecast_change_pct', 0.0)
        confidence = forecast.get('confidence', 0.0)
        
        # Convert signal string to Signal enum
        if signal_str == 'bullish' and forecast_change > 1.0:
            signal = Signal.BUY
            score = min(1.0, forecast_change / 10.0)  # Normalize to -1 to 1 range
        elif signal_str == 'bearish' and forecast_change < -1.0:
            signal = Signal.SELL
            score = max(-1.0, forecast_change / 10.0)  # Normalize to -1 to 1 range
        else:
            signal = Signal.HOLD
            score = 0.0
        
        # Adjust confidence based on forecast strength
        # Strong forecasts (>5% change) get confidence boost
        if abs(forecast_change) > 5.0:
            confidence = min(0.9, confidence * 1.1)
        elif abs(forecast_change) < 2.0:
            confidence = confidence * 0.8  # Reduce confidence for weak forecasts
        
        return signal, score, confidence
    
    def get_pattern_statistics(self, symbol: str, timeframe: str = '1h') -> Dict:
        """
        Get statistics about available patterns for a symbol
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
        
        Returns:
            Statistics dictionary
        """
        historical_data = self._get_historical_data(symbol, timeframe)
        
        if historical_data is None or len(historical_data) < self.pattern_length:
            return {
                'total_patterns': 0,
                'data_points': len(historical_data) if historical_data is not None else 0,
                'coverage_days': 0
            }
        
        extractor = PatternExtractor(pattern_length=self.pattern_length)
        patterns = extractor.extract_multiple_patterns(historical_data)
        
        if historical_data.index[0] and historical_data.index[-1]:
            coverage = (historical_data.index[-1] - historical_data.index[0]).days
        else:
            coverage = 0
        
        return {
            'total_patterns': len(patterns),
            'data_points': len(historical_data),
            'coverage_days': coverage,
            'pattern_length': self.pattern_length,
            'lookahead_periods': self.forecaster.lookahead_periods
        }

