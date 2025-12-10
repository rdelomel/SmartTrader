"""Pattern matching for historical correlation forecasting"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PatternMatch:
    """Represents a matched historical pattern"""
    pattern_id: str
    similarity_score: float
    correlation: float
    historical_date: datetime
    pattern_length: int
    future_outcome: Dict  # What happened after this pattern
    pattern_features: Dict


class PatternExtractor:
    """
    Extracts normalized patterns from price data
    Similar to forecaster.biz pattern extraction
    """
    
    def __init__(self, pattern_length: int = 20, normalize: bool = True):
        """
        Initialize pattern extractor
        
        Args:
            pattern_length: Number of bars in a pattern
            normalize: Whether to normalize patterns for comparison
        """
        self.pattern_length = pattern_length
        self.normalize = normalize
    
    def extract_pattern(self, data: pd.DataFrame, end_index: int = None) -> Dict:
        """
        Extract a pattern from price data
        
        Args:
            data: OHLCV DataFrame
            end_index: End index for pattern (None = most recent)
        
        Returns:
            Dictionary with pattern features
        """
        if end_index is None:
            end_index = len(data)
        
        start_index = max(0, end_index - self.pattern_length)
        pattern_data = data.iloc[start_index:end_index].copy()
        
        if len(pattern_data) < self.pattern_length:
            return None
        
        # Extract price pattern (normalized)
        prices = pattern_data['close'].values
        volumes = pattern_data['volume'].values
        
        # Normalize prices to percentage changes from start
        if self.normalize:
            price_pattern = ((prices - prices[0]) / prices[0]) * 100  # Percentage change
        else:
            price_pattern = prices
        
        # Normalize volumes
        if self.normalize and volumes.max() > 0:
            volume_pattern = (volumes / volumes.mean()) - 1  # Relative to mean
        else:
            volume_pattern = volumes
        
        # Calculate pattern features
        features = {
            'price_pattern': price_pattern.tolist(),
            'volume_pattern': volume_pattern.tolist(),
            'price_range': float(prices.max() - prices.min()),
            'price_range_pct': float((prices.max() - prices.min()) / prices[0] * 100),
            'volatility': float(prices.std() / prices.mean() * 100),
            'trend': float((prices[-1] - prices[0]) / prices[0] * 100),
            'volume_trend': float((volumes[-1] - volumes[0]) / volumes[0] * 100) if volumes[0] > 0 else 0,
            'high_low_ratio': float((pattern_data['high'].max() - pattern_data['low'].min()) / prices[0] * 100),
            'start_price': float(prices[0]),
            'end_price': float(prices[-1]),
            'pattern_length': len(pattern_data)
        }
        
        return features
    
    def extract_multiple_patterns(self, data: pd.DataFrame, step: int = 5) -> List[Dict]:
        """
        Extract multiple patterns from historical data
        
        Args:
            data: OHLCV DataFrame
            step: Step size between patterns
        
        Returns:
            List of pattern dictionaries with timestamps
        """
        patterns = []
        
        for i in range(self.pattern_length, len(data), step):
            pattern = self.extract_pattern(data, i)
            if pattern:
                pattern['timestamp'] = data.index[i] if hasattr(data.index[i], 'timestamp') else i
                pattern['end_index'] = i
                patterns.append(pattern)
        
        return patterns


class PatternMatcher:
    """
    Matches current patterns with historical patterns
    Uses correlation and similarity metrics
    """
    
    def __init__(self, similarity_threshold: float = 0.7, correlation_threshold: float = 0.6):
        """
        Initialize pattern matcher
        
        Args:
            similarity_threshold: Minimum similarity score (0-1)
            correlation_threshold: Minimum correlation coefficient (0-1)
        """
        self.similarity_threshold = similarity_threshold
        self.correlation_threshold = correlation_threshold
        self.extractor = PatternExtractor()
    
    def calculate_similarity(self, pattern1: Dict, pattern2: Dict) -> float:
        """
        Calculate similarity between two patterns
        
        Uses multiple metrics:
        1. Price pattern correlation
        2. Euclidean distance (normalized)
        3. Feature similarity
        
        Returns:
            Similarity score (0-1, higher = more similar)
        """
        try:
            # Price pattern correlation
            price1 = np.array(pattern1['price_pattern'])
            price2 = np.array(pattern2['price_pattern'])
            
            if len(price1) != len(price2):
                return 0.0
            
            # Pearson correlation
            if len(price1) > 1 and price1.std() > 0 and price2.std() > 0:
                correlation, _ = pearsonr(price1, price2)
                correlation = max(0, correlation)  # Only positive correlations
            else:
                correlation = 0.0
            
            # Normalized Euclidean distance
            if price1.std() > 0 and price2.std() > 0:
                # Normalize for distance calculation
                price1_norm = (price1 - price1.mean()) / price1.std()
                price2_norm = (price2 - price2.mean()) / price2.std()
                distance = euclidean(price1_norm, price2_norm)
                # Convert distance to similarity (0-1)
                max_distance = np.sqrt(len(price1)) * 2  # Theoretical max
                distance_similarity = max(0, 1 - (distance / max_distance))
            else:
                distance_similarity = 0.0
            
            # Feature similarity
            feature_similarity = self._compare_features(pattern1, pattern2)
            
            # Weighted combination
            similarity = (
                correlation * 0.4 +
                distance_similarity * 0.3 +
                feature_similarity * 0.3
            )
            
            return float(similarity)
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
    
    def _compare_features(self, pattern1: Dict, pattern2: Dict) -> float:
        """Compare pattern features"""
        features = ['volatility', 'trend', 'volume_trend', 'high_low_ratio']
        similarities = []
        
        for feature in features:
            if feature in pattern1 and feature in pattern2:
                val1 = abs(pattern1[feature])
                val2 = abs(pattern2[feature])
                
                if val1 == 0 and val2 == 0:
                    similarities.append(1.0)
                elif val1 == 0 or val2 == 0:
                    similarities.append(0.0)
                else:
                    # Similarity based on ratio
                    ratio = min(val1, val2) / max(val1, val2)
                    similarities.append(ratio)
        
        return np.mean(similarities) if similarities else 0.0
    
    def find_similar_patterns(
        self,
        current_pattern: Dict,
        historical_patterns: List[Dict],
        max_matches: int = 10
    ) -> List[PatternMatch]:
        """
        Find similar historical patterns
        
        Args:
            current_pattern: Current pattern to match
            historical_patterns: List of historical patterns with outcomes
            max_matches: Maximum number of matches to return
        
        Returns:
            List of PatternMatch objects sorted by similarity
        """
        matches = []
        
        for hist_pattern in historical_patterns:
            similarity = self.calculate_similarity(current_pattern, hist_pattern)
            correlation = self._calculate_correlation(
                current_pattern['price_pattern'],
                hist_pattern['price_pattern']
            )
            
            if (similarity >= self.similarity_threshold and 
                correlation >= self.correlation_threshold):
                
                match = PatternMatch(
                    pattern_id=hist_pattern.get('pattern_id', 'unknown'),
                    similarity_score=similarity,
                    correlation=correlation,
                    historical_date=hist_pattern.get('timestamp', datetime.now()),
                    pattern_length=hist_pattern.get('pattern_length', 0),
                    future_outcome=hist_pattern.get('future_outcome', {}),
                    pattern_features=hist_pattern
                )
                matches.append(match)
        
        # Sort by similarity (highest first)
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return matches[:max_matches]
    
    def _calculate_correlation(self, pattern1: List, pattern2: List) -> float:
        """Calculate Pearson correlation between patterns"""
        try:
            if len(pattern1) != len(pattern2) or len(pattern1) < 2:
                return 0.0
            
            arr1 = np.array(pattern1)
            arr2 = np.array(pattern2)
            
            if arr1.std() == 0 or arr2.std() == 0:
                return 0.0
            
            correlation, _ = pearsonr(arr1, arr2)
            return max(0, float(correlation))  # Only positive correlations
        except:
            return 0.0


class PatternForecaster:
    """
    Forecasts future price movements based on historical pattern matches
    Similar to forecaster.biz approach
    """
    
    def __init__(self, lookahead_periods: int = 10):
        """
        Initialize pattern forecaster
        
        Args:
            lookahead_periods: Number of periods to forecast ahead
        """
        self.lookahead_periods = lookahead_periods
        self.extractor = PatternExtractor()
        self.matcher = PatternMatcher()
    
    def forecast(
        self,
        current_data: pd.DataFrame,
        historical_data: pd.DataFrame,
        min_matches: int = 3
    ) -> Dict:
        """
        Forecast future price movement based on pattern matching
        
        Args:
            current_data: Current market data
            historical_data: Historical data for pattern matching
            min_matches: Minimum number of similar patterns required
        
        Returns:
            Forecast dictionary with predicted direction, magnitude, and confidence
        """
        # Extract current pattern
        current_pattern = self.extractor.extract_pattern(current_data)
        if not current_pattern:
            return {
                'signal': None,
                'forecast_price': None,
                'forecast_change_pct': None,
                'confidence': 0.0,
                'matches_found': 0,
                'reason': 'Insufficient data for pattern extraction'
            }
        
        # Extract historical patterns with outcomes
        historical_patterns = self._extract_historical_patterns_with_outcomes(historical_data)
        
        if len(historical_patterns) < min_matches:
            return {
                'signal': None,
                'forecast_price': None,
                'forecast_change_pct': None,
                'confidence': 0.0,
                'matches_found': len(historical_patterns),
                'reason': f'Insufficient historical patterns (found {len(historical_patterns)}, need {min_matches})'
            }
        
        # Find similar patterns
        matches = self.matcher.find_similar_patterns(
            current_pattern,
            historical_patterns,
            max_matches=20
        )
        
        if len(matches) < min_matches:
            return {
                'signal': None,
                'forecast_price': None,
                'forecast_change_pct': None,
                'confidence': 0.0,
                'matches_found': len(matches),
                'reason': f'Insufficient pattern matches (found {len(matches)}, need {min_matches})'
            }
        
        # Aggregate outcomes from matched patterns
        forecast = self._aggregate_outcomes(matches, current_data['close'].iloc[-1])
        
        forecast['matches_found'] = len(matches)
        forecast['top_matches'] = [
            {
                'similarity': m.similarity_score,
                'correlation': m.correlation,
                'date': m.historical_date.isoformat() if hasattr(m.historical_date, 'isoformat') else str(m.historical_date),
                'outcome': m.future_outcome
            }
            for m in matches[:5]
        ]
        
        return forecast
    
    def _extract_historical_patterns_with_outcomes(
        self,
        historical_data: pd.DataFrame
    ) -> List[Dict]:
        """
        Extract patterns from historical data and calculate their outcomes
        
        Args:
            historical_data: Historical OHLCV data
        
        Returns:
            List of patterns with future outcomes
        """
        patterns = []
        extractor = PatternExtractor(pattern_length=self.extractor.pattern_length)
        
        # Extract patterns with step size
        step = max(1, self.extractor.pattern_length // 2)
        
        for i in range(self.extractor.pattern_length, len(historical_data) - self.lookahead_periods, step):
            # Extract pattern
            pattern = extractor.extract_pattern(historical_data, i)
            if not pattern:
                continue
            
            # Calculate future outcome
            future_start = i
            future_end = min(i + self.lookahead_periods, len(historical_data))
            
            if future_end > future_start:
                future_data = historical_data.iloc[future_start:future_end]
                outcome = self._calculate_outcome(
                    historical_data.iloc[i-1]['close'],  # Price at pattern end
                    future_data
                )
                
                pattern['timestamp'] = historical_data.index[i] if hasattr(historical_data.index[i], 'timestamp') else i
                pattern['future_outcome'] = outcome
                pattern['pattern_id'] = f"pattern_{i}"
                
                patterns.append(pattern)
        
        return patterns
    
    def _calculate_outcome(self, start_price: float, future_data: pd.DataFrame) -> Dict:
        """
        Calculate what happened after a pattern
        
        Args:
            start_price: Price at end of pattern
            future_data: Future price data
        
        Returns:
            Outcome dictionary
        """
        if len(future_data) == 0:
            return {
                'price_change_pct': 0.0,
                'max_gain_pct': 0.0,
                'max_loss_pct': 0.0,
                'final_price': start_price,
                'direction': 'neutral'
            }
        
        final_price = future_data['close'].iloc[-1]
        max_price = future_data['high'].max()
        min_price = future_data['low'].min()
        
        price_change_pct = ((final_price - start_price) / start_price) * 100
        max_gain_pct = ((max_price - start_price) / start_price) * 100
        max_loss_pct = ((min_price - start_price) / start_price) * 100
        
        if price_change_pct > 1.0:
            direction = 'bullish'
        elif price_change_pct < -1.0:
            direction = 'bearish'
        else:
            direction = 'neutral'
        
        return {
            'price_change_pct': float(price_change_pct),
            'max_gain_pct': float(max_gain_pct),
            'max_loss_pct': float(max_loss_pct),
            'final_price': float(final_price),
            'direction': direction,
            'periods': len(future_data)
        }
    
    def _aggregate_outcomes(
        self,
        matches: List[PatternMatch],
        current_price: float
    ) -> Dict:
        """
        Aggregate outcomes from matched patterns to create forecast
        
        Args:
            matches: List of matched patterns
            current_price: Current price
        
        Returns:
            Forecast dictionary
        """
        if not matches:
            return {
                'signal': None,
                'forecast_price': None,
                'forecast_change_pct': None,
                'confidence': 0.0
            }
        
        # Weight outcomes by similarity
        weighted_changes = []
        weights = []
        directions = []
        
        for match in matches:
            outcome = match.future_outcome
            similarity = match.similarity_score
            
            if 'price_change_pct' in outcome:
                weighted_changes.append(outcome['price_change_pct'] * similarity)
                weights.append(similarity)
                directions.append(outcome.get('direction', 'neutral'))
        
        if not weights:
            return {
                'signal': None,
                'forecast_price': None,
                'forecast_change_pct': None,
                'confidence': 0.0
            }
        
        # Calculate weighted average forecast
        total_weight = sum(weights)
        forecast_change_pct = sum(weighted_changes) / total_weight if total_weight > 0 else 0.0
        forecast_price = current_price * (1 + forecast_change_pct / 100)
        
        # Calculate confidence based on:
        # 1. Number of matches
        # 2. Average similarity
        # 3. Consistency of directions
        avg_similarity = np.mean([m.similarity_score for m in matches])
        direction_consistency = self._calculate_direction_consistency(directions)
        
        confidence = min(0.9, (
            min(len(matches) / 10, 1.0) * 0.3 +  # More matches = higher confidence
            avg_similarity * 0.4 +  # Higher similarity = higher confidence
            direction_consistency * 0.3  # More consistent = higher confidence
        ))
        
        # Determine signal
        if forecast_change_pct > 1.0:
            signal = 'bullish'
        elif forecast_change_pct < -1.0:
            signal = 'bearish'
        else:
            signal = 'neutral'
        
        return {
            'signal': signal,
            'forecast_price': float(forecast_price),
            'forecast_change_pct': float(forecast_change_pct),
            'confidence': float(confidence),
            'avg_similarity': float(avg_similarity),
            'direction_consistency': float(direction_consistency),
            'matches_count': len(matches)
        }
    
    def _calculate_direction_consistency(self, directions: List[str]) -> float:
        """Calculate how consistent the directions are"""
        if not directions:
            return 0.0
        
        bullish_count = directions.count('bullish')
        bearish_count = directions.count('bearish')
        neutral_count = directions.count('neutral')
        
        total = len(directions)
        max_count = max(bullish_count, bearish_count, neutral_count)
        
        return max_count / total if total > 0 else 0.0

