"""Chart pattern recognition for technical analysis"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum


class PatternType(Enum):
    """Chart pattern types"""
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIANGLE_ASCENDING = "triangle_ascending"
    TRIANGLE_DESCENDING = "triangle_descending"
    TRIANGLE_SYMMETRICAL = "triangle_symmetrical"
    WEDGE_RISING = "wedge_rising"
    WEDGE_FALLING = "wedge_falling"
    FLAG_BULLISH = "flag_bullish"
    FLAG_BEARISH = "flag_bearish"
    PENNANT_BULLISH = "pennant_bullish"
    PENNANT_BEARISH = "pennant_bearish"
    CUP_AND_HANDLE = "cup_and_handle"
    RECTANGLE = "rectangle"


class ChartPatternRecognizer:
    """
    Chart pattern recognition based on Investopedia technical analysis principles
    - Identifies common chart patterns that suggest future price movements
    - Uses price action, volume, and trend analysis
    """
    
    def __init__(self, min_pattern_bars: int = 20, max_pattern_bars: int = 100):
        """
        Initialize pattern recognizer
        
        Args:
            min_pattern_bars: Minimum bars to form a pattern
            max_pattern_bars: Maximum bars to look back for patterns
        """
        self.min_pattern_bars = min_pattern_bars
        self.max_pattern_bars = max_pattern_bars
    
    def detect_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """
        Detect chart patterns in price data
        
        Args:
            data: OHLCV DataFrame
        
        Returns:
            List of detected patterns with type, direction, and confidence
        """
        if len(data) < self.min_pattern_bars:
            return []
        
        patterns = []
        
        # Look at recent data (last max_pattern_bars)
        lookback = min(self.max_pattern_bars, len(data))
        recent_data = data.iloc[-lookback:].copy()
        
        # Detect various patterns
        patterns.extend(self._detect_head_and_shoulders(recent_data))
        patterns.extend(self._detect_double_top_bottom(recent_data))
        patterns.extend(self._detect_triangles(recent_data))
        patterns.extend(self._detect_flags_pennants(recent_data))
        patterns.extend(self._detect_cup_and_handle(recent_data))
        
        # Sort by confidence (highest first)
        patterns.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return patterns[:5]  # Return top 5 patterns
    
    def _detect_head_and_shoulders(self, data: pd.DataFrame) -> List[Dict]:
        """Detect head and shoulders pattern (bearish reversal)"""
        patterns = []
        if len(data) < 30:
            return patterns
        
        # Find local peaks
        peaks = self._find_peaks(data, window=5)
        
        if len(peaks) < 3:
            return patterns
        
        # Look for three peaks: left shoulder, head, right shoulder
        for i in range(len(peaks) - 2):
            left_shoulder = peaks[i]
            head = peaks[i + 1]
            right_shoulder = peaks[i + 2]
            
            # Head should be highest
            if (head['price'] > left_shoulder['price'] and 
                head['price'] > right_shoulder['price']):
                
                # Shoulders should be roughly equal
                shoulder_diff = abs(left_shoulder['price'] - right_shoulder['price']) / head['price']
                
                if shoulder_diff < 0.05:  # Within 5%
                    # Check for neckline (support level between shoulders)
                    neckline = min(data.iloc[left_shoulder['index']:right_shoulder['index']]['low'].min(),
                                 data.iloc[left_shoulder['index']:right_shoulder['index']]['close'].min())
                    
                    confidence = 0.7 if shoulder_diff < 0.03 else 0.5
                    
                    patterns.append({
                        'type': PatternType.HEAD_AND_SHOULDERS,
                        'direction': 'bearish',
                        'confidence': confidence,
                        'entry_price': data['close'].iloc[-1],
                        'target': neckline * 0.95,  # 5% below neckline
                        'stop_loss': head['price'] * 1.02
                    })
        
        return patterns
    
    def _detect_double_top_bottom(self, data: pd.DataFrame) -> List[Dict]:
        """Detect double top (bearish) and double bottom (bullish) patterns"""
        patterns = []
        if len(data) < 20:
            return patterns
        
        peaks = self._find_peaks(data, window=5)
        troughs = self._find_troughs(data, window=5)
        
        # Double top
        if len(peaks) >= 2:
            for i in range(len(peaks) - 1):
                peak1 = peaks[i]
                peak2 = peaks[i + 1]
                
                # Peaks should be similar (within 2%)
                price_diff = abs(peak1['price'] - peak2['price']) / peak1['price']
                
                if price_diff < 0.02:
                    # Find valley between peaks
                    valley_idx = data.iloc[peak1['index']:peak2['index']]['low'].idxmin()
                    valley_price = data.loc[valley_idx, 'low']
                    
                    confidence = 0.65 if price_diff < 0.01 else 0.5
                    
                    patterns.append({
                        'type': PatternType.DOUBLE_TOP,
                        'direction': 'bearish',
                        'confidence': confidence,
                        'entry_price': data['close'].iloc[-1],
                        'target': valley_price * 0.95,
                        'stop_loss': max(peak1['price'], peak2['price']) * 1.02
                    })
        
        # Double bottom
        if len(troughs) >= 2:
            for i in range(len(troughs) - 1):
                trough1 = troughs[i]
                trough2 = troughs[i + 1]
                
                # Troughs should be similar
                price_diff = abs(trough1['price'] - trough2['price']) / trough1['price']
                
                if price_diff < 0.02:
                    # Find peak between troughs
                    peak_idx = data.iloc[trough1['index']:trough2['index']]['high'].idxmax()
                    peak_price = data.loc[peak_idx, 'high']
                    
                    confidence = 0.65 if price_diff < 0.01 else 0.5
                    
                    patterns.append({
                        'type': PatternType.DOUBLE_BOTTOM,
                        'direction': 'bullish',
                        'confidence': confidence,
                        'entry_price': data['close'].iloc[-1],
                        'target': peak_price * 1.05,
                        'stop_loss': min(trough1['price'], trough2['price']) * 0.98
                    })
        
        return patterns
    
    def _detect_triangles(self, data: pd.DataFrame) -> List[Dict]:
        """Detect ascending, descending, and symmetrical triangles"""
        patterns = []
        if len(data) < 20:
            return patterns
        
        # Use recent data for triangle detection
        recent = data.iloc[-20:]
        
        # Find trendlines
        highs = recent['high'].values
        lows = recent['low'].values
        
        # Calculate slopes
        high_slope = np.polyfit(range(len(highs)), highs, 1)[0]
        low_slope = np.polyfit(range(len(lows)), lows, 1)[0]
        
        # Ascending triangle: horizontal resistance, rising support
        if abs(high_slope) < 0.01 and low_slope > 0.01:
            patterns.append({
                'type': PatternType.TRIANGLE_ASCENDING,
                'direction': 'bullish',
                'confidence': 0.6,
                'entry_price': data['close'].iloc[-1],
                'target': recent['high'].max() * 1.03,
                'stop_loss': recent['low'].min() * 0.98
            })
        
        # Descending triangle: falling resistance, horizontal support
        elif high_slope < -0.01 and abs(low_slope) < 0.01:
            patterns.append({
                'type': PatternType.TRIANGLE_DESCENDING,
                'direction': 'bearish',
                'confidence': 0.6,
                'entry_price': data['close'].iloc[-1],
                'target': recent['low'].min() * 0.97,
                'stop_loss': recent['high'].max() * 1.02
            })
        
        # Symmetrical triangle: converging trendlines
        elif abs(high_slope) > 0.01 and abs(low_slope) > 0.01 and np.sign(high_slope) != np.sign(low_slope):
            # Determine direction based on current trend
            current_trend = 'bullish' if data['close'].iloc[-1] > data['close'].iloc[-10] else 'bearish'
            patterns.append({
                'type': PatternType.TRIANGLE_SYMMETRICAL,
                'direction': current_trend,
                'confidence': 0.55,
                'entry_price': data['close'].iloc[-1],
                'target': data['close'].iloc[-1] * (1.03 if current_trend == 'bullish' else 0.97),
                'stop_loss': data['close'].iloc[-1] * (0.98 if current_trend == 'bullish' else 1.02)
            })
        
        return patterns
    
    def _detect_flags_pennants(self, data: pd.DataFrame) -> List[Dict]:
        """Detect flag and pennant patterns (continuation patterns)"""
        patterns = []
        if len(data) < 15:
            return patterns
        
        # Flags: small rectangular consolidation after strong move
        recent = data.iloc[-15:]
        price_range = recent['high'].max() - recent['low'].min()
        price_change = abs((recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0])
        
        # Strong move followed by consolidation
        if price_change > 0.03 and price_range / recent['close'].mean() < 0.02:
            direction = 'bullish' if recent['close'].iloc[-1] > recent['close'].iloc[0] else 'bearish'
            
            patterns.append({
                'type': PatternType.FLAG_BULLISH if direction == 'bullish' else PatternType.FLAG_BEARISH,
                'direction': direction,
                'confidence': 0.6,
                'entry_price': data['close'].iloc[-1],
                'target': data['close'].iloc[-1] * (1.04 if direction == 'bullish' else 0.96),
                'stop_loss': data['close'].iloc[-1] * (0.98 if direction == 'bullish' else 1.02)
            })
        
        return patterns
    
    def _detect_cup_and_handle(self, data: pd.DataFrame) -> List[Dict]:
        """Detect cup and handle pattern (bullish continuation)"""
        patterns = []
        if len(data) < 40:
            return patterns
        
        # Look for U-shaped cup followed by small handle
        recent = data.iloc[-40:]
        mid_point = len(recent) // 2
        
        # Cup: U-shape (decline then rise)
        first_half = recent.iloc[:mid_point]
        second_half = recent.iloc[mid_point:]
        
        cup_decline = (first_half['low'].min() - first_half['high'].iloc[0]) / first_half['high'].iloc[0]
        cup_rise = (second_half['high'].iloc[-1] - first_half['low'].min()) / first_half['low'].min()
        
        if cup_decline < -0.05 and cup_rise > 0.05:
            # Handle: small pullback
            handle = recent.iloc[-10:]
            handle_pullback = (handle['low'].min() - handle['high'].iloc[0]) / handle['high'].iloc[0]
            
            if -0.03 < handle_pullback < -0.01:  # Small pullback
                patterns.append({
                    'type': PatternType.CUP_AND_HANDLE,
                    'direction': 'bullish',
                    'confidence': 0.65,
                    'entry_price': data['close'].iloc[-1],
                    'target': recent['high'].max() * 1.05,
                    'stop_loss': handle['low'].min() * 0.98
                })
        
        return patterns
    
    def _find_peaks(self, data: pd.DataFrame, window: int = 5) -> List[Dict]:
        """Find local peaks in price data"""
        peaks = []
        for i in range(window, len(data) - window):
            if data['high'].iloc[i] == data.iloc[i-window:i+window+1]['high'].max():
                peaks.append({
                    'index': i,
                    'price': data['high'].iloc[i],
                    'time': data.index[i] if hasattr(data.index[i], 'timestamp') else i
                })
        return peaks
    
    def _find_troughs(self, data: pd.DataFrame, window: int = 5) -> List[Dict]:
        """Find local troughs in price data"""
        troughs = []
        for i in range(window, len(data) - window):
            if data['low'].iloc[i] == data.iloc[i-window:i+window+1]['low'].min():
                troughs.append({
                    'index': i,
                    'price': data['low'].iloc[i],
                    'time': data.index[i] if hasattr(data.index[i], 'timestamp') else i
                })
        return troughs
    
    def find_support_resistance(self, data: pd.DataFrame, lookback: int = 50) -> Dict:
        """
        Find support and resistance levels based on Investopedia principles
        
        Args:
            data: OHLCV DataFrame
            lookback: Number of bars to analyze
        
        Returns:
            Dictionary with support and resistance levels
        """
        if len(data) < lookback:
            lookback = len(data)
        
        recent = data.iloc[-lookback:]
        
        # Find significant price levels (where price bounced multiple times)
        price_levels = []
        for i in range(len(recent)):
            price_levels.append(recent['high'].iloc[i])
            price_levels.append(recent['low'].iloc[i])
        
        # Cluster price levels (within 1% of each other)
        clusters = []
        sorted_levels = sorted(set(price_levels))
        
        for level in sorted_levels:
            found_cluster = False
            for cluster in clusters:
                if abs(level - cluster['center']) / cluster['center'] < 0.01:
                    cluster['levels'].append(level)
                    cluster['count'] += 1
                    found_cluster = True
                    break
            
            if not found_cluster:
                clusters.append({
                    'center': level,
                    'levels': [level],
                    'count': 1
                })
        
        # Find strongest support (most touches at lower levels)
        support_levels = sorted([c for c in clusters if c['center'] < recent['close'].iloc[-1]], 
                               key=lambda x: (x['count'], -x['center']), reverse=True)
        resistance_levels = sorted([c for c in clusters if c['center'] > recent['close'].iloc[-1]], 
                                  key=lambda x: (x['count'], x['center']), reverse=True)
        
        return {
            'support': support_levels[0]['center'] if support_levels else None,
            'resistance': resistance_levels[0]['center'] if resistance_levels else None,
            'support_strength': support_levels[0]['count'] if support_levels else 0,
            'resistance_strength': resistance_levels[0]['count'] if resistance_levels else 0
        }

