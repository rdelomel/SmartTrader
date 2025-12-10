"""Signal aggregation from multiple sources"""

from typing import Dict, List, Optional
from ..strategies.base_strategy import Signal
import numpy as np


class SignalAggregator:
    """Aggregates signals from multiple strategies/models"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize signal aggregator
        
        Config parameters:
            aggregation_method: 'weighted_average', 'majority_vote', 'confidence_weighted'
            min_confidence: Minimum confidence threshold for signal
        """
        self.config = config or {}
        self.aggregation_method = self.config.get('aggregation_method', 'weighted_average')
        self.min_confidence = self.config.get('min_confidence', 0.3)
    
    def aggregate(self, signals: List[Dict]) -> Dict:
        """
        Aggregate multiple signals into final decision
        
        Args:
            signals: List of signal dictionaries, each with:
                - 'signal': Signal (BUY/SELL/HOLD)
                - 'confidence': float (0-1)
                - 'weight': float (strategy weight)
                - 'source': str (strategy/model name)
        
        Returns:
            Aggregated signal dictionary
        """
        if not signals:
            return {
                'signal': Signal.HOLD,
                'confidence': 0.0,
                'reason': 'No signals available'
            }
        
        # Filter enabled signals
        enabled_signals = [s for s in signals if s.get('enabled', True)]
        
        if not enabled_signals:
            return {
                'signal': Signal.HOLD,
                'confidence': 0.0,
                'reason': 'No enabled signals'
            }
        
        if self.aggregation_method == 'weighted_average':
            return self._weighted_average(enabled_signals)
        elif self.aggregation_method == 'majority_vote':
            return self._majority_vote(enabled_signals)
        elif self.aggregation_method == 'confidence_weighted':
            return self._confidence_weighted(enabled_signals)
        else:
            return self._weighted_average(enabled_signals)
    
    def _weighted_average(self, signals: List[Dict]) -> Dict:
        """Weighted average aggregation"""
        # Convert signals to numeric values
        signal_values = []
        weights = []
        confidences = []
        sources = []
        
        for sig in signals:
            signal_val = 1 if sig['signal'] == Signal.BUY else (-1 if sig['signal'] == Signal.SELL else 0)
            weight = sig.get('weight', 1.0)
            confidence = sig.get('confidence', 0.5)
            
            signal_values.append(signal_val * confidence * weight)
            weights.append(weight)
            confidences.append(confidence)
            sources.append(sig.get('source', 'unknown'))
        
        # Calculate weighted average
        total_weight = sum(weights)
        if total_weight == 0:
            return {
                'signal': Signal.HOLD,
                'confidence': 0.0,
                'reason': 'Zero total weight'
            }
        
        weighted_signal = sum(signal_values) / total_weight
        
        # Determine final signal
        if weighted_signal > self.min_confidence:
            final_signal = Signal.BUY
            confidence = min(abs(weighted_signal), 1.0)
        elif weighted_signal < -self.min_confidence:
            final_signal = Signal.SELL
            confidence = min(abs(weighted_signal), 1.0)
        else:
            final_signal = Signal.HOLD
            confidence = abs(weighted_signal)
        
        # Build reason
        buy_count = sum(1 for s in signals if s['signal'] == Signal.BUY)
        sell_count = sum(1 for s in signals if s['signal'] == Signal.SELL)
        reason = f"Weighted average: {weighted_signal:.3f} (BUY: {buy_count}, SELL: {sell_count})"
        
        return {
            'signal': final_signal,
            'confidence': confidence,
            'weighted_value': weighted_signal,
            'reason': reason,
            'sources': sources
        }
    
    def _majority_vote(self, signals: List[Dict]) -> Dict:
        """Majority vote aggregation"""
        buy_votes = sum(1 for s in signals if s['signal'] == Signal.BUY)
        sell_votes = sum(1 for s in signals if s['signal'] == Signal.SELL)
        hold_votes = sum(1 for s in signals if s['signal'] == Signal.HOLD)
        
        total_votes = len(signals)
        
        if buy_votes > sell_votes and buy_votes > hold_votes:
            signal = Signal.BUY
            confidence = buy_votes / total_votes
        elif sell_votes > buy_votes and sell_votes > hold_votes:
            signal = Signal.SELL
            confidence = sell_votes / total_votes
        else:
            signal = Signal.HOLD
            confidence = 0.0
        
        reason = f"Majority vote: BUY={buy_votes}, SELL={sell_votes}, HOLD={hold_votes}"
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': reason
        }
    
    def _confidence_weighted(self, signals: List[Dict]) -> Dict:
        """Confidence-weighted aggregation"""
        buy_strength = sum(
            s.get('confidence', 0) * s.get('weight', 1.0)
            for s in signals if s['signal'] == Signal.BUY
        )
        sell_strength = sum(
            s.get('confidence', 0) * s.get('weight', 1.0)
            for s in signals if s['signal'] == Signal.SELL
        )
        
        if buy_strength > sell_strength and buy_strength > self.min_confidence:
            signal = Signal.BUY
            confidence = min(buy_strength / max(buy_strength + sell_strength, 0.001), 1.0)
        elif sell_strength > buy_strength and sell_strength > self.min_confidence:
            signal = Signal.SELL
            confidence = min(sell_strength / max(buy_strength + sell_strength, 0.001), 1.0)
        else:
            signal = Signal.HOLD
            confidence = 0.0
        
        reason = f"Confidence weighted: BUY={buy_strength:.3f}, SELL={sell_strength:.3f}"
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': reason
        }

