"""Portfolio-level risk management with correlation and diversification"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict


class PortfolioRiskManager:
    """Manage portfolio-level risk including correlation and diversification"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize portfolio risk manager
        
        Config parameters:
            max_portfolio_exposure: Maximum total portfolio exposure as % of capital (default: 50.0)
            max_correlation: Maximum correlation between positions (default: 0.7)
            max_sector_concentration: Maximum exposure to single sector/asset class (default: 30.0)
            diversification_threshold: Minimum diversification score (default: 0.6)
        """
        self.config = config or {}
        # Support both config key names for compatibility
        self.max_portfolio_exposure = self.config.get('max_exposure_percent', 
                                                      self.config.get('max_portfolio_exposure', 20.0))
        self.max_correlation = self.config.get('max_correlation', 0.7)
        self.max_sector_concentration = self.config.get('max_sector_concentration', 30.0)
        self.diversification_threshold = self.config.get('diversification_threshold', 0.6)
        
        # Track current positions
        self.positions = {}  # symbol -> position info
        self.position_history = []  # Historical positions for correlation calculation
    
    def add_position(self, symbol: str, side: str, quantity: float, 
                    entry_price: float, asset_class: str = 'unknown'):
        """
        Add a position to portfolio tracking
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Position quantity
            entry_price: Entry price
            asset_class: Asset class (crypto, forex, stocks)
        """
        position_value = quantity * entry_price
        self.positions[symbol] = {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'entry_price': entry_price,
            'value': position_value,
            'asset_class': asset_class,
            'added_at': datetime.now()
        }
    
    def remove_position(self, symbol: str):
        """Remove a position from portfolio tracking"""
        if symbol in self.positions:
            # Add to history before removing
            self.position_history.append(self.positions[symbol])
            del self.positions[symbol]
    
    def check_portfolio_limits(self, account_balance: float, 
                               proposed_symbol: str, proposed_value: float,
                               proposed_asset_class: str = 'unknown') -> Dict:
        """
        Check if proposed position violates portfolio-level limits
        
        Args:
            account_balance: Current account balance
            proposed_symbol: Symbol of proposed position
            proposed_value: Value of proposed position
            proposed_asset_class: Asset class of proposed position
        
        Returns:
            Dictionary with check results and recommendations
        """
        violations = []
        warnings = []
        
        # Calculate current portfolio exposure
        total_exposure = sum(pos['value'] for pos in self.positions.values())
        total_exposure_pct = (total_exposure / account_balance * 100) if account_balance > 0 else 0.0
        
        # Check maximum portfolio exposure
        if (total_exposure + proposed_value) / account_balance * 100 > self.max_portfolio_exposure:
            violations.append({
                'type': 'max_portfolio_exposure',
                'current': total_exposure_pct,
                'proposed': (total_exposure + proposed_value) / account_balance * 100,
                'limit': self.max_portfolio_exposure,
                'message': f'Portfolio exposure would exceed {self.max_portfolio_exposure}%'
            })
        
        # Check sector/asset class concentration
        asset_class_exposure = defaultdict(float)
        for pos in self.positions.values():
            asset_class_exposure[pos['asset_class']] += pos['value']
        
        proposed_asset_exposure = asset_class_exposure[proposed_asset_class] + proposed_value
        asset_exposure_pct = (proposed_asset_exposure / account_balance * 100) if account_balance > 0 else 0.0
        
        if asset_exposure_pct > self.max_sector_concentration:
            violations.append({
                'type': 'sector_concentration',
                'asset_class': proposed_asset_class,
                'exposure_pct': asset_exposure_pct,
                'limit': self.max_sector_concentration,
                'message': f'{proposed_asset_class} exposure would exceed {self.max_sector_concentration}%'
            })
        
        # Check correlation with existing positions
        correlation_warning = self._check_correlation(proposed_symbol, proposed_asset_class)
        if correlation_warning:
            warnings.append(correlation_warning)
        
        # Calculate diversification score
        diversification_score = self._calculate_diversification_score()
        if diversification_score < self.diversification_threshold:
            warnings.append({
                'type': 'low_diversification',
                'score': diversification_score,
                'threshold': self.diversification_threshold,
                'message': f'Portfolio diversification score ({diversification_score:.2f}) below threshold'
            })
        
        return {
            'approved': len(violations) == 0,
            'violations': violations,
            'warnings': warnings,
            'current_exposure_pct': total_exposure_pct,
            'diversification_score': diversification_score,
            'recommendation': 'APPROVED' if len(violations) == 0 else 'REJECTED'
        }
    
    def _check_correlation(self, proposed_symbol: str, proposed_asset_class: str) -> Optional[Dict]:
        """
        Check correlation with existing positions
        
        Returns:
            Warning dictionary if high correlation detected, None otherwise
        """
        if not self.positions:
            return None
        
        # Simplified correlation check based on asset class and symbol similarity
        # In production, would use actual price correlation data
        
        # Check for same asset class concentration
        same_class_positions = [pos for pos in self.positions.values() 
                              if pos['asset_class'] == proposed_asset_class]
        
        if len(same_class_positions) >= 3:  # Too many positions in same asset class
            return {
                'type': 'high_correlation_risk',
                'message': f'Multiple positions in {proposed_asset_class} may be highly correlated',
                'existing_positions': len(same_class_positions)
            }
        
        # Check for similar symbols (e.g., BTC/USD and ETH/USD are correlated)
        if proposed_asset_class == 'crypto':
            crypto_positions = [pos for pos in self.positions.values() 
                              if pos['asset_class'] == 'crypto']
            if len(crypto_positions) >= 2:
                return {
                    'type': 'crypto_correlation',
                    'message': 'Multiple crypto positions may be highly correlated',
                    'existing_crypto': len(crypto_positions)
                }
        
        return None
    
    def _calculate_diversification_score(self) -> float:
        """
        Calculate portfolio diversification score (0-1)
        
        Higher score = better diversification
        """
        if not self.positions:
            return 1.0  # Empty portfolio is perfectly diversified
        
        num_positions = len(self.positions)
        if num_positions == 0:
            return 1.0
        
        # Count unique asset classes
        asset_classes = set(pos['asset_class'] for pos in self.positions.values())
        num_asset_classes = len(asset_classes)
        
        # Calculate position size distribution (more even = better diversification)
        position_values = [pos['value'] for pos in self.positions.values()]
        total_value = sum(position_values)
        
        if total_value == 0:
            return 0.0
        
        # Calculate Gini coefficient (lower = more even distribution)
        sorted_values = sorted(position_values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        gini = (2 * sum((i + 1) * val for i, val in enumerate(sorted_values))) / (n * total_value) - (n + 1) / n
        
        # Diversification score combines:
        # - Number of positions (more = better, up to a point)
        # - Number of asset classes (more = better)
        # - Even distribution (lower Gini = better)
        position_score = min(1.0, num_positions / 5.0)  # Optimal at 5+ positions
        asset_class_score = min(1.0, num_asset_classes / 3.0)  # Optimal at 3+ asset classes
        distribution_score = 1.0 - gini  # Lower Gini = higher score
        
        # Weighted average
        diversification_score = (position_score * 0.3 + 
                                asset_class_score * 0.4 + 
                                distribution_score * 0.3)
        
        return diversification_score
    
    def get_portfolio_summary(self, account_balance: float) -> Dict:
        """
        Get portfolio summary with risk metrics
        
        Args:
            account_balance: Current account balance
        
        Returns:
            Portfolio summary dictionary
        """
        if not self.positions:
            return {
                'total_positions': 0,
                'total_exposure': 0.0,
                'exposure_pct': 0.0,
                'asset_class_distribution': {},
                'diversification_score': 1.0
            }
        
        total_exposure = sum(pos['value'] for pos in self.positions.values())
        exposure_pct = (total_exposure / account_balance * 100) if account_balance > 0 else 0.0
        
        # Asset class distribution
        asset_class_dist = defaultdict(float)
        for pos in self.positions.values():
            asset_class_dist[pos['asset_class']] += pos['value']
        
        # Convert to percentages
        asset_class_pct = {k: (v / account_balance * 100) if account_balance > 0 else 0.0 
                          for k, v in asset_class_dist.items()}
        
        diversification_score = self._calculate_diversification_score()
        
        return {
            'total_positions': len(self.positions),
            'total_exposure': total_exposure,
            'exposure_pct': exposure_pct,
            'asset_class_distribution': asset_class_pct,
            'diversification_score': diversification_score,
            'positions': list(self.positions.keys())
        }
    
    def recommend_position_size(self, account_balance: float, 
                                proposed_value: float,
                                proposed_asset_class: str) -> Dict:
        """
        Recommend position size based on portfolio constraints
        
        Args:
            account_balance: Current account balance
            proposed_value: Proposed position value
            proposed_asset_class: Asset class of proposed position
        
        Returns:
            Recommendation with adjusted position size if needed
        """
        # Check current exposure
        total_exposure = sum(pos['value'] for pos in self.positions.values())
        remaining_capacity = (account_balance * self.max_portfolio_exposure / 100) - total_exposure
        
        # Check asset class exposure
        asset_class_exposure = sum(pos['value'] for pos in self.positions.values() 
                                  if pos['asset_class'] == proposed_asset_class)
        asset_class_capacity = (account_balance * self.max_sector_concentration / 100) - asset_class_exposure
        
        # Recommended value is minimum of proposed, remaining capacity, and asset class capacity
        recommended_value = min(proposed_value, remaining_capacity, asset_class_capacity)
        recommended_value = max(0.0, recommended_value)  # Ensure non-negative
        
        return {
            'original_value': proposed_value,
            'recommended_value': recommended_value,
            'reduction_pct': ((proposed_value - recommended_value) / proposed_value * 100) if proposed_value > 0 else 0.0,
            'reason': 'Portfolio limits' if recommended_value < proposed_value else 'Within limits'
        }

