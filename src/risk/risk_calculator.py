"""Portfolio risk calculations"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional


class RiskCalculator:
    """Calculate portfolio risk metrics"""
    
    @staticmethod
    def calculate_portfolio_value(positions: List[Dict], current_prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value
        
        Args:
            positions: List of position dictionaries
            current_prices: Dictionary of current prices by symbol
        
        Returns:
            Total portfolio value
        """
        total_value = 0.0
        
        for position in positions:
            symbol = position.get('symbol')
            quantity = position.get('quantity', 0)
            entry_price = position.get('entry_price', 0)
            
            if symbol in current_prices:
                current_price = current_prices[symbol]
                total_value += quantity * current_price
            else:
                total_value += quantity * entry_price
        
        return total_value
    
    @staticmethod
    def calculate_portfolio_risk(
        positions: List[Dict],
        current_prices: Dict[str, float],
        account_balance: float
    ) -> Dict:
        """
        Calculate portfolio risk metrics
        
        Args:
            positions: List of open positions
            current_prices: Dictionary of current prices
            account_balance: Current account balance
        
        Returns:
            Dictionary with risk metrics
        """
        if account_balance == 0:
            return {
                'total_risk': 0.0,
                'risk_percent': 0.0,
                'position_count': 0,
                'largest_position_percent': 0.0
            }
        
        total_risk = 0.0
        position_values = []
        
        for position in positions:
            symbol = position.get('symbol')
            quantity = position.get('quantity', 0)
            entry_price = position.get('entry_price', 0)
            stop_loss = position.get('stop_loss', entry_price)
            
            if symbol in current_prices:
                current_price = current_prices[symbol]
            else:
                current_price = entry_price
            
            # Calculate risk for this position
            risk_per_unit = abs(entry_price - stop_loss)
            position_risk = quantity * risk_per_unit
            total_risk += position_risk
            
            # Calculate position value
            position_value = quantity * current_price
            position_values.append(position_value)
        
        risk_percent = (total_risk / account_balance) * 100
        
        largest_position = max(position_values) if position_values else 0.0
        largest_position_percent = (largest_position / account_balance) * 100
        
        return {
            'total_risk': total_risk,
            'risk_percent': risk_percent,
            'position_count': len(positions),
            'largest_position_percent': largest_position_percent,
            'average_position_percent': (sum(position_values) / account_balance * 100) / len(positions) if positions else 0.0
        }
    
    @staticmethod
    def calculate_correlation_risk(positions: List[Dict]) -> float:
        """
        Calculate correlation risk (simplified - assumes high correlation for same asset class)
        
        Args:
            positions: List of positions
        
        Returns:
            Correlation risk score (0-1)
        """
        if len(positions) <= 1:
            return 0.0
        
        # Group positions by asset class (simplified)
        asset_classes = {}
        for position in positions:
            symbol = position.get('symbol', '')
            # Simple classification
            if '/' in symbol and 'USD' in symbol:
                asset_class = 'forex'
            elif '/' in symbol:
                asset_class = 'crypto'
            else:
                asset_class = 'stock'
            
            if asset_class not in asset_classes:
                asset_classes[asset_class] = []
            asset_classes[asset_class].append(position)
        
        # High correlation risk if many positions in same asset class
        max_positions_in_class = max(len(positions) for positions in asset_classes.values()) if asset_classes else 0
        correlation_risk = min(max_positions_in_class / len(positions), 1.0)
        
        return correlation_risk
    
    @staticmethod
    def calculate_max_portfolio_risk(
        account_balance: float,
        max_risk_per_trade: float,
        max_concurrent_positions: int = 5
    ) -> float:
        """
        Calculate maximum portfolio risk
        
        Args:
            account_balance: Account balance
            max_risk_per_trade: Maximum risk per trade
            max_concurrent_positions: Maximum concurrent positions
        
        Returns:
            Maximum portfolio risk
        """
        return max_risk_per_trade * max_concurrent_positions

