"""Position sizing calculations"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from ..indicators.technical import TechnicalIndicators


class PositionSizer:
    """Calculate position sizes based on risk parameters"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize position sizer
        
        Config parameters:
            position_size_percent: Percentage of capital to risk per trade (default: 1.5)
            max_position_size_percent: Maximum position size as % of capital (default: 5.0)
            method: 'fixed_fractional', 'volatility_scaled', 'kelly'
            risk_per_trade: Risk amount per trade in base currency
            dynamic_scaling: Enable dynamic position sizing based on performance
        """
        self.config = config or {}
        self.position_size_percent = self.config.get('position_size_percent', 1.5)
        self.max_position_size_percent = self.config.get('max_position_size_percent', 5.0)
        self.method = self.config.get('method', 'fixed_fractional')
        self.risk_per_trade = self.config.get('risk_per_trade', None)
        self.indicators = TechnicalIndicators()
        
        # Dynamic scaling configuration
        dynamic_config = self.config.get('dynamic_scaling', {})
        self.dynamic_scaling_enabled = dynamic_config.get('enabled', False)
        self.performance_threshold_sharpe = dynamic_config.get('performance_threshold_sharpe', 1.0)
        self.max_risk_per_trade = dynamic_config.get('max_risk_per_trade', 2.5)
        self.win_streak_bonus = dynamic_config.get('win_streak_bonus', 0.1)
        
        # Performance tracking
        self.recent_returns = []  # Track recent returns for Sharpe calculation
        self.consecutive_wins = 0  # Track consecutive wins
        self.recent_sharpe = 0.0  # Recent Sharpe ratio
    
    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss: float,
        data: Optional[pd.DataFrame] = None,
        recent_sharpe: Optional[float] = None,
        consecutive_wins: Optional[int] = None,
        symbol: str = ""
    ) -> Dict:
        """
        Calculate position size based on risk parameters with dynamic scaling
        
        Args:
            account_balance: Current account balance
            entry_price: Entry price for trade
            stop_loss: Stop loss price
            data: Optional market data for volatility-based sizing
            recent_sharpe: Recent Sharpe ratio for performance-based scaling (optional)
            consecutive_wins: Number of consecutive wins for win streak bonus (optional)
            symbol: Trading symbol to determine quote currency
        
        Returns:
            Dictionary with position size information
        """
        # Calculate risk per share/unit
        risk_per_unit = abs(entry_price - stop_loss)
        
        # Normalize risk_per_unit for pairs where quote currency is not USD
        # e.g. USD/JPY, EUR/GBP
        if symbol:
            symbol_upper = symbol.upper()
            is_quote_usd = True
            
            if '/' in symbol_upper:
                quote = symbol_upper.split('/')[1]
                if quote != 'USD':
                    is_quote_usd = False
            elif len(symbol_upper) >= 6 and not symbol_upper.endswith('USD'):
                is_quote_usd = False
                
            if not is_quote_usd and entry_price > 0:
                # Convert risk from quote currency back to base currency (approx USD)
                risk_per_unit = risk_per_unit / entry_price

        
        if risk_per_unit == 0:
            return {
                'quantity': 0.0,
                'value': 0.0,
                'risk_amount': 0.0,
                'method': self.method,
                'reason': 'Invalid stop loss'
            }
        
        # Calculate base risk amount with dynamic scaling
        base_risk_percent = self.position_size_percent
        if self.dynamic_scaling_enabled:
            # Performance-based scaling
            sharpe = recent_sharpe if recent_sharpe is not None else self.recent_sharpe
            if sharpe > self.performance_threshold_sharpe:
                # Increase risk when performing well
                base_risk_percent = min(
                    self.position_size_percent * 1.5,  # Up to 1.5x base risk
                    self.max_risk_per_trade
                )
            
            # Win streak bonus
            wins = consecutive_wins if consecutive_wins is not None else self.consecutive_wins
            if wins > 0:
                win_bonus = min(wins * self.win_streak_bonus, 1.0)  # Max 100% bonus
                base_risk_percent = min(
                    base_risk_percent * (1.0 + win_bonus),
                    self.max_risk_per_trade
                )
        
        # Calculate risk amount
        if self.risk_per_trade:
            risk_amount = self.risk_per_trade
        else:
            risk_amount = account_balance * (base_risk_percent / 100)
        
        # Calculate base position size
        if self.method == 'fixed_fractional':
            quantity = self._fixed_fractional(risk_amount, risk_per_unit)
        elif self.method == 'volatility_scaled':
            quantity = self._volatility_scaled(account_balance, entry_price, stop_loss, data, risk_amount)
        elif self.method == 'kelly':
            quantity = self._kelly_criterion(account_balance, entry_price, stop_loss, data)
        else:
            quantity = self._fixed_fractional(risk_amount, risk_per_unit)
        
        # Apply maximum position size limit
        max_position_value = account_balance * (self.max_position_size_percent / 100)
        max_quantity = max_position_value / entry_price
        
        quantity = min(quantity, max_quantity)
        
        # Ensure minimum position size (at least 0.5% of account balance to avoid tiny positions)
        min_position_value = account_balance * 0.005  # 0.5% minimum
        min_quantity = min_position_value / entry_price
        
        # Only enforce minimum if calculated quantity is meaningful (not zero or very small)
        if quantity > 0 and quantity < min_quantity:
            # Check if minimum would exceed max - if so, use calculated quantity
            if min_quantity <= max_quantity:
                quantity = min_quantity
                print(f"PositionSizer: Enforcing minimum position size ({min_position_value:.2f} = 0.5% of balance)")
        
        position_value = quantity * entry_price
        
        return {
            'quantity': quantity,
            'value': position_value,
            'risk_amount': risk_amount,
            'risk_percent': (risk_amount / account_balance) * 100,
            'base_risk_percent': base_risk_percent,
            'method': self.method,
            'reason': 'Position size calculated',
            'dynamic_scaling_applied': self.dynamic_scaling_enabled
        }
    
    def update_performance(self, trade_return: float, is_win: bool):
        """
        Update performance tracking for dynamic position sizing
        
        Args:
            trade_return: Return from the trade (as decimal, e.g., 0.02 for 2%)
            is_win: Whether the trade was profitable
        """
        # Track recent returns (keep last 30 trades)
        self.recent_returns.append(trade_return)
        if len(self.recent_returns) > 30:
            self.recent_returns.pop(0)
        
        # Calculate Sharpe ratio
        if len(self.recent_returns) > 10:
            returns_array = np.array(self.recent_returns)
            if returns_array.std() > 0:
                self.recent_sharpe = returns_array.mean() / returns_array.std() * np.sqrt(252)  # Annualized
            else:
                self.recent_sharpe = 0.0
        
        # Update consecutive wins
        if is_win:
            self.consecutive_wins += 1
        else:
            self.consecutive_wins = 0
    
    def _fixed_fractional(self, risk_amount: float, risk_per_unit: float) -> float:
        """Fixed fractional position sizing"""
        return risk_amount / risk_per_unit
    
    def _volatility_scaled(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss: float,
        data: Optional[pd.DataFrame],
        risk_amount: float,
        confidence_multiplier: float = 1.0
    ) -> float:
        """
        Volatility-scaled position sizing
        
        Formula: Position Size = Risk Capital / (Current Volatility × Multiplier)
        
        Args:
            account_balance: Account balance
            entry_price: Entry price
            stop_loss: Stop loss price
            data: Market data for volatility calculation
            risk_amount: Risk amount per trade
            confidence_multiplier: Multiplier adjusted by DRL agent confidence (default: 1.0)
        
        Returns:
            Position quantity
        """
        if data is None or len(data) < 14:
            # Fallback to fixed fractional
            risk_per_unit = abs(entry_price - stop_loss)
            return risk_amount / risk_per_unit
        
        # Calculate current volatility (ATR normalized by price)
        atr = self.indicators.atr(data)
        atr_value = atr.iloc[-1] if not atr.empty else abs(entry_price - stop_loss)
        current_volatility = atr_value / entry_price
        
        # Ensure minimum volatility to avoid division by zero
        current_volatility = max(current_volatility, 0.001)
        
        # Calculate position size using volatility scaling formula
        # Position Size = Risk Capital / (Current Volatility × Multiplier)
        position_size = risk_amount / (current_volatility * confidence_multiplier)
        
        # Convert to quantity
        quantity = position_size / entry_price
        
        # Apply stop loss risk constraint
        risk_per_unit = abs(entry_price - stop_loss)
        max_units = risk_amount / risk_per_unit
        
        # Return minimum of volatility-scaled size and stop-loss constrained size
        return min(quantity, max_units)
    
    def calculate_volatility_scaled_size(
        self,
        account_balance: float,
        volatility: float,
        confidence_multiplier: float,
        entry_price: float,
        stop_loss: float
    ) -> Dict:
        """
        Calculate volatility-scaled position size (aggressive mode)
        
        Args:
            account_balance: Account balance
            volatility: Current volatility (normalized, e.g., ATR/price)
            confidence_multiplier: DRL agent confidence multiplier
            entry_price: Entry price
            stop_loss: Stop loss price
        
        Returns:
            Position size information
        """
        risk_capital = account_balance * (self.position_size_percent / 100)
        
        # Ensure minimum volatility
        volatility = max(volatility, 0.001)
        
        # Calculate position size: Risk Capital / (Volatility × Multiplier)
        position_size = risk_capital / (volatility * confidence_multiplier)
        
        # Convert to quantity
        quantity = position_size / entry_price
        
        # Apply stop loss risk constraint
        risk_per_unit = abs(entry_price - stop_loss)
        max_units = risk_capital / risk_per_unit
        
        # Final quantity is minimum of volatility-scaled and stop-loss constrained
        quantity = min(quantity, max_units)
        
        # Apply maximum position size limit
        max_position_value = account_balance * (self.max_position_size_percent / 100)
        max_quantity = max_position_value / entry_price
        quantity = min(quantity, max_quantity)
        
        position_value = quantity * entry_price
        
        return {
            'quantity': quantity,
            'value': position_value,
            'risk_amount': risk_capital,
            'volatility': volatility,
            'confidence_multiplier': confidence_multiplier,
            'method': 'volatility_scaled'
        }
    
    def _kelly_criterion(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss: float,
        data: Optional[pd.DataFrame]
    ) -> float:
        """Kelly Criterion position sizing (simplified)"""
        # This is a simplified version
        # Full Kelly requires win rate and average win/loss
        
        # For now, use conservative fractional Kelly (25%)
        risk_per_unit = abs(entry_price - stop_loss)
        risk_amount = account_balance * (self.position_size_percent / 100)
        
        # Kelly fraction (simplified - assumes 50% win rate)
        kelly_fraction = 0.25  # Conservative: 25% of full Kelly
        
        quantity = (account_balance * kelly_fraction) / entry_price
        
        return quantity

