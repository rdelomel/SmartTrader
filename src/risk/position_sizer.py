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
        # Accept either 'method' or 'position_sizing_method' from trading_config.yaml
        raw_method = self.config.get('method') or self.config.get('position_sizing_method', 'fixed_fractional')
        method_aliases = {
            'fixed': 'fixed_fractional',
            'fixed_fractional': 'fixed_fractional',
            'kelly': 'kelly',
            'volatility_adjusted': 'volatility_scaled',
            'volatility_scaled': 'volatility_scaled',
        }
        self.method = method_aliases.get(str(raw_method).lower(), 'fixed_fractional')
        self.risk_per_trade = self.config.get('risk_per_trade', None)
        self.indicators = TechnicalIndicators()

        # Expectancy clamp: fall back to fixed % risk when recent PF is weak/unknown
        clamp_cfg = self.config.get('expectancy_clamp', {}) or {}
        self.expectancy_clamp_enabled = clamp_cfg.get('enabled', True)
        self.expectancy_lookback = int(clamp_cfg.get('lookback_trades', 30))
        self.min_pf_for_kelly = float(clamp_cfg.get('min_profit_factor_for_kelly', 1.3))
        self.fallback_risk_percent = float(clamp_cfg.get('fallback_risk_percent', 0.75))
        self._recent_closed_pnls = []  # filled via update_recent_trade_pnls()
        
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

    def update_recent_trade_pnls(self, pnls: list) -> None:
        """Update recent closed-trade PnLs used by expectancy clamp."""
        if not pnls:
            self._recent_closed_pnls = []
            return
        self._recent_closed_pnls = [float(p) for p in pnls[-self.expectancy_lookback:]]

    def _recent_profit_factor(self) -> Optional[float]:
        """Profit factor over recent closed trades, or None if insufficient data."""
        if len(self._recent_closed_pnls) < 5:
            return None
        wins = sum(p for p in self._recent_closed_pnls if p > 0)
        losses = abs(sum(p for p in self._recent_closed_pnls if p < 0))
        if losses == 0:
            return float('inf') if wins > 0 else None
        return wins / losses

    def _should_use_kelly(self) -> bool:
        if self.method != 'kelly':
            return False
        if not self.expectancy_clamp_enabled:
            return True
        pf = self._recent_profit_factor()
        # Unknown or weak expectancy -> fixed conservative risk instead of Kelly
        if pf is None:
            return False
        return pf >= self.min_pf_for_kelly

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
        sizing_method = self.method
        if self.method == 'fixed_fractional':
            quantity = self._fixed_fractional(risk_amount, risk_per_unit)
        elif self.method == 'volatility_scaled':
            quantity = self._volatility_scaled(account_balance, entry_price, stop_loss, data, risk_amount)
        elif self.method == 'kelly':
            if self._should_use_kelly():
                quantity = self._kelly_criterion(account_balance, entry_price, stop_loss, data)
            else:
                # Negative/unknown expectancy: conservative fixed % of equity at risk
                sizing_method = 'fixed_fractional_expectancy_clamp'
                clamped_risk = account_balance * (self.fallback_risk_percent / 100.0)
                quantity = self._fixed_fractional(clamped_risk, risk_per_unit)
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
            'risk_percent': (risk_amount / account_balance) * 100 if account_balance else 0.0,
            'base_risk_percent': base_risk_percent,
            'method': sizing_method,
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
        data
    ) -> float:
        """
        Risk-based position sizing scaled by Kelly fraction.

        Previous implementation was BROKEN — it used:
            quantity = (account_balance * 0.25) / entry_price
        which completely ignored stop distance and sized positions at
        25% of account regardless of risk.

        Correct formula: size the position so that if the stop is hit,
        you lose exactly (position_size_percent * kelly_fraction) % of
        account balance.

            risk_amount = account * position_size_pct/100 * kelly_fraction
            quantity    = risk_amount / |entry - stop|

        Example: $300k account, 1.5% risk, 0.25 Kelly, $2k BTC stop:
            risk_amount = $300k * 0.015 * 0.25 = $1,125
            quantity    = $1,125 / $2,000 = 0.5625 BTC (~$43k)
            → capped by max_position_size_percent (4%) → $12k max
        """
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit == 0:
            # No stop distance defined — fall back to fixed fractional (safety)
            risk_amount = account_balance * (self.position_size_percent / 100)
            return risk_amount / entry_price if entry_price > 0 else 0.0

        kelly_fraction = self.config.get('kelly_fraction', 0.25)
        # Base risk: what % of account are we willing to lose if stop hits
        risk_amount = account_balance * (self.position_size_percent / 100) * kelly_fraction
        quantity = risk_amount / risk_per_unit
        return quantity

