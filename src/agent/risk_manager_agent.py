"""Enhanced risk manager agent with liquidity checks"""

import pandas as pd
from typing import Dict, Optional
from datetime import datetime
from .base_agent import BaseAgent
from ..risk.position_sizer import PositionSizer
from ..risk.drawdown_manager import DrawdownManager
from ..risk.stop_loss import StopLossManager
from ..risk.circuit_breaker import CircuitBreaker
from ..risk.portfolio_risk_manager import PortfolioRiskManager


class RiskManagerAgent(BaseAgent):
    """Risk management agent with veto power"""
    
    def __init__(self, config: Optional[Dict] = None,
                 position_sizer: Optional[PositionSizer] = None,
                 drawdown_manager: Optional[DrawdownManager] = None,
                 stop_loss_manager: Optional[StopLossManager] = None,
                 circuit_breaker: Optional[CircuitBreaker] = None,
                 portfolio_risk_manager: Optional[PortfolioRiskManager] = None):
        """
        Initialize risk manager agent
        
        Args:
            config: Risk manager configuration
            position_sizer: PositionSizer instance
            drawdown_manager: DrawdownManager instance
            stop_loss_manager: StopLossManager instance
            circuit_breaker: CircuitBreaker instance
            portfolio_risk_manager: PortfolioRiskManager instance
        """
        super().__init__("RiskManagerAgent", config)
        self.position_sizer = position_sizer
        self.drawdown_manager = drawdown_manager
        self.stop_loss_manager = stop_loss_manager
        self.circuit_breaker = circuit_breaker or CircuitBreaker(self.config.get('circuit_breaker', {}))
        self.portfolio_risk_manager = portfolio_risk_manager or PortfolioRiskManager(
            self.config.get('portfolio_risk', {})
        )
        
        # Risk thresholds
        self.max_position_size_percent = self.config.get('max_position_size_percent', 5.0)
        self.max_drawdown_percent = self.config.get('max_drawdown_percent', 10.0)
        self.min_liquidity_ratio = self.config.get('min_liquidity_ratio', 0.1)  # 10% of order size
        self.max_slippage_bps = self.config.get('max_slippage_bps', 10)  # 10 basis points

        # Performance state defaults (used by dynamic position sizing)
        # Prevents attribute errors before first performance update cycle.
        self.recent_sharpe = 1.0
        self.recent_win_rate = 50.0
    
    def analyze(self, symbol: str, data: pd.DataFrame, account_balance: float, 
                proposed_signal=None, proposed_position_size: float = 0.0) -> Dict:
        """
        Perform risk analysis
        
        Args:
            symbol: Trading symbol
            data: Market data
            account_balance: Current account balance
            proposed_signal: Proposed trading signal (optional)
            proposed_position_size: Proposed position size (optional)
        
        Returns:
            Dictionary with risk analysis results
        """
        if not self.enabled:
            return {
                'risk_score': 0.0,
                'approved': True,
                'reason': 'Risk manager disabled'
            }
        
        # Perform veto check
        veto_result = self.check_veto(symbol, data, account_balance)
        
        if veto_result.get('veto', False):
            return {
                'risk_score': 1.0,  # High risk
                'approved': False,
                'reason': veto_result.get('reason', 'Risk check failed'),
                'veto': True
            }
        
        # Calculate risk score based on various factors
        risk_score = 0.0
        
        # Drawdown risk
        if self.drawdown_manager:
            drawdown = self.drawdown_manager.calculate_drawdown()
            if drawdown > self.max_drawdown_percent * 0.5:  # 50% of max
                risk_score += 0.3
        
        # Volatility risk
        volatility_check = self._check_volatility(data)
        if volatility_check.get('reduce_position', False):
            risk_score += 0.2
        
        # Liquidity risk
        liquidity_check = self._check_liquidity(symbol, data)
        if not liquidity_check.get('sufficient', True):
            risk_score += 0.3
        else:
            volume_ratio = liquidity_check.get('volume_ratio', 1.0)
            if volume_ratio < 0.5:
                risk_score += 0.1
        
        # Position size risk
        if proposed_position_size > 0:
            position_pct = (proposed_position_size / account_balance) * 100
            if position_pct > self.max_position_size_percent:
                risk_score += 0.4
        
        risk_score = min(risk_score, 1.0)
        approved = risk_score < 0.7  # Approve if risk score < 70%
        
        self._update_timestamp()
        
        return {
            'risk_score': risk_score,
            'approved': approved,
            'reason': 'Risk analysis complete',
            'drawdown': self.drawdown_manager.calculate_drawdown() if self.drawdown_manager else 0.0,
            'liquidity': liquidity_check,
            'volatility': volatility_check
        }
    
    def check_veto(self, symbol: str, data: pd.DataFrame, account_balance: float) -> Dict:
        """
        Check if trade should be vetoed
        
        Args:
            symbol: Trading symbol
            data: Market data
            account_balance: Current account balance
        
        Returns:
            Dictionary with veto decision and reason
        """
        veto_reasons = []
        
        # Check 0: Circuit breaker (highest priority)
        if self.circuit_breaker:
            # Update circuit breaker with current equity
            self.circuit_breaker.update_equity(account_balance)
            circuit_check = self.circuit_breaker.check_limits(account_balance)
            if circuit_check.get('triggered', False):
                return {
                    'veto': True,
                    'reason': f"Circuit Breaker: {circuit_check.get('reason', 'Limit exceeded')}",
                    'circuit_breaker': circuit_check
                }
        
        # Check 1: Kill switch (drawdown)
        if self.drawdown_manager:
            if not self.drawdown_manager.is_trading_allowed():
                return {
                    'veto': True,
                    'reason': f'Kill switch active: Drawdown {self.drawdown_manager.calculate_drawdown():.2f}%',
                    'drawdown': self.drawdown_manager.calculate_drawdown()
                }
        
        # Check 2: Liquidity (if broker supports it)
        liquidity_check = self._check_liquidity(symbol, data)
        if not liquidity_check.get('sufficient', True):
            return {
                'veto': True,
                'reason': f'Insufficient liquidity: {liquidity_check.get("reason", "Unknown")}',
                'liquidity_info': liquidity_check
            }
        
        # Check 3: Volatility-based position reduction
        if len(data) >= 14:
            volatility_check = self._check_volatility(data)
            if volatility_check.get('reduce_position', False):
                # Don't veto, but flag for position reduction
                pass
        
        return {
            'veto': False,
            'reason': 'All risk checks passed'
        }
    
    def calculate_position_size(self, account_balance: float, entry_price: float,
                               stop_loss: float, data: pd.DataFrame,
                               regime: str = 'Neutral', 
                               performance_tracker: Optional[object] = None,
                               symbol: str = "") -> Dict:
        """
        Calculate position size with regime-based adjustments and dynamic scaling
        
        Args:
            account_balance: Account balance
            entry_price: Entry price
            stop_loss: Stop loss price
            data: Market data
            regime: Current market regime
            performance_tracker: Optional performance tracker for dynamic scaling
        
        Returns:
            Position size information
        """
        if not self.position_sizer:
            return {
                'quantity': 0.0,
                'value': 0.0,
                'reason': 'Position sizer not available'
            }
        
        # Get performance metrics for dynamic scaling
        recent_sharpe = None
        consecutive_wins = None
        if performance_tracker:
            try:
                metrics = performance_tracker.calculate_metrics()
                recent_sharpe = metrics.get('sharpe_ratio', 0.0)
                # Get consecutive wins from position sizer if available
                consecutive_wins = getattr(self.position_sizer, 'consecutive_wins', None)
            except:
                pass
        
        # Base position size calculation with dynamic scaling
        position_info = self.position_sizer.calculate_position_size(
            account_balance, entry_price, stop_loss, data,
            recent_sharpe=recent_sharpe,
            consecutive_wins=consecutive_wins,
            symbol=symbol
        )
        
        # Regime-based adjustment
        regime_adjustment = self._get_regime_adjustment(regime)
        adjusted_quantity = position_info['quantity'] * regime_adjustment
        adjusted_value = adjusted_quantity * entry_price
        
        # Performance-based risk adjustment
        # Increase position limits when Sharpe ratio > 1.5
        # Reduce position limits when Sharpe ratio < 0.5
        performance_multiplier = 1.0
        if performance_tracker:
            try:
                metrics = performance_tracker.calculate_metrics()
                sharpe = metrics.get('sharpe_ratio', 0.0)
                win_rate = metrics.get('win_rate', 0.0)
                
                # Adjust based on Sharpe ratio
                if sharpe > 1.5:
                    performance_multiplier = min(1.3, 1.0 + (sharpe - 1.5) * 0.2)  # Up to 30% increase
                elif sharpe < 0.5:
                    performance_multiplier = max(0.7, 1.0 - (0.5 - sharpe) * 0.4)  # Up to 30% decrease
                
                # Additional adjustment based on win rate
                if win_rate > 60:
                    performance_multiplier = min(1.2, performance_multiplier * 1.1)  # 10% bonus
                elif win_rate < 40:
                    performance_multiplier = max(0.8, performance_multiplier * 0.9)  # 10% reduction
                
                self.recent_sharpe = sharpe
                self.recent_win_rate = win_rate
            except:
                pass
        
        # Apply performance multiplier
        adjusted_quantity = adjusted_quantity * performance_multiplier
        adjusted_value = adjusted_quantity * entry_price
        
        # Dynamic max position size based on recent performance
        dynamic_max_position_percent = self.max_position_size_percent
        if self.recent_sharpe > 1.5:
            dynamic_max_position_percent = min(7.0, self.max_position_size_percent * 1.2)  # Up to 20% increase
        elif self.recent_sharpe < 0.5:
            dynamic_max_position_percent = max(3.0, self.max_position_size_percent * 0.8)  # Up to 20% decrease
        
        # Ensure we don't exceed dynamic max position size
        max_value = account_balance * (dynamic_max_position_percent / 100)
        if adjusted_value > max_value:
            adjusted_quantity = max_value / entry_price
            adjusted_value = max_value
        
        return {
            'quantity': adjusted_quantity,
            'value': adjusted_value,
            'risk_amount': position_info.get('risk_amount', 0.0),
            'regime_adjustment': regime_adjustment,
            'performance_multiplier': performance_multiplier,
            'dynamic_max_position_percent': dynamic_max_position_percent,
            'base_quantity': position_info['quantity'],
            'method': position_info.get('method', 'fixed_fractional')
        }
    
    def _check_liquidity(self, symbol: str, data: pd.DataFrame) -> Dict:
        """
        Check market liquidity
        
        Args:
            symbol: Trading symbol
            data: Market data
        
        Returns:
            Liquidity check result
        """
        # Major cryptos and stocks are always considered liquid
        major_cryptos = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'BTCUSD', 'ETHUSD', 'SOLUSD']
        is_major_crypto = any(major in symbol.upper() for major in major_cryptos)
        
        if is_major_crypto:
            return {
                'sufficient': True,
                'volume_ratio': 1.0,
                'reason': f'Major crypto {symbol} - always liquid'
            }
        
        # Simplified liquidity check based on volume
        # In production, would check order book depth via broker API
        
        if len(data) < 20:
            return {
                'sufficient': True,
                'reason': 'Insufficient data for liquidity check'
            }
        
        # Check if volume column exists and has data
        if 'volume' not in data.columns:
            print(f"RiskManager: Warning - No volume column for {symbol}, assuming liquid")
            return {
                'sufficient': True,
                'volume_ratio': 1.0,
                'reason': 'No volume data - assuming liquid'
            }
        
        # Get volume data, handling NaN values
        volume_data = data['volume'].dropna()
        
        if len(volume_data) < 10:
            print(f"RiskManager: Warning - Insufficient volume data for {symbol} ({len(volume_data)} valid points), assuming liquid")
            return {
                'sufficient': True,
                'volume_ratio': 1.0,
                'reason': 'Insufficient valid volume data - assuming liquid'
            }
        
        # Check recent volume (last 20 valid points)
        recent_volume_data = volume_data.tail(20)
        recent_volume = recent_volume_data.mean()
        avg_volume = volume_data.mean()
        
        # Debug logging
        print(f"RiskManager: Liquidity check for {symbol}:")
        print(f"  Recent volume (last 20): {recent_volume:.2f}")
        print(f"  Average volume (all): {avg_volume:.2f}")
        print(f"  Valid volume points: {len(volume_data)}/{len(data)}")
        
        # If average volume is 0 or very small, assume liquid (data issue, not liquidity issue)
        if avg_volume <= 0 or pd.isna(avg_volume):
            print(f"RiskManager: Average volume is {avg_volume}, assuming liquid (data issue)")
            return {
                'sufficient': True,
                'volume_ratio': 1.0,
                'reason': 'Volume data issue - assuming liquid'
            }
        
        # If recent volume is 0 or NaN, check if it's a data issue
        if recent_volume <= 0 or pd.isna(recent_volume):
            print(f"RiskManager: Recent volume is {recent_volume}, checking if data issue...")
            # Check if ANY recent volume exists
            if recent_volume_data.sum() > 0:
                # Some volume exists, recalculate mean excluding zeros
                recent_volume = recent_volume_data[recent_volume_data > 0].mean()
                if pd.isna(recent_volume) or recent_volume <= 0:
                    print(f"RiskManager: No valid recent volume, but historical volume exists - assuming temporary data gap")
                    return {
                        'sufficient': True,
                        'volume_ratio': 0.5,  # Conservative but not blocking
                        'reason': 'Temporary volume data gap - assuming liquid'
                    }
            else:
                # All recent volumes are 0 - likely data issue for major assets
                print(f"RiskManager: All recent volumes are 0 - likely data issue, not liquidity issue")
                return {
                    'sufficient': True,
                    'volume_ratio': 0.5,
                    'reason': 'Recent volume data missing - assuming liquid'
                }
        
        # Calculate volume ratio
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Handle NaN
        if pd.isna(volume_ratio):
            print(f"RiskManager: Volume ratio is NaN, assuming liquid")
            return {
                'sufficient': True,
                'volume_ratio': 1.0,
                'reason': 'Volume ratio calculation error - assuming liquid'
            }
        
        print(f"RiskManager: Volume ratio: {volume_ratio:.4f} (threshold: {self.min_liquidity_ratio:.4f})")
        
        if volume_ratio < self.min_liquidity_ratio:
            print(f"RiskManager: ⚠️  Low volume ratio detected: {volume_ratio:.4f} < {self.min_liquidity_ratio:.4f}")
            return {
                'sufficient': False,
                'reason': f'Low volume ratio: {volume_ratio:.4f}',
                'volume_ratio': volume_ratio
            }
        
        print(f"RiskManager: ✅ Liquidity check passed")
        return {
            'sufficient': True,
            'volume_ratio': volume_ratio,
            'reason': 'Liquidity check passed'
        }
    
    def _check_volatility(self, data: pd.DataFrame) -> Dict:
        """Check volatility conditions"""
        if len(data) < 14:
            return {'reduce_position': False}
        
        returns = data['close'].pct_change()
        current_volatility = returns.tail(14).std()
        avg_volatility = returns.std()
        
        # High volatility flag
        if current_volatility > avg_volatility * 2.0:
            return {
                'reduce_position': True,
                'volatility_ratio': current_volatility / avg_volatility if avg_volatility > 0 else 1.0
            }
        
        return {'reduce_position': False}
    
    def _get_regime_adjustment(self, regime: str) -> float:
        """
        Get position size adjustment factor based on regime
        
        Returns:
            Adjustment multiplier (1.0 = no change, <1.0 = reduce, >1.0 = increase)
        """
        regime_adjustments = {
            'Bullish Trend': 1.2,  # More aggressive
            'Bearish Trend': 0.8,  # More conservative
            'Consolidation': 1.0,  # Normal
            'High Volatility': 0.6,  # Much more conservative
            'Neutral': 1.0  # Normal
        }
        
        return regime_adjustments.get(regime, 1.0)
    
    def get_status(self) -> Dict:
        """Get risk manager status"""
        status = super().get_status()
        
        if self.drawdown_manager:
            status['drawdown'] = self.drawdown_manager.get_status()
        
        return status
