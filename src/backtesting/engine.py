"""Backtesting engine"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from ..strategies.base_strategy import BaseStrategy, Signal
from ..risk.position_sizer import PositionSizer
from ..risk.stop_loss import StopLossManager
from ..indicators.technical import TechnicalIndicators
from .metrics import BacktestMetrics


class BacktestEngine:
    """Backtesting engine for strategy validation"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize backtest engine
        
        Config parameters:
            initial_capital: Starting capital (default: 10000)
            commission_rate: Commission rate per trade (default: 0.001)
            slippage_bps: Slippage in basis points (default: 5)
        """
        self.config = config or {}
        self.initial_capital = self.config.get('initial_capital', 10000.0)
        self.commission_rate = self.config.get('commission_rate', 0.001)
        self.slippage_bps = self.config.get('slippage_bps', 5)
        
        self.position_sizer = PositionSizer()
        self.stop_loss_manager = StopLossManager()
        self.indicators = TechnicalIndicators()
        self.metrics = BacktestMetrics()
    
    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy: BaseStrategy,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """
        Run backtest on historical data
        
        Args:
            data: Historical OHLCV data
            strategy: Trading strategy to test
            start_date: Start date for backtest
            end_date: End date for backtest
        
        Returns:
            Dictionary with backtest results
        """
        # Filter data by date range
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        if data.empty:
            return {'error': 'No data in specified date range'}
        
        # Add indicators
        data = self.indicators.add_all_indicators(data)
        
        # Initialize state
        capital = self.initial_capital
        equity_curve = [capital]
        trades = []
        open_position = None
        
        # Iterate through data
        for i in range(len(data)):
            current_data = data.iloc[:i+1]
            current_bar = data.iloc[i]
            current_price = current_bar['close']
            timestamp = current_data.index[-1]
            
            # Check if we need to exit position
            if open_position:
                # Check stop loss
                stop_loss = open_position['stop_loss']
                if self.stop_loss_manager.check_stop_loss(
                    current_price, stop_loss, open_position['side']
                ):
                    # Exit at stop loss
                    trade = self._close_position(
                        open_position, current_price, timestamp, 'stop_loss'
                    )
                    trades.append(trade)
                    capital = trade['exit_equity']
                    equity_curve.append(capital)
                    open_position = None
                    continue
                
                # Check take profit
                take_profit = open_position.get('take_profit')
                if take_profit:
                    if (open_position['side'] == 'buy' and current_price >= take_profit) or \
                       (open_position['side'] == 'sell' and current_price <= take_profit):
                        trade = self._close_position(
                            open_position, current_price, timestamp, 'take_profit'
                        )
                        trades.append(trade)
                        capital = trade['exit_equity']
                        equity_curve.append(capital)
                        open_position = None
                        continue
                
                # Check strategy exit signal
                if strategy.should_exit(current_data, open_position):
                    trade = self._close_position(
                        open_position, current_price, timestamp, 'strategy_exit'
                    )
                    trades.append(trade)
                    capital = trade['exit_equity']
                    equity_curve.append(capital)
                    open_position = None
                    continue
            
            # Generate signal if no open position
            if not open_position:
                signal_data = strategy.generate_signal(current_data)
                
                if signal_data['signal'] != Signal.HOLD:
                    # Calculate position size
                    entry_price = signal_data['entry_price']
                    stop_loss = signal_data.get('stop_loss', entry_price * 0.98)
                    
                    position_info = self.position_sizer.calculate_position_size(
                        capital, entry_price, stop_loss, current_data, symbol='TEST'
                    )
                    
                    if position_info['quantity'] > 0:
                        # Open position
                        side = 'buy' if signal_data['signal'] == Signal.BUY else 'sell'
                        
                        # Apply slippage
                        if side == 'buy':
                            execution_price = entry_price * (1 + self.slippage_bps / 10000)
                        else:
                            execution_price = entry_price * (1 - self.slippage_bps / 10000)
                        
                        # Calculate commission
                        position_value = position_info['quantity'] * execution_price
                        commission = position_value * self.commission_rate
                        
                        open_position = {
                            'entry_time': timestamp,
                            'symbol': 'TEST',  # Placeholder
                            'side': side,
                            'quantity': position_info['quantity'],
                            'entry_price': execution_price,
                            'stop_loss': stop_loss,
                            'take_profit': signal_data.get('take_profit'),
                            'commission': commission,
                            'entry_equity': capital - commission
                        }
            
            # Update equity curve
            if open_position:
                # Calculate unrealized P&L
                if open_position['side'] == 'buy':
                    unrealized_pnl = (current_price - open_position['entry_price']) * open_position['quantity']
                else:
                    unrealized_pnl = (open_position['entry_price'] - current_price) * open_position['quantity']
                
                current_equity = open_position['entry_equity'] + unrealized_pnl
            else:
                current_equity = capital
            
            equity_curve.append(current_equity)
        
        # Close any remaining position
        if open_position:
            final_price = data['close'].iloc[-1]
            final_timestamp = data.index[-1]
            trade = self._close_position(
                open_position, final_price, final_timestamp, 'end_of_data'
            )
            trades.append(trade)
            equity_curve[-1] = trade['exit_equity']
        
        # Calculate metrics
        equity_series = pd.Series(equity_curve, index=[data.index[0]] + list(data.index))
        metrics = self.metrics.calculate_all_metrics(equity_series, trades, self.initial_capital)
        
        return {
            'equity_curve': equity_series,
            'trades': trades,
            'metrics': metrics,
            'initial_capital': self.initial_capital,
            'final_capital': equity_series.iloc[-1]
        }
    
    def _close_position(
        self,
        position: Dict,
        exit_price: float,
        timestamp: datetime,
        exit_reason: str
    ) -> Dict:
        """Close a position and calculate P&L"""
        # Apply slippage
        if position['side'] == 'buy':
            execution_price = exit_price * (1 - self.slippage_bps / 10000)
        else:
            execution_price = exit_price * (1 + self.slippage_bps / 10000)
        
        # Calculate P&L
        if position['side'] == 'buy':
            pnl = (execution_price - position['entry_price']) * position['quantity']
        else:
            pnl = (position['entry_price'] - execution_price) * position['quantity']
        
        # Calculate commission
        exit_value = position['quantity'] * execution_price
        exit_commission = exit_value * self.commission_rate
        
        # Net P&L
        net_pnl = pnl - position['commission'] - exit_commission
        
        # Calculate exit equity
        exit_equity = position['entry_equity'] + net_pnl
        
        return {
            'entry_time': position['entry_time'],
            'exit_time': timestamp,
            'symbol': position['symbol'],
            'side': position['side'],
            'quantity': position['quantity'],
            'entry_price': position['entry_price'],
            'exit_price': execution_price,
            'pnl': net_pnl,
            'pnl_percent': (net_pnl / position['entry_equity']) * 100,
            'commission': position['commission'] + exit_commission,
            'exit_reason': exit_reason,
            'entry_equity': position['entry_equity'],
            'exit_equity': exit_equity
        }

