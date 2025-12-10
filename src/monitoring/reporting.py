"""Comprehensive trading performance reporting similar to MyFxBook"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from ..data.data_storage import DataStorage


class TradingReporter:
    """Generate comprehensive trading performance reports"""
    
    def __init__(self, storage: DataStorage, initial_equity: Optional[float] = None):
        self.storage = storage
        self.initial_equity = initial_equity or 10000.0  # Default fallback
    
    def generate_report(self) -> Dict:
        """
        Generate comprehensive trading performance report
        
        Returns:
            Dictionary with all report data
        """
        # Get all closed trades
        all_trades = self.storage.get_all_trades()
        closed_trades = [t for t in all_trades if t.get('status') == 'closed']
        open_trades = [t for t in all_trades if t.get('status') == 'open']
        
        # Calculate time periods
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = today_start - timedelta(days=now.weekday())
        month_start = today_start.replace(day=1)
        year_start = today_start.replace(month=1, day=1)
        
        # Filter trades by period
        def filter_by_period(trades, start_date):
            return [t for t in trades if self._parse_time(t.get('exit_time') or t.get('entry_time')) >= start_date]
        
        today_trades = filter_by_period(closed_trades, today_start)
        week_trades = filter_by_period(closed_trades, week_start)
        month_trades = filter_by_period(closed_trades, month_start)
        year_trades = filter_by_period(closed_trades, year_start)
        
        # Calculate general account info
        general_info = self._calculate_general_info(closed_trades, open_trades)
        
        # Calculate period statistics
        periods = {
            'today': self._calculate_period_stats(today_trades, closed_trades),
            'week': self._calculate_period_stats(week_trades, closed_trades),
            'month': self._calculate_period_stats(month_trades, closed_trades),
            'year': self._calculate_period_stats(year_trades, closed_trades),
            'all': self._calculate_period_stats(closed_trades, closed_trades)
        }
        
        # Calculate advanced statistics
        advanced_stats = self._calculate_advanced_stats(closed_trades)
        
        # Calculate equity curve
        equity_curve = self._calculate_equity_curve(closed_trades, open_trades)
        
        # Calculate monthly analytics
        monthly_analytics = self._calculate_monthly_analytics(closed_trades)
        
        return {
            'general': general_info,
            'periods': periods,
            'advanced': advanced_stats,
            'equity_curve': equity_curve,
            'monthly_analytics': monthly_analytics,
            'last_updated': now.isoformat()
        }
    
    def _parse_time(self, time_str: Optional[str]) -> datetime:
        """Parse time string to datetime"""
        if not time_str:
            return datetime.min
        if isinstance(time_str, datetime):
            return time_str
        try:
            from dateutil import parser
            return parser.parse(time_str)
        except:
            return datetime.min
    
    def _calculate_general_info(self, closed_trades: List[Dict], open_trades: List[Dict]) -> Dict:
        """Calculate general account information"""
        if not closed_trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'loss_rate': 0.0,
                'total_profit': 0.0,
                'total_loss': 0.0,
                'net_profit': 0.0,
                'profit_factor': 0.0,
                'average_win': 0.0,
                'average_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'longs_won': 0,
                'longs_total': 0,
                'shorts_won': 0,
                'shorts_total': 0
            }
        
        winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in closed_trades if t.get('pnl', 0) <= 0]  # Include break-even (P&L = 0) in losses
        
        total_profit = sum(t.get('pnl', 0) for t in winning_trades)
        total_loss = abs(sum(t.get('pnl', 0) for t in losing_trades))
        
        longs = [t for t in closed_trades if t.get('side', '').lower() == 'buy']
        shorts = [t for t in closed_trades if t.get('side', '').lower() == 'sell']
        
        longs_won = len([t for t in longs if t.get('pnl', 0) > 0])
        shorts_won = len([t for t in shorts if t.get('pnl', 0) > 0])
        
        return {
            'total_trades': len(closed_trades),
            'win_rate': (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0.0,
            'loss_rate': (len(losing_trades) / len(closed_trades) * 100) if closed_trades else 0.0,  # Now includes break-even trades
            'total_profit': total_profit,
            'total_loss': total_loss,
            'net_profit': total_profit - total_loss,
            'profit_factor': total_profit / total_loss if total_loss > 0 else 0.0,
            'average_win': total_profit / len(winning_trades) if winning_trades else 0.0,
            'average_loss': total_loss / len(losing_trades) if losing_trades else 0.0,
            'largest_win': max([t.get('pnl', 0) for t in closed_trades], default=0.0),
            'largest_loss': min([t.get('pnl', 0) for t in closed_trades], default=0.0),
            'longs_won': longs_won,
            'longs_total': len(longs),
            'shorts_won': shorts_won,
            'shorts_total': len(shorts),
            'open_positions': len(open_trades)
        }
    
    def _calculate_period_stats(self, period_trades: List[Dict], all_trades: List[Dict]) -> Dict:
        """Calculate statistics for a specific period"""
        if not period_trades:
            return {
                'gain': 0.0,
                'gain_pct': 0.0,
                'profit': 0.0,
                'trades': 0,
                'win_rate': 0.0,
                'winning_trades': 0,
                'losing_trades': 0
            }
        
        winning = [t for t in period_trades if t.get('pnl', 0) > 0]
        losing = [t for t in period_trades if t.get('pnl', 0) <= 0]  # Include break-even (P&L = 0) in losses
        
        profit = sum(t.get('pnl', 0) for t in period_trades)
        
        # Calculate gain percentage (simplified - would need initial equity)
        # For now, use profit as gain
        gain_pct = 0.0  # Would need to calculate based on equity
        
        return {
            'gain': profit,
            'gain_pct': gain_pct,
            'profit': profit,
            'trades': len(period_trades),
            'win_rate': (len(winning) / len(period_trades) * 100) if period_trades else 0.0,
            'winning_trades': len(winning),
            'losing_trades': len(losing)
        }
    
    def _calculate_advanced_stats(self, closed_trades: List[Dict]) -> Dict:
        """Calculate advanced statistics"""
        if not closed_trades:
            return {
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'standard_deviation': 0.0,
                'average_trade': 0.0,
                'expectancy': 0.0
            }
        
        pnls = [t.get('pnl', 0) for t in closed_trades]
        
        # Sharpe ratio (simplified)
        if len(pnls) > 1:
            mean_return = np.mean(pnls)
            std_return = np.std(pnls)
            sharpe = (mean_return / std_return) if std_return > 0 else 0.0
        else:
            sharpe = 0.0
        
        # Sortino ratio (only downside deviation)
        downside_returns = [r for r in pnls if r < 0]
        if len(downside_returns) > 1:
            downside_std = np.std(downside_returns)
            sortino = (mean_return / downside_std) if downside_std > 0 else 0.0
        else:
            sortino = 0.0
        
        # Calculate equity curve for drawdown
        equity_curve = []
        running_equity = self.initial_equity  # Use actual initial equity
        for trade in sorted(closed_trades, key=lambda x: self._parse_time(x.get('exit_time') or x.get('entry_time'))):
            running_equity += trade.get('pnl', 0)
            # Prevent negative equity (shouldn't happen, but safety check)
            running_equity = max(0.0, running_equity)
            equity_curve.append(running_equity)
        
        max_drawdown = 0.0
        if equity_curve:
            peak = self.initial_equity  # Start with initial equity as peak
            for equity in equity_curve:
                if equity > peak:
                    peak = equity
                # Drawdown should never exceed 100% (can't lose more than 100% of peak)
                drawdown = ((peak - equity) / peak * 100) if peak > 0 else 0.0
                drawdown = min(100.0, max(0.0, drawdown))  # Clamp between 0-100%
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        
        # Expectancy
        winning = [t.get('pnl', 0) for t in closed_trades if t.get('pnl', 0) > 0]
        losing = [abs(t.get('pnl', 0)) for t in closed_trades if t.get('pnl', 0) < 0]
        
        win_rate = len(winning) / len(closed_trades) if closed_trades else 0.0
        avg_win = np.mean(winning) if winning else 0.0
        avg_loss = np.mean(losing) if losing else 0.0
        
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        return {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'standard_deviation': np.std(pnls) if len(pnls) > 1 else 0.0,
            'average_trade': np.mean(pnls),
            'expectancy': expectancy
        }
    
    def _calculate_equity_curve(self, closed_trades: List[Dict], open_trades: List[Dict]) -> List[Dict]:
        """Calculate equity curve over time"""
        if not closed_trades:
            return []
        
        # Sort trades by exit time
        sorted_trades = sorted(
            closed_trades,
            key=lambda x: self._parse_time(x.get('exit_time') or x.get('entry_time'))
        )
        
        equity_curve = []
        running_equity = self.initial_equity  # Use actual initial equity
        
        for trade in sorted_trades:
            running_equity += trade.get('pnl', 0)
            # Prevent negative equity (shouldn't happen, but safety check)
            running_equity = max(0.0, running_equity)
            exit_time = self._parse_time(trade.get('exit_time') or trade.get('entry_time'))
            equity_curve.append({
                'date': exit_time.isoformat(),
                'equity': running_equity,
                'drawdown': 0.0  # Will calculate separately
            })
        
        # Calculate drawdown
        if equity_curve:
            peak = self.initial_equity  # Start with initial equity as peak
            for point in equity_curve:
                if point['equity'] > peak:
                    peak = point['equity']
                # Drawdown should never exceed 100% (can't lose more than 100% of peak)
                drawdown = ((peak - point['equity']) / peak * 100) if peak > 0 else 0.0
                point['drawdown'] = min(100.0, max(0.0, drawdown))  # Clamp between 0-100%
        
        return equity_curve
    
    def _calculate_monthly_analytics(self, closed_trades: List[Dict]) -> List[Dict]:
        """Calculate monthly performance analytics"""
        if not closed_trades:
            return []
        
        # Group trades by month
        monthly_data = {}
        
        for trade in closed_trades:
            exit_time = self._parse_time(trade.get('exit_time') or trade.get('entry_time'))
            month_key = exit_time.strftime('%Y-%m')
            
            if month_key not in monthly_data:
                monthly_data[month_key] = {
                    'month': exit_time.strftime('%b %Y'),
                    'trades': [],
                    'profit': 0.0
                }
            
            monthly_data[month_key]['trades'].append(trade)
            monthly_data[month_key]['profit'] += trade.get('pnl', 0)
        
        # Convert to list and calculate percentages
        monthly_list = []
        for month_key in sorted(monthly_data.keys()):
            data = monthly_data[month_key]
            # Calculate gain percentage (simplified - would need monthly starting equity)
            gain_pct = 0.0  # Placeholder
            
            monthly_list.append({
                'month': data['month'],
                'gain_pct': gain_pct,
                'profit': data['profit'],
                'trades': len(data['trades']),
                'win_rate': (len([t for t in data['trades'] if t.get('pnl', 0) > 0]) / len(data['trades']) * 100) if data['trades'] else 0.0
            })
        
        return monthly_list

