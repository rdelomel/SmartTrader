"""Comprehensive trading performance reporting similar to MyFxBook"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from ..data.data_storage import DataStorage


def _is_broker_sync_trade(trade: Dict) -> bool:
    """True for broker-imported fills (strategy contains _sync)."""
    return '_sync' in str(trade.get('strategy') or '').lower()


def _agent_trades_only(trades: List[Dict]) -> List[Dict]:
    """Exclude broker-sync imports from agent performance math."""
    return [t for t in trades if not _is_broker_sync_trade(t)]


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
        # KPIs / equity curve use agent trades only; leaderboard keeps sync for visibility
        agent_closed = _agent_trades_only(closed_trades)
        
        # Calculate time periods
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = today_start - timedelta(days=now.weekday())
        month_start = today_start.replace(day=1)
        year_start = today_start.replace(month=1, day=1)
        
        # Filter trades by period
        def filter_by_period(trades, start_date):
            return [t for t in trades if self._parse_time(t.get('exit_time') or t.get('entry_time')) >= start_date]
        
        today_trades = filter_by_period(agent_closed, today_start)
        week_trades = filter_by_period(agent_closed, week_start)
        month_trades = filter_by_period(agent_closed, month_start)
        year_trades = filter_by_period(agent_closed, year_start)
        
        # Calculate general account info
        general_info = self._calculate_general_info(agent_closed, open_trades)
        
        # Calculate period statistics
        periods = {
            'today': self._calculate_period_stats(today_trades, agent_closed),
            'week': self._calculate_period_stats(week_trades, agent_closed),
            'month': self._calculate_period_stats(month_trades, agent_closed),
            'year': self._calculate_period_stats(year_trades, agent_closed),
            'all': self._calculate_period_stats(agent_closed, agent_closed)
        }
        
        # Calculate advanced statistics
        advanced_stats = self._calculate_advanced_stats(agent_closed)
        
        # Calculate equity curve
        equity_curve = self._calculate_equity_curve(agent_closed, open_trades)
        
        # Calculate monthly analytics
        monthly_analytics = self._calculate_monthly_analytics(agent_closed)
        
        # Leaderboard keeps all strategies (including *_sync) for transparency
        strategy_leaderboard = self._calculate_strategy_leaderboard(closed_trades)

        return {
            'general': general_info,
            'periods': periods,
            'advanced': advanced_stats,
            'equity_curve': equity_curve,
            'monthly_analytics': monthly_analytics,
            'strategy_leaderboard': strategy_leaderboard,
            'agent_trade_count': len(agent_closed),
            'sync_trade_count': len(closed_trades) - len(agent_closed),
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
    
    def _calculate_strategy_leaderboard(self, closed_trades: List[Dict]) -> List[Dict]:
        """
        Break down performance by originating strategy so the dashboard can show
        which strategies are actually profitable (or not) in paper trading.

        Uses the same win-rate / profit-factor / expectancy formulas as
        _calculate_general_info, just grouped by trade['strategy'].
        """
        # Exclude $0-P&L unresolved broker-sync imports (same rule as general info)
        resolved_trades = [t for t in closed_trades if abs(t.get('pnl', 0) or 0) >= 0.001]

        by_strategy: Dict[str, List[Dict]] = {}
        for t in resolved_trades:
            name = t.get('strategy') or 'unknown'
            by_strategy.setdefault(name, []).append(t)

        leaderboard = []
        for name, trades in by_strategy.items():
            winning = [t for t in trades if t.get('pnl', 0) > 0]
            losing = [t for t in trades if t.get('pnl', 0) < 0]
            total_profit = sum(t.get('pnl', 0) for t in winning)
            total_loss = abs(sum(t.get('pnl', 0) for t in losing))
            pnls = [t.get('pnl', 0) for t in trades]
            net_profit = total_profit - total_loss
            win_rate = (len(winning) / len(trades) * 100) if trades else 0.0
            profit_factor = (total_profit / total_loss) if total_loss > 0 else (float('inf') if total_profit > 0 else 0.0)
            expectancy = (sum(pnls) / len(pnls)) if pnls else 0.0
            avg_win = (total_profit / len(winning)) if winning else 0.0
            avg_loss = (total_loss / len(losing)) if losing else 0.0

            leaderboard.append({
                'strategy': name,
                'total_trades': len(trades),
                'winning_trades': len(winning),
                'losing_trades': len(losing),
                'win_rate': round(win_rate, 2),
                'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 999.0,
                'expectancy': round(expectancy, 2),
                'net_profit': round(net_profit, 2),
                'total_profit': round(total_profit, 2),
                'total_loss': round(total_loss, 2),
                'average_win': round(avg_win, 2),
                'average_loss': round(avg_loss, 2),
            })

        # Rank by net profit descending — most profitable strategies first
        leaderboard.sort(key=lambda x: x['net_profit'], reverse=True)
        return leaderboard

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
        
        # Exclude $0-P&L unresolved broker-sync imports from statistics
        resolved_trades = [t for t in closed_trades if abs(t.get('pnl', 0) or 0) >= 0.001]
        winning_trades = [t for t in resolved_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in resolved_trades if t.get('pnl', 0) < 0]  # True losses only (P&L < 0)
        
        total_profit = sum(t.get('pnl', 0) for t in winning_trades)
        total_loss = abs(sum(t.get('pnl', 0) for t in losing_trades))
        
        longs = [t for t in resolved_trades if t.get('side', '').lower() == 'buy']
        shorts = [t for t in resolved_trades if t.get('side', '').lower() == 'sell']
        
        longs_won = len([t for t in longs if t.get('pnl', 0) > 0])
        shorts_won = len([t for t in shorts if t.get('pnl', 0) > 0])
        
        return {
            'total_trades': len(resolved_trades),   # Only resolved trades; use 'total_imported' for $0-P&L count
            'win_rate': (len(winning_trades) / len(resolved_trades) * 100) if resolved_trades else 0.0,
            'loss_rate': (len(losing_trades) / len(resolved_trades) * 100) if resolved_trades else 0.0,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'net_profit': total_profit - total_loss,
            'profit_factor': total_profit / total_loss if total_loss > 0 else 0.0,
            'average_win': total_profit / len(winning_trades) if winning_trades else 0.0,
            'average_loss': total_loss / len(losing_trades) if losing_trades else 0.0,
            'largest_win': max([t.get('pnl', 0) for t in resolved_trades], default=0.0),
            'largest_loss': min([t.get('pnl', 0) for t in resolved_trades], default=0.0),
            'longs_won': longs_won,
            'total_imported': len(closed_trades) - len(resolved_trades),
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
        
        # Exclude $0-P&L unresolved imports from period stats
        eff = [t for t in period_trades if abs(t.get('pnl', 0) or 0) >= 0.001]
        winning = [t for t in eff if t.get('pnl', 0) > 0]
        losing  = [t for t in eff if t.get('pnl', 0) < 0]
        
        profit = sum(t.get('pnl', 0) for t in eff)
        
        # Calculate gain % vs initial equity
        gain_pct = (profit / self.initial_equity * 100) if self.initial_equity else 0.0
        
        return {
            'gain': profit,
            'gain_pct': gain_pct,
            'profit': profit,
            'trades': len(eff),
            'win_rate': (len(winning) / len(eff) * 100) if eff else 0.0,
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
        
        # Exclude $0-P&L unresolved imports from advanced stats
        resolved = [t for t in closed_trades if abs(t.get('pnl', 0) or 0) >= 0.001]
        pnls = [t.get('pnl', 0) for t in resolved]
        
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
        for trade in sorted(resolved, key=lambda x: self._parse_time(x.get('exit_time') or x.get('entry_time'))):
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
        # Calculate running equity by summing all trades before each month to get accurate starting equity
        for month_key in sorted(monthly_data.keys()):
            data = monthly_data[month_key]
            
            # Find starting equity for this month
            month_start_date = datetime.strptime(month_key, '%Y-%m')
            # Exclude $0-P&L unresolved imports from equity base
            prior_profit = sum(t.get('pnl', 0) for t in closed_trades
                               if abs(t.get('pnl', 0) or 0) >= 0.001
                               and self._parse_time(t.get('exit_time') or t.get('entry_time')) < month_start_date)
            starting_equity = self.initial_equity + prior_profit
            
            gain_pct = (data['profit'] / starting_equity * 100) if starting_equity > 0 else 0.0
            
            monthly_list.append({
                'month': data['month'],
                'year': month_key[:4],  # Add year for grouping
                'gain_pct': gain_pct,
                'profit': data['profit'],
                'trades': len(data['trades']),
                'win_rate': (len([t for t in data['trades'] if t.get('pnl', 0) > 0]) / len(data['trades']) * 100) if data['trades'] else 0.0
            })
        
        return monthly_list

