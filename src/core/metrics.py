"""Performance metrics computed from daily returns and closed trades."""

from typing import Dict, Optional

import numpy as np
import pandas as pd


def to_daily_returns(equity: pd.Series) -> pd.Series:
    daily = equity.resample('1D').last().dropna()
    return daily.pct_change().dropna()


def _monthly(returns: pd.Series) -> pd.Series:
    try:
        return (1 + returns).resample('ME').prod() - 1
    except ValueError:  # pandas < 2.2
        return (1 + returns).resample('M').prod() - 1


def max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    curve = (1 + returns).cumprod()
    return float((1 - curve / curve.cummax()).max())


def summarize(daily_returns: pd.Series, trades: Optional[pd.DataFrame] = None) -> Dict:
    r = daily_returns.dropna()
    out: Dict = {
        'days': int(len(r)), 'total_return_pct': 0.0, 'cagr_pct': 0.0, 'avg_month_pct': 0.0,
        'median_month_pct': 0.0, 'worst_month_pct': 0.0, 'pct_positive_months': 0.0,
        'max_drawdown_pct': 0.0, 'sharpe': 0.0,
    }
    if len(r) >= 2:
        total = float((1 + r).prod() - 1)
        years = max((r.index[-1] - r.index[0]).days / 365.25, 1e-9)
        months = _monthly(r)
        std = float(r.std())
        out.update({
            'total_return_pct': total * 100,
            'cagr_pct': ((1 + total) ** (1 / years) - 1) * 100 if total > -1 else -100.0,
            'avg_month_pct': float(months.mean()) * 100,
            'median_month_pct': float(months.median()) * 100,
            'worst_month_pct': float(months.min()) * 100,
            'pct_positive_months': float((months > 0).mean()) * 100,
            'max_drawdown_pct': max_drawdown(r) * 100,
            'sharpe': float(r.mean() / std * np.sqrt(252)) if std > 0 else 0.0,
        })
    out.update(trade_stats(trades))
    return out


def trade_stats(trades: Optional[pd.DataFrame]) -> Dict:
    if trades is None or trades.empty:
        return {'trades': 0, 'win_rate_pct': 0.0, 'profit_factor': 0.0, 'expectancy_r': 0.0, 'sqn': 0.0}
    r = trades['r'].astype(float)
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    std = float(r.std()) if len(r) > 1 else 0.0
    return {
        'trades': int(len(r)),
        'win_rate_pct': float((r > 0).mean()) * 100,
        'profit_factor': float(wins / losses) if losses > 0 else (float('inf') if wins > 0 else 0.0),
        'expectancy_r': float(r.mean()),
        'sqn': float(r.mean() / std * np.sqrt(len(r))) if std > 0 else 0.0,
    }
