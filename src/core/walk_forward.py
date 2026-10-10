"""Walk-forward validation.

Each sampled parameter set is backtested once over the full history (strategies
are causal, so this equals running fold by fold). For every fold, parameters are
chosen on the training window only, preferring plateaus over spikes, and judged
on trades that were entered in the following unseen test window.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .backtest import BacktestResult, run_backtest
from .costs import CostModel
from .metrics import summarize, to_daily_returns, trade_stats
from .strategies import generate, normalize_params


@dataclass
class WalkForwardResult:
    symbol: str
    asset_class: str
    strategy: str
    timeframe: str
    oos: Dict                       # summary metrics on concatenated out-of-sample data
    folds: List[Dict]
    live_params: Optional[Dict]     # params chosen on the most recent training window
    oos_daily_returns: pd.Series = field(repr=False, default_factory=pd.Series)
    exposure: float = 0.0


def sample_params(space: Dict, n: int, rng: np.random.Generator) -> List[Dict]:
    samples = []
    for _ in range(n):
        p = {}
        for name, spec in space.items():
            lo, hi = spec['min'], spec['max']
            if spec.get('type') == 'int':
                p[name] = int(rng.integers(int(lo), int(hi) + 1))
            else:
                p[name] = round(float(rng.uniform(lo, hi)), 3)
        samples.append(p)
    return samples


def _normalized_matrix(params: List[Dict], space: Dict) -> np.ndarray:
    names = list(space)
    m = np.array([[float(p[k]) for k in names] for p in params])
    lo = np.array([space[k]['min'] for k in names], dtype=float)
    hi = np.array([space[k]['max'] for k in names], dtype=float)
    return (m - lo) / np.where(hi > lo, hi - lo, 1.0)


def _objective(trades: pd.DataFrame, min_trades: int) -> float:
    stats = trade_stats(trades)
    if stats['trades'] < min_trades:
        return -np.inf
    return stats['sqn']


def _fold_windows(index: pd.DatetimeIndex, train_days: int, test_days: int) -> List[tuple]:
    start, end = index[0], index[-1]
    windows = []
    test_start = start + pd.Timedelta(days=train_days)
    while test_start < end:
        test_end = min(test_start + pd.Timedelta(days=test_days), end)
        windows.append((test_start - pd.Timedelta(days=train_days), test_start, test_end))
        test_start = test_end
    return windows


def _entries_between(trades: pd.DataFrame, start, end) -> pd.DataFrame:
    if trades.empty:
        return trades
    mask = (trades['entry_time'] >= start) & (trades['entry_time'] < end)
    return trades[mask]


def walk_forward(
    df: pd.DataFrame,
    symbol: str,
    asset_class: str,
    strategy: str,
    timeframe: str,
    space: Dict,
    cost: CostModel,
    train_days: int,
    test_days: int,
    samples: int = 48,
    smoothing_k: int = 5,
    min_train_trades: int = 15,
    allow_short: bool = True,
    max_leverage: float = 10.0,
    risk_frac: float = 0.01,
    seed: int = 7,
) -> WalkForwardResult:
    rng = np.random.default_rng(seed)
    params = [normalize_params(strategy, p) for p in sample_params(space, samples, rng)]
    runs: List[BacktestResult] = [
        run_backtest(df, generate(strategy, df, p), cost, risk_frac=risk_frac,
                     allow_short=allow_short, max_leverage=max_leverage)
        for p in params
    ]
    norm = _normalized_matrix(params, space)
    dist = np.linalg.norm(norm[:, None, :] - norm[None, :, :], axis=2)
    neighbors = np.argsort(dist, axis=1)[:, :max(1, smoothing_k)]

    def pick(start, end) -> Optional[int]:
        raw = np.array([_objective(_entries_between(r.trades, start, end), min_train_trades) for r in runs])
        if not np.isfinite(raw).any():
            return None
        filled = np.where(np.isfinite(raw), raw, -5.0)
        smoothed = filled[neighbors].mean(axis=1)
        smoothed[~np.isfinite(raw)] = -np.inf
        return int(np.argmax(smoothed))

    folds, oos_trades, oos_returns = [], [], []
    for train_start, test_start, test_end in _fold_windows(df.index, train_days, test_days):
        best = pick(train_start, test_start)
        if best is None:
            folds.append({'test_start': str(test_start.date()), 'params': None, 'trades': 0, 'return_pct': 0.0})
            continue
        run = runs[best]
        t = _entries_between(run.trades, test_start, test_end)
        eq = run.equity[(run.equity.index >= test_start) & (run.equity.index < test_end)]
        rets = to_daily_returns(eq) if len(eq) > 1 else pd.Series(dtype=float)
        oos_trades.append(t)
        oos_returns.append(rets)
        folds.append({
            'test_start': str(test_start.date()), 'params': params[best], 'trades': int(len(t)),
            'return_pct': float((1 + rets).prod() - 1) * 100 if len(rets) else 0.0,
        })

    all_trades = pd.concat(oos_trades) if oos_trades else pd.DataFrame(columns=['r'])
    all_returns = pd.concat(oos_returns).sort_index() if oos_returns else pd.Series(dtype=float)
    all_returns = all_returns[~all_returns.index.duplicated(keep='first')]
    oos = summarize(all_returns, all_trades)
    active = [f for f in folds if f['params'] is not None]
    oos['folds'] = len(folds)
    oos['pct_profitable_folds'] = (
        100.0 * sum(1 for f in active if f['return_pct'] > 0) / len(active) if active else 0.0
    )

    latest = pick(df.index[-1] - pd.Timedelta(days=train_days), df.index[-1] + pd.Timedelta(days=1))
    exposure = runs[latest].exposure if latest is not None else 0.0
    return WalkForwardResult(
        symbol=symbol, asset_class=asset_class, strategy=strategy, timeframe=timeframe,
        oos=oos, folds=folds, live_params=params[latest] if latest is not None else None,
        oos_daily_returns=all_returns, exposure=exposure,
    )
