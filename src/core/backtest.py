"""Single-symbol backtester.

Rules (kept identical to the live engine):
- Signals are read at bar close and filled at the next bar's open.
- Position size = equity * risk / (stop distance / entry), capped by max leverage.
- Stops/targets are checked against each bar's high/low. If a bar touches both,
  the stop is assumed to fill first. Gaps through a stop fill at the open.
- Costs: half spread + commission + slippage on each side, plus daily financing.
P&L is tracked in return space, so results are independent of quote currency.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .costs import CostModel
from .strategies import SignalFrame


@dataclass
class BacktestResult:
    trades: pd.DataFrame   # one row per closed trade
    equity: pd.Series      # mark-to-market equity, starts at 1.0
    exposure: float        # fraction of bars with an open position


def _bar_days(index: pd.Index) -> float:
    if len(index) < 3:
        return 1.0
    deltas = np.diff(index.values).astype('timedelta64[s]').astype(float)
    return float(np.median(deltas)) / 86400.0


def run_backtest(
    df: pd.DataFrame,
    sig: SignalFrame,
    cost: CostModel,
    risk_frac: float = 0.01,
    allow_short: bool = True,
    max_leverage: float = 10.0,
    trade_start: int = 0,
) -> BacktestResult:
    o = df['open'].to_numpy(float)
    h = df['high'].to_numpy(float)
    l = df['low'].to_numpy(float)
    c = df['close'].to_numpy(float)
    idx = df.index
    n = len(df)
    per_side = cost.per_side_frac
    financing = cost.financing_frac_per_day * _bar_days(idx)
    first_entry_bar = max(trade_start, sig.warmup + 1)

    cash = 1.0
    eq = np.ones(n)
    trades: List[dict] = []
    bars_in_market = 0

    pos = 0
    entry_px = stop = tp = notional = risk_amt = extreme = 0.0
    entry_i = bars_held = 0
    pending_exit = False
    pending_entry = 0
    pending_sd = 0.0

    def close(i: int, px: float, reason: str) -> None:
        nonlocal cash, pos
        gross = pos * notional * (px / entry_px - 1.0)
        exit_cost = notional * (px / entry_px) * per_side
        pnl = gross - exit_cost
        cash += pnl
        trades.append({
            'entry_time': idx[entry_i], 'exit_time': idx[i], 'direction': pos,
            'entry': entry_px, 'exit': px, 'bars': bars_held,
            'pnl_frac': pnl / (cash - pnl) if cash - pnl > 0 else 0.0,
            'r': pnl / risk_amt if risk_amt > 0 else 0.0,
            'reason': reason,
        })
        pos = 0

    for i in range(n):
        # 1. Orders queued at the previous close fill at this open.
        if pos != 0 and pending_exit:
            close(i, o[i], 'signal')
        if pos == 0 and pending_entry != 0 and i >= first_entry_bar:
            d = pending_entry
            if (d > 0 or allow_short) and np.isfinite(pending_sd) and pending_sd > 0 and o[i] > 0:
                entry_px = o[i]
                stop_frac = pending_sd / entry_px
                notional = cash * min(risk_frac / stop_frac, max_leverage)
                risk_amt = notional * stop_frac
                cash -= notional * per_side
                stop = entry_px - d * pending_sd
                tp = entry_px + d * pending_sd * sig.take_profit_r if sig.take_profit_r > 0 else 0.0
                extreme = entry_px
                pos, entry_i, bars_held = d, i, 0
        pending_exit, pending_entry = False, 0

        # 2. Intrabar stop / target.
        if pos > 0:
            if l[i] <= stop:
                close(i, min(stop, o[i]), 'stop')
            elif tp and h[i] >= tp:
                close(i, max(tp, o[i]), 'target')
        elif pos < 0:
            if h[i] >= stop:
                close(i, max(stop, o[i]), 'stop')
            elif tp and l[i] <= tp:
                close(i, min(tp, o[i]), 'target')

        # 3. End of bar: carry costs, trailing stop, exits, new entries.
        if pos != 0:
            bars_in_market += 1
            bars_held += 1
            cash -= notional * financing
            a = sig.atr[i]
            if sig.trail_atr > 0 and np.isfinite(a):
                if pos > 0:
                    extreme = max(extreme, h[i])
                    stop = max(stop, extreme - sig.trail_atr * a)
                else:
                    extreme = min(extreme, l[i])
                    stop = min(stop, extreme + sig.trail_atr * a)
            if sig.max_bars and bars_held >= sig.max_bars:
                pending_exit = True
            if (pos > 0 and sig.exit_long[i]) or (pos < 0 and sig.exit_short[i]):
                pending_exit = True
            if sig.entry[i] == -pos:
                pending_exit = True
                pending_entry, pending_sd = -pos, sig.stop_dist[i]
        elif sig.entry[i] != 0:
            pending_entry, pending_sd = int(sig.entry[i]), sig.stop_dist[i]

        eq[i] = cash + (pos * notional * (c[i] / entry_px - 1.0) if pos != 0 else 0.0)

    if pos != 0:
        close(n - 1, c[-1], 'end')
        eq[-1] = cash

    trades_df = pd.DataFrame(trades, columns=[
        'entry_time', 'exit_time', 'direction', 'entry', 'exit', 'bars', 'pnl_frac', 'r', 'reason',
    ])
    return BacktestResult(trades=trades_df, equity=pd.Series(eq, index=idx), exposure=bars_in_market / max(n, 1))
