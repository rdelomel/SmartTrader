#!/usr/bin/env python3
"""Paper-run report: realized win rate, expectancy, drawdown and engine health.

Reads the engine's own logs (no database needed):
  data/live_engine_events.jsonl   entries / exits / rejections / errors
  data/equity_curve.jsonl         broker equity every 15 min (added with this report)

Usage:  python3 scripts/paper_report.py [--events PATH] [--equity PATH] [--since YYYY-MM-DD]
Pass criteria are the ones in docs/PAPER_RUN.md.
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime

CRITERIA = {'max_dd_pct': 5.0, 'min_pf': 1.15, 'min_expectancy_r': 0.05, 'min_win_rate_pct': 50.0, 'min_trades': 30}


def load(path):
    rows = []
    try:
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        pass
    except FileNotFoundError:
        pass
    return rows


def parse_ts(s):
    return datetime.fromisoformat(s.replace('Z', '+00:00')).replace(tzinfo=None)


def closed_trades(events, curve):
    open_by_sym, trades = defaultdict(list), []
    for e in events:
        if e['event'] == 'entry':
            open_by_sym[e['symbol']].append(e)
        elif e['event'] == 'exit' and open_by_sym[e['symbol']]:
            en = open_by_sym[e['symbol']].pop(0)
            equity = en.get('equity')
            if equity is None:
                t = parse_ts(en['at'])
                prior = [c for c in curve if parse_ts(c['at']) <= t]
                equity = sum(prior[-1]['equity'].values()) if prior else None
            risk_usd = en['risk_pct'] / 100 * equity if equity else None
            trades.append({
                'symbol': e['symbol'], 'strategy': en.get('strategy') or '?', 'pnl': e['pnl'],
                'r': e['pnl'] / risk_usd if risk_usd else None, 'reason': e['reason'], 'at': e['at'],
            })
    return trades, sum(len(v) for v in open_by_sym.values())


def stats(ts):
    n = len(ts)
    if not n:
        return None
    wins = [t['pnl'] for t in ts if t['pnl'] > 0]
    losses = [-t['pnl'] for t in ts if t['pnl'] < 0]
    rs = [t['r'] for t in ts if t['r'] is not None]
    return {
        'n': n, 'win': 100 * len(wins) / n, 'pnl': sum(t['pnl'] for t in ts),
        'pf': sum(wins) / sum(losses) if losses else math.inf if wins else 0.0,
        'exp_r': sum(rs) / len(rs) if rs else float('nan'),
        'avg_win': sum(wins) / len(wins) if wins else 0.0, 'avg_loss': -sum(losses) / len(losses) if losses else 0.0,
    }


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (100 * (c - a) / d, 100 * (c + a) / d)


def equity_stats(curve):
    out = {}
    for broker in sorted({b for c in curve for b in c['equity']}):
        pts = [(parse_ts(c['at']), c['equity'][broker]) for c in curve if broker in c['equity']]
        peak, dd, day_open, worst_day = 0.0, 0.0, {}, 0.0
        for t, eq in pts:
            peak = max(peak, eq)
            dd = max(dd, (1 - eq / peak) * 100)
            day_open.setdefault(t.date(), eq)
        last_by_day = {}
        for t, eq in pts:
            last_by_day[t.date()] = eq
        prev = None
        for d in sorted(last_by_day):
            if prev:
                worst_day = min(worst_day, (last_by_day[d] / prev - 1) * 100)
            prev = last_by_day[d]
        out[broker] = {'start': pts[0][1], 'end': pts[-1][1], 'ret': (pts[-1][1] / pts[0][1] - 1) * 100,
                       'max_dd': dd, 'worst_day': worst_day, 'since': pts[0][0].date()}
    return out


def fmt(x, nd=2):
    return 'n/a' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:.{nd}f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', default='data/live_engine_events.jsonl')
    ap.add_argument('--equity', default='data/equity_curve.jsonl')
    ap.add_argument('--since', default=None)
    a = ap.parse_args()
    events, curve = load(a.events), load(a.equity)
    if a.since:
        cut = datetime.fromisoformat(a.since)
        events = [e for e in events if parse_ts(e['at']) >= cut]
        curve = [c for c in curve if parse_ts(c['at']) >= cut]
    if not events and not curve:
        print('No data found. Check the paths (docker bind mount is ~/SmartTrader/data).')
        return 1

    trades, still_open = closed_trades(events, curve)
    total = stats(trades)
    print(f'=== Paper report: {events[0]["at"][:10] if events else "-"} to {events[-1]["at"][:10] if events else "-"} ===')
    print(f'Closed trades: {len(trades)}   still open: {still_open}')
    if total:
        lo, hi = wilson(round(total['win'] * total['n'] / 100), total['n'])
        print(f'Win rate {fmt(total["win"],1)}% (95% range {lo:.0f}-{hi:.0f}%)  PF {fmt(total["pf"])}  '
              f'expectancy {fmt(total["exp_r"],3)}R  P&L {fmt(total["pnl"])}  avg win {fmt(total["avg_win"])}  avg loss {fmt(total["avg_loss"])}')
        print('\nPer symbol:')
        by = defaultdict(list)
        for t in trades:
            by[(t['symbol'], t['strategy'])].append(t)
        for (sym, strat), ts in sorted(by.items()):
            s = stats(ts)
            print(f'  {sym:9s} {strat:16s} n={s["n"]:3d} win={s["win"]:5.1f}% PF={fmt(s["pf"]):>5} exp={fmt(s["exp_r"],3):>6}R pnl={s["pnl"]:9.2f}')
        print('\nExits by reason:', dict(Counter(t['reason'] for t in trades)))

    eq = equity_stats(curve)
    print('\nEquity (per broker account):')
    for b, s in eq.items():
        print(f'  {b:7s} {s["start"]:.2f} -> {s["end"]:.2f}  return {s["ret"]:+.2f}%  max drawdown {s["max_dd"]:.2f}%  worst day {s["worst_day"]:+.2f}%  (since {s["since"]})')
    if not eq:
        print('  no equity_curve.jsonl yet (needs the updated image)')

    kinds = Counter(e['event'] for e in events)
    print('\nEngine health:', {k: kinds[k] for k in ('order_failed', 'close_failed', 'exit_failed', 'data_error', 'equity_error',
                                                    'positions_error', 'config_rejected', 'flatten_all', 'broker_added', 'broker_only_synced') if kinds[k]} or 'clean')
    rej = Counter(r for e in events if e['event'] == 'entry_rejected' for r in e.get('reasons', []))
    if rej:
        print('Top entry rejections:', rej.most_common(5))

    print('\nCriteria (docs/PAPER_RUN.md):')
    checks = []
    if total:
        checks += [('trades >= %d' % CRITERIA['min_trades'], total['n'] >= CRITERIA['min_trades']),
                   ('win rate >= %.0f%%' % CRITERIA['min_win_rate_pct'], total['win'] >= CRITERIA['min_win_rate_pct']),
                   ('profit factor >= %.2f' % CRITERIA['min_pf'], total['pf'] >= CRITERIA['min_pf']),
                   ('expectancy >= %.2fR' % CRITERIA['min_expectancy_r'], total['exp_r'] >= CRITERIA['min_expectancy_r'])]
    for b, s in eq.items():
        checks.append((f'{b} max drawdown <= {CRITERIA["max_dd_pct"]:.0f}%', s['max_dd'] <= CRITERIA['max_dd_pct']))
        checks.append((f'{b} return > 0', s['ret'] > 0))
    for name, ok in checks:
        print(f'  [{"PASS" if ok else "FAIL"}] {name}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
