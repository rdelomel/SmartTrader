"""Analyze closed-trade expectancy, exit reasons, and per-strategy profit factor.

Usage:
  python scripts/analyze_trade_performance.py
  python scripts/analyze_trade_performance.py --db sqlite:///./data/trading.db

Expects DATABASE_URL or ./data/trading.db from the NAS/deploy copy.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_storage import DataStorage


CLOSED_STATUSES = {"closed", "stopped", "taken_profit"}


def _classify_exit(trade: Dict) -> str:
    status = (trade.get("status") or "").lower()
    if status == "taken_profit":
        return "take_profit"
    if status == "stopped":
        return "stop_loss"
    entry = trade.get("entry_price") or 0.0
    exit_px = trade.get("exit_price")
    stop = trade.get("stop_loss")
    tp = trade.get("take_profit")
    if exit_px is None or entry <= 0:
        return status or "unknown"
    # Heuristic when status is generic 'closed'
    if stop and abs(exit_px - stop) / max(entry, 1e-9) < 0.002:
        return "stop_loss"
    if tp and abs(exit_px - tp) / max(entry, 1e-9) < 0.002:
        return "take_profit"
    pnl = trade.get("pnl") or 0.0
    if pnl > 0:
        return "closed_win"
    if pnl < 0:
        return "closed_loss"
    return "closed_flat"


def _metrics(trades: List[Dict]) -> Dict:
    if not trades:
        return {
            "n": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "net_pnl": 0.0,
            "realized_rr": 0.0,
        }
    pnls = [float(t.get("pnl") or 0.0) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    avg_win = (gross_win / len(wins)) if wins else 0.0
    avg_loss = (gross_loss / len(losses)) if losses else 0.0
    win_rate = len(wins) / len(pnls) if pnls else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
    expectancy = (sum(pnls) / len(pnls)) if pnls else 0.0
    realized_rr = (avg_win / avg_loss) if avg_loss > 0 else 0.0
    return {
        "n": len(pnls),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": pf,
        "expectancy": expectancy,
        "net_pnl": sum(pnls),
        "realized_rr": realized_rr,
    }


def _print_metrics(title: str, m: Dict) -> None:
    pf = m["profit_factor"]
    pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
    print(f"\n{title}")
    print(f"  Trades: {m['n']}  Wins: {m['wins']}  Losses: {m['losses']}")
    print(f"  Win rate: {m['win_rate']*100:.1f}%")
    print(f"  Avg win: ${m['avg_win']:.2f}  Avg loss: ${m['avg_loss']:.2f}")
    print(f"  Realized R:R (avg win/avg loss): 1:{m['realized_rr']:.2f}")
    print(f"  Profit factor: {pf_s}  Expectancy/trade: ${m['expectancy']:.2f}")
    print(f"  Net PnL: ${m['net_pnl']:.2f}")


def analyze(db_url: Optional[str] = None) -> int:
    if db_url:
        os.environ["DATABASE_URL"] = db_url

    db_path = Path("data/trading.db")
    if not db_url and not os.getenv("DATABASE_URL") and not db_path.exists():
        print("No trading.db found at data/trading.db and DATABASE_URL is unset.")
        print("Copy the NAS/deploy DB into data/trading.db then re-run:")
        print("  python scripts/analyze_trade_performance.py")
        return 1

    try:
        storage = DataStorage()
        trades = storage.get_all_trades() or []
    except Exception as e:
        print(f"Failed to load trades from database: {e}")
        print("If Railway/Postgres is unreachable, copy trading.db locally and run:")
        print("  python scripts/analyze_trade_performance.py --db sqlite:///./data/trading.db")
        return 1

    closed = [t for t in trades if (t.get("status") or "").lower() in CLOSED_STATUSES]
    open_trades = [t for t in trades if (t.get("status") or "").lower() == "open"]

    print("=" * 72)
    print("SMARTTRADER TRADE PERFORMANCE REPORT")
    print("=" * 72)
    print(f"Total trades: {len(trades)}  Closed: {len(closed)}  Open: {len(open_trades)}")

    if not closed:
        print("\nNo closed trades to analyze.")
        return 0

    overall = _metrics(closed)
    _print_metrics("OVERALL (closed)", overall)

    # Exit reasons
    by_exit: Dict[str, List[Dict]] = defaultdict(list)
    for t in closed:
        by_exit[_classify_exit(t)].append(t)
    print("\nEXIT REASONS")
    for reason, group in sorted(by_exit.items(), key=lambda x: -len(x[1])):
        m = _metrics(group)
        print(f"  {reason}: n={m['n']}  WR={m['win_rate']*100:.1f}%  net=${m['net_pnl']:.2f}")

    # Per strategy
    by_strat: Dict[str, List[Dict]] = defaultdict(list)
    for t in closed:
        by_strat[str(t.get("strategy") or "unknown")].append(t)
    print("\nPER-STRATEGY LEADERBOARD (closed)")
    ranked: List[Tuple[str, Dict]] = []
    for name, group in by_strat.items():
        ranked.append((name, _metrics(group)))
    ranked.sort(key=lambda x: (x[1]["profit_factor"] if x[1]["profit_factor"] != float("inf") else 999, x[1]["net_pnl"]), reverse=True)
    for name, m in ranked:
        pf = m["profit_factor"]
        pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
        flag = ""
        if m["n"] >= 15 and (pf < 1.0 if pf != float("inf") else False):
            flag = "  <-- prune candidate (PF<1, n>=15)"
        print(
            f"  {name:28s} n={m['n']:3d}  WR={m['win_rate']*100:5.1f}%  "
            f"PF={pf_s:>5s}  exp=${m['expectancy']:7.2f}  net=${m['net_pnl']:8.2f}{flag}"
        )

    # Live gates checklist
    print("\nLIVE TRADING GATES (from config targets)")
    print(f"  min_win_rate 0.48: {'PASS' if overall['win_rate'] >= 0.48 else 'FAIL'} ({overall['win_rate']*100:.1f}%)")
    pf = overall["profit_factor"]
    print(f"  min_profit_factor 1.3: {'PASS' if (pf >= 1.3 if pf != float('inf') else True) else 'FAIL'} ({pf if pf != float('inf') else 'inf'})")
    print("=" * 72)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze SmartTrader trade performance")
    parser.add_argument("--db", default=None, help="SQLAlchemy DB URL override")
    args = parser.parse_args()
    return analyze(args.db)


if __name__ == "__main__":
    raise SystemExit(main())
