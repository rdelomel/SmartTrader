"""Check paper performance against live_trading_gates in trading_config.yaml.

Usage:
  python scripts/validate_live_gates.py
"""

from __future__ import annotations

import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_storage import DataStorage

CLOSED_STATUSES = {"closed", "stopped", "taken_profit"}


def _load_gates(config_path: Path) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("live_trading_gates", {})


def _closed_trades(storage: DataStorage) -> List[Dict]:
    trades = storage.get_all_trades()
    return [t for t in trades if (t.get("status") or "").lower() in CLOSED_STATUSES]


def _win_rate(trades: List[Dict]) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if (t.get("pnl") or 0) > 0)
    return wins / len(trades)


def _profit_factor(trades: List[Dict]) -> float:
    gross_win = sum(float(t.get("pnl") or 0) for t in trades if (t.get("pnl") or 0) > 0)
    gross_loss = abs(sum(float(t.get("pnl") or 0) for t in trades if (t.get("pnl") or 0) < 0))
    if gross_loss == 0:
        return float("inf") if gross_win > 0 else 0.0
    return gross_win / gross_loss


def _max_drawdown_pct(trades: List[Dict]) -> float:
    """Approximate equity DD from cumulative closed PnL (starting at 0)."""
    if not trades:
        return 0.0
    ordered = sorted(
        trades,
        key=lambda t: t.get("exit_time") or t.get("entry_time") or "",
    )
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in ordered:
        equity += float(t.get("pnl") or 0.0)
        peak = max(peak, equity)
        if peak > 0:
            dd = (peak - equity) / peak * 100.0
            max_dd = max(max_dd, dd)
        elif equity < 0:
            max_dd = max(max_dd, abs(equity))  # absolute when peak is 0
    return max_dd


def _paper_days(trades: List[Dict]) -> float:
    times = []
    for t in trades:
        for key in ("entry_time", "exit_time"):
            raw = t.get(key)
            if not raw:
                continue
            try:
                times.append(datetime.fromisoformat(str(raw).replace("Z", "+00:00")))
            except Exception:
                pass
    if len(times) < 2:
        return 0.0
    span = max(times) - min(times)
    return span.total_seconds() / 86400.0


def _sharpe_approx(trades: List[Dict]) -> Optional[float]:
    """Simple per-trade Sharpe (mean/std of pnl). Not annualized — gate uses relative bar."""
    if len(trades) < 5:
        return None
    pnls = [float(t.get("pnl") or 0.0) for t in trades]
    mean = sum(pnls) / len(pnls)
    var = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
    std = math.sqrt(var) if var > 0 else 0.0
    if std == 0:
        return None
    # Scale lightly so magnitude is comparable to a ratio gate
    return mean / std


def main() -> int:
    root = Path(__file__).parent.parent
    config_path = root / "config" / "trading_config.yaml"
    gates = _load_gates(config_path)

    db_path = root / "data" / "trading.db"
    if not db_path.exists() and not __import__("os").getenv("DATABASE_URL"):
        print("No trading.db — cannot validate gates. Copy deploy DB to data/trading.db first.")
        print("Gates configured:")
        for k, v in gates.items():
            print(f"  {k}: {v}")
        return 1

    storage = DataStorage()
    closed = _closed_trades(storage)
    wr = _win_rate(closed)
    pf = _profit_factor(closed)
    dd = _max_drawdown_pct(closed)
    days = _paper_days(closed)
    sharpe = _sharpe_approx(closed)

    checks = [
        ("min_paper_trading_days", days >= float(gates.get("min_paper_trading_days", 60)), f"{days:.1f}"),
        ("min_win_rate", wr >= float(gates.get("min_win_rate", 0.48)), f"{wr*100:.1f}%"),
        ("min_profit_factor", (pf >= float(gates.get("min_profit_factor", 1.3)) if pf != float("inf") else True), f"{pf if pf != float('inf') else 'inf'}"),
        ("max_drawdown_percent", dd <= float(gates.get("max_drawdown_percent", 8.0)), f"{dd:.1f}%"),
    ]
    if sharpe is not None:
        checks.append(
            ("min_sharpe_ratio", sharpe >= float(gates.get("min_sharpe_ratio", 1.0)), f"{sharpe:.2f}")
        )
    else:
        checks.append(("min_sharpe_ratio", False, "n/a (<5 trades)"))

    print("=" * 60)
    print("LIVE TRADING GATES VALIDATION")
    print("=" * 60)
    print(f"Closed trades: {len(closed)}")
    all_pass = True
    for name, ok, value in checks:
        target = gates.get(name, "?")
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] {name}: actual={value}  target={target}")
    print("=" * 60)
    print("RESULT:", "READY for size-up" if all_pass else "NOT READY — keep paper trading / prune")
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
