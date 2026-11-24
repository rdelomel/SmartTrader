"""Keep local trade storage aligned with the broker.

- Mirror broker-only opens into storage (after restart).
- Close local opens that are no longer at the broker (manual close, stop fill, etc.)
  so Total Trades / realized P&L update without waiting for the live engine tick.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


def symbol_key(symbol: str) -> str:
    return symbol.upper().replace("/", "").replace("_", "")


def _side_key(side: str) -> str:
    return "buy" if str(side or "buy").lower() in ("buy", "long") else "sell"


def _estimate_pnl(symbol: str, side: str, qty: float, entry: float, exit_px: float) -> float:
    """Rough USD P&L for display; fine for paper. Forex quote≈USD for major pairs."""
    if entry <= 0 or qty <= 0 or exit_px <= 0:
        return 0.0
    if _side_key(side) == "buy":
        return (exit_px - entry) * qty
    return (entry - exit_px) * qty


def _broker_open_index(brokers: Dict[str, Any]) -> Tuple[Dict[str, Dict], bool]:
    """
    Returns (index by symbol_key+side, ok).
    ok=False if every broker failed — do not close locals in that case.
    """
    index: Dict[str, Dict] = {}
    any_ok = False
    any_attempt = False
    for name, broker in (brokers or {}).items():
        if not hasattr(broker, "get_open_positions"):
            continue
        any_attempt = True
        try:
            raw = broker.get_open_positions() or []
            any_ok = True
            for p in raw:
                key = symbol_key(p.get("symbol", ""))
                side = _side_key(p.get("side", "buy"))
                index[f"{key}:{side}"] = {**p, "broker": name}
                # also index by symbol only for matching flexibility
                index.setdefault(key, {**p, "broker": name})
        except Exception:
            continue
    if not any_attempt:
        return {}, False
    return index, any_ok


def close_stale_local_opens(
    storage: Any,
    brokers: Dict[str, Any],
    now: Optional[datetime] = None,
) -> List[Dict]:
    """
    Mark local open trades closed when the broker no longer holds them.
    Returns list of closed trade summaries.
    """
    if not storage or not brokers:
        return []
    now = now or datetime.now(timezone.utc)
    broker_idx, ok = _broker_open_index(brokers)
    if not ok:
        # Broker unreachable — leave local opens alone (avoid mass false closes)
        return []

    try:
        local_open = storage.get_open_trades() or []
    except Exception:
        return []

    closed: List[Dict] = []
    for t in local_open:
        sym = t.get("symbol") or ""
        key = symbol_key(sym)
        side = _side_key(t.get("side", "buy"))
        if f"{key}:{side}" in broker_idx or key in broker_idx:
            # still open at broker (same symbol; side match preferred)
            if f"{key}:{side}" in broker_idx:
                continue
            # same symbol opposite side rare — treat as still open if any position on symbol
            bp = broker_idx.get(key)
            if bp and _side_key(bp.get("side", "buy")) == side:
                continue
            if bp:
                continue

        entry = float(t.get("entry_price") or 0)
        qty = float(t.get("quantity") or 0)
        exit_px = float(t.get("exit_price") or 0)
        # try live price from any broker
        if exit_px <= 0:
            for broker in brokers.values():
                try:
                    if hasattr(broker, "get_current_price"):
                        px = float(broker.get_current_price(sym) or 0)
                        if px > 0:
                            exit_px = px
                            break
                except Exception:
                    continue
        if exit_px <= 0:
            exit_px = float(t.get("stop_loss") or entry or 0)

        pnl = _estimate_pnl(sym, side, qty, entry, exit_px)
        trade_id = t.get("trade_id")
        if not trade_id:
            continue
        try:
            storage.update_trade(
                str(trade_id),
                {
                    "status": "closed",
                    "exit_price": exit_px,
                    "exit_time": now.replace(tzinfo=None) if hasattr(now, "replace") else now,
                    "pnl": pnl,
                },
            )
            closed.append(
                {
                    "trade_id": trade_id,
                    "symbol": sym,
                    "side": side,
                    "pnl": pnl,
                    "exit_price": exit_px,
                }
            )
        except Exception:
            continue
    return closed


def sync_broker_only_to_storage(engine: Any, now: Optional[datetime] = None) -> int:
    """Write open broker positions missing from storage. Returns count synced."""
    storage = getattr(engine, "storage", None)
    brokers = getattr(engine, "brokers", None) or {}
    if not storage or not brokers:
        return 0

    now = now or datetime.now(timezone.utc)
    try:
        local_open = storage.get_open_trades() or []
    except Exception:
        return 0

    local_keys = {symbol_key(t.get("symbol", "")) for t in local_open}
    positions = {}
    try:
        positions = engine._positions() if hasattr(engine, "_positions") else {}
    except Exception:
        positions = {}
    managed_keys = {symbol_key(s) for s in positions}

    synced = 0
    for broker_name, broker in brokers.items():
        try:
            raw = broker.get_open_positions() if hasattr(broker, "get_open_positions") else None
        except Exception:
            continue
        if not raw:
            continue
        for p in raw:
            sym = p.get("symbol", "")
            key = symbol_key(sym)
            if not key or key in managed_keys or key in local_keys:
                continue
            trade_id = f"v2-broker-{key}"
            if any(t.get("trade_id") == trade_id for t in local_open):
                continue
            side = "buy" if str(p.get("side", "buy")).lower() == "buy" else "sell"
            try:
                storage.store_trade(
                    {
                        "trade_id": trade_id,
                        "symbol": sym or key,
                        "side": side,
                        "quantity": float(p.get("quantity") or 0),
                        "entry_price": float(p.get("entry_price") or 0),
                        "stop_loss": p.get("stop_loss"),
                        "take_profit": p.get("take_profit"),
                        "entry_time": now.replace(tzinfo=None),
                        "status": "open",
                        "strategy": "v2_broker_sync",
                    }
                )
                local_keys.add(key)
                synced += 1
                if hasattr(engine, "_event"):
                    engine._event("broker_only_synced", symbol=sym, broker=broker_name)
            except Exception:
                continue
    return synced


def full_integrity_pass(engine: Any, now: Optional[datetime] = None) -> Dict[str, Any]:
    """Run both directions: close stale locals, then import broker-only opens."""
    storage = getattr(engine, "storage", None)
    brokers = getattr(engine, "brokers", None) or {}
    closed = close_stale_local_opens(storage, brokers, now=now)
    synced = sync_broker_only_to_storage(engine, now=now)
    return {"closed_stale": closed, "synced_broker_only": synced}
