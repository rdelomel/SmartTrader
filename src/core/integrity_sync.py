"""Mirror broker-only open positions into local trade storage.

Keeps dashboard Data Integrity from sitting at 0% when the broker still has
positions that are not tracked in engine state (e.g. after a restart).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional


def symbol_key(symbol: str) -> str:
    return symbol.upper().replace('/', '').replace('_', '')


def sync_broker_only_to_storage(engine: Any, now: Optional[datetime] = None) -> int:
    """Write open broker positions missing from storage. Returns count synced."""
    storage = getattr(engine, 'storage', None)
    brokers = getattr(engine, 'brokers', None) or {}
    if not storage or not brokers:
        return 0

    now = now or datetime.now(timezone.utc)
    try:
        local_open = storage.get_open_trades() or []
    except Exception:
        return 0

    local_keys = {symbol_key(t.get('symbol', '')) for t in local_open}
    positions = {}
    try:
        positions = engine._positions() if hasattr(engine, '_positions') else {}
    except Exception:
        positions = {}
    managed_keys = {symbol_key(s) for s in positions}

    synced = 0
    for broker_name, broker in brokers.items():
        try:
            raw = broker.get_open_positions() if hasattr(broker, 'get_open_positions') else None
        except Exception:
            continue
        if not raw:
            continue
        for p in raw:
            sym = p.get('symbol', '')
            key = symbol_key(sym)
            if not key or key in managed_keys or key in local_keys:
                continue
            trade_id = f'v2-broker-{key}'
            if any(t.get('trade_id') == trade_id for t in local_open):
                continue
            side = 'buy' if str(p.get('side', 'buy')).lower() == 'buy' else 'sell'
            try:
                storage.store_trade({
                    'trade_id': trade_id,
                    'symbol': sym or key,
                    'side': side,
                    'quantity': float(p.get('quantity') or 0),
                    'entry_price': float(p.get('entry_price') or 0),
                    'entry_time': now.replace(tzinfo=None),
                    'status': 'open',
                    'strategy': 'v2_broker_sync',
                })
                local_keys.add(key)
                synced += 1
                if hasattr(engine, '_event'):
                    engine._event('broker_only_synced', symbol=sym, broker=broker_name)
            except Exception as e:
                if hasattr(engine, '_event'):
                    engine._event('broker_only_sync_error', symbol=key, error=str(e))
    return synced
