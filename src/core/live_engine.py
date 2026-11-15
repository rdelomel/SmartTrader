"""Live engine for the v2 core.

Trades config/live_strategy.yaml with the same rules as core.backtest:
- Signals are evaluated once per newly closed bar, then executed at the next
  opportunity the market is open (the live equivalent of "next bar open").
- Initial stop = entry -/+ stop_dist; trailing stops ratchet once per closed bar.
- Every entry goes through the RiskGovernor of the broker account it trades in.
The broker is the source of truth for open positions. State survives restarts.
"""

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from ..data.brokers.base_broker import OrderSide, OrderType
from .config_manager import LIVE_PATH, load_space, validate_live_config
from .data import HistoryProvider, latest_closed_bars
from .risk_governor import OpenPosition, RiskGovernor, TradeProposal, load_limits
from .strategies import generate

log = logging.getLogger('smarttrader.v2')

BAR_SPAN = {'1d': pd.Timedelta(days=1), '4h': pd.Timedelta(hours=4), '1h': pd.Timedelta(hours=1)}
REFETCH_EVERY = {'1d': pd.Timedelta(minutes=60), '4h': pd.Timedelta(minutes=15), '1h': pd.Timedelta(minutes=5)}
# Long enough that EMA(250) values match the research backtests closely.
LIVE_HISTORY_YEARS = {'1d': 6.0, '4h': 1.9, '1h': 0.5}
BROKER_FOR = {'forex': 'oanda', 'metals': 'oanda', 'crypto': 'alpaca', 'stocks': 'alpaca'}
FOREIGN_POSITION_RISK_PCT = 0.5
MISSING_CHECKS_BEFORE_CLOSED = 2
MAX_ORDER_ATTEMPTS = 3
STATE_PATH = os.path.join('data', 'live_engine_state.json')
EVENT_LOG = os.path.join('data', 'live_engine_events.jsonl')


def symbol_key(symbol: str) -> str:
    return symbol.upper().replace('/', '').replace('_', '')


@dataclass
class Position:
    symbol: str
    asset_class: str
    broker: str
    strategy: str
    timeframe: str
    params: Dict
    allow_short: bool
    direction: int
    units: float
    entry_price: float
    entry_time: str
    stop: float
    initial_stop_dist: float
    extreme: float
    risk_pct: float
    trade_id: str
    bars_held: int = 0
    missing_checks: int = 0

    def open_risk_pct(self) -> float:
        if self.initial_stop_dist <= 0:
            return self.risk_pct
        at_risk = (self.entry_price - self.stop) * self.direction
        return self.risk_pct * max(0.0, at_risk) / self.initial_stop_dist


@dataclass
class EngineState:
    positions: Dict[str, Dict] = field(default_factory=dict)
    last_bar: Dict[str, str] = field(default_factory=dict)
    last_fetch: Dict[str, str] = field(default_factory=dict)
    pending: Dict[str, Dict] = field(default_factory=dict)
    counters: Dict[str, int] = field(default_factory=dict)


# ── pure decision logic (mirrors core.backtest step 3) ──────────────────────
def evaluate_bar(strategy: str, params: Dict, allow_short: bool, df: pd.DataFrame,
                 pos: Optional[Position]) -> Optional[Dict]:
    """Decide what to do after the last bar of df closed. Mutates pos (trail, bars_held)."""
    sig = generate(strategy, df, params)
    i = len(df) - 1
    if i < sig.warmup:
        return None
    if pos is not None:
        pos.bars_held += 1
        a = sig.atr[i]
        if sig.trail_atr > 0 and np.isfinite(a):
            if pos.direction > 0:
                pos.extreme = max(pos.extreme, float(df['high'].iloc[i]))
                pos.stop = max(pos.stop, pos.extreme - sig.trail_atr * a)
            else:
                pos.extreme = min(pos.extreme, float(df['low'].iloc[i]))
                pos.stop = min(pos.stop, pos.extreme + sig.trail_atr * a)
        if sig.entry[i] == -pos.direction:
            new_dir = -pos.direction
            if new_dir > 0 or allow_short:
                return {'type': 'reverse', 'direction': new_dir, 'stop_dist': float(sig.stop_dist[i]),
                        'reason': 'reverse signal'}
            return {'type': 'exit', 'reason': 'opposite signal'}
        if sig.max_bars and pos.bars_held >= sig.max_bars:
            return {'type': 'exit', 'reason': 'time stop'}
        if (pos.direction > 0 and sig.exit_long[i]) or (pos.direction < 0 and sig.exit_short[i]):
            return {'type': 'exit', 'reason': 'exit signal'}
        return None
    d = int(sig.entry[i])
    sd = float(sig.stop_dist[i])
    if d != 0 and (d > 0 or allow_short) and np.isfinite(sd) and sd > 0:
        return {'type': 'enter', 'direction': d, 'stop_dist': sd, 'reason': 'entry signal'}
    return None


def order_units(symbol: str, asset_class: str, notional_usd: float, price: float) -> float:
    if price <= 0 or notional_usd <= 0:
        return 0.0
    if asset_class in ('forex', 'metals'):
        base, quote = symbol.upper().replace('/', '_').split('_')
        if quote == 'USD':
            return float(math.floor(notional_usd / price))
        if base == 'USD':
            return float(math.floor(notional_usd))
        return 0.0
    if asset_class == 'stocks':
        return float(math.floor(notional_usd / price))
    return math.floor(notional_usd / price * 1e6) / 1e6


def pnl_usd(symbol: str, asset_class: str, direction: int, units: float, entry: float, exit_px: float) -> float:
    raw = direction * units * (exit_px - entry)
    if asset_class in ('forex', 'metals'):
        base, quote = symbol.upper().replace('/', '_').split('_')
        if quote != 'USD' and base == 'USD' and exit_px > 0:
            return raw / exit_px
    return raw


def market_open(asset_class: str, brokers: Dict, now: datetime) -> bool:
    if asset_class == 'crypto':
        return True
    if asset_class == 'stocks':
        broker = brokers.get('alpaca')
        return bool(broker and broker.is_market_open('SPY'))
    wd, h = now.weekday(), now.hour  # OANDA: closed ~Fri 21:00 to Sun 22:00 UTC
    return not (wd == 5 or (wd == 4 and h >= 21) or (wd == 6 and h < 22))


class LiveEngine:
    def __init__(self, brokers: Dict, storage=None, provider=None, config_path: str = LIVE_PATH,
                 state_path: str = STATE_PATH, event_log: str = EVENT_LOG,
                 limits: Optional[Dict] = None, space: Optional[Dict] = None, governor_dir: str = 'data'):
        self.brokers = brokers
        self.storage = storage
        self.limits = limits or load_limits()
        self.space = space or load_space()
        self.config_path = config_path
        self.state_path = state_path
        self.event_log = event_log
        self.governors = {
            name: RiskGovernor(self.limits, os.path.join(governor_dir, f'governor_state_{name}.json'))
            for name in brokers
        }
        self.provider = provider or HistoryProvider(
            cache_dir=os.path.join('data', 'live_history'), max_age_hours=0.05,
            oanda_broker=brokers.get('oanda'),
        )
        self.config: Dict = {'strategies': []}
        self._config_mtime: Optional[float] = None
        self.state = self._load_state()

    # ── persistence / logging ────────────────────────────────────────────────
    def _load_state(self) -> EngineState:
        if os.path.exists(self.state_path):
            with open(self.state_path, 'r', encoding='utf-8') as f:
                return EngineState(**json.load(f))
        return EngineState()

    def _save_state(self) -> None:
        os.makedirs(os.path.dirname(self.state_path) or '.', exist_ok=True)
        tmp = self.state_path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.state), f, indent=2, default=str)
        os.replace(tmp, self.state_path)

    def _event(self, kind: str, **data) -> None:
        rec = {'at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'event': kind, **data}
        log.info('%s %s', kind, data)
        os.makedirs(os.path.dirname(self.event_log) or '.', exist_ok=True)
        with open(self.event_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, default=str) + '\n')

    def _count(self, key: str) -> None:
        self.state.counters[key] = self.state.counters.get(key, 0) + 1

    def _reload_config(self) -> None:
        if not os.path.exists(self.config_path):
            if self._config_mtime is not None or not self.config['strategies']:
                self._config_mtime = None
            return
        mtime = os.path.getmtime(self.config_path)
        if mtime == self._config_mtime:
            return
        with open(self.config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        try:
            validate_live_config(cfg, self.space, self.limits)
        except ValueError as e:
            self._event('config_rejected', error=str(e))
            self._config_mtime = mtime
            return
        self.config = cfg
        self._config_mtime = mtime
        self._event('config_loaded', version=cfg.get('version'), strategies=len(cfg.get('strategies', [])))

    def _positions(self) -> Dict[str, Position]:
        return {s: Position(**p) for s, p in self.state.positions.items()}

    def _put(self, pos: Position) -> None:
        self.state.positions[pos.symbol] = asdict(pos)

    # ── main loop step ───────────────────────────────────────────────────────
    def run_once(self, now: Optional[datetime] = None) -> Dict:
        now = now or datetime.now(timezone.utc)
        self._reload_config()

        equities: Dict[str, float] = {}
        for name, broker in self.brokers.items():
            try:
                bal = broker.get_account_balance() or {}
                eq = float(bal.get('equity') or bal.get('balance') or 0.0)
            except Exception as e:
                eq = 0.0
                self._event('equity_error', broker=name, error=str(e))
            if eq <= 0:
                continue
            equities[name] = eq
            if 'flatten_all' in self.governors[name].update_equity(eq, now):
                self._flatten(name, now, self.governors[name].state.halt_reason)

        broker_positions = {n: self._broker_positions(n) for n in equities}
        self._reconcile(broker_positions, now)
        self._check_stops(now)
        self._evaluate_new_bars(now)
        self._execute_pending(equities, broker_positions, now)
        self._save_state()
        return self.status(equities)

    def status(self, equities: Optional[Dict[str, float]] = None) -> Dict:
        return {
            'config_version': self.config.get('version'),
            'strategies': len(self.config.get('strategies', [])),
            'open_positions': list(self.state.positions),
            'pending': {s: a['type'] for s, a in self.state.pending.items()},
            'equity': equities or {},
            'governor': {n: {'halted': g.state.halted, 'reason': g.state.halt_reason,
                             'peak_equity': g.state.peak_equity} for n, g in self.governors.items()},
        }

    # ── broker views ─────────────────────────────────────────────────────────
    def _broker_positions(self, name: str) -> Optional[Dict[str, Dict]]:
        try:
            raw = self.brokers[name].get_open_positions()
        except Exception as e:
            self._event('positions_error', broker=name, error=str(e))
            return None
        if raw is None:
            return None
        out = {}
        for p in raw:
            out[symbol_key(p.get('symbol', ''))] = {
                'symbol': p.get('symbol', ''),
                'direction': 1 if str(p.get('side', 'buy')).lower() == 'buy' else -1,
                'units': float(p.get('quantity', 0) or 0),
                'entry_price': float(p.get('entry_price', 0) or 0),
            }
        return out

    def _price(self, pos_or_symbol, broker_name: str) -> float:
        symbol = pos_or_symbol if isinstance(pos_or_symbol, str) else pos_or_symbol.symbol
        try:
            return float(self.brokers[broker_name].get_current_price(symbol) or 0.0)
        except Exception:
            return 0.0

    def _asset_class_of(self, sym_key: str, broker_name: str) -> str:
        for ac, u in self.space['universe'].items():
            if any(symbol_key(s) == sym_key for s in u['symbols']):
                return ac
        if broker_name == 'oanda':
            return 'metals' if sym_key.startswith(('XAU', 'XAG')) else 'forex'
        return 'crypto' if sym_key.endswith('USD') and len(sym_key) <= 7 else 'stocks'

    # ── reconciliation and stops ─────────────────────────────────────────────
    def _reconcile(self, broker_positions: Dict[str, Optional[Dict]], now: datetime) -> None:
        for sym, pos in self._positions().items():
            seen = broker_positions.get(pos.broker)
            if seen is None:
                continue
            if symbol_key(sym) in seen:
                pos.missing_checks = 0
                self._put(pos)
                continue
            pos.missing_checks += 1
            if pos.missing_checks < MISSING_CHECKS_BEFORE_CLOSED:
                self._put(pos)
                continue
            exit_px = self._price(pos, pos.broker) or pos.stop
            self._record_close(pos, exit_px, 'closed at broker', now)

    def _check_stops(self, now: datetime) -> None:
        for sym, pos in self._positions().items():
            price = self._price(pos, pos.broker)
            if price <= 0:
                continue
            hit = price <= pos.stop if pos.direction > 0 else price >= pos.stop
            if hit and market_open(pos.asset_class, self.brokers, now):
                self.state.pending.pop(sym, None)
                self._close(pos, 'stop', now)

    # ── signals ──────────────────────────────────────────────────────────────
    def _evaluate_new_bars(self, now: datetime) -> None:
        now_naive = now.astimezone(timezone.utc).replace(tzinfo=None)
        configured = {e['symbol']: e for e in self.config.get('strategies', [])}
        positions = self._positions()
        work = dict(configured)
        for sym, pos in positions.items():
            if sym not in work:  # removed from config: keep managing exits with stored params
                work[sym] = {'symbol': sym, 'asset_class': pos.asset_class, 'strategy': pos.strategy,
                             'timeframe': pos.timeframe, 'params': pos.params, 'allow_short': pos.allow_short,
                             'risk_pct': pos.risk_pct, 'exit_only': True}

        for sym, entry in work.items():
            tf = entry['timeframe']
            last = self.state.last_bar.get(sym)
            if last and pd.Timestamp(last) + 2 * BAR_SPAN[tf] > now_naive:
                continue
            fetched = self.state.last_fetch.get(sym)
            if fetched and pd.Timestamp(fetched) + REFETCH_EVERY[tf] > now_naive:
                continue
            self.state.last_fetch[sym] = now_naive.isoformat()
            try:
                df = self.provider.load(sym, entry['asset_class'], tf, LIVE_HISTORY_YEARS[tf])
            except Exception as e:
                self._event('data_error', symbol=sym, error=str(e))
                continue
            df = latest_closed_bars(df, tf, now)
            if len(df) < 60:
                continue
            bar = df.index[-1]
            if last and bar <= pd.Timestamp(last):
                continue
            self.state.last_bar[sym] = bar.isoformat()

            pos = positions.get(sym)
            strategy, params, allow_short = entry['strategy'], entry['params'], entry['allow_short']
            if pos is not None:
                strategy, params, allow_short = pos.strategy, pos.params, pos.allow_short
            action = evaluate_bar(strategy, params, allow_short, df, pos)
            if pos is not None:
                self._put(pos)

            if action and action['type'] in ('enter', 'reverse') and (last is None or entry.get('exit_only')):
                # Never act on an entry seen on the first evaluation after start-up (stale) or for
                # symbols no longer in the config.
                action = {'type': 'exit', 'reason': action['reason']} if action['type'] == 'reverse' else None

            existing = self.state.pending.get(sym)
            if action:
                action.update(bar=bar.isoformat(), attempts=0, symbol=sym, asset_class=entry['asset_class'],
                              strategy=entry['strategy'], timeframe=tf, params=entry['params'],
                              allow_short=entry['allow_short'], risk_pct=entry['risk_pct'])
                self.state.pending[sym] = action
                self._count(f'signal_{action["type"]}')
                self._event('signal', symbol=sym, action=action['type'], reason=action['reason'], bar=bar)
            elif existing and existing['type'] == 'enter':
                self.state.pending.pop(sym)  # entry signal went stale without a fill
                self._count('entry_expired')

    # ── execution ────────────────────────────────────────────────────────────
    def _execute_pending(self, equities: Dict[str, float], broker_positions: Dict, now: datetime) -> None:
        for sym, act in list(self.state.pending.items()):
            ac = act['asset_class']
            broker_name = BROKER_FOR[ac]
            if broker_name not in equities or not market_open(ac, self.brokers, now):
                continue
            positions = self._positions()
            if act['type'] in ('exit', 'reverse') and sym in positions:
                if not self._close(positions[sym], act['reason'], now):
                    act['attempts'] += 1
                    if act['attempts'] >= MAX_ORDER_ATTEMPTS:
                        self._event('exit_failed', symbol=sym)
                    continue
            if act['type'] == 'exit':
                self.state.pending.pop(sym, None)
                continue
            done = self._open(act, equities[broker_name], broker_positions.get(broker_name) or {}, now)
            act['attempts'] += 1
            if done or act['attempts'] >= MAX_ORDER_ATTEMPTS:
                self.state.pending.pop(sym, None)

    def _governor_view(self, broker_name: str, broker_positions: Dict[str, Dict]) -> List[OpenPosition]:
        managed = [p for p in self._positions().values() if p.broker == broker_name]
        view = [OpenPosition(p.symbol, p.asset_class, p.direction, p.open_risk_pct()) for p in managed]
        managed_keys = {symbol_key(p.symbol) for p in managed}
        for key, bp in broker_positions.items():
            if key not in managed_keys:
                view.append(OpenPosition(bp['symbol'], self._asset_class_of(key, broker_name),
                                         bp['direction'], FOREIGN_POSITION_RISK_PCT))
        return view

    def _open(self, act: Dict, equity: float, broker_positions: Dict[str, Dict], now: datetime) -> bool:
        sym, ac, d = act['symbol'], act['asset_class'], int(act['direction'])
        broker_name = BROKER_FOR[ac]
        broker = self.brokers[broker_name]
        price = self._price(sym, broker_name)
        if price <= 0:
            return False
        stop = price - d * act['stop_dist']
        decision = self.governors[broker_name].approve(
            TradeProposal(sym, ac, d, price, stop, act['risk_pct']),
            self._governor_view(broker_name, broker_positions), equity,
        )
        if not decision.approved:
            self._count('rejected: ' + decision.reasons[0])
            self._event('entry_rejected', symbol=sym, reasons=decision.reasons)
            return True
        units = order_units(sym, ac, decision.notional, price)
        if units <= 0:
            self._count('rejected: size rounds to zero')
            return True

        hard_stop = round(stop, 5) if broker_name == 'oanda' else None
        res = broker.place_order(symbol=sym, side=OrderSide.BUY if d > 0 else OrderSide.SELL,
                                 quantity=units, order_type=OrderType.MARKET, stop_loss=hard_stop)
        if not res or not res.get('success'):
            self._count('order_failed')
            self._event('order_failed', symbol=sym, error=(res or {}).get('error'))
            return False

        fill = float(res.get('price') or 0.0) or price
        stop = fill - d * act['stop_dist']
        trade_id = f'v2-{symbol_key(sym)}-{now.strftime("%Y%m%d%H%M%S")}'
        pos = Position(
            symbol=sym, asset_class=ac, broker=broker_name, strategy=act['strategy'], timeframe=act['timeframe'],
            params=act['params'], allow_short=act['allow_short'], direction=d, units=units, entry_price=fill,
            entry_time=now.isoformat(), stop=stop, initial_stop_dist=act['stop_dist'], extreme=fill,
            risk_pct=decision.risk_pct, trade_id=trade_id,
        )
        self._put(pos)
        self._count('entries')
        self._event('entry', symbol=sym, direction=d, units=units, price=fill, stop=stop,
                    risk_pct=round(decision.risk_pct, 3), notes=decision.reasons)
        if self.storage:
            self.storage.store_trade({
                'trade_id': trade_id, 'symbol': sym, 'side': 'buy' if d > 0 else 'sell', 'quantity': units,
                'entry_price': fill, 'stop_loss': stop, 'entry_time': now.replace(tzinfo=None),
                'status': 'open', 'strategy': f'v2_{act["strategy"]}',
            })
        return True

    def _close(self, pos: Position, reason: str, now: datetime) -> bool:
        broker = self.brokers.get(pos.broker)
        if broker is None:
            return False
        broker_symbol = pos.symbol.replace('/', '') if pos.broker == 'alpaca' else pos.symbol
        price = self._price(pos, pos.broker)
        ok = broker.close_position(broker_symbol, side=OrderSide.BUY if pos.direction > 0 else OrderSide.SELL)
        if not ok:
            self._count('close_failed')
            self._event('close_failed', symbol=pos.symbol, reason=reason)
            return False
        self._record_close(pos, price or pos.stop, reason, now)
        return True

    def _record_close(self, pos: Position, exit_px: float, reason: str, now: datetime) -> None:
        pnl = pnl_usd(pos.symbol, pos.asset_class, pos.direction, pos.units, pos.entry_price, exit_px)
        notional = pos.units * pos.entry_price
        self.state.positions.pop(pos.symbol, None)
        self._count(f'exit: {reason}')
        self._event('exit', symbol=pos.symbol, reason=reason, price=exit_px, pnl=round(pnl, 2))
        if self.storage:
            self.storage.update_trade(pos.trade_id, {
                'status': 'closed', 'exit_price': exit_px, 'exit_time': now.replace(tzinfo=None),
                'pnl': pnl, 'pnl_percent': pnl / notional * 100 if notional else 0.0,
            })

    def _flatten(self, broker_name: str, now: datetime, reason: str) -> None:
        self._event('flatten_all', broker=broker_name, reason=reason)
        for pos in [p for p in self._positions().values() if p.broker == broker_name]:
            self._close(pos, 'drawdown halt', now)
        broker = self.brokers[broker_name]
        for bp in (self._broker_positions(broker_name) or {}).values():
            sym = bp['symbol'].replace('/', '') if broker_name == 'alpaca' else bp['symbol']
            broker.close_position(sym)
        for sym, act in list(self.state.pending.items()):
            if BROKER_FOR[act['asset_class']] == broker_name:
                self.state.pending.pop(sym)
