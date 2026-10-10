"""Account-level risk governor.

Hard limits come from config/risk_limits.yaml (human-owned). Nothing in the
strategy layer or the config manager can raise them. A drawdown halt persists
across restarts and is only cleared by an explicit human reset.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import yaml

DEFAULT_LIMITS_PATH = os.path.join('config', 'risk_limits.yaml')
DEFAULT_STATE_PATH = os.path.join('data', 'governor_state.json')


def load_limits(path: str = DEFAULT_LIMITS_PATH) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@dataclass
class TradeProposal:
    symbol: str
    asset_class: str
    direction: int          # +1 long, -1 short
    entry_price: float
    stop_price: float
    risk_pct: float         # requested loss-at-stop as % of equity


@dataclass
class OpenPosition:
    symbol: str
    asset_class: str
    direction: int
    risk_pct: float         # current loss-at-stop as % of equity (0 if stop is at breakeven or better)


@dataclass
class Decision:
    approved: bool
    risk_pct: float = 0.0
    notional: float = 0.0   # in account currency
    reasons: List[str] = field(default_factory=list)


@dataclass
class GovernorState:
    peak_equity: float = 0.0
    halted: bool = False
    halt_reason: str = ''
    halted_at: str = ''
    day_key: str = ''
    day_start_equity: float = 0.0
    week_key: str = ''
    week_start_equity: float = 0.0


def currency_legs(symbol: str, asset_class: str) -> List[str]:
    """Exposure buckets a long position adds to (a short adds the negatives)."""
    s = symbol.upper().replace('/', '_')
    if asset_class in ('forex', 'metals') and '_' in s:
        base, quote = s.split('_', 1)
        return [base, '-' + quote]
    if asset_class == 'crypto':
        return ['CRYPTO']
    return ['US_EQUITY']


class RiskGovernor:
    def __init__(self, limits: Optional[Dict] = None, state_path: Optional[str] = DEFAULT_STATE_PATH):
        self.limits = limits or load_limits()
        self.state_path = state_path
        self.state = self._load_state()

    # ── persistence ──────────────────────────────────────────────────────────
    def _load_state(self) -> GovernorState:
        if self.state_path and os.path.exists(self.state_path):
            with open(self.state_path, 'r', encoding='utf-8') as f:
                return GovernorState(**json.load(f))
        return GovernorState()

    def _save(self) -> None:
        if not self.state_path:
            return
        os.makedirs(os.path.dirname(self.state_path) or '.', exist_ok=True)
        with open(self.state_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.state), f, indent=2)

    # ── equity tracking ──────────────────────────────────────────────────────
    def drawdown_pct(self, equity: float) -> float:
        peak = self.state.peak_equity
        return round(max(0.0, (1 - equity / peak) * 100), 9) if peak > 0 else 0.0

    def update_equity(self, equity: float, now: Optional[datetime] = None) -> List[str]:
        """Record equity. Returns actions the engine must take, e.g. ['flatten_all']."""
        now = now or datetime.now(timezone.utc)
        acct = self.limits['account']
        s = self.state
        actions: List[str] = []

        day_key = now.strftime('%Y-%m-%d')
        week_key = '%d-W%02d' % now.isocalendar()[:2]
        if s.day_key != day_key:
            s.day_key, s.day_start_equity = day_key, equity
        if s.week_key != week_key:
            s.week_key, s.week_start_equity = week_key, equity
        s.peak_equity = max(s.peak_equity, equity)

        if not s.halted and self.drawdown_pct(equity) >= acct['max_drawdown_halt_pct']:
            s.halted = True
            s.halt_reason = f'drawdown {self.drawdown_pct(equity):.2f}% >= {acct["max_drawdown_halt_pct"]}%'
            s.halted_at = now.isoformat()
            actions.append('flatten_all')
        self._save()
        return actions

    def reset_halt(self, equity: float, confirm: str) -> None:
        """Human-only: clears a drawdown halt and restarts peak tracking."""
        if confirm != 'I understand the risk':
            raise PermissionError("Pass confirm='I understand the risk' to clear a drawdown halt")
        self.state = GovernorState(peak_equity=equity)
        self._save()

    def risk_multiplier(self, equity: float) -> float:
        acct = self.limits['account']
        dd, start, halt = self.drawdown_pct(equity), acct['drawdown_derisk_start_pct'], acct['max_drawdown_halt_pct']
        if self.state.halted or dd >= halt:
            return 0.0
        if dd <= start:
            return 1.0
        return 1.0 - 0.75 * (dd - start) / (halt - start)

    def entries_paused(self, equity: float) -> Optional[str]:
        acct, s = self.limits['account'], self.state
        if s.halted:
            return f'halted: {s.halt_reason}'
        if s.day_start_equity > 0 and (1 - equity / s.day_start_equity) * 100 >= acct['daily_loss_pause_pct']:
            return 'daily loss limit reached'
        if s.week_start_equity > 0 and (1 - equity / s.week_start_equity) * 100 >= acct['weekly_loss_pause_pct']:
            return 'weekly loss limit reached'
        return None

    # ── trade approval ───────────────────────────────────────────────────────
    def approve(self, p: TradeProposal, open_positions: List[OpenPosition], equity: float) -> Decision:
        pt, pf = self.limits['per_trade'], self.limits['portfolio']
        paused = self.entries_paused(equity)
        if paused:
            return Decision(False, reasons=[paused])
        if p.direction not in (1, -1) or p.entry_price <= 0 or equity <= 0:
            return Decision(False, reasons=['invalid proposal'])
        stop_dist = (p.entry_price - p.stop_price) * p.direction
        if stop_dist <= 0:
            return Decision(False, reasons=['stop is on the wrong side of entry'])

        if any(o.symbol == p.symbol for o in open_positions):
            return Decision(False, reasons=['position already open for symbol'])
        if len(open_positions) >= pf['max_positions']:
            return Decision(False, reasons=['max positions reached'])
        class_cap = pf['max_positions_per_asset_class'].get(p.asset_class, 0)
        if sum(1 for o in open_positions if o.asset_class == p.asset_class) >= class_cap:
            return Decision(False, reasons=[f'max {p.asset_class} positions reached'])

        exposure: Dict[str, int] = {}
        for o in open_positions + [OpenPosition(p.symbol, p.asset_class, p.direction, 0.0)]:
            for leg in currency_legs(o.symbol, o.asset_class):
                sign = -1 if leg.startswith('-') else 1
                key = leg.lstrip('-')
                exposure[key] = exposure.get(key, 0) + sign * o.direction
        for key, net in exposure.items():
            if abs(net) > pf['max_same_currency_exposure'] and key in [
                l.lstrip('-') for l in currency_legs(p.symbol, p.asset_class)
            ]:
                return Decision(False, reasons=[f'too much same-direction {key} exposure'])

        reasons: List[str] = []
        risk = min(max(p.risk_pct, pt['min_risk_pct']), pt['max_risk_pct'])
        if risk != p.risk_pct:
            reasons.append(f'risk clamped to {risk:.2f}%')
        mult = self.risk_multiplier(equity)
        if mult < 1.0:
            risk *= mult
            reasons.append(f'drawdown de-risk x{mult:.2f}')

        open_risk = sum(o.risk_pct for o in open_positions)
        room = pf['max_open_risk_pct'] - open_risk
        if room < pt['min_risk_pct']:
            return Decision(False, reasons=['portfolio open-risk cap reached'])
        if risk > room:
            risk = room
            reasons.append(f'risk cut to {risk:.2f}% by open-risk cap')

        stop_frac = stop_dist / p.entry_price
        notional = equity * (risk / 100) / stop_frac
        max_notional = equity * pf['max_leverage'].get(p.asset_class, 1.0)
        if notional > max_notional:
            notional = max_notional
            risk = notional * stop_frac / equity * 100
            reasons.append(f'leverage cap; risk now {risk:.2f}%')
        if risk < pt['min_risk_pct'] * 0.5:
            return Decision(False, reasons=reasons + ['risk too small after caps'])
        return Decision(True, risk_pct=risk, notional=notional, reasons=reasons)
