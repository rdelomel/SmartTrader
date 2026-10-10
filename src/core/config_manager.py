"""Bounded config manager.

Turns walk-forward research into config/live_strategy.yaml:
1. Runs walk-forward for every symbol x strategy x timeframe in the search space.
2. Enables only candidates that pass the out-of-sample gates in risk_limits.yaml.
3. Keeps the best candidate per symbol and sizes risk so the combined
   out-of-sample portfolio drawdown lands near the target.
4. Applies safety rails: per-trade risk bounds, probation for new strategies,
   and a cap on how fast any strategy's risk can grow between updates.
The optimiser can only choose values inside config/strategy_space.yaml and can
never modify config/risk_limits.yaml. An LLM, if configured, only writes the
human-readable explanation of a change.
"""

import json
import os
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from .costs import cost_for
from .metrics import summarize
from .risk_governor import load_limits
from .strategies import STRATEGIES
from .walk_forward import WalkForwardResult, walk_forward

SPACE_PATH = os.path.join('config', 'strategy_space.yaml')
LIVE_PATH = os.path.join('config', 'live_strategy.yaml')
HISTORY_DIR = os.path.join('data', 'config_history')
RESEARCH_RISK_PCT = 1.0  # walk-forward runs at 1% risk per trade; allocation rescales


def load_space(path: str = SPACE_PATH) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _py(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, dict):
        return {k: _py(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_py(x) for x in v]
    return v


def validate_live_config(cfg: Dict, space: Dict, limits: Dict) -> None:
    """Raises ValueError if the config steps outside human-owned bounds."""
    pt = limits['per_trade']
    universe = {
        s.upper().replace('/', '_'): ac
        for ac, u in space['universe'].items() for s in u['symbols']
    }
    for e in cfg.get('strategies', []):
        sym = e['symbol'].upper().replace('/', '_')
        if universe.get(sym) != e['asset_class']:
            raise ValueError(f'{e["symbol"]} is not in the {e["asset_class"]} universe')
        if e['strategy'] not in STRATEGIES or e['strategy'] not in space['strategies']:
            raise ValueError(f'unknown strategy {e["strategy"]}')
        if e['timeframe'] not in space['timeframes']:
            raise ValueError(f'timeframe {e["timeframe"]} not allowed')
        bounds = space['strategies'][e['strategy']]
        for name, spec in bounds.items():
            v = e['params'].get(name)
            if v is None or not (spec['min'] <= v <= spec['max']):
                raise ValueError(f'{e["symbol"]} {name}={v} outside [{spec["min"]}, {spec["max"]}]')
        if not (pt['min_risk_pct'] <= e['risk_pct'] <= pt['max_risk_pct']):
            raise ValueError(f'{e["symbol"]} risk {e["risk_pct"]}% outside per-trade bounds')
        if e.get('allow_short') and not space['universe'][e['asset_class']].get('allow_short', False):
            raise ValueError(f'{e["symbol"]} shorting not allowed for {e["asset_class"]}')


class ConfigManager:
    def __init__(self, space: Optional[Dict] = None, limits: Optional[Dict] = None,
                 live_path: str = LIVE_PATH, history_dir: str = HISTORY_DIR):
        self.space = space or load_space()
        self.limits = limits or load_limits()
        self.live_path = live_path
        self.history_dir = history_dir

    # ── 1. research ──────────────────────────────────────────────────────────
    def research(self, provider, asset_classes: Optional[List[str]] = None,
                 timeframes: Optional[List[str]] = None, samples: Optional[int] = None,
                 log: Callable[[str], None] = print) -> List[WalkForwardResult]:
        opt = self.space.get('optimizer', {})
        results: List[WalkForwardResult] = []
        for ac, u in self.space['universe'].items():
            if asset_classes and ac not in asset_classes:
                continue
            for tf in timeframes or self.space['timeframes']:
                wf = self.space['walk_forward'][tf]
                for symbol in u['symbols']:
                    try:
                        df = provider.load(symbol, ac, tf, wf['history_years'])
                    except Exception as e:
                        log(f'  [data] {symbol} {tf}: {e}')
                        continue
                    if len(df) < 300:
                        log(f'  [data] {symbol} {tf}: only {len(df)} bars, skipped')
                        continue
                    for strat, bounds in self.space['strategies'].items():
                        res = walk_forward(
                            df, symbol, ac, strat, tf, bounds, cost_for(ac),
                            train_days=wf['train_days'], test_days=wf['test_days'],
                            samples=samples or opt.get('samples_per_strategy', 48),
                            smoothing_k=opt.get('neighbor_smoothing_k', 5),
                            min_train_trades=opt.get('min_train_trades', 15),
                            allow_short=u.get('allow_short', False),
                            max_leverage=self.limits['portfolio']['max_leverage'].get(ac, 1.0),
                            risk_frac=RESEARCH_RISK_PCT / 100, seed=opt.get('random_seed', 7),
                        )
                        o = res.oos
                        log(f'  {symbol:8} {tf:3} {strat:15} trades={o["trades"]:4d} '
                            f'PF={o["profit_factor"]:.2f} expR={o["expectancy_r"]:+.3f} '
                            f'DD={o["max_drawdown_pct"]:.1f}% folds+={o["pct_profitable_folds"]:.0f}%')
                        results.append(res)
        return results

    # ── 2. acceptance gates ──────────────────────────────────────────────────
    def gate(self, res: WalkForwardResult) -> List[str]:
        g, o = self.limits['strategy_acceptance'], res.oos
        fails = []
        if res.live_params is None:
            fails.append('no tradable params on latest window')
        if o['trades'] < g['min_oos_trades']:
            fails.append(f'{o["trades"]} OOS trades < {g["min_oos_trades"]}')
        if o['profit_factor'] < g['min_oos_profit_factor']:
            fails.append(f'PF {o["profit_factor"]:.2f} < {g["min_oos_profit_factor"]}')
        if o['expectancy_r'] < g['min_oos_expectancy_r']:
            fails.append(f'expectancy {o["expectancy_r"]:+.3f}R < {g["min_oos_expectancy_r"]}')
        if o['pct_profitable_folds'] < g['min_pct_profitable_folds'] * 100:
            fails.append(f'{o["pct_profitable_folds"]:.0f}% profitable folds')
        if o['max_drawdown_pct'] > g['max_oos_drawdown_pct_at_1pct_risk']:
            fails.append(f'DD {o["max_drawdown_pct"]:.1f}% too deep')
        return fails

    def select(self, results: List[WalkForwardResult]) -> Tuple[List[WalkForwardResult], List[Tuple[WalkForwardResult, List[str]]]]:
        passed, rejected = [], []
        for r in results:
            fails = self.gate(r)
            (rejected.append((r, fails)) if fails else passed.append(r))
        best: Dict[str, WalkForwardResult] = {}
        for r in passed:
            cur = best.get(r.symbol)
            if cur is None or r.oos['sqn'] > cur.oos['sqn']:
                best[r.symbol] = r
        for r in passed:
            if best[r.symbol] is not r:
                rejected.append((r, ['a stronger strategy was chosen for this symbol']))
        return list(best.values()), rejected

    # ── 3. allocation ────────────────────────────────────────────────────────
    @staticmethod
    def _portfolio_returns(chosen: List[WalkForwardResult], risks: Dict[str, float]) -> pd.Series:
        series = []
        for r in chosen:
            s = r.oos_daily_returns
            if s.empty:
                continue
            series.append(s * (risks[r.symbol] / RESEARCH_RISK_PCT))
        if not series:
            return pd.Series(dtype=float)
        latest_start = max(s.index[0] for s in series)
        frame = pd.concat(series, axis=1).fillna(0.0)
        if (frame.index[-1] - latest_start).days >= 365:
            frame = frame[frame.index >= latest_start]
        return frame.sum(axis=1)

    def allocate(self, chosen: List[WalkForwardResult], previous: Optional[Dict] = None) -> Tuple[Dict[str, float], Dict]:
        pt, alloc = self.limits['per_trade'], self.limits['allocation']
        if not chosen:
            return {}, summarize(pd.Series(dtype=float))
        prev = {e['symbol']: e for e in (previous or {}).get('strategies', [])}

        w = np.array([min(max(r.oos['sharpe'], 0.2), 2.0) for r in chosen])
        base = pt['default_risk_pct']
        risks = {r.symbol: base * wi / w.mean() for r, wi in zip(chosen, w)}

        dd = summarize(self._portfolio_returns(chosen, risks))['max_drawdown_pct']
        scale = alloc['target_portfolio_drawdown_pct'] / dd if dd > 0 else 1.0
        scale = min(scale, pt['max_risk_pct'] / max(risks.values()))
        for r in chosen:
            v = risks[r.symbol] * scale
            old = prev.get(r.symbol)
            if old is None or old.get('strategy') != r.strategy:
                v *= alloc['probation_risk_multiplier']
            else:
                v = min(v, old['risk_pct'] * alloc['max_risk_increase_per_update'])
            risks[r.symbol] = round(min(max(v, pt['min_risk_pct']), pt['max_risk_pct']), 3)

        estimate = summarize(self._portfolio_returns(chosen, risks))
        estimate['expected_open_risk_pct'] = float(sum(r.exposure * risks[r.symbol] for r in chosen))
        return risks, estimate

    # ── 4. build / apply ─────────────────────────────────────────────────────
    def load_live(self) -> Optional[Dict]:
        if not os.path.exists(self.live_path):
            return None
        with open(self.live_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def build(self, chosen: List[WalkForwardResult], risks: Dict[str, float], estimate: Dict,
              previous: Optional[Dict], data_source: str) -> Dict:
        prev_symbols = {e['symbol']: e for e in (previous or {}).get('strategies', [])}
        entries = []
        for r in sorted(chosen, key=lambda x: (x.asset_class, x.symbol)):
            old = prev_symbols.get(r.symbol)
            entries.append(_py({
                'symbol': r.symbol, 'asset_class': r.asset_class, 'strategy': r.strategy,
                'timeframe': r.timeframe, 'params': r.live_params, 'risk_pct': risks[r.symbol],
                'allow_short': bool(self.space['universe'][r.asset_class].get('allow_short', False)),
                'probation': old is None or old.get('strategy') != r.strategy,
                'oos': {k: round(float(r.oos[k]), 3) for k in (
                    'trades', 'profit_factor', 'expectancy_r', 'win_rate_pct', 'sharpe',
                    'max_drawdown_pct', 'pct_profitable_folds')},
            }))
        return {
            'version': int((previous or {}).get('version', 0)) + 1,
            'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'generated_by': 'core.config_manager',
            'data_source': data_source,
            'portfolio_estimate_oos': _py({k: round(float(v), 3) for k, v in estimate.items()}),
            'strategies': entries,
        }

    @staticmethod
    def diff(old: Optional[Dict], new: Dict) -> List[str]:
        o = {e['symbol']: e for e in (old or {}).get('strategies', [])}
        n = {e['symbol']: e for e in new.get('strategies', [])}
        lines = []
        for s in sorted(set(o) | set(n)):
            a, b = o.get(s), n.get(s)
            if a is None:
                lines.append(f'+ enable {s} {b["strategy"]} {b["timeframe"]} at {b["risk_pct"]}% risk')
            elif b is None:
                lines.append(f'- disable {s} ({a["strategy"]}) — no longer passes OOS gates')
            else:
                if a['strategy'] != b['strategy'] or a['timeframe'] != b['timeframe']:
                    lines.append(f'~ {s}: {a["strategy"]} {a["timeframe"]} -> {b["strategy"]} {b["timeframe"]}')
                if a['risk_pct'] != b['risk_pct']:
                    lines.append(f'~ {s}: risk {a["risk_pct"]}% -> {b["risk_pct"]}%')
                if a['params'] != b['params']:
                    lines.append(f'~ {s}: params {a["params"]} -> {b["params"]}')
        return lines

    def explain(self, changes: List[str], estimate: Dict) -> str:
        summary = (
            f'{len(changes)} change(s). Out-of-sample portfolio estimate: '
            f'avg month {estimate.get("avg_month_pct", 0):+.2f}%, '
            f'max drawdown {estimate.get("max_drawdown_pct", 0):.1f}%, '
            f'Sharpe {estimate.get("sharpe", 0):.2f}.'
        )
        if not os.getenv('OPENAI_API_KEY') or not changes:
            return summary
        try:
            from openai import OpenAI

            client = OpenAI()
            msg = client.chat.completions.create(
                model=os.getenv('CONFIG_EXPLAINER_MODEL', 'gpt-4o-mini'),
                messages=[
                    {'role': 'system', 'content': 'Explain trading-config changes to a non-expert in under 120 words. '
                                                  'Do not recommend changes; only describe these ones and their risk.'},
                    {'role': 'user', 'content': summary + '\n' + '\n'.join(changes)},
                ],
                temperature=0.2,
            )
            return msg.choices[0].message.content.strip()
        except Exception as e:
            return f'{summary} (LLM explanation unavailable: {e})'

    def apply(self, new: Dict, previous: Optional[Dict]) -> Tuple[str, List[str], str]:
        validate_live_config(new, self.space, self.limits)
        changes = self.diff(previous, new)
        explanation = self.explain(changes, new.get('portfolio_estimate_oos', {}))
        new = dict(new, change_explanation=explanation)
        os.makedirs(self.history_dir, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        with open(os.path.join(self.history_dir, f'live_strategy_v{new["version"]}_{stamp}.yaml'), 'w', encoding='utf-8') as f:
            yaml.safe_dump(new, f, sort_keys=False)
        with open(os.path.join(self.history_dir, 'changelog.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps({'version': new['version'], 'at': stamp, 'changes': changes,
                                'explanation': explanation}) + '\n')
        tmp = self.live_path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            f.write('# GENERATED by core.config_manager — do not hand-edit; bounds live in\n'
                    '# config/strategy_space.yaml and config/risk_limits.yaml.\n')
            yaml.safe_dump(new, f, sort_keys=False)
        os.replace(tmp, self.live_path)
        return self.live_path, changes, explanation
