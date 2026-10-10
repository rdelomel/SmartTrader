"""Tests for the v2 core: no lookahead, conservative fills, hard risk limits, bounded config."""

import copy

import numpy as np
import pandas as pd
import pytest

from src.core.backtest import run_backtest
from src.core.config_manager import ConfigManager, load_space, validate_live_config
from src.core.costs import CostModel
from src.core.risk_governor import OpenPosition, RiskGovernor, TradeProposal, load_limits
from src.core.strategies import SignalFrame, generate
from src.core.walk_forward import walk_forward

ZERO_COST = CostModel(0.0, 0.0, 0.0)


def _df(closes, spread=0.5):
    idx = pd.date_range('2020-01-01', periods=len(closes), freq='D')
    c = np.asarray(closes, float)
    o = np.concatenate([[c[0]], c[:-1]])
    return pd.DataFrame({'open': o, 'high': np.maximum(o, c) + spread, 'low': np.minimum(o, c) - spread,
                         'close': c, 'volume': 0}, index=idx)


def _sig(n, entries=None, stop=5.0, exit_long=None):
    e = np.zeros(n, np.int8)
    for i, d in (entries or {}).items():
        e[i] = d
    xl = np.zeros(n, bool)
    for i in exit_long or []:
        xl[i] = True
    return SignalFrame(entry=e, exit_long=xl, exit_short=np.zeros(n, bool),
                       stop_dist=np.full(n, stop), atr=np.full(n, 1.0), warmup=0)


def test_entry_fills_at_next_open_not_signal_close():
    df = _df([100, 101, 102, 103, 104, 105])
    res = run_backtest(df, _sig(6, {1: 1}, exit_long=[3]), ZERO_COST)
    t = res.trades.iloc[0]
    assert t['entry_time'] == df.index[2] and t['entry'] == df['open'].iloc[2]
    assert t['exit_time'] == df.index[4] and t['exit'] == df['open'].iloc[4]


def test_stop_wins_when_bar_hits_stop_and_target():
    df = _df([100, 100, 100, 100])
    df.iloc[2, df.columns.get_loc('high')] = 120
    df.iloc[2, df.columns.get_loc('low')] = 80
    sig = _sig(4, {0: 1}, stop=5.0)
    sig.take_profit_r = 2.0
    t = run_backtest(df, sig, ZERO_COST).trades.iloc[0]
    assert t['reason'] == 'stop' and t['r'] == pytest.approx(-1.0)


def test_one_r_loss_equals_risk_fraction():
    df = _df([100, 100, 100, 100])
    df.iloc[2, df.columns.get_loc('low')] = 90
    res = run_backtest(df, _sig(4, {0: 1}, stop=5.0), ZERO_COST, risk_frac=0.01)
    assert res.equity.iloc[-1] == pytest.approx(0.99)


def test_costs_reduce_pnl():
    df = _df(np.linspace(100, 120, 30))
    sig = _sig(30, {2: 1}, exit_long=[25])
    free = run_backtest(df, sig, ZERO_COST).equity.iloc[-1]
    paid = run_backtest(df, sig, CostModel(10, 10, 10, 1.0)).equity.iloc[-1]
    assert paid < free


def test_strategy_signals_do_not_use_future_bars():
    rng = np.random.default_rng(0)
    df = _df(100 + np.cumsum(rng.normal(0, 1, 400)))
    params = {'entry_n': 30, 'exit_n': 15, 'stop_atr': 3.0, 'trail_atr': 0.0, 'trend_ema': 100}
    full = generate('donchian_trend', df, params)
    cut = generate('donchian_trend', df.iloc[:300], params)
    assert np.array_equal(full.entry[:300], cut.entry)


def test_walk_forward_only_scores_out_of_sample_entries():
    rng = np.random.default_rng(1)
    df = _df(100 + np.cumsum(rng.normal(0.05, 1, 1500)))
    space = load_space()['strategies']['donchian_trend']
    res = walk_forward(df, 'X', 'stocks', 'donchian_trend', '1d', space, ZERO_COST,
                       train_days=500, test_days=250, samples=8)
    first_test = pd.Timestamp(res.folds[0]['test_start'])
    assert res.oos_daily_returns.index.min() >= first_test


@pytest.fixture
def governor(tmp_path):
    return RiskGovernor(load_limits(), state_path=str(tmp_path / 'state.json'))


def _proposal(**kw):
    p = dict(symbol='EUR_USD', asset_class='forex', direction=1, entry_price=1.10, stop_price=1.09, risk_pct=0.5)
    p.update(kw)
    return TradeProposal(**p)


def test_drawdown_halt_flattens_and_persists(governor, tmp_path):
    governor.update_equity(100_000)
    assert governor.update_equity(90_000) == ['flatten_all']
    reloaded = RiskGovernor(load_limits(), state_path=str(tmp_path / 'state.json'))
    assert not reloaded.approve(_proposal(), [], 95_000).approved
    with pytest.raises(PermissionError):
        reloaded.reset_halt(95_000, confirm='yes')
    reloaded.reset_halt(95_000, confirm='I understand the risk')
    assert reloaded.approve(_proposal(), [], 95_000).approved


def test_risk_is_clamped_and_derisked(governor):
    from datetime import datetime, timezone

    governor.update_equity(100_000, now=datetime(2026, 1, 5, tzinfo=timezone.utc))
    assert governor.approve(_proposal(risk_pct=5.0), [], 100_000).risk_pct == pytest.approx(1.0)
    # 7% DD a few weeks later (fresh day/week, so only the drawdown de-risk applies)
    governor.update_equity(93_000, now=datetime(2026, 2, 2, tzinfo=timezone.utc))
    d = governor.approve(_proposal(risk_pct=1.0), [], 93_000)
    assert d.approved and d.risk_pct < 1.0


def test_daily_loss_pauses_entries(governor):
    from datetime import datetime, timezone

    governor.update_equity(100_000, now=datetime(2026, 1, 5, 1, tzinfo=timezone.utc))
    governor.update_equity(97_500, now=datetime(2026, 1, 5, 9, tzinfo=timezone.utc))
    d = governor.approve(_proposal(), [], 97_500)
    assert not d.approved and 'daily' in d.reasons[0]


def test_position_and_exposure_caps(governor):
    governor.update_equity(100_000)
    opens = [OpenPosition(s, 'forex', 1, 0.5) for s in ('GBP_USD', 'AUD_USD', 'NZD_USD')]
    d = governor.approve(_proposal(), opens, 100_000)
    assert not d.approved and 'USD' in d.reasons[0]
    full = [OpenPosition(f'S{i}', 'stocks', 1, 0.5) for i in range(4)]
    assert not governor.approve(_proposal(symbol='SPY', asset_class='stocks', entry_price=500,
                                          stop_price=490), full, 100_000).approved


def test_leverage_cap_limits_notional(governor):
    governor.update_equity(100_000)
    d = governor.approve(_proposal(symbol='BTC/USD', asset_class='crypto', entry_price=60_000,
                                   stop_price=59_900, risk_pct=1.0), [], 100_000)
    assert d.notional <= 100_000 + 1e-6 and d.risk_pct < 1.0


def _valid_cfg():
    return {'strategies': [{
        'symbol': 'EUR_USD', 'asset_class': 'forex', 'strategy': 'donchian_trend', 'timeframe': '1d',
        'params': {'entry_n': 55, 'exit_n': 20, 'stop_atr': 3.0, 'trail_atr': 0.0, 'trend_ema': 0},
        'risk_pct': 0.5, 'allow_short': True,
    }]}


def test_config_validation_rejects_out_of_bounds():
    space, limits = load_space(), load_limits()
    validate_live_config(_valid_cfg(), space, limits)
    for mutate in (
        lambda e: e.update(risk_pct=3.0),
        lambda e: e['params'].update(entry_n=500),
        lambda e: e.update(symbol='BTC/USD'),
        lambda e: e.update(asset_class='crypto', symbol='BTC/USD', allow_short=True),
    ):
        cfg = copy.deepcopy(_valid_cfg())
        mutate(cfg['strategies'][0])
        with pytest.raises(ValueError):
            validate_live_config(cfg, space, limits)


def test_allocation_respects_probation_and_growth_cap():
    rng = np.random.default_rng(3)
    df = _df(100 + np.cumsum(rng.normal(0.08, 1, 2500)))
    space = load_space()
    cm = ConfigManager(space=space, limits=load_limits())
    res = walk_forward(df, 'SPY', 'stocks', 'donchian_trend', '1d', space['strategies']['donchian_trend'],
                       ZERO_COST, train_days=700, test_days=365, samples=8, allow_short=False)
    risks, _ = cm.allocate([res], previous=None)
    assert risks['SPY'] <= load_limits()['per_trade']['max_risk_pct'] * 0.5 + 1e-9
    prev = {'strategies': [{'symbol': 'SPY', 'strategy': 'donchian_trend', 'risk_pct': 0.2}]}
    risks2, _ = cm.allocate([res], previous=prev)
    assert risks2['SPY'] <= 0.2 * 1.5 + 1e-9
