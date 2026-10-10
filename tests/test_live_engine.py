"""Live engine tests with a fake broker: parity with the backtester and safety behaviour."""

from datetime import timedelta, timezone

import numpy as np
import pandas as pd
import pytest
import yaml

from src.core.backtest import run_backtest
from src.core.costs import CostModel
from src.core.live_engine import LiveEngine, order_units, pnl_usd
from src.core.strategies import generate

PARAMS = {'entry_n': 20, 'exit_n': 10, 'stop_atr': 5.0, 'trail_atr': 0.0, 'trend_ema': 0}


def _df(n=600, seed=4):
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0.05, 1, n))
    o = np.concatenate([[c[0]], c[:-1]]) + rng.normal(0, 0.2, n)
    idx = pd.date_range('2020-01-01', periods=n, freq='D')
    return pd.DataFrame({'open': o, 'high': np.maximum(o, c) + 0.5, 'low': np.minimum(o, c) - 0.5,
                         'close': c, 'volume': 0}, index=idx)


class FakeBroker:
    def __init__(self, equity=100_000.0):
        self.equity = equity
        self.prices = {}
        self.positions = {}
        self.orders = []

    def get_account_balance(self):
        return {'equity': self.equity}

    def get_open_positions(self):
        return [{'symbol': s, 'side': 'buy' if p['d'] > 0 else 'sell', 'quantity': p['units'],
                 'entry_price': p['price']} for s, p in self.positions.items()]

    def get_current_price(self, symbol):
        return self.prices.get(symbol, 0.0)

    def place_order(self, symbol, side, quantity, order_type, stop_loss=None, **kw):
        d = 1 if side.value == 'buy' else -1
        price = self.prices[symbol]
        self.orders.append((symbol, d, quantity, price))
        self.positions[symbol] = {'d': d, 'units': quantity, 'price': price}
        return {'success': True, 'price': price}

    def close_position(self, symbol, side=None):
        return self.positions.pop(symbol, None) is not None

    def is_market_open(self, symbol=None):
        return True


class FakeProvider:
    def __init__(self, df):
        self.df = df

    def load(self, symbol, asset_class, timeframe, years):
        return self.df


class FakeStorage:
    def __init__(self):
        self.trades = {}

    def store_trade(self, t):
        self.trades[t['trade_id']] = dict(t)

    def update_trade(self, trade_id, upd):
        self.trades[trade_id].update(upd)


def _engine(tmp_path, df, broker, storage=None, params=PARAMS):
    cfg = {'version': 1, 'strategies': [{
        'symbol': 'SPY', 'asset_class': 'stocks', 'strategy': 'donchian_trend', 'timeframe': '1d',
        'params': params, 'risk_pct': 0.5, 'allow_short': False,
    }]}
    path = tmp_path / 'live_strategy.yaml'
    path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    return LiveEngine({'alpaca': broker}, storage=storage, provider=FakeProvider(df), config_path=str(path),
                      state_path=str(tmp_path / 'state.json'), event_log=str(tmp_path / 'events.jsonl'),
                      governor_dir=str(tmp_path))


def _at(ts):
    return ts.to_pydatetime().replace(tzinfo=timezone.utc)


def test_engine_trades_match_backtest(tmp_path):
    df = _df()
    broker, storage = FakeBroker(), FakeStorage()
    engine = _engine(tmp_path, df, broker, storage)
    for t in range(len(df) - 1):
        broker.prices['SPY'] = float(df['open'].iloc[t + 1])
        engine.run_once(_at(df.index[t] + timedelta(days=1, minutes=1)))

    bt = run_backtest(df, generate('donchian_trend', df, PARAMS), CostModel(0, 0, 0), allow_short=False).trades
    expected = {(r.entry_time, r.exit_time) for r in bt.itertuples()
                if r.reason == 'signal' and r.entry_time >= df.index[70]}
    live = {(pd.Timestamp(t['entry_time']) - timedelta(minutes=1),
             pd.Timestamp(t['exit_time']) - timedelta(minutes=1))
            for t in storage.trades.values() if t.get('status') == 'closed'}
    assert len(expected) >= 3
    assert expected <= live


def test_first_evaluation_never_enters(tmp_path):
    df = _df()
    sig = generate('donchian_trend', df, PARAMS)
    t = int(np.flatnonzero(sig.entry[100:] == 1)[0]) + 100
    broker = FakeBroker()
    broker.prices['SPY'] = float(df['open'].iloc[t + 1])
    engine = _engine(tmp_path, df, broker)
    engine.run_once(_at(df.index[t] + timedelta(days=1, minutes=1)))
    assert broker.orders == [] and engine.state.pending == {}


def test_drawdown_halt_flattens_everything(tmp_path):
    df = _df()
    broker = FakeBroker()
    broker.positions['AAPL'] = {'d': 1, 'units': 10, 'price': 200.0}
    engine = _engine(tmp_path, df, broker)
    now = _at(df.index[100])
    engine.run_once(now)
    broker.equity = 89_000.0
    engine.run_once(now + timedelta(minutes=1))
    assert broker.positions == {}
    assert engine.governors['alpaca'].state.halted


def test_position_needs_two_missing_checks_before_closed(tmp_path):
    df = _df()
    broker, storage = FakeBroker(), FakeStorage()
    engine = _engine(tmp_path, df, broker, storage)
    for t in range(len(df) - 1):
        broker.prices['SPY'] = float(df['open'].iloc[t + 1])
        engine.run_once(_at(df.index[t] + timedelta(days=1, minutes=1)))
        if engine.state.positions:
            break
    assert engine.state.positions
    broker.positions.clear()
    now = _at(df.index[t] + timedelta(days=1, minutes=2))
    engine.run_once(now)
    assert engine.state.positions
    engine.run_once(now + timedelta(minutes=1))
    assert not engine.state.positions


def test_units_and_pnl_conversion():
    assert order_units('EUR_USD', 'forex', 11_000, 1.10) == 10_000
    assert order_units('USD_JPY', 'forex', 10_000, 150.0) == 10_000
    assert order_units('AAPL', 'stocks', 1_000, 300.0) == 3
    assert pnl_usd('USD_JPY', 'forex', 1, 10_000, 150.0, 151.5) == pytest.approx(10_000 * 1.5 / 151.5)
    assert pnl_usd('EUR_USD', 'forex', -1, 10_000, 1.10, 1.09) == pytest.approx(100.0)
