"""Tests for dashboard reporting metrics population and manual close flows."""

from src.monitoring.dashboard import create_dashboard_app, _generate_unified_dashboard_html
from fastapi.testclient import TestClient


class FakeStorage:
    def __init__(self):
        self.trades = [
            {
                'trade_id': 't1',
                'symbol': 'BTC/USD',
                'side': 'buy',
                'quantity': 0.1,
                'entry_price': 100.0,
                'exit_price': 110.0,
                'pnl': 10.0,
                'status': 'closed',
                'entry_time': '2024-01-01T00:00:00',
                'exit_time': '2024-01-01T01:00:00',
            },
            {
                'trade_id': 't2',
                'symbol': 'ETH/USD',
                'side': 'sell',
                'quantity': 1.0,
                'entry_price': 200.0,
                'exit_price': 205.0,
                'pnl': -5.0,
                'status': 'closed',
                'entry_time': '2024-01-02T00:00:00',
                'exit_time': '2024-01-02T01:00:00',
            },
            {
                'trade_id': 't3',
                'symbol': 'SOL/USD',
                'side': 'buy',
                'quantity': 2.0,
                'entry_price': 50.0,
                'pnl': 3.0,
                'status': 'open',
                'entry_time': '2024-01-03T00:00:00',
            },
        ]

    def get_all_trades(self, limit=500):
        return self.trades[:limit]

    def get_open_trades(self, symbol=None):
        rows = [t for t in self.trades if t.get('status') == 'open']
        if symbol:
            return [t for t in rows if t.get('symbol') == symbol]
        return rows

    def update_trade(self, trade_id, update_data):
        for t in self.trades:
            if str(t.get('trade_id')) == str(trade_id):
                t.update(update_data)
                return True
        return False

    def delete_trade(self, trade_id):
        before = len(self.trades)
        self.trades = [t for t in self.trades if str(t.get('trade_id')) != str(trade_id)]
        return len(self.trades) < before


class FakeBroker:
    def __init__(self):
        self.closed = []

    def get_account_balance(self):
        return {'total': 10050.0}

    def get_open_positions(self):
        return [
            {
                'symbol': 'SOLUSD',
                'side': 'buy',
                'quantity': 2.0,
                'entry_price': 50.0,
            }
        ]

    def close_position(self, symbol, side=None):
        self.closed.append((symbol, side.value if hasattr(side, 'value') else side))
        return True

    def get_current_price(self, symbol):
        return 52.0


def test_dashboard_api_populates_performance_and_risk_metrics():
    app = create_dashboard_app(storage=FakeStorage(), brokers={'alpaca': FakeBroker()})
    client = TestClient(app)

    response = client.get('/api/dashboard')
    assert response.status_code == 200

    data = response.json()

    # Performance metrics should be populated from reporting + live balance.
    assert 'performance' in data
    assert data['performance']['total_trades'] == 2
    assert data['performance']['realized_pnl'] == 5.0
    assert data['performance']['win_rate'] == 50.0
    assert data['performance']['loss_rate'] == 50.0
    assert 'profit_factor' in data['performance']
    assert 'expectancy' in data['performance']

    # Risk metrics should be populated for dashboard risk panel.
    assert 'risk_metrics' in data
    assert 'drawdown_percent' in data['risk_metrics']
    assert 'peak_equity' in data['risk_metrics']
    assert 'current_equity' in data['risk_metrics']
    assert 'data_quality' in data
    assert 'integrity_score' in data['data_quality']

    # Recent Trades lists closed trades only; the open one appears under positions.
    assert len(data['trades']) == 2
    first_trade = data['trades'][0]
    assert isinstance(first_trade['quantity'], float)
    assert isinstance(first_trade['price'], float)
    assert isinstance(first_trade['pnl'], float)


def test_reconcile_open_trades_endpoint_reports_mismatches():
    storage = FakeStorage()
    app = create_dashboard_app(storage=storage, brokers={'alpaca': FakeBroker()})
    client = TestClient(app)

    response = client.get('/api/reconcile_open_trades')
    assert response.status_code == 200
    data = response.json()

    assert data['success'] is True
    assert data['local_open_count'] >= 1
    assert data['broker_open_count'] >= 1
    assert isinstance(data['stale_local_trades'], list)
    assert isinstance(data['broker_only_positions'], list)


def test_close_position_accepts_symbol_when_trade_id_missing():
    storage = FakeStorage()
    broker = FakeBroker()
    app = create_dashboard_app(storage=storage, brokers={'alpaca': broker})
    client = TestClient(app)

    # Use symbol in path (mirrors UI fallback when trade_id is absent).
    response = client.post('/api/close_position/SOL_USD')
    assert response.status_code == 200
    data = response.json()

    assert data['success'] is True
    assert data['method'] == 'broker_api'
    assert broker.closed  # broker close call executed


def test_report_endpoint_returns_myfxbook_style_sections():
    app = create_dashboard_app(storage=FakeStorage(), brokers={'alpaca': FakeBroker()})
    client = TestClient(app)

    response = client.get('/api/report')
    assert response.status_code == 200
    payload = response.json()

    assert payload['success'] is True
    assert 'report' in payload
    assert 'general' in payload['report']
    assert 'periods' in payload['report']
    assert 'advanced' in payload['report']
    assert 'equity_curve' in payload['report']
    assert 'monthly_analytics' in payload['report']
    assert 'data_quality' in payload


def test_unified_dashboard_html_handles_malformed_report_payload():
    malformed = {
        'equity_curve': [None, {'equity': None}, {'date': '2024-01-01'}],
        'monthly_analytics': [None, {'month': '2024-01'}],
        'general': {'total_trades': 1},
        'periods': {},
        'advanced': {}
    }

    html = _generate_unified_dashboard_html(malformed)
    assert isinstance(html, str)
    assert 'SmartTrader - Unified Dashboard' in html


def test_dashboard_position_includes_broker_verification_flag():
    app = create_dashboard_app(storage=FakeStorage(), brokers={'alpaca': FakeBroker()})
    client = TestClient(app)

    response = client.get('/api/dashboard')
    assert response.status_code == 200
    data = response.json()

    # Open SOL/USD trade should match FakeBroker SOLUSD position and be marked verified.
    sol_positions = [p for p in data.get('positions', []) if p.get('symbol') == 'SOL/USD']
    assert sol_positions, 'Expected SOL/USD open position in dashboard payload'
    assert sol_positions[0].get('verified_at_broker') is True
