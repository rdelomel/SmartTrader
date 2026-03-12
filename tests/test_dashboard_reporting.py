"""Tests for dashboard reporting metrics population."""

from src.monitoring.dashboard import create_dashboard_app
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
        return trades[:limit]


class FakeBroker:
    def get_account_balance(self):
        return {'total': 10050.0}


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

    # Risk metrics should be populated for dashboard risk panel.
    assert 'risk_metrics' in data
    assert 'drawdown_percent' in data['risk_metrics']
    assert 'peak_equity' in data['risk_metrics']
    assert 'current_equity' in data['risk_metrics']

    # Trades should be normalized with numeric fields expected by UI.
    assert len(data['trades']) == 3
    first_trade = data['trades'][0]
    assert isinstance(first_trade['quantity'], float)
    assert isinstance(first_trade['price'], float)
    assert isinstance(first_trade['pnl'], float)
