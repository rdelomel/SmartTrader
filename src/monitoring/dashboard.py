"""FastAPI-based monitoring dashboard"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Dict, List, Optional
import json
from datetime import datetime
import asyncio
from ..data.brokers.base_broker import OrderSide
from .reporting import TradingReporter


def _serialize_datetime(obj):
    """Convert datetime objects to ISO strings for JSON serialization"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: _serialize_datetime(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_serialize_datetime(item) for item in obj]
    return obj


def create_dashboard_app(storage=None, brokers=None, initial_equity: Optional[float] = None) -> FastAPI:
    """Create FastAPI dashboard application"""
    app = FastAPI(title="SmartTrader Dashboard")
    @app.get("/health")
    async def health():
        return {"status": "ok"}
    
    # Store dashboard data
    dashboard_data = {
        'equity': 10000.0,
        'balance': 10000.0,
        'positions': [],
        'trades': [],
        'performance': {},
        'risk_metrics': {},
        'signals': []
    }
    
    # Store storage and brokers references for API endpoints
    app.storage = storage
    app.brokers = brokers or {}
    
    # Initialize reporter
    reporter = TradingReporter(storage) if storage else None

    def _normalize_symbol(symbol: Optional[str]) -> str:
        if not symbol:
            return ''
        return str(symbol).replace('_', '/').replace('-', '/').upper()

    def _symbols_match(a: Optional[str], b: Optional[str]) -> bool:
        na = _normalize_symbol(a).replace('/', '')
        nb = _normalize_symbol(b).replace('/', '')
        return bool(na and nb and na == nb)

    def _collect_broker_open_positions() -> List[Dict]:
        positions: List[Dict] = []
        if not app.brokers:
            return positions
        for broker_name, broker in app.brokers.items():
            try:
                for pos in broker.get_open_positions() or []:
                    positions.append({
                        'broker': broker_name,
                        'symbol': _normalize_symbol(pos.get('symbol', '')),
                        'side': str(pos.get('side', 'buy')).lower(),
                        'raw': pos
                    })
            except Exception as e:
                print(f"Error collecting positions from broker {broker_name}: {e}")
        return positions

    def _build_trusted_dashboard_snapshot() -> Dict:
        """Build a broker-reconciled, reporting-backed dashboard snapshot."""

        def _normalize_trade(trade: Dict) -> Dict:
            quantity = trade.get('quantity') or trade.get('filled_qty') or 0.0
            entry_price = trade.get('entry_price') or trade.get('price') or 0.0
            exit_price = trade.get('exit_price') or trade.get('price') or entry_price
            status = (trade.get('status') or '').lower()
            display_price = exit_price if status == 'closed' else entry_price
            return {
                'trade_id': trade.get('trade_id') or trade.get('order_id') or '',
                'symbol': trade.get('symbol', ''),
                'side': trade.get('side', ''),
                'quantity': float(quantity or 0.0),
                'price': float(display_price or 0.0),
                'pnl': float(trade.get('pnl', 0.0) or 0.0),
                'time': trade.get('exit_time') or trade.get('entry_time') or datetime.now().isoformat(),
                'status': status,
                'entry_price': float(entry_price or 0.0),
                'exit_price': float(exit_price or 0.0)
            }

        all_trades = app.storage.get_all_trades(limit=1000) if (app.storage and hasattr(app.storage, 'get_all_trades')) else []
        local_open = app.storage.get_open_trades() if (app.storage and hasattr(app.storage, 'get_open_trades')) else []
        closed_trades = [t for t in all_trades if t.get('status') == 'closed']
        broker_open = _collect_broker_open_positions()

        # Refresh balance from brokers if possible; fallback to cached value.
        total_balance = 0.0
        if app.brokers:
            for broker in app.brokers.values():
                try:
                    balance_info = broker.get_account_balance()
                    total_balance += float(balance_info.get('equity', 0.0) or balance_info.get('portfolio_value', 0.0) or balance_info.get('balance', 0.0))
                except Exception:
                    pass
        if total_balance <= 0:
            total_balance = float(dashboard_data.get('balance', 0.0) or 0.0)

        # Build broker-verified position view from local open trades.
        positions = []
        matched_count = 0
        stale_local = []
        for t in local_open:
            t_symbol = _normalize_symbol(t.get('symbol'))
            t_side = str(t.get('side', 'buy')).lower()
            broker_match = next(
                (p for p in broker_open if _symbols_match(p.get('symbol'), t_symbol) and p.get('side') == t_side),
                None
            )
            verified = broker_match is not None
            if verified:
                matched_count += 1
            else:
                stale_local.append({'trade_id': t.get('trade_id'), 'symbol': t.get('symbol'), 'side': t.get('side')})

            entry_price = float(t.get('entry_price', 0.0) or 0.0)
            current_price = entry_price
            pnl = float(t.get('pnl', 0.0) or 0.0)

            if broker_match:
                raw = broker_match.get('raw', {})
                # Prefer broker's entry price (more accurate - OANDA averagePrice)
                broker_entry = float(raw.get('entry_price', raw.get('averagePrice', entry_price)) or entry_price)
                if broker_entry > 0:
                    entry_price = broker_entry
                # Use broker's current/market price
                broker_current = float(raw.get('current_price', raw.get('market_price', 0)) or 0)
                if broker_current > 0:
                    current_price = broker_current
                # Use broker's unrealized P&L directly (most accurate - includes spread/swap)
                broker_pnl = raw.get('unrealized_pnl')
                if broker_pnl is not None:
                    pnl = float(broker_pnl)
                elif entry_price > 0 and current_price > 0:
                    quantity = float(t.get('quantity', 0.0) or 0.0)
                    pnl = (current_price - entry_price) * quantity if t_side == 'buy' else (entry_price - current_price) * quantity
            elif entry_price > 0 and current_price > 0:
                quantity = float(t.get('quantity', 0.0) or 0.0)
                pnl = (current_price - entry_price) * quantity if t_side == 'buy' else (entry_price - current_price) * quantity

            quantity = float(t.get('quantity', 0.0) or 0.0)

            positions.append({
                'trade_id': t.get('trade_id'),
                'symbol': t.get('symbol'),
                'side': t.get('side', 'buy'),
                'quantity': quantity,
                'entry_price': entry_price,
                'current_price': current_price,
                'stop_loss': t.get('stop_loss'),
                'take_profit': t.get('take_profit'),
                'pnl': pnl,
                'strategy': t.get('strategy', 'smarttrader'),
                'verified_at_broker': verified,
                'broker': broker_match.get('broker') if broker_match else None
            })

        broker_only = []
        for p in broker_open:
            exists = any(_symbols_match(t.get('symbol'), p.get('symbol')) and str(t.get('side', 'buy')).lower() == p.get('side') for t in local_open)
            if not exists:
                broker_only.append({'broker': p.get('broker'), 'symbol': p.get('symbol'), 'side': p.get('side')})

        report_data = reporter.generate_report() if reporter else None
        general = report_data.get('general', {}) if report_data else {}
        advanced = report_data.get('advanced', {}) if report_data else {}
        periods = report_data.get('periods', {}) if report_data else {}
        monthly = report_data.get('monthly_analytics', []) if report_data else []
        equity_curve = report_data.get('equity_curve', []) if report_data else []

        realized_pnl = float(general.get('net_profit', sum((t.get('pnl', 0) or 0) for t in closed_trades)) or 0.0)
        unrealized_pnl = float(sum((p.get('pnl', 0) or 0) for p in positions) or 0.0)
        equity = total_balance + unrealized_pnl
        # Derive initial equity by reversing realized P&L from current balance
        # (avoids reset-on-restart bug; works without a separate stored starting balance)
        _derived_initial = total_balance - realized_pnl
        initial_eq = float(_derived_initial if _derived_initial > 1000.0 else
                           ((reporter.initial_equity if reporter else 0) or total_balance or 10000.0))
        total_return = ((equity - initial_eq) / initial_eq * 100) if initial_eq > 0 else 0.0

        peak_equity = max([point.get('equity', 0.0) for point in equity_curve], default=initial_eq)
        current_drawdown = max(0.0, min(100.0, ((peak_equity - equity) / peak_equity * 100))) if peak_equity > 0 else 0.0

        local_open_count = len(local_open)
        broker_open_count = len(broker_open)
        stale_local_count = len(stale_local)
        broker_only_count = len(broker_only)
        mismatch_total = stale_local_count + broker_only_count
        denom = max(local_open_count + broker_open_count, 1)
        integrity_score = max(0.0, 100.0 - (mismatch_total / denom) * 100.0)

        normalized_recent_trades = [_normalize_trade(t) for t in sorted(
            all_trades,
            key=lambda x: x.get('exit_time') or x.get('entry_time') or '',
            reverse=True
        )[:50]]

        return {
            'equity': float(equity),
            'balance': float(total_balance),
            'positions': positions,
            'trades': normalized_recent_trades,
            'performance': {
                'total_return': float(total_return),
                'realized_pnl': float(realized_pnl),
                'unrealized_pnl': float(unrealized_pnl),
                'total_trades': int(general.get('total_trades', len(closed_trades)) or 0),
                'win_rate': float(general.get('win_rate', 0.0) or 0.0),
                'loss_rate': float(general.get('loss_rate', 0.0) or 0.0),
                'sharpe_ratio': float(advanced.get('sharpe_ratio', 0.0) or 0.0),
                'sortino_ratio': float(advanced.get('sortino_ratio', 0.0) or 0.0),
                'profit_factor': float(general.get('profit_factor', 0.0) or 0.0),
                'expectancy': float(advanced.get('expectancy', 0.0) or 0.0),
                'average_trade': float(advanced.get('average_trade', 0.0) or 0.0),
                'initial_equity': float(initial_eq)
            },
            'risk_metrics': {
                'drawdown_percent': float(current_drawdown),
                'peak_equity': float(peak_equity),
                'current_equity': float(equity),
                'max_drawdown_percent': float(advanced.get('max_drawdown', 0.0) or 0.0),
                'standard_deviation': float(advanced.get('standard_deviation', 0.0) or 0.0)
            },
            'data_quality': {
                'integrity_score': float(integrity_score),
                'matched_open_trades': int(matched_count),
                'local_open_count': int(local_open_count),
                'broker_open_count': int(broker_open_count),
                'stale_local_count': int(stale_local_count),
                'broker_only_count': int(broker_only_count),
                'stale_local_trades': stale_local,
                'broker_only_positions': broker_only,
                'last_reconciled_at': datetime.now().isoformat()
            },
            'report': {
                'general': general,
                'periods': periods,
                'advanced': advanced,
                'equity_curve': equity_curve[-300:],
                'monthly_analytics': monthly[-36:]
            }
        }

    
    @app.get("/")
    async def get_dashboard():
        """Serve unified dashboard HTML with tabs"""
        try:
            # Try to load initial data from database if dashboard is empty
            if (dashboard_data.get('equity', 0) == 10000.0 and
                dashboard_data.get('balance', 0) == 10000.0 and
                len(dashboard_data.get('positions', [])) == 0 and
                storage):
                try:
                    open_trades = storage.get_open_trades()
                    if open_trades or (hasattr(storage, 'get_all_trades') and storage.get_all_trades(limit=1)):
                        pass
                except Exception:
                    pass

            report_data = None
            if reporter:
                try:
                    report_data = reporter.generate_report()
                except Exception:
                    report_data = None

            html = _generate_unified_dashboard_html(report_data)
            return HTMLResponse(content=html)
        except Exception as e:
            # Never break root dashboard endpoint; render minimal fallback instead.
            fallback_html = f"""
            <html><body style='font-family:Arial;padding:20px'>
              <h1>SmartTrader Dashboard</h1>
              <p>Dashboard rendering fallback active.</p>
              <p>Error: {str(e)}</p>
            </body></html>
            """
            return HTMLResponse(content=fallback_html, status_code=200)
    
    @app.get("/dashboard-old")
    async def get_old_dashboard():
        """Legacy dashboard endpoint (for backward compatibility)"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>SmartTrader Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; }
                .card { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }
                .metric { text-align: center; }
                .metric-value { font-size: 24px; font-weight: bold; color: #333; }
                .metric-label { font-size: 12px; color: #666; margin-top: 5px; }
                .positive { color: #28a745; }
                .negative { color: #dc3545; }
                table { width: 100%; border-collapse: collapse; }
                th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background: #f8f9fa; }
                .btn-close { background: #dc3545; color: white; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer; }
                .btn-close:hover { background: #c82333; }
                .btn-close:disabled { background: #ccc; cursor: not-allowed; }
            
/* ============================================================
   MOBILE-RESPONSIVE  (injected)
   ============================================================ */

/* — Topbar — */
@media(max-width:600px){{
  .topbar{{padding:0 12px;height:50px;gap:8px}}
  .logo{{font-size:14px}}
  .logo-glyph{{width:26px;height:26px;font-size:13px}}
  #lastUpdate{{display:none}}
  .btn-sm{{padding:4px 10px;font-size:11px}}
  .live-pill{{padding:2px 7px;font-size:10px}}
}}

/* — Main padding — */
@media(max-width:640px){{
  .main{{padding:12px 10px}}
}}

/* — Tab nav: scrollable row, no wrap — */
.tab-nav{{overflow-x:auto;-webkit-overflow-scrolling:touch;white-space:nowrap}}
@media(max-width:600px){{
  .tab-nav{{width:100%;border-radius:8px}}
  .tab-btn{{padding:6px 14px;font-size:12px}}
}}

/* — Metric cards — */
@media(max-width:640px){{
  .mc{{padding:10px 12px}}
  .mc-val{{font-size:17px}}
  .mc-lbl{{font-size:9px}}
}}
@media(max-width:360px){{
  .mrow{{grid-template-columns:1fr}}
}}

/* — Cards — */
@media(max-width:640px){{
  .card{{padding:14px 12px;margin-bottom:14px}}
  .card-hdr{{margin-bottom:12px;font-size:11px}}
}}

/* — Tables: always scrollable, minimum widths keep columns readable — */
.tbl-wrap{{
  overflow-x:auto;
  -webkit-overflow-scrolling:touch;
  /* fade-right hint so user knows it scrolls */
  background:
    linear-gradient(to right,var(--surface) 0%,transparent 5%),
    linear-gradient(to left, var(--surface) 0%,transparent 5%)
    right center / 40px 100% no-repeat;
}}
/* min widths prevent column crush */
#posBody ~ * , .card .tbl-wrap table{{min-width:540px}}
@media(max-width:768px){{
  table{{font-size:12px}}
  thead th{{padding:7px 10px;font-size:9px}}
  tbody td{{padding:8px 10px}}
  /* hide less-critical columns in positions table on mobile */
  .pos-col-qty{{display:none}}
  .pos-col-tp{{display:none}}
  /* hide less-critical columns in trades table on mobile */
  .trd-col-qty{{display:none}}
}}

/* — Reports grid: 2 col on mobile, 1 col on very small — */
@media(max-width:480px){{
  .rpt-grid{{grid-template-columns:1fr 1fr}}
}}
@media(max-width:360px){{
  .rpt-grid{{grid-template-columns:1fr}}
}}

/* — Charts: shorter on mobile — */
@media(max-width:640px){{
  .chart-wrap{{height:160px}}
  .chart-wrap2{{height:190px}}
}}

/* — Close position button: full width touch target on mobile — */
@media(max-width:640px){{
  .btn-close-pos{{padding:6px 10px;font-size:11px}}
}}

/* — Safe area for notched phones — */
@media(max-width:640px){{
  .main{{padding-bottom:max(12px,env(safe-area-inset-bottom))}}
}}
</style>
        </head>
        <body>
            <div class="container">
                <h1>SmartTrader Dashboard</h1>
                
                <div class="card">
                    <h2>Performance Metrics</h2>
                    <div class="metrics" id="metrics"></div>
                </div>
                
                <div class="card">
                    <h2>Open Positions</h2>
                    <table id="positions">
                        <thead>
                            <tr>
                                <th>Symbol</th>
                                <th>Side</th>
                                <th>Quantity</th>
                                <th>Entry Price</th>
                                <th>Current Price</th>
                                <th>Stop Loss</th>
                                <th>Take Profit</th>
                                <th>P&L</th>
                                <th>Action</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
                
                <div class="card">
                    <h2>Recent Trades</h2>
                    <table id="trades">
                        <thead>
                            <tr>
                                <th>Time</th>
                                <th>Symbol</th>
                                <th>Side</th>
                                <th>Quantity</th>
                                <th>Price</th>
                                <th>P&L</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
                
                <div class="card">
                    <h2>Risk Metrics</h2>
                    <div id="risk-metrics"></div>
                </div>
            </div>
            
            <script>
                const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsHost = window.location.host;
                const ws = new WebSocket(`${wsProtocol}//${wsHost}/ws`);
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    updateDashboard(data);
                };
                
                async function closePosition(tradeId, symbol, side) {
                    if (!confirm(`Are you sure you want to close/delete this position?\\n\\nSymbol: ${symbol}\\nSide: ${side}\\n\\nIf the position exists in Alpaca/OANDA, it will be closed via API.\\nIf it's a phantom position, it will be deleted from the database.`)) {
                        return;
                    }
                    
                    // Find the button that was clicked
                    const buttons = document.querySelectorAll('.btn-close');
                    let button = null;
                    for (let btn of buttons) {
                        if (btn.getAttribute('onclick').includes(tradeId)) {
                            button = btn;
                            break;
                        }
                    }
                    
                    if (button) {
                        button.disabled = true;
                        button.textContent = 'Processing...';
                    }
                    
                    try {
                        const response = await fetch(`/api/close_position/${tradeId}`, {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            }
                        });
                        
                        const result = await response.json();
                        
                        if (result.success) {
                            const action = result.method === 'broker_api' ? 'closed' : 'deleted';
                            alert(result.message || `Position ${action} successfully`);
                            // Position will be removed on next dashboard update (within 1 second)
                        } else {
                            alert('Error: ' + (result.error || 'Failed to close/delete position'));
                            if (button) {
                                button.disabled = false;
                                // Restore original button text
                                const originalText = button.getAttribute('title')?.includes('phantom') ? 'Delete' : 'Close';
                                button.textContent = originalText;
                            }
                        }
                    } catch (error) {
                        alert('Error: ' + error.message);
                        if (button) {
                            button.disabled = false;
                            // Restore original button text
                            const originalText = button.getAttribute('title')?.includes('phantom') ? 'Delete' : 'Close';
                            button.textContent = originalText;
                        }
                    }
                }
                
                function updateDashboard(data) {
                    // Update metrics
                    const metricsDiv = document.getElementById('metrics');
                    metricsDiv.innerHTML = `
                        <div class="metric">
                            <div class="metric-value">$${data.equity.toFixed(2)}</div>
                            <div class="metric-label">Equity</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">$${data.balance.toFixed(2)}</div>
                            <div class="metric-label">Balance</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value ${data.performance.total_return >= 0 ? 'positive' : 'negative'}">
                                ${data.performance.total_return?.toFixed(2) || 0}%
                            </div>
                            <div class="metric-label">Total Return</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value ${data.performance.realized_pnl >= 0 ? 'positive' : 'negative'}">
                                $${data.performance.realized_pnl?.toFixed(2) || 0}
                            </div>
                            <div class="metric-label">Realized P&L</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value ${data.performance.unrealized_pnl >= 0 ? 'positive' : 'negative'}">
                                $${data.performance.unrealized_pnl?.toFixed(2) || 0}
                            </div>
                            <div class="metric-label">Unrealized P&L</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${data.performance.sharpe_ratio?.toFixed(2) || 0}</div>
                            <div class="metric-label">Sharpe Ratio</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value positive">${data.performance.win_rate?.toFixed(1) || 0}%</div>
                            <div class="metric-label">Win Rate</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value negative">${data.performance.loss_rate?.toFixed(1) || 0}%</div>
                            <div class="metric-label">Loss Rate</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${data.performance.total_trades || 0}</div>
                            <div class="metric-label">Total Trades</div>
                        </div>
                    `;
                    
                    // Update positions
                    const positionsBody = document.querySelector('#positions tbody');
                    positionsBody.innerHTML = data.positions.map(pos => {
                        // Determine if this is a phantom position (local only) or broker position
                        const isPhantom = !pos.strategy || pos.strategy === 'smarttrader' || (pos.strategy && !pos.strategy.includes('_sync') && pos.strategy !== 'unknown');
                        const buttonText = isPhantom ? 'Delete' : 'Close';
                        const buttonTitle = isPhantom ? 'Delete this phantom position (local only)' : 'Close position via broker API';
                        
                        return `
                        <tr>
                            <td>${pos.symbol}</td>
                            <td>${pos.side}</td>
                            <td>${pos.quantity}</td>
                            <td>$${pos.entry_price}</td>
                            <td>$${pos.current_price || pos.entry_price}</td>
                            <td class="${pos.pnl >= 0 ? 'positive' : 'negative'}">$${pos.pnl.toFixed(2)}</td>
                            <td>
                                <button class="btn-close" onclick="closePosition('${pos.trade_id || pos.symbol.replace('/', '_')}', '${pos.symbol}', '${pos.side}')" title="${buttonTitle}">
                                    ${buttonText}
                                </button>
                            </td>
                        </tr>
                    `;
                    }).join('');
                    
                    // Update trades (show closed trades with realized P&L)
                    const tradesBody = document.querySelector('#trades tbody');
                    if (data.trades && data.trades.length > 0) {
                        tradesBody.innerHTML = data.trades.slice(0, 10).map(trade => `
                            <tr>
                                <td>${new Date(trade.time || Date.now()).toLocaleString()}</td>
                                <td>${trade.symbol}</td>
                                <td>${trade.side}</td>
                                <td>${trade.quantity.toFixed(4)}</td>
                                <td>$${trade.price.toFixed(2)}</td>
                                <td class="${trade.pnl >= 0 ? 'positive' : 'negative'}">$${trade.pnl.toFixed(2)}</td>
                            </tr>
                        `).join('');
                    } else {
                        tradesBody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #999;">No closed trades yet</td></tr>';
                    }
                }
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html)
    
    @app.get("/api/status")
    async def get_status():
        """Get current status"""
        return dashboard_data
    
    @app.get("/api/dashboard")
    async def get_dashboard_api():
        """Get trusted dashboard data as JSON, reconciled with broker positions."""
        try:
            snapshot = _build_trusted_dashboard_snapshot()
            dashboard_data.update(snapshot)
        except Exception as e:
            print(f"Error loading dashboard data from database: {e}")

        return _serialize_datetime(dashboard_data)

    @app.get("/api/reconcile_open_trades")
    async def reconcile_open_trades():
        """Reconcile local open trades with broker open positions."""
        if not app.storage:
            return {
                'success': False,
                'error': 'Storage not available',
                'stale_local_trades': [],
                'broker_only_positions': []
            }

        snapshot = _build_trusted_dashboard_snapshot()
        dq = snapshot.get('data_quality', {})
        return {
            'success': True,
            'local_open_count': dq.get('local_open_count', 0),
            'broker_open_count': dq.get('broker_open_count', 0),
            'matched_open_trades': dq.get('matched_open_trades', 0),
            'integrity_score': dq.get('integrity_score', 0.0),
            'stale_local_trades': dq.get('stale_local_trades', []),
            'broker_only_positions': dq.get('broker_only_positions', []),
            'last_reconciled_at': dq.get('last_reconciled_at')
        }

    @app.get("/api/report")
    async def get_report_api():
        """Return report payload used by MyFxBook-style analytics tabs."""
        snapshot = _build_trusted_dashboard_snapshot()
        return _serialize_datetime({
            'success': True,
            'report': snapshot.get('report', {}),
            'performance': snapshot.get('performance', {}),
            'risk_metrics': snapshot.get('risk_metrics', {}),
            'data_quality': snapshot.get('data_quality', {})
        })

    @app.post("/api/close_position/{trade_id}")
    async def close_position(trade_id: str):
        """
        Close/delete a position by trade identifier.
        - If position exists in broker (Alpaca/OANDA), close it via API
        - If position is phantom (local only), delete from database
        - If trade_id is missing, supports symbol-based fallback
        """
        if not app.storage:
            return {"success": False, "error": "Storage not available"}

        try:
            open_trades = app.storage.get_open_trades() if hasattr(app.storage, 'get_open_trades') else []
            all_trades = app.storage.get_all_trades(limit=500) if hasattr(app.storage, 'get_all_trades') else []

            trade = None
            normalized_input = _normalize_symbol(trade_id)

            # 1) Primary lookup by trade/order id
            if trade_id:
                trade = next((t for t in open_trades if str(t.get('trade_id', '')) == str(trade_id)), None)
                if not trade:
                    trade = next(
                        (t for t in all_trades if str(t.get('trade_id', '')) == str(trade_id) or str(t.get('order_id', '')) == str(trade_id)),
                        None,
                    )

            # 2) Fallback lookup by symbol when UI had no trade_id
            if not trade and normalized_input:
                symbol_matches = [
                    t for t in open_trades
                    if _symbols_match(t.get('symbol'), normalized_input)
                ]
                if symbol_matches:
                    trade = symbol_matches[0]

            # 3) If still missing, attempt reconciliation and close directly from broker positions by symbol
            if not trade:
                broker_positions = _collect_broker_open_positions()
                symbol_guess = normalized_input if normalized_input else None
                if symbol_guess:
                    broker_match = next((p for p in broker_positions if _symbols_match(p.get('symbol'), symbol_guess)), None)
                    if broker_match:
                        # Build synthetic trade shape so close flow can continue.
                        trade = {
                            'trade_id': trade_id or symbol_guess,
                            'symbol': symbol_guess,
                            'side': broker_match.get('side', 'buy'),
                            'strategy': f"{broker_match.get('broker', '')}_sync"
                        }

            if not trade:
                return {
                    "success": False,
                    "error": "Trade not found",
                    "hint": "Run /api/reconcile_open_trades to inspect local/broker mismatches"
                }

            symbol = trade.get('symbol')
            side = str(trade.get('side', 'buy')).lower()
            strategy = trade.get('strategy') or ''

            # Check if position exists in any broker
            position_found_in_broker = False
            broker_used = None

            if strategy and '_sync' in strategy:
                broker_name = strategy.replace('_sync', '')
                broker = app.brokers.get(broker_name)
                if broker:
                    try:
                        for pos in broker.get_open_positions() or []:
                            if _symbols_match(pos.get('symbol'), symbol) and str(pos.get('side', 'buy')).lower() == side:
                                position_found_in_broker = True
                                broker_used = broker
                                break
                    except Exception as e:
                        print(f"Error checking broker {broker_name} for position: {e}")

            if not position_found_in_broker and app.brokers:
                for broker_name, broker in app.brokers.items():
                    try:
                        for pos in broker.get_open_positions() or []:
                            if _symbols_match(pos.get('symbol'), symbol) and str(pos.get('side', 'buy')).lower() == side:
                                position_found_in_broker = True
                                broker_used = broker
                                break
                        if position_found_in_broker:
                            break
                    except Exception as e:
                        print(f"Error checking broker {broker_name} for position: {e}")
                        continue

            if position_found_in_broker and broker_used:
                try:
                    order_side = OrderSide.BUY if side == 'buy' else OrderSide.SELL
                    closed = broker_used.close_position(symbol, order_side)

                    if closed:
                        try:
                            current_price = broker_used.get_current_price(symbol)
                        except Exception:
                            current_price = trade.get('entry_price', 0)

                        entry_price = float(trade.get('entry_price', 0) or 0)
                        quantity = float(trade.get('quantity', 0) or 0)
                        if side == 'buy':
                            final_pnl = (current_price - entry_price) * quantity
                        else:
                            final_pnl = (entry_price - current_price) * quantity

                        # Update local DB record if known, otherwise best-effort symbol update.
                        target_trade_id = trade.get('trade_id')
                        if target_trade_id:
                            app.storage.update_trade(str(target_trade_id), {
                                'status': 'closed',
                                'exit_price': current_price,
                                'exit_time': datetime.now(),
                                'pnl': final_pnl
                            })

                        return {
                            "success": True,
                            "message": f"Position {symbol} {side} closed successfully via broker API",
                            "method": "broker_api"
                        }

                    return {"success": False, "error": "Failed to close position via broker API"}
                except Exception as e:
                    return {"success": False, "error": f"Error closing position via broker: {str(e)}"}

            # Not at broker -> delete local phantom trade.
            target_trade_id = str(trade.get('trade_id') or trade_id)
            success = app.storage.delete_trade(target_trade_id) if target_trade_id else False
            if success:
                return {
                    "success": True,
                    "message": f"Phantom position {symbol} {side} deleted from local database",
                    "method": "local_delete"
                }
            return {"success": False, "error": "Failed to delete trade"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @app.post("/api/purge_zero_trades")
    async def purge_zero_trades():
        """Delete all $0-P&L broker-sync imported trades immediately."""
        if not app.storage:
            return {"success": False, "error": "Storage not available"}
        try:
            all_trades = app.storage.get_all_trades(limit=5000)
            to_purge = [
                t for t in all_trades
                if abs(float(t.get('pnl', 0) or 0)) < 0.001
                and (t.get('status') or '').lower() not in ('open', 'pending', 'partial')
            ]
            count = len(to_purge)
            for t in to_purge:
                app.storage.delete_trade(t['trade_id'])
            return {"success": True, "deleted": count, "message": f"Deleted {count} SYNCED $0 trades"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time updates"""
        await websocket.accept()
        try:
            while True:
                # Serialize datetime objects before sending
                serialized_data = _serialize_datetime(dashboard_data)
                await websocket.send_json(serialized_data)
                await asyncio.sleep(1)  # Update every second
        except WebSocketDisconnect:
            pass
    
    # Store update function reference
    app.update_dashboard_data = lambda data: dashboard_data.update(data)
    
    # Initialize reporter for reports page
    # Use provided initial_equity or try to get from brokers
    reporter_initial_equity = initial_equity
    if not reporter_initial_equity and storage and brokers:
        try:
            # Get current balance as approximation of initial equity
            # (In a real system, you'd track this separately)
            total_balance = 0.0
            for broker in brokers.values():
                try:
                    balance_info = broker.get_account_balance()
                    total_balance += (balance_info.get('equity', 0.0) or balance_info.get('portfolio_value', 0.0) or balance_info.get('balance', 0.0))
                except:
                    pass
            # Use a reasonable default if we can't get balance
            reporter_initial_equity = total_balance if total_balance > 0 else 10000.0
        except:
            reporter_initial_equity = 10000.0
    
    reporter = TradingReporter(storage, reporter_initial_equity) if storage else None
    
    @app.get("/reports")
    async def get_reports():
        """Serve comprehensive trading reports page"""
        if not reporter:
            return HTMLResponse(content="<html><body><h1>Reports not available - storage not initialized</h1></body></html>")
        
        report_data = reporter.generate_report()
        
        # Generate comprehensive HTML report page
        html = _generate_reports_html(report_data)
        return HTMLResponse(content=html)
    
    @app.get("/api/reports")
    async def get_reports_api():
        """Get reports data as JSON"""
        if not reporter:
            return JSONResponse({"error": "Reports not available"})
        report_data = reporter.generate_report()
        return JSONResponse(_serialize_datetime(report_data))
    
    return app


def _generate_unified_dashboard_html(report_data: Optional[Dict] = None) -> str:
    """Generate modern dark-themed SmartTrader dashboard"""

    general   = report_data.get('general',           {}) if report_data else {}
    periods   = report_data.get('periods',            {}) if report_data else {}
    advanced  = report_data.get('advanced',           {}) if report_data else {}
    equity_curve  = report_data.get('equity_curve',   []) if report_data else []
    monthly       = report_data.get('monthly_analytics', []) if report_data else []
    monthly_data_json = json.dumps([m for m in monthly if isinstance(m, dict)])

    # Build Strategy Leaderboard table rows (ranked by net profit, most profitable first)
    def _pf_badge(pf):
        if pf >= 1.5:
            return '#10b981'  # green
        if pf >= 1.0:
            return '#f59e0b'  # amber
        return '#ef4444'      # red

    leaderboard_rows = ""
    if strategy_leaderboard:
        for row in strategy_leaderboard:
            pf_display = "∞" if row.get('profit_factor', 0) >= 999 else f"{row.get('profit_factor', 0):.2f}"
            pf_color = _pf_badge(row.get('profit_factor', 0))
            net_color = '#10b981' if row.get('net_profit', 0) >= 0 else '#ef4444'
            leaderboard_rows += f'''
                <tr>
                    <td style="font-weight:600;">{row.get('strategy', 'unknown')}</td>
                    <td>{row.get('total_trades', 0)}</td>
                    <td>{row.get('win_rate', 0):.1f}%</td>
                    <td style="color:{pf_color}; font-weight:600;">{pf_display}</td>
                    <td style="color:{net_color}; font-weight:600;">${row.get('net_profit', 0):,.2f}</td>
                    <td>${row.get('expectancy', 0):,.2f}</td>
                    <td>${row.get('average_win', 0):,.2f}</td>
                    <td>${row.get('average_loss', 0):,.2f}</td>
                </tr>'''
    else:
        leaderboard_rows = '<tr><td colspan="8" style="text-align:center; color:#999; padding:20px;">No resolved trades yet — leaderboard will populate as strategies close positions.</td></tr>'

    equity_labels   = [str(point.get('date',''))[:10] for point in equity_curve[-120:] if isinstance(point, dict)]
    equity_values   = [float(point.get('equity',0.0) or 0.0) for point in equity_curve[-120:] if isinstance(point, dict)]
    drawdown_values = [float(point.get('drawdown',0.0) or 0.0) for point in equity_curve[-120:] if isinstance(point, dict)]
    monthly_labels  = [str(m.get('month','')) for m in monthly if isinstance(m, dict)]
    monthly_gains   = [float(m.get('gain_pct',0.0) or 0.0) for m in monthly if isinstance(m, dict)]

    # Pre-render period stats rows
    def period_row(label, data):
        d = periods.get(data, {})
        wr = d.get('win_rate', 0)
        pnl = d.get('profit', d.get('pnl', d.get('net_profit', 0)))
        trades = d.get('trades', d.get('total_trades', d.get('winning_trades',0) + d.get('losing_trades',0)))
        pf = d.get('profit_factor', 0)
        sign = '+' if pnl >= 0 else ''
        clr = 'pos' if pnl >= 0 else 'neg'
        return (f'<tr id="period-{data}"><td>{label}</td><td>{trades}</td>' f'<td><span class="{clr}">{sign}${pnl:.2f}</span></td>' f'<td>{wr:.1f}%</td><td>{pf:.2f}</td></tr>')

    periods_html = period_row('Today','today') + period_row('This Week','week') + period_row('This Month','month') + period_row('This Year','year') + period_row('All Time','all')

    gnet   = general.get('net_profit', 0)
    gwin   = general.get('win_rate',   0)
    gloss  = general.get('loss_rate',  0)
    gtot   = general.get('total_trades', 0)
    gpf    = general.get('profit_factor', 0)
    gaw    = general.get('average_win',  0)
    gal    = general.get('average_loss', 0)
    glw    = general.get('largest_win',  0)
    gll    = general.get('largest_loss', 0)
    gmdd   = advanced.get('max_drawdown', 0)
    gsharpe= advanced.get('sharpe_ratio', 0)
    gsort  = advanced.get('sortino_ratio', 0)
    gexp   = advanced.get('expectancy', 0)
    gstd   = advanced.get('standard_deviation', 0)

    pnl_class = 'pos' if gnet >= 0 else 'neg'
    pf_class  = 'pos' if gpf  >= 1  else 'neg'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="theme-color" content="#0a0e1a">
<title>SmartTrader Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*,*::before,*::after{{margin:0;padding:0;box-sizing:border-box}}
:root{{
  --bg:#0a0e1a;
  --surface:#0f1624;
  --surface2:#131d2e;
  --border:#1e2d45;
  --border2:#243347;
  --text:#e0e6f1;
  --text2:#8898aa;
  --text3:#4a5568;
  --pos:#00e676;
  --neg:#ff4d6d;
  --warn:#f6ad55;
  --blue:#4dabf7;
  --purple:#7950f2;
  --radius:12px;
}}
body{{font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;font-size:14px;line-height:1.5}}
a{{color:var(--blue);text-decoration:none}}

/* = Topbar = */
.topbar{{
  background:var(--surface);border-bottom:1px solid var(--border);
  padding:0 24px;height:56px;display:flex;align-items:center;
  justify-content:space-between;position:sticky;top:0;z-index:200;
  gap:16px;
}}
.logo{{display:flex;align-items:center;gap:10px;font-size:17px;font-weight:700;letter-spacing:-.3px;color:var(--text)}}
.logo-glyph{{
  width:30px;height:30px;border-radius:8px;
  background:linear-gradient(135deg,#4dabf7,#7950f2);
  display:flex;align-items:center;justify-content:center;
  font-size:16px;
}}
.live-pill{{
  display:inline-flex;align-items:center;gap:6px;
  background:rgba(0,230,118,.08);border:1px solid rgba(0,230,118,.2);
  color:var(--pos);padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;
}}
.live-dot{{width:6px;height:6px;background:var(--pos);border-radius:50%;animation:blink 1.6s ease-in-out infinite}}
@keyframes blink{{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:.4;transform:scale(.7)}}}}
.topbar-right{{display:flex;align-items:center;gap:12px}}
#lastUpdate{{color:var(--text2);font-size:12px;min-width:130px;text-align:right}}
.btn-sm{{
  background:var(--surface2);border:1px solid var(--border);
  color:var(--text2);padding:5px 14px;border-radius:7px;cursor:pointer;
  font-size:12px;font-weight:500;font-family:inherit;transition:all .15s;
}}
.btn-sm:hover{{border-color:var(--border2);color:var(--text)}}

/* = Layout = */
.main{{padding:24px;max-width:1440px;margin:0 auto}}

/* = Tabs = */
.tab-nav{{
  display:flex;gap:2px;background:var(--surface);border:1px solid var(--border);
  border-radius:9px;padding:3px;width:fit-content;margin-bottom:24px;
}}
.tab-btn{{
  padding:7px 22px;border:none;background:transparent;color:var(--text2);
  font-size:13px;font-weight:500;cursor:pointer;border-radius:7px;
  transition:all .15s;font-family:inherit;
}}
.tab-btn:hover{{color:var(--text)}}
.tab-btn.active{{background:var(--surface2);color:var(--blue);box-shadow:0 1px 4px rgba(0,0,0,.4)}}
.tab-pane{{display:none}}.tab-pane.active{{display:block}}

/* = Cards = */
.card{{
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);padding:20px 24px;margin-bottom:20px;
}}
.card-hdr{{
  font-size:12px;font-weight:600;color:var(--text2);text-transform:uppercase;
  letter-spacing:.7px;margin-bottom:18px;display:flex;align-items:center;
  justify-content:space-between;
}}
.card-hdr span{{font-weight:400;font-size:11px;text-transform:none;letter-spacing:0}}

/* = Metric grid = */
.mrow{{display:grid;grid-template-columns:repeat(6,1fr);gap:14px;margin-bottom:14px}}
@media(max-width:1100px){{.mrow{{grid-template-columns:repeat(3,1fr)}}}}
@media(max-width:640px){{.mrow{{grid-template-columns:repeat(2,1fr)}}}}
.mc{{
  background:var(--surface2);border:1px solid var(--border);border-radius:10px;
  padding:14px 16px;transition:border-color .2s,transform .15s;cursor:default;
}}
.mc:hover{{border-color:var(--border2);transform:translateY(-1px)}}
.mc.warn-state{{border-color:rgba(246,173,85,.3);background:rgba(246,173,85,.04)}}
.mc-lbl{{font-size:10px;font-weight:600;color:var(--text3);text-transform:uppercase;letter-spacing:.8px;margin-bottom:7px}}
.mc-val{{font-size:21px;font-weight:700;font-variant-numeric:tabular-nums;color:var(--text)}}
.pos{{color:var(--pos)!important}}.neg{{color:var(--neg)!important}}.warn{{color:var(--warn)!important}}
.muted{{color:var(--text2)!important}}

/* = 2-col = */
.grid2{{display:grid;grid-template-columns:2fr 1fr;gap:20px}}
@media(max-width:1024px){{.grid2{{grid-template-columns:1fr}}}}

/* = Tables = */
.tbl-wrap{{overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
thead th{{
  padding:9px 14px;text-align:left;font-size:10px;font-weight:600;
  color:var(--text3);text-transform:uppercase;letter-spacing:.6px;
  border-bottom:1px solid var(--border);white-space:nowrap;
}}
tbody tr{{border-bottom:1px solid rgba(30,45,69,.5);transition:background .12s}}
tbody tr:hover{{background:rgba(31,45,70,.4)}}
tbody td{{padding:11px 14px;color:var(--text)}}
tbody tr:last-child{{border-bottom:none}}
.sym{{font-weight:600;font-size:13px}}
.mono{{font-variant-numeric:tabular-nums;font-size:13px}}

/* = Badges = */
.badge{{
  display:inline-flex;align-items:center;gap:4px;
  padding:2px 8px;border-radius:20px;font-size:10px;font-weight:700;
  text-transform:uppercase;letter-spacing:.4px;white-space:nowrap;
}}
.badge-buy{{background:rgba(0,230,118,.12);color:var(--pos);border:1px solid rgba(0,230,118,.2)}}
.badge-sell{{background:rgba(255,77,109,.12);color:var(--neg);border:1px solid rgba(255,77,109,.2)}}
.badge-synced{{background:rgba(74,85,104,.15);color:var(--text3);border:1px solid var(--border);font-size:9px}}
.badge-warn{{background:rgba(246,173,85,.1);color:var(--warn);border:1px solid rgba(246,173,85,.2)}}

/* P&L colors in tables */
.ppos{{color:var(--pos);font-weight:600}}.pneg{{color:var(--neg);font-weight:600}}.pzero{{color:var(--text3)}}

/* = Close btn = */
.btn-close-pos{{
  background:rgba(255,77,109,.08);border:1px solid rgba(255,77,109,.25);
  color:var(--neg);padding:4px 12px;border-radius:6px;cursor:pointer;
  font-size:11px;font-weight:600;font-family:inherit;transition:all .15s;
}}
.btn-close-pos:hover{{background:rgba(255,77,109,.16)}}
.btn-close-pos:disabled{{opacity:.35;cursor:not-allowed}}

/* = Chart = */
.chart-wrap{{position:relative;height:230px}}

/* = Info banner = */
.info-banner{{
  background:rgba(77,171,247,.06);border:1px solid rgba(77,171,247,.18);
  border-radius:8px;padding:10px 14px;font-size:12px;color:var(--text2);
  margin-bottom:16px;display:flex;gap:10px;align-items:flex-start;
}}
.info-banner .ib-icon{{color:var(--blue);font-size:14px;flex-shrink:0;margin-top:1px}}

/* = Reports tab = */
.rpt-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:20px}}
@media(max-width:900px){{.rpt-grid{{grid-template-columns:repeat(2,1fr)}}}}
.rpt-card{{
  background:var(--surface2);border:1px solid var(--border);border-radius:10px;
  padding:16px;text-align:center;
}}
.rpt-card .rc-val{{font-size:20px;font-weight:700;margin-bottom:4px}}
.rpt-card .rc-lbl{{font-size:10px;color:var(--text3);text-transform:uppercase;letter-spacing:.7px}}
.rpt-section{{margin-bottom:24px}}
.rpt-section h3{{font-size:13px;font-weight:600;color:var(--text2);text-transform:uppercase;letter-spacing:.6px;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid var(--border)}}
.rpt-tbl thead th{{background:transparent;color:var(--text3)}}
.rpt-tbl tbody tr:hover{{background:rgba(31,45,70,.4)}}
.chart-wrap2{{position:relative;height:280px}}

/* = Scrollbar = */
::-webkit-scrollbar{{width:5px;height:5px}}
::-webkit-scrollbar-track{{background:var(--bg)}}
::-webkit-scrollbar-thumb{{background:var(--border2);border-radius:3px}}
::-webkit-scrollbar-thumb:hover{{background:var(--text3)}}
</style>
</head>
<body>

<!-- = Top bar = -->
<header class="topbar">
  <div class="logo">
    <div class="logo-glyph">&#9650;</div>
    SmartTrader
  </div>
  <div class="live-pill"><div class="live-dot"></div>LIVE</div>
  <div class="topbar-right">
    <button class="btn-sm" onclick="refreshData()">&#8635; Refresh</button>
    <span id="lastUpdate">Connecting...</span>
  </div>
</header>

<main class="main">

<!-- = Tab nav = -->
<div class="tab-nav" role="tablist" aria-label="Dashboard sections">
  <button class="tab-btn active" onclick="switchTab('dashboard',this)">&#128200; Dashboard</button>
  <button class="tab-btn" onclick="switchTab('reports',this)">&#128202; Reports</button>
</div>

<!-- = DASHBOARD TAB = -->
<div id="tab-dashboard" class="tab-pane active">

  <!-- Row 1: Core financials -->
  <div class="mrow">
    <div class="mc"><div class="mc-lbl">Equity</div><div class="mc-val" id="mEquity">&mdash;</div></div>
    <div class="mc"><div class="mc-lbl">Balance</div><div class="mc-val" id="mBalance">&mdash;</div></div>
    <div class="mc"><div class="mc-lbl">Total Return</div><div class="mc-val" id="mTotalReturn">&mdash;</div></div>
    <div class="mc"><div class="mc-lbl">Realized P&amp;L</div><div class="mc-val" id="mRealizedPnl">&mdash;</div></div>
    <div class="mc"><div class="mc-lbl">Unrealized P&amp;L</div><div class="mc-val" id="mUnrealizedPnl">&mdash;</div></div>
    <div class="mc"><div class="mc-lbl">Sharpe Ratio</div><div class="mc-val" id="mSharpe">&mdash;</div></div>
  </div>
  <!-- Row 2: Trade stats -->
  <div class="mrow">
    <div class="mc" id="mcWin"><div class="mc-lbl">Win Rate</div><div class="mc-val" id="mWinRate">&mdash;</div></div>
    <div class="mc" id="mcLoss"><div class="mc-lbl">Loss Rate</div><div class="mc-val" id="mLossRate">&mdash;</div></div>
    <div class="mc"><div class="mc-lbl">Total Trades</div><div class="mc-val" id="mTrades">&mdash;</div></div>
    <div class="mc"><div class="mc-lbl">Profit Factor</div><div class="mc-val" id="mPF">&mdash;</div></div>
    <div class="mc"><div class="mc-lbl">Expectancy</div><div class="mc-val" id="mExpectancy">&mdash;</div></div>
    <div class="mc"><div class="mc-lbl">Data Integrity</div><div class="mc-val" id="mIntegrity">&mdash;</div></div>
  </div>

  <!-- Positions + Chart -->
  <div class="grid2">
    <div class="card">
      <div class="card-hdr">Open Positions</div>
      <div class="tbl-wrap">
        <table>
          <thead><tr><th>Symbol</th><th>Side</th><th class="pos-col-qty">Qty</th><th>Entry</th><th>Current</th><th>Stop Loss</th><th class="pos-col-tp">Take Profit</th><th>P&amp;L</th><th></th></tr></thead>
          <tbody id="posBody"><tr><td colspan="9" style="text-align:center;padding:28px;color:var(--text3)">Loading...</td></tr></tbody>
        </table>
      </div>
    </div>
    <div class="card">
      <div class="card-hdr">Equity Curve</div>
      <div class="chart-wrap"><canvas id="miniChart"></canvas></div>
    </div>
  </div>

  <!-- Recent Trades -->
  <div class="card">
    <div class="card-hdr">Recent Trades <span>Last 50 trades</span></div>
    <div id="importedBanner" class="info-banner" style="display:none">
      <span class="ib-icon">&#8505;</span>
      <div>Trades marked <strong>SYNCED</strong> were imported from your broker on startup &mdash; their P&amp;L shows <strong>$0.00</strong> because SmartTrader only calculates realized P&amp;L for trades it opens and closes itself. These do <em>not</em> count as wins or losses in performance statistics.</div>
    </div>
    <div class="tbl-wrap">
      <table>
        <thead><tr><th>Time</th><th>Symbol</th><th>Side</th><th class="trd-col-qty">Qty</th><th>Price</th><th>P&amp;L</th><th>Source</th></tr></thead>
        <tbody id="tradesBody"><tr><td colspan="7" style="text-align:center;padding:28px;color:var(--text3)">Loading...</td></tr></tbody>
      </table>
    </div>
  </div>

</div><!-- /dashboard -->

<!-- = REPORTS TAB = -->
<div id="tab-reports" class="tab-pane">

  <!-- Summary metrics -->
  <div class="rpt-grid">
    <div class="rpt-card"><div class="rc-val {pnl_class}">${gnet:,.2f}</div><div class="rc-lbl">Net Profit</div></div>
    <div class="rpt-card"><div class="rc-val pos">{gwin:.1f}%</div><div class="rc-lbl">Win Rate</div></div>
    <div class="rpt-card"><div class="rc-val">{gtot}</div><div class="rc-lbl">Total Trades</div></div>
    <div class="rpt-card"><div class="rc-val {pf_class}">{gpf:.2f}</div><div class="rc-lbl">Profit Factor</div></div>
    <div class="rpt-card"><div class="rc-val neg">{gmdd:.2f}%</div><div class="rc-lbl">Max Drawdown</div></div>
    <div class="rpt-card"><div class="rc-val">{gsharpe:.2f}</div><div class="rc-lbl">Sharpe Ratio</div></div>
    <div class="rpt-card"><div class="rc-val">{gsort:.2f}</div><div class="rc-lbl">Sortino Ratio</div></div>
    <div class="rpt-card"><div class="rc-val">{gstd:.4f}</div><div class="rc-lbl">Std Deviation</div></div>
  </div>

  <!-- Period breakdown -->
  <div class="card rpt-section">
    <h3>Performance by Period</h3>
    <div class="tbl-wrap">
      <table class="rpt-tbl">
        <thead><tr><th>Period</th><th>Trades</th><th>Net P&amp;L</th><th>Win Rate</th><th>Profit Factor</th></tr></thead>
        <tbody>{periods_html}</tbody>
      </table>
    </div>
  </div>

  <!-- Trade stats -->
  <div class="card rpt-section">
    <h3>Trade Statistics</h3>
    <div class="tbl-wrap">
      <table class="rpt-tbl">
        <thead><tr><th>Metric</th><th>Value</th></tr></thead>
        <tbody>
          <tr><td>Average Win</td><td class="ppos">${gaw:,.2f}</td></tr>
          <tr><td>Average Loss</td><td class="pneg">${gal:,.2f}</td></tr>
          <tr><td>Largest Win</td><td class="ppos">${glw:,.2f}</td></tr>
          <tr><td>Largest Loss</td><td class="pneg">${gll:,.2f}</td></tr>
          <tr><td>Expectancy</td><td class="{'ppos' if gexp >= 0 else 'pneg'}">${gexp:,.2f}</td></tr>
          <tr><td>Loss Rate</td><td>{gloss:.1f}%</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- Equity chart (full) -->
  <div class="card rpt-section">
    <h3>Equity Curve</h3>
    <div class="chart-wrap2"><canvas id="rptEquityChart"></canvas></div>
  </div>

  <!-- Monthly chart -->
  <div class="card rpt-section">
    <h3>Monthly Returns</h3>
    <div class="chart-wrap2"><canvas id="rptMonthlyChart"></canvas></div>
  </div>

</div><!-- /reports -->
</main>

<script>
// = Globals =
let miniChart=null, rptEquity=null, rptMonthly=null;
const eqLabels = {json.dumps(equity_labels)};
const eqValues = {json.dumps(equity_values)};
const ddValues = {json.dumps(drawdown_values)};
const mthLabels= {json.dumps(monthly_labels)};
const mthGains = {json.dumps(monthly_gains)};
const CS = Chart.defaults; CS.font.family = "'Inter',system-ui,sans-serif";

// = Tab switch =
function switchTab(name, btn) {{
  document.querySelectorAll('.tab-pane').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  btn.classList.add('active');
  if (name==='reports') initReportCharts();
}}

// = Formatters =
const fmt$ = n => n==null?'-':(n<0?'-$':'$')+Math.abs(n).toLocaleString('en-US',{{minimumFractionDigits:2,maximumFractionDigits:2}});
const fmtPct = n => n==null?'-':(n>=0?'+':'')+parseFloat(n).toFixed(2)+'%';
const fmtN = (n,d=2) => n==null?'-':parseFloat(n).toFixed(d);
function pClass(n) {{ return n>0.001?'ppos':n<-0.001?'pneg':'pzero'; }}
function mClass(n) {{ return n>0.001?'pos':n<-0.001?'neg':''; }}
function fmtDate(s) {{ if(!s) return '-'; const d=new Date(s); return isNaN(d)?s:d.toLocaleTimeString('en-AU',{{hour:'2-digit',minute:'2-digit',second:'2-digit'}}); }}

// = Metric update =
function setM(id,val,cls) {{ const el=document.getElementById(id); if(!el) return; el.textContent=val; el.className='mc-val '+(cls||''); }}

function updateMetrics(data) {{
  const p=data.performance||{{}}, dq=data.data_quality||{{}};
  const eq=data.equity||0, bal=data.balance||0;
  const ret=p.total_return||0, rpnl=p.realized_pnl||0, upnl=p.unrealized_pnl||0;
  const sharpe=p.sharpe_ratio||0, wr=p.win_rate||0, lr=p.loss_rate||0;
  const trades=p.total_trades||0, pf=p.profit_factor||0, exp=p.expectancy||0;
  const integ=dq.integrity_score||0;

  setM('mEquity',    fmt$(eq),     mClass(eq-bal));
  setM('mBalance',   fmt$(bal),    '');
  setM('mTotalReturn', fmtPct(ret),  mClass(ret));
  setM('mRealizedPnl', fmt$(rpnl), mClass(rpnl));
  setM('mUnrealizedPnl',fmt$(upnl),mClass(upnl));
  setM('mSharpe',    fmtN(sharpe), sharpe>=1?'pos':sharpe<0?'neg':'');
  setM('mWinRate',   fmtN(wr,1)+'%', wr>=50?'pos':wr>0?'':'muted');
  setM('mLossRate',  fmtN(lr,1)+'%', lr>60?'neg':lr>40?'warn':'');
  setM('mTrades',    trades, '');
  setM('mPF',        fmtN(pf,2),   pf>=1.5?'pos':pf>=1?'':'neg');
  setM('mExpectancy',fmt$(exp),    mClass(exp));
  setM('mIntegrity', fmtN(integ,1)+'%', integ>=90?'pos':integ>=70?'warn':'neg');

  // Flag suspicious 100% loss rate (all imported, no real trades)
  const artifact = lr>=99.9 && wr<0.1;
  document.getElementById('mcLoss').className = 'mc'+(artifact?' warn-state':'');
  if (artifact) document.getElementById('importedBanner').style.display='flex';
}}

// = Positions =
function updatePositions(pos) {{
  const tb=document.getElementById('posBody');
  if (!pos||!pos.length) {{ tb.innerHTML='<tr><td colspan="9" style="text-align:center;padding:32px;color:var(--text3)">No open positions</td></tr>'; return; }}
  tb.innerHTML=pos.map(p=>{{
    const sym=p.symbol||'-', side=(p.side||'buy').toLowerCase();
    const qty=parseFloat(p.quantity||0).toLocaleString('en-US',{{maximumFractionDigits:4}});
    const entry=p.entry_price?'$'+parseFloat(p.entry_price).toFixed(5):'-';
    const cur=p.current_price?'$'+parseFloat(p.current_price).toFixed(5):'-';
    const sl=p.stop_loss?'$'+parseFloat(p.stop_loss).toFixed(5):'-';
    const tp=p.take_profit?'$'+parseFloat(p.take_profit).toFixed(5):'-';
    const pnl=parseFloat(p.pnl||0), tid=p.trade_id;
    const warn=p.verified_at_broker===false?'&#9888; ':'';
    return `<tr>
      <td><span class="sym">${{warn}}${{sym}}</span></td>
      <td><span class="badge badge-${{side}}">${{side.toUpperCase()}}</span></td>
      <td class="mono">${{qty}}</td>
      <td class="mono">${{entry}}</td>
      <td class="mono">${{cur}}</td>
      <td class="mono" style="color:var(--neg)">${{sl}}</td>
      <td class="mono" style="color:var(--pos)">${{tp}}</td>
      <td class="${{pClass(pnl)}}">${{fmt$(pnl)}}</td>
      <td><button class="btn-close-pos" onclick="closePos('${{tid}}','${{sym}}','${{side}}')">Close</button></td>
    </tr>`;
  }}).join('');
}}

// = Trades =
function updateTrades(trades) {{
  const tb=document.getElementById('tradesBody');
  if (!trades||!trades.length) {{ tb.innerHTML='<tr><td colspan="7" style="text-align:center;padding:32px;color:var(--text3)">No trades recorded</td></tr>'; return; }}
  let hasZero=false;
  tb.innerHTML=trades.map(t=>{{
    const sym=t.symbol||'-', side=(t.side||'buy').toLowerCase();
    const qty=parseFloat(t.quantity||0).toLocaleString('en-US',{{maximumFractionDigits:4}});
    const price=t.price?'$'+parseFloat(t.price).toFixed(2):'-';
    const pnl=parseFloat(t.pnl||0);
    const ts=fmtDate(t.exit_time||t.entry_time||t.time||t.timestamp);
    const synced=Math.abs(pnl)<0.001;
    if(synced) hasZero=true;
    return `<tr>
      <td style="color:var(--text3);font-size:12px">${{ts}}</td>
      <td><span class="sym">${{sym}}</span></td>
      <td><span class="badge badge-${{side}}">${{side.toUpperCase()}}</span></td>
      <td class="mono">${{qty}}</td>
      <td class="mono">${{price}}</td>
      <td class="${{pClass(pnl)}}">${{fmt$(pnl)}}</td>
      <td>${{synced?'<span class="badge badge-synced">SYNCED</span>':''}}</td>
    </tr>`;
  }}).join('');
  if(hasZero) document.getElementById('importedBanner').style.display='flex';
}}

// = Mini equity chart =
function initMiniChart(labels, values) {{
  const ctx=document.getElementById('miniChart'); if(!ctx) return;
  if(miniChart) {{
    miniChart.data.labels=labels;
    miniChart.data.datasets[0].data=values;
    miniChart.update('none');
    return;
  }}
  miniChart=new Chart(ctx,{{
    type:'line',
    data:{{labels,datasets:[{{
      data:values,borderColor:'#00e676',borderWidth:2,
      backgroundColor:'rgba(0,230,118,.06)',fill:true,tension:.35,
      pointRadius:0,pointHoverRadius:4,pointHoverBackgroundColor:'#00e676'
    }}]}},
    options:{{
      responsive:true,maintainAspectRatio:false,
      interaction:{{mode:'index',intersect:false}},
      plugins:{{legend:{{display:false}},tooltip:{{
        backgroundColor:'#0f1624',borderColor:'#1e2d45',borderWidth:1,
        callbacks:{{label:ctx=>'$'+parseFloat(ctx.raw).toFixed(2)}}
      }}}},
      scales:{{
        x:{{display:false}},
        y:{{grid:{{color:'rgba(30,45,69,.5)'}},
            ticks:{{color:'#4a5568',font:{{size:10}},callback:v=>'$'+v.toFixed(0)}},
            border:{{color:'#1e2d45'}}}}
      }}
    }}
  }});
}}

// = Report charts (lazy) =
function initReportCharts() {{
  if (!rptEquity) {{
    const ctx=document.getElementById('rptEquityChart'); if(!ctx) return;
    rptEquity=new Chart(ctx,{{
      type:'line',
      data:{{labels:eqLabels,datasets:[
        {{label:'Equity',data:eqValues,borderColor:'#4dabf7',borderWidth:2,backgroundColor:'rgba(77,171,247,.06)',fill:true,tension:.3,pointRadius:0}},
        {{label:'Drawdown %',data:ddValues,borderColor:'#ff4d6d',borderWidth:1.5,backgroundColor:'rgba(255,77,109,.05)',fill:true,tension:.3,pointRadius:0,yAxisID:'y2'}}
      ]}},
      options:{{
        responsive:true,maintainAspectRatio:false,
        interaction:{{mode:'index',intersect:false}},
        plugins:{{legend:{{labels:{{color:'#8898aa',boxWidth:12}},position:'top'}},
                 tooltip:{{backgroundColor:'#0f1624',borderColor:'#1e2d45',borderWidth:1}}}},
        scales:{{
          x:{{ticks:{{color:'#4a5568',maxTicksLimit:8,font:{{size:10}}}},grid:{{color:'rgba(30,45,69,.4)'}},border:{{color:'#1e2d45'}}}},
          y:{{ticks:{{color:'#4a5568',font:{{size:10}},callback:v=>'$'+v.toFixed(0)}},grid:{{color:'rgba(30,45,69,.4)'}},border:{{color:'#1e2d45'}}}},
          y2:{{position:'right',ticks:{{color:'#4a5568',font:{{size:10}},callback:v=>v.toFixed(1)+'%'}},grid:{{display:false}},border:{{color:'#1e2d45'}}}}
        }}
      }}
    }});
  }}
  if (!rptMonthly && mthLabels.length>0) {{
    const ctx2=document.getElementById('rptMonthlyChart'); if(!ctx2) return;
    const colors=mthGains.map(v=>v>=0?'rgba(0,230,118,.7)':'rgba(255,77,109,.7)');
    rptMonthly=new Chart(ctx2,{{
      type:'bar',
      data:{{labels:mthLabels,datasets:[{{
        label:'Monthly Return %',data:mthGains,backgroundColor:colors,borderRadius:4
      }}]}},
      options:{{
        responsive:true,maintainAspectRatio:false,
        plugins:{{legend:{{display:false}},tooltip:{{backgroundColor:'#0f1624',borderColor:'#1e2d45',borderWidth:1,callbacks:{{label:ctx=>ctx.raw.toFixed(2)+'%'}}}}}},
        scales:{{
          x:{{ticks:{{color:'#4a5568',font:{{size:10}}}},grid:{{color:'rgba(30,45,69,.3)'}},border:{{color:'#1e2d45'}}}},
          y:{{ticks:{{color:'#4a5568',font:{{size:10}},callback:v=>v+'%'}},grid:{{color:'rgba(30,45,69,.4)'}},border:{{color:'#1e2d45'}}}}
        }}
      }}
    }});
  }}
}}

// = API / WebSocket =
async function refreshData() {{
  try {{
    const r=await fetch('/api/dashboard'); if(!r.ok) return;
    handleData(await r.json());
  }} catch(e){{}}
}}

function handleData(data) {{
  updateMetrics(data);
  updatePositions(data.positions||[]);
  updateTrades(data.trades||[]);
  document.getElementById('lastUpdate').textContent='Updated '+new Date().toLocaleTimeString();
  if (data.report&&data.report.equity_curve&&data.report.equity_curve.length>1) {{
    const pts=data.report.equity_curve.slice(-120);
    initMiniChart(pts.map(p=>(p.date||'').toString().slice(0,10)), pts.map(p=>parseFloat(p.equity||0)));
    if (rptEquity) {{
      rptEquity.data.labels=pts.map(p=>(p.date||'').toString().slice(0,10));
      rptEquity.data.datasets[0].data=pts.map(p=>parseFloat(p.equity||0));
      rptEquity.data.datasets[1].data=pts.map(p=>parseFloat(p.drawdown||0));
      rptEquity.update('none');
    }}
  }}
  if (data.report&&data.report.periods) {{
    const per=data.report.periods;
    [['today'],['week'],['month'],['year'],['all']].forEach(([k])=>{{
      const row=document.getElementById('period-'+k);
      if (!row||!per[k]) return;
      const d=per[k];
      const pnl=parseFloat(d.profit||d.pnl||d.net_profit||0);
      const trades=parseInt(d.trades||d.total_trades||0);
      const wr=parseFloat(d.win_rate||0);
      const pf=parseFloat(d.profit_factor||0);
      const sign=pnl>=0?'+':'';
      const clr=pnl>=0?'pos':'neg';
      const cells=row.querySelectorAll('td');
      if(cells.length>=5){{
        cells[1].textContent=trades;
        cells[2].innerHTML='<span class="'+clr+'">'+sign+'$'+pnl.toFixed(2)+'</span>';
        cells[3].textContent=wr.toFixed(1)+'%';
        cells[4].textContent=pf.toFixed(2);
      }}
    }});
  }}
}}

async function closePos(tid,sym,side) {{
  if(!confirm('Close position?\\n\\nSymbol: '+sym+'\\nSide: '+side+'\\n\\nCloses via broker API, or deletes if phantom.')) return;
  try {{
    const r=await fetch('/api/close_position/'+tid,{{method:'POST',headers:{{'Content-Type':'application/json'}}}});
    const res=await r.json();
    if(res.success) {{ alert(res.message||'Position closed'); setTimeout(refreshData,1200); }}
    else alert('Error: '+(res.error||'Failed'));
  }} catch(e){{alert('Error: '+e.message)}}
}}

async function purgeZeroTrades() {{
  try {{
    const r=await fetch('/api/purge_zero_trades',{{method:'POST',headers:{{'Content-Type':'application/json'}}}});
    const res=await r.json();
    if(res.success && res.deleted>0) {{
      document.getElementById('importedBanner').style.display='none';
      setTimeout(refreshData,600);
    }}
  }} catch(e){{}}
}}

let ws=null, wsRetries=0;
function connectWS() {{
  const proto=location.protocol==='https:'?'wss:':'ws:';
  ws=new WebSocket(proto+'//'+location.host+'/ws');
  ws.onopen=()=>{{ wsRetries=0; document.getElementById('lastUpdate').textContent='Connected'; setTimeout(purgeZeroTrades,2000); }};
  ws.onmessage=e=>{{ try{{handleData(JSON.parse(e.data))}}catch(ex){{}} }};
  ws.onclose=()=>{{
    wsRetries++; document.getElementById('lastUpdate').textContent='Reconnecting...';
    setTimeout(connectWS, Math.min(wsRetries*2000,30000));
  }};
}}

// = Bootstrap =
initMiniChart(eqLabels, eqValues);
connectWS();
refreshData();
setInterval(refreshData, 15000);
</script>
</body>
</html>"""
    return html


def _generate_reports_html(report_data: Dict) -> str:
    """Generate comprehensive HTML reports page similar to MyFxBook"""
    
    general = report_data.get('general', {})
    periods = report_data.get('periods', {})
    advanced = report_data.get('advanced', {})
    equity_curve = report_data.get('equity_curve', [])
    monthly = report_data.get('monthly_analytics', [])
    strategy_leaderboard = report_data.get('strategy_leaderboard', [])
    
    # Prepare equity curve data for Chart.js
    equity_labels = [str(point.get('date', ''))[:10] for point in equity_curve[-100:] if isinstance(point, dict)] if equity_curve else []
    equity_values = [float(point.get('equity', 0.0) or 0.0) for point in equity_curve[-100:] if isinstance(point, dict)] if equity_curve else []
    drawdown_values = [float(point.get('drawdown', 0.0) or 0.0) for point in equity_curve[-100:] if isinstance(point, dict)] if equity_curve else []
    
    # Prepare monthly data
    monthly_labels = [str(m.get('month', '')) for m in monthly if isinstance(m, dict)]
    monthly_gains = [float(m.get('gain_pct', 0.0) or 0.0) for m in monthly if isinstance(m, dict)]
    monthly_data_json = json.dumps([m for m in monthly if isinstance(m, dict)])

    # Build Strategy Leaderboard table rows (ranked by net profit, most profitable first)
    def _pf_badge(pf):
        if pf >= 1.5:
            return '#10b981'  # green
        if pf >= 1.0:
            return '#f59e0b'  # amber
        return '#ef4444'      # red

    leaderboard_rows = ""
    if strategy_leaderboard:
        for row in strategy_leaderboard:
            pf_display = "∞" if row.get('profit_factor', 0) >= 999 else f"{row.get('profit_factor', 0):.2f}"
            pf_color = _pf_badge(row.get('profit_factor', 0))
            net_color = '#10b981' if row.get('net_profit', 0) >= 0 else '#ef4444'
            leaderboard_rows += f'''
                <tr>
                    <td style="font-weight:600;">{row.get('strategy', 'unknown')}</td>
                    <td>{row.get('total_trades', 0)}</td>
                    <td>{row.get('win_rate', 0):.1f}%</td>
                    <td style="color:{pf_color}; font-weight:600;">{pf_display}</td>
                    <td style="color:{net_color}; font-weight:600;">${row.get('net_profit', 0):,.2f}</td>
                    <td>${row.get('expectancy', 0):,.2f}</td>
                    <td>${row.get('average_win', 0):,.2f}</td>
                    <td>${row.get('average_loss', 0):,.2f}</td>
                </tr>'''
    else:
        leaderboard_rows = '<tr><td colspan="8" style="text-align:center; color:#999; padding:20px;">No resolved trades yet — leaderboard will populate as strategies close positions.</td></tr>'
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SmartTrader - Performance Reports</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2.0.0"></script>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                background: #f5f7fa;
                color: #333;
                padding: 20px;
            }}
            .container {{ max-width: 1400px; margin: 0 auto; }}
            .header {{ 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .header h1 {{ font-size: 32px; margin-bottom: 10px; }}
            .header p {{ opacity: 0.9; }}
            .nav {{ 
                background: white;
                padding: 15px 30px;
                border-radius: 10px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .nav a {{
                color: #667eea;
                text-decoration: none;
                margin-right: 20px;
                font-weight: 500;
                padding: 8px 16px;
                border-radius: 5px;
                transition: background 0.2s;
            }}
            .nav a:hover {{ background: #f0f0f0; }}
            .nav a.active {{ background: #667eea; color: white; }}
            .card {{
                background: white;
                border-radius: 10px;
                padding: 25px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .card h2 {{
                font-size: 20px;
                margin-bottom: 20px;
                color: #333;
                border-bottom: 2px solid #667eea;
                padding-bottom: 10px;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }}
            .metric {{
                text-align: center;
                padding: 15px;
                background: #f8f9fa;
                border-radius: 8px;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 5px;
            }}
            .metric-label {{
                font-size: 12px;
                color: #666;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .positive {{ color: #28a745; }}
            .negative {{ color: #dc3545; }}
            .neutral {{ color: #666; }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #e0e0e0;
            }}
            th {{
                background: #f8f9fa;
                font-weight: 600;
                color: #333;
            }}
            tr:hover {{ background: #f8f9fa; }}
            .chart-container {{
                position: relative;
                height: 400px;
                margin-top: 20px;
            }}
            .tabs {{
                display: flex;
                border-bottom: 2px solid #e0e0e0;
                margin-bottom: 20px;
            }}
            .tab {{
                padding: 12px 24px;
                cursor: pointer;
                border: none;
                background: none;
                font-size: 14px;
                font-weight: 500;
                color: #666;
                transition: all 0.2s;
            }}
            .tab:hover {{ color: #667eea; }}
            .tab.active {{
                color: #667eea;
                border-bottom: 2px solid #667eea;
            }}
            .tab-content {{
                display: none;
            }}
            .tab-content.active {{
                display: block;
            }}
            .progress-bar {{
                height: 20px;
                background: #e0e0e0;
                border-radius: 10px;
                overflow: hidden;
                margin: 10px 0;
            }}
            .progress-fill {{
                height: 100%;
                background: linear-gradient(90deg, #28a745, #20c997);
                transition: width 0.3s;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1> SmartTrader Performance Reports</h1>
                <p>Comprehensive trading analytics and statistics</p>
            </div>
            
            <div class="nav">
                <a href="/">Dashboard</a>
                <a href="/reports" class="active">Reports</a>
            </div>
            
            <!-- General Account Info -->
            <div class="card">
                <h2> General Account Information</h2>
                <div class="metrics-grid">
                    <div class="metric">
                        <div class="metric-value {('positive' if general.get('net_profit', 0) >= 0 else 'negative')}">
                            ${general.get('net_profit', 0):,.2f}
                        </div>
                        <div class="metric-label">Net Profit</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value positive">
                            {general.get('win_rate', 0):.1f}%
                        </div>
                        <div class="metric-label">Win Rate</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value negative">
                            {general.get('loss_rate', 0):.1f}%
                        </div>
                        <div class="metric-label">Loss Rate</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value neutral">
                            {general.get('total_trades', 0)}
                        </div>
                        <div class="metric-label">Total Trades</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value {('positive' if general.get('profit_factor', 0) >= 1 else 'negative')}">
                            {general.get('profit_factor', 0):.2f}
                        </div>
                        <div class="metric-label">Profit Factor</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value negative">
                            {advanced.get('max_drawdown', 0):.2f}%
                        </div>
                        <div class="metric-label">Max Drawdown</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value neutral">
                            {advanced.get('sharpe_ratio', 0):.2f}
                        </div>
                        <div class="metric-label">Sharpe Ratio</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value {('positive' if general.get('average_win', 0) >= abs(general.get('average_loss', 0)) else 'negative')}">
                            ${general.get('average_win', 0):,.2f}
                        </div>
                        <div class="metric-label">Avg Win</div>
                    </div>
                </div>
                
                <div style="margin-top: 30px;">
                    <h3 style="margin-bottom: 15px;">Longs vs Shorts</h3>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                        <div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                <span>Longs Won:</span>
                                <strong>{general.get('longs_won', 0)}/{general.get('longs_total', 0)} ({general.get('longs_total', 0) and (general.get('longs_won', 0)/general.get('longs_total', 0)*100) or 0:.1f}%)</strong>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: {general.get('longs_total', 0) and (general.get('longs_won', 0)/general.get('longs_total', 0)*100) or 0}%"></div>
                            </div>
                        </div>
                        <div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                <span>Shorts Won:</span>
                                <strong>{general.get('shorts_won', 0)}/{general.get('shorts_total', 0)} ({general.get('shorts_total', 0) and (general.get('shorts_won', 0)/general.get('shorts_total', 0)*100) or 0:.1f}%)</strong>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: {general.get('shorts_total', 0) and (general.get('shorts_won', 0)/general.get('shorts_total', 0)*100) or 0}%"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Performance Chart -->
            <div class="card">
                <h2> Equity Curve</h2>
                <div class="tabs">
                    <button class="tab active" onclick="switchTab('equity')">Equity</button>
                    <button class="tab" onclick="switchTab('drawdown')">Drawdown</button>
                </div>
                <div id="equity" class="tab-content active">
                    <div class="chart-container">
                        <canvas id="equityChart"></canvas>
                    </div>
                </div>
                <div id="drawdown" class="tab-content">
                    <div class="chart-container">
                        <canvas id="drawdownChart"></canvas>
                    </div>
                </div>
            </div>
            
            <!-- Period Statistics -->
            <div class="card">
                <h2> Trading Periods</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Period</th>
                            <th>Gain</th>
                            <th>Profit</th>
                            <th>Win Rate</th>
                            <th>Trades</th>
                            <th>Winning</th>
                            <th>Losing</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Today</strong></td>
                            <td class="{('positive' if periods.get('today', {}).get('profit', 0) >= 0 else 'negative')}">
                                ${periods.get('today', {}).get('profit', 0):,.2f}
                            </td>
                            <td class="{('positive' if periods.get('today', {}).get('profit', 0) >= 0 else 'negative')}">
                                ${periods.get('today', {}).get('profit', 0):,.2f}
                            </td>
                            <td>{periods.get('today', {}).get('win_rate', 0):.1f}%</td>
                            <td>{periods.get('today', {}).get('trades', 0)}</td>
                            <td class="positive">{periods.get('today', {}).get('winning_trades', 0)}</td>
                            <td class="negative">{periods.get('today', {}).get('losing_trades', 0)}</td>
                        </tr>
                        <tr>
                            <td><strong>This Week</strong></td>
                            <td class="{('positive' if periods.get('week', {}).get('profit', 0) >= 0 else 'negative')}">
                                ${periods.get('week', {}).get('profit', 0):,.2f}
                            </td>
                            <td class="{('positive' if periods.get('week', {}).get('profit', 0) >= 0 else 'negative')}">
                                ${periods.get('week', {}).get('profit', 0):,.2f}
                            </td>
                            <td>{periods.get('week', {}).get('win_rate', 0):.1f}%</td>
                            <td>{periods.get('week', {}).get('trades', 0)}</td>
                            <td class="positive">{periods.get('week', {}).get('winning_trades', 0)}</td>
                            <td class="negative">{periods.get('week', {}).get('losing_trades', 0)}</td>
                        </tr>
                        <tr>
                            <td><strong>This Month</strong></td>
                            <td class="{('positive' if periods.get('month', {}).get('profit', 0) >= 0 else 'negative')}">
                                ${periods.get('month', {}).get('profit', 0):,.2f}
                            </td>
                            <td class="{('positive' if periods.get('month', {}).get('profit', 0) >= 0 else 'negative')}">
                                ${periods.get('month', {}).get('profit', 0):,.2f}
                            </td>
                            <td>{periods.get('month', {}).get('win_rate', 0):.1f}%</td>
                            <td>{periods.get('month', {}).get('trades', 0)}</td>
                            <td class="positive">{periods.get('month', {}).get('winning_trades', 0)}</td>
                            <td class="negative">{periods.get('month', {}).get('losing_trades', 0)}</td>
                        </tr>
                        <tr>
                            <td><strong>This Year</strong></td>
                            <td class="{('positive' if periods.get('year', {}).get('profit', 0) >= 0 else 'negative')}">
                                ${periods.get('year', {}).get('profit', 0):,.2f}
                            </td>
                            <td class="{('positive' if periods.get('year', {}).get('profit', 0) >= 0 else 'negative')}">
                                ${periods.get('year', {}).get('profit', 0):,.2f}
                            </td>
                            <td>{periods.get('year', {}).get('win_rate', 0):.1f}%</td>
                            <td>{periods.get('year', {}).get('trades', 0)}</td>
                            <td class="positive">{periods.get('year', {}).get('winning_trades', 0)}</td>
                            <td class="negative">{periods.get('year', {}).get('losing_trades', 0)}</td>
                        </tr>
                        <tr>
                            <td><strong>All Time</strong></td>
                            <td class="{('positive' if periods.get('all', {}).get('profit', 0) >= 0 else 'negative')}">
                                ${periods.get('all', {}).get('profit', 0):,.2f}
                            </td>
                            <td class="{('positive' if periods.get('all', {}).get('profit', 0) >= 0 else 'negative')}">
                                ${periods.get('all', {}).get('profit', 0):,.2f}
                            </td>
                            <td>{periods.get('all', {}).get('win_rate', 0):.1f}%</td>
                            <td>{periods.get('all', {}).get('trades', 0)}</td>
                            <td class="positive">{periods.get('all', {}).get('winning_trades', 0)}</td>
                            <td class="negative">{periods.get('all', {}).get('losing_trades', 0)}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <!-- Advanced Statistics -->
            <div class="card">
                <h2> Advanced Statistics</h2>
                <div class="metrics-grid">
                    <div class="metric">
                        <div class="metric-value neutral">
                            {advanced.get('sharpe_ratio', 0):.2f}
                        </div>
                        <div class="metric-label">Sharpe Ratio</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value neutral">
                            {advanced.get('sortino_ratio', 0):.2f}
                        </div>
                        <div class="metric-label">Sortino Ratio</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value negative">
                            {advanced.get('max_drawdown', 0):.2f}%
                        </div>
                        <div class="metric-label">Max Drawdown</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value neutral">
                            ${advanced.get('standard_deviation', 0):,.2f}
                        </div>
                        <div class="metric-label">Std Deviation</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value {('positive' if advanced.get('expectancy', 0) >= 0 else 'negative')}">
                            ${advanced.get('expectancy', 0):,.2f}
                        </div>
                        <div class="metric-label">Expectancy</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value {('positive' if advanced.get('average_trade', 0) >= 0 else 'negative')}">
                            ${advanced.get('average_trade', 0):,.2f}
                        </div>
                        <div class="metric-label">Avg Trade</div>
                    </div>
                </div>
            </div>
            
            <!-- Monthly Analytics -->
            <div class="card">
                <div style="display: flex; justify-content: flex-start; align-items: center; border-bottom: 2px solid #e0e0e0; margin-bottom: 20px;">
                    <h2 style="margin-bottom: 0; border-bottom: none; padding-right: 20px;"> Monthly Analytics</h2>
                    <div class="tabs" style="border-bottom: none; margin-bottom: 0;" id="monthly-tabs-reports">
                    </div>
                </div>
                <div class="chart-container">
                    <canvas id="monthlyChartReports"></canvas>
                </div>
            </div>

            <!-- Strategy Leaderboard -->
            <div class="card">
                <h2>🏆 Strategy Leaderboard</h2>
                <p style="color:#888; margin-bottom:16px; font-size:14px;">
                    Per-strategy performance across all resolved paper trades — use this to see which
                    strategies are actually working so you can reweight or disable underperformers.
                </p>
                <div style="overflow-x:auto;">
                    <table style="width:100%; border-collapse:collapse; font-size:14px;">
                        <thead>
                            <tr style="text-align:left; border-bottom:2px solid #e0e0e0;">
                                <th style="padding:10px;">Strategy</th>
                                <th style="padding:10px;">Trades</th>
                                <th style="padding:10px;">Win Rate</th>
                                <th style="padding:10px;">Profit Factor</th>
                                <th style="padding:10px;">Net P&L</th>
                                <th style="padding:10px;">Expectancy</th>
                                <th style="padding:10px;">Avg Win</th>
                                <th style="padding:10px;">Avg Loss</th>
                            </tr>
                        </thead>
                        <tbody>
                            {leaderboard_rows}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <script>
            function switchTab(tabName) {{
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                event.target.classList.add('active');
                document.getElementById(tabName).classList.add('active');
            }}
            
            // Equity Chart
            const equityCtx = document.getElementById('equityChart').getContext('2d');
            new Chart(equityCtx, {{
                type: 'line',
                data: {{
                    labels: {json.dumps(equity_labels)},
                    datasets: [{{
                        label: 'Equity',
                        data: {json.dumps(equity_values)},
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        tension: 0.4,
                        fill: true
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: true }},
                        title: {{ display: true, text: 'Equity Curve' }}
                    }},
                    scales: {{
                        y: {{ beginAtZero: false }}
                    }}
                }}
            }});
            
            // Drawdown Chart
            const drawdownCtx = document.getElementById('drawdownChart').getContext('2d');
            new Chart(drawdownCtx, {{
                type: 'line',
                data: {{
                    labels: {json.dumps(equity_labels)},
                    datasets: [{{
                        label: 'Drawdown %',
                        data: {json.dumps(drawdown_values)},
                        borderColor: '#dc3545',
                        backgroundColor: 'rgba(220, 53, 69, 0.1)',
                        tension: 0.4,
                        fill: true
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: true }},
                        title: {{ display: true, text: 'Drawdown %' }}
                    }},
                    scales: {{
                        y: {{ beginAtZero: true }}
                    }}
                }}
            }});
            
            // Monthly Data Global
            window.monthlyDataReports = {monthly_data_json};
            window.monthlyChartInstanceReports = null;

            function renderMonthlyChartReports(year) {{
                const yearData = window.monthlyDataReports.filter(d => d.year === String(year));
                const labels = yearData.map(d => d.month);
                const gains = yearData.map(d => parseFloat(d.gain_pct).toFixed(2));
                const colors = ['#b388b8', '#e38484', '#61b3b1', '#fdb68b', '#b9d66f', '#77a4e6', '#c9c27f', '#d78ec5', '#8cd2b8', '#e3b26c', '#67a3a1', '#e88f8f'];
                
                const ctx = document.getElementById('monthlyChartReports').getContext('2d');
                if (window.monthlyChartInstanceReports) {{
                    window.monthlyChartInstanceReports.destroy();
                }}
                
                window.monthlyChartInstanceReports = new Chart(ctx, {{
                    type: 'bar',
                    data: {{
                        labels: labels,
                        datasets: [{{
                            label: 'Monthly Gain(Change)',
                            data: gains,
                            backgroundColor: labels.map((l, i) => colors[i % colors.length])
                        }}]
                    }},
                    plugins: [ChartDataLabels],
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{ display: false }},
                            title: {{ display: true, text: 'Monthly Gain(Change)', font: {{ size: 16 }} }},
                            datalabels: {{
                                anchor: 'end',
                                align: 'top',
                                formatter: function(value) {{ return value + "%"; }},
                                font: {{ weight: 'bold' }}
                            }}
                        }},
                        scales: {{
                            y: {{
                                beginAtZero: true,
                                ticks: {{ callback: function(value) {{ return value + "%" }} }},
                                suggestedMax: Math.max(...gains.map(Number)) * 1.2
                            }}
                        }}
                    }}
                }});
                
                document.querySelectorAll('#monthly-tabs-reports .tab').forEach(t => {{
                    if (t.textContent === String(year)) {{ t.classList.add('active'); }}
                    else {{ t.classList.remove('active'); }}
                }});
            }}

            // Setup tabs and render initial chart
            if (window.monthlyDataReports && window.monthlyDataReports.length > 0) {{
                const monthlyTabsDiv = document.getElementById('monthly-tabs-reports');
                const years = [...new Set(window.monthlyDataReports.map(d => d.year))].sort();
                if (years.length > 0 && monthlyTabsDiv) {{
                    years.forEach(year => {{
                        const btn = document.createElement('button');
                        btn.className = 'tab';
                        btn.textContent = year;
                        btn.onclick = () => renderMonthlyChartReports(year);
                        monthlyTabsDiv.appendChild(btn);
                    }});
                    renderMonthlyChartReports(years[years.length - 1]);
                }}
            }}
        </script>
    </body>
    </html>
    """
    return html
