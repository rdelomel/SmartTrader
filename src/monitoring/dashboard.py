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
    
    @app.get("/")
    async def get_dashboard():
        """Serve unified dashboard HTML with tabs"""
        # Try to load initial data from database if dashboard is empty
        if (dashboard_data.get('equity', 0) == 10000.0 and 
            dashboard_data.get('balance', 0) == 10000.0 and 
            len(dashboard_data.get('positions', [])) == 0 and
            storage):
            try:
                # Load initial data from database
                open_trades = storage.get_open_trades()
                if open_trades or (hasattr(storage, 'get_all_trades') and storage.get_all_trades(limit=1)):
                    # We have data, trigger a load
                    # This will be handled by the /api/dashboard endpoint when called
                    pass
            except:
                pass
        
        # Get report data if available
        report_data = None
        if reporter:
            try:
                report_data = reporter.generate_report()
            except:
                pass
        
        html = _generate_unified_dashboard_html(report_data)
        return HTMLResponse(content=html)
    
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
                                <button class="btn-close" onclick="closePosition('${pos.trade_id}', '${pos.symbol}', '${pos.side}')" title="${buttonTitle}">
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
        """Get dashboard data as JSON - with fallback to load from database if empty"""
        # If dashboard data is still default/empty, try to load from database
        if (dashboard_data.get('equity', 0) == 10000.0 and 
            dashboard_data.get('balance', 0) == 10000.0 and 
            len(dashboard_data.get('positions', [])) == 0 and
            len(dashboard_data.get('trades', [])) == 0 and
            app.storage):
            # Try to load data from database
            try:
                # Get open trades
                open_trades = app.storage.get_open_trades()
                positions = []
                for trade in open_trades:
                    positions.append({
                        'symbol': trade.get('symbol'),
                        'side': trade.get('side', 'buy'),
                        'quantity': trade.get('quantity', 0),
                        'entry_price': trade.get('entry_price', 0),
                        'current_price': trade.get('entry_price', 0),  # Fallback
                        'pnl': trade.get('pnl', 0),
                        'trade_id': trade.get('trade_id')
                    })
                
                # Get recent closed trades
                all_trades = app.storage.get_all_trades(limit=50) if hasattr(app.storage, 'get_all_trades') else []
                closed_trades = [t for t in all_trades if t.get('status') == 'closed']
                
                # Calculate totals
                total_balance = 0.0
                if app.brokers:
                    for broker in app.brokers.values():
                        try:
                            balance_info = broker.get_account_balance()
                            total_balance += balance_info.get('total', 0.0)
                        except:
                            pass
                
                # Calculate realized P&L from closed trades
                realized_pnl = sum(t.get('pnl', 0) for t in closed_trades)
                
                # Get initial equity (try to get from reporter, or use balance as fallback)
                initial_equity = 10000.0  # Default
                if reporter and hasattr(reporter, 'initial_equity'):
                    initial_equity = reporter.initial_equity
                if not initial_equity or initial_equity == 0:
                    initial_equity = total_balance if total_balance > 0 else 10000.0
                
                # Calculate equity (balance + unrealized P&L)
                unrealized_pnl = sum(p.get('pnl', 0) for p in positions)
                equity = total_balance + unrealized_pnl
                
                # Calculate total return
                total_return = ((equity - initial_equity) / initial_equity * 100) if initial_equity > 0 else 0.0
                
                # Calculate win/loss rates
                winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
                losing_trades = [t for t in closed_trades if t.get('pnl', 0) < 0]
                win_rate = (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0.0
                loss_rate = (len(losing_trades) / len(closed_trades) * 100) if closed_trades else 0.0
                
                # Update dashboard data with loaded information
                dashboard_data.update({
                    'equity': equity,
                    'balance': total_balance,
                    'positions': positions,
                    'trades': closed_trades[-10:],  # Last 10 closed trades
                    'performance': {
                        'total_return': total_return,
                        'realized_pnl': realized_pnl,
                        'unrealized_pnl': unrealized_pnl,
                        'total_trades': len(closed_trades),
                        'win_rate': win_rate,
                        'loss_rate': loss_rate,
                        'sharpe_ratio': 0.0,
                        'initial_equity': initial_equity
                    }
                })
            except Exception as e:
                print(f"Error loading dashboard data from database: {e}")
                import traceback
                traceback.print_exc()
        
        return _serialize_datetime(dashboard_data)
    
    @app.post("/api/close_position/{trade_id}")
    async def close_position(trade_id: str):
        """
        Close/delete a position by trade_id
        - If position exists in broker (Alpaca/OANDA), close it via API
        - If position is phantom (local only), delete from database
        """
        if not app.storage:
            return {"success": False, "error": "Storage not available"}
        
        try:
            # Get the trade to check if it exists
            open_trades = app.storage.get_open_trades()
            trade = next((t for t in open_trades if t.get('trade_id') == trade_id), None)

            # Fallback: also search all trades (some synced trades may not be marked open)
            if not trade:
                all_trades = app.storage.get_all_trades(limit=500) if hasattr(app.storage, "get_all_trades") else []
                trade = next(
                    (
                        t
                        for t in all_trades
                        if t.get("trade_id") == trade_id or t.get("order_id") == trade_id
                    ),
                    None,
                )

            if not trade:
                # As a last resort, try to treat trade_id as a symbol (UI may send symbol when trade_id missing)
                symbol_guess = trade_id if "/" in trade_id else None
                side_guess = "buy"
                strategy_guess = ""

                if symbol_guess:
                    trade = {"trade_id": trade_id, "symbol": symbol_guess, "side": side_guess, "strategy": strategy_guess}
                else:
                    return {"success": False, "error": "Trade not found"}
            
            symbol = trade.get('symbol')
            side = trade.get('side', 'buy')
            strategy = trade.get('strategy') or ''  # Handle None case
            
            # Check if position exists in any broker
            position_found_in_broker = False
            broker_used = None
            
            # First, try to identify broker from strategy field
            if strategy and '_sync' in strategy:
                broker_name = strategy.replace('_sync', '')
                broker = app.brokers.get(broker_name)
                if broker:
                    try:
                        broker_positions = broker.get_open_positions()
                        # Check if this position exists in broker
                        for pos in broker_positions:
                            broker_symbol = pos.get('symbol', '').replace('_', '/')
                            broker_side = pos.get('side', 'buy')
                            # Match by symbol and side
                            if broker_symbol == symbol and broker_side == side:
                                position_found_in_broker = True
                                broker_used = broker
                                break
                    except Exception as e:
                        print(f"Error checking broker {broker_name} for position: {e}")
            
            # If not found by strategy, check all brokers
            if not position_found_in_broker and app.brokers:
                for broker_name, broker in app.brokers.items():
                    try:
                        broker_positions = broker.get_open_positions()
                        for pos in broker_positions:
                            broker_symbol = pos.get('symbol', '').replace('_', '/')
                            broker_side = pos.get('side', 'buy')
                            if broker_symbol == symbol and broker_side == side:
                                position_found_in_broker = True
                                broker_used = broker
                                break
                        if position_found_in_broker:
                            break
                    except Exception as e:
                        print(f"Error checking broker {broker_name} for position: {e}")
                        continue
            
            # If position exists in broker, close it via API
            if position_found_in_broker and broker_used:
                try:
                    # Convert side to OrderSide enum
                    order_side = OrderSide.SELL if side == 'buy' else OrderSide.BUY
                    
                    # Close position via broker API
                    closed = broker_used.close_position(symbol, order_side)
                    
                    if closed:
                        # Update local database to mark as closed
                        # Get current price for exit price
                        try:
                            current_price = broker_used.get_current_price(symbol)
                        except:
                            current_price = trade.get('entry_price', 0)
                        
                        # Calculate final P&L
                        entry_price = trade.get('entry_price', 0)
                        quantity = trade.get('quantity', 0)
                        if side == 'buy':
                            final_pnl = (current_price - entry_price) * quantity
                        else:
                            final_pnl = (entry_price - current_price) * quantity
                        
                        # Update trade in database
                        app.storage.update_trade(trade_id, {
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
                    else:
                        return {
                            "success": False,
                            "error": f"Failed to close position via broker API"
                        }
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Error closing position via broker: {str(e)}"
                    }
            
            # Position doesn't exist in broker - delete from local database (phantom position)
            success = app.storage.delete_trade(trade_id)
            
            if success:
                return {
                    "success": True,
                    "message": f"Phantom position {symbol} {side} deleted from local database",
                    "method": "local_delete"
                }
            else:
                return {"success": False, "error": "Failed to delete trade"}
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
                    total_balance += balance_info.get('total', 0.0)
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
    """Generate unified dashboard with tabs for Dashboard and Reports"""
    
    # Prepare report data if available
    general = report_data.get('general', {}) if report_data else {}
    periods = report_data.get('periods', {}) if report_data else {}
    advanced = report_data.get('advanced', {}) if report_data else {}
    equity_curve = report_data.get('equity_curve', []) if report_data else []
    monthly = report_data.get('monthly_analytics', []) if report_data else []
    
    # Prepare equity curve data for Chart.js
    equity_labels = [point['date'][:10] for point in equity_curve[-100:]] if equity_curve else []
    equity_values = [point['equity'] for point in equity_curve[-100:]] if equity_curve else []
    drawdown_values = [point['drawdown'] for point in equity_curve[-100:]] if equity_curve else []
    
    # Prepare monthly data
    monthly_labels = [m['month'] for m in monthly]
    monthly_gains = [m['gain_pct'] for m in monthly]
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SmartTrader - Unified Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
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
            .tabs {{
                display: flex;
                background: white;
                padding: 0;
                border-radius: 10px 10px 0 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 0;
            }}
            .tab {{
                padding: 15px 30px;
                cursor: pointer;
                border: none;
                background: none;
                font-size: 16px;
                font-weight: 500;
                color: #666;
                transition: all 0.2s;
                border-bottom: 3px solid transparent;
            }}
            .tab:hover {{ color: #667eea; }}
            .tab.active {{
                color: #667eea;
                border-bottom: 3px solid #667eea;
            }}
            .tab-content {{
                display: none;
                background: white;
                padding: 25px;
                border-radius: 0 0 10px 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .tab-content.active {{
                display: block;
            }}
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
            .metrics {{
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
            .btn-close {{
                background: #dc3545;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 4px;
                cursor: pointer;
            }}
            .btn-close:hover {{ background: #c82333; }}
            .btn-close:disabled {{
                background: #ccc;
                cursor: not-allowed;
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
                <h1>📊 SmartTrader - Unified Dashboard</h1>
                <p>Real-time monitoring and comprehensive performance analytics</p>
            </div>
            
            <div class="tabs">
                <button class="tab active" onclick="switchTab('dashboard')">📈 Dashboard</button>
                <button class="tab" onclick="switchTab('reports')">📊 Reports</button>
            </div>
            
            <!-- Dashboard Tab -->
            <div id="dashboard" class="tab-content active">
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
            
            <!-- Reports Tab -->
            <div id="reports" class="tab-content">
                <div class="card">
                    <h2>📈 General Account Information</h2>
                    <div class="metrics">
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
                
                <div class="card">
                    <h2>📊 Equity Curve</h2>
                    <div class="tabs" style="margin-top: 0;">
                        <button class="tab active" onclick="switchChartTab('equity')">Equity</button>
                        <button class="tab" onclick="switchChartTab('drawdown')">Drawdown</button>
                    </div>
                    <div id="equity-chart" class="tab-content active">
                        <div class="chart-container">
                            <canvas id="equityChart"></canvas>
                        </div>
                    </div>
                    <div id="drawdown-chart" class="tab-content">
                        <div class="chart-container">
                            <canvas id="drawdownChart"></canvas>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <h2>📅 Trading Periods</h2>
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
                
                <div class="card">
                    <h2>🔬 Advanced Statistics</h2>
                    <div class="metrics">
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
                
                <div class="card">
                    <h2>📆 Monthly Analytics</h2>
                    <div class="chart-container">
                        <canvas id="monthlyChart"></canvas>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // Tab switching
            function switchTab(tabName) {{
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                event.target.classList.add('active');
                document.getElementById(tabName).classList.add('active');
                
                // Initialize charts when Reports tab is opened
                if (tabName === 'reports' && !window.chartsInitialized) {{
                    initializeCharts();
                    window.chartsInitialized = true;
                }}
            }}
            
            function switchChartTab(tabName) {{
                document.querySelectorAll('#reports .tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('#reports .tab-content').forEach(c => c.classList.remove('active'));
                event.target.classList.add('active');
                document.getElementById(tabName + '-chart').classList.add('active');
            }}
            
            // Load initial dashboard data via API (fallback if WebSocket hasn't connected yet)
            async function loadInitialData() {{
                try {{
                    const response = await fetch('/api/dashboard');
                    const data = await response.json();
                    updateDashboard(data);
                }} catch (error) {{
                    console.error('Error loading initial dashboard data:', error);
                    // Try fallback endpoint
                    try {{
                        const response = await fetch('/api/status');
                        const data = await response.json();
                        updateDashboard(data);
                    }} catch (e) {{
                        console.error('Error loading dashboard status:', e);
                    }}
                }}
            }}
            
            // Load data immediately on page load
            loadInitialData();
            
            // WebSocket for real-time dashboard updates
            const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsHost = window.location.host;
            const ws = new WebSocket(`${{wsProtocol}}//${{wsHost}}/ws`);
            
            ws.onopen = function() {{
                console.log('WebSocket connected');
            }};
            
            ws.onerror = function(error) {{
                console.error('WebSocket error:', error);
                // Fallback to polling if WebSocket fails
                setInterval(loadInitialData, 5000); // Poll every 5 seconds
            }};
            
            ws.onmessage = function(event) {{
                const data = JSON.parse(event.data);
                updateDashboard(data);
            }};
            
            ws.onclose = function() {{
                console.log('WebSocket closed, falling back to polling');
                // Fallback to polling if WebSocket closes
                setInterval(loadInitialData, 5000); // Poll every 5 seconds
            }};
            
            function updateDashboard(data) {{
                // Only update if Dashboard tab is active
                if (!document.getElementById('dashboard').classList.contains('active')) return;
                
                // Update metrics
                const metricsDiv = document.getElementById('metrics');
                metricsDiv.innerHTML = `
                    <div class="metric">
                        <div class="metric-value">$${{data.equity.toFixed(2)}}</div>
                        <div class="metric-label">Equity</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">$${{data.balance.toFixed(2)}}</div>
                        <div class="metric-label">Balance</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value ${{data.performance.total_return >= 0 ? 'positive' : 'negative'}}">
                            ${{data.performance.total_return?.toFixed(2) || 0}}%
                        </div>
                        <div class="metric-label">Total Return</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value ${{data.performance.realized_pnl >= 0 ? 'positive' : 'negative'}}">
                            $${{data.performance.realized_pnl?.toFixed(2) || 0}}
                        </div>
                        <div class="metric-label">Realized P&L</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value ${{data.performance.unrealized_pnl >= 0 ? 'positive' : 'negative'}}">
                            $${{data.performance.unrealized_pnl?.toFixed(2) || 0}}
                        </div>
                        <div class="metric-label">Unrealized P&L</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${{data.performance.sharpe_ratio?.toFixed(2) || 0}}</div>
                        <div class="metric-label">Sharpe Ratio</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value positive">${{data.performance.win_rate?.toFixed(1) || 0}}%</div>
                        <div class="metric-label">Win Rate</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value negative">${{data.performance.loss_rate?.toFixed(1) || 0}}%</div>
                        <div class="metric-label">Loss Rate</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${{data.performance.total_trades || 0}}</div>
                        <div class="metric-label">Total Trades</div>
                    </div>
                `;
                
                // Update positions
                const positionsBody = document.querySelector('#positions tbody');
                positionsBody.innerHTML = data.positions.map(pos => {{
                    const isPhantom = !pos.strategy || pos.strategy === 'smarttrader' || (pos.strategy && !pos.strategy.includes('_sync') && pos.strategy !== 'unknown');
                    const buttonText = isPhantom ? 'Delete' : 'Close';
                    const buttonTitle = isPhantom ? 'Delete this phantom position (local only)' : 'Close position via broker API';
                    
                    return `
                        <tr>
                            <td>${{pos.symbol}}</td>
                            <td>${{pos.side}}</td>
                            <td>${{pos.quantity}}</td>
                            <td>$${{pos.entry_price}}</td>
                            <td>$${{pos.current_price || pos.entry_price}}</td>
                            <td class="${{pos.pnl >= 0 ? 'positive' : 'negative'}}">$${{pos.pnl.toFixed(2)}}</td>
                            <td>
                                <button class="btn-close" onclick="closePosition('${{pos.trade_id}}', '${{pos.symbol}}', '${{pos.side}}')" title="${{buttonTitle}}">
                                    ${{buttonText}}
                                </button>
                            </td>
                        </tr>
                    `;
                }}).join('');
                
                // Update trades
                const tradesBody = document.querySelector('#trades tbody');
                if (data.trades && data.trades.length > 0) {{
                    tradesBody.innerHTML = data.trades.slice(0, 10).map(trade => `
                        <tr>
                            <td>${{new Date(trade.time || Date.now()).toLocaleString()}}</td>
                            <td>${{trade.symbol}}</td>
                            <td>${{trade.side}}</td>
                            <td>${{trade.quantity.toFixed(4)}}</td>
                            <td>$${{trade.price.toFixed(2)}}</td>
                            <td class="${{trade.pnl >= 0 ? 'positive' : 'negative'}}">$${{trade.pnl.toFixed(2)}}</td>
                        </tr>
                    `).join('');
                }} else {{
                    tradesBody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #999;">No closed trades yet</td></tr>';
                }}
                
                // Update risk metrics
                const riskMetricsDiv = document.getElementById('risk-metrics');
                if (data.risk_metrics) {{
                    riskMetricsDiv.innerHTML = `
                        <p><strong>Drawdown:</strong> ${{data.risk_metrics.drawdown_percent?.toFixed(2) || 0}}%</p>
                        <p><strong>Peak Equity:</strong> $${{data.risk_metrics.peak_equity?.toFixed(2) || 0}}</p>
                        <p><strong>Current Equity:</strong> $${{data.risk_metrics.current_equity?.toFixed(2) || 0}}</p>
                    `;
                }}
            }}
            
            async function closePosition(tradeId, symbol, side) {{
                if (!confirm(`Are you sure you want to close/delete this position?\\n\\nSymbol: ${{symbol}}\\nSide: ${{side}}\\n\\nIf the position exists in Alpaca/OANDA, it will be closed via API.\\nIf it's a phantom position, it will be deleted from the database.`)) {{
                    return;
                }}
                
                const buttons = document.querySelectorAll('.btn-close');
                let button = null;
                for (let btn of buttons) {{
                    if (btn.getAttribute('onclick').includes(tradeId)) {{
                        button = btn;
                        break;
                    }}
                }}
                
                if (button) {{
                    button.disabled = true;
                    button.textContent = 'Processing...';
                }}
                
                try {{
                    const response = await fetch(`/api/close_position/${{tradeId}}`, {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }}
                    }});
                    
                    const result = await response.json();
                    
                    if (result.success) {{
                        const action = result.method === 'broker_api' ? 'closed' : 'deleted';
                        alert(result.message || `Position ${{action}} successfully`);
                    }} else {{
                        alert('Error: ' + (result.error || 'Failed to close/delete position'));
                        if (button) {{
                            button.disabled = false;
                            button.textContent = button.getAttribute('title')?.includes('phantom') ? 'Delete' : 'Close';
                        }}
                    }}
                }} catch (error) {{
                    alert('Error: ' + error.message);
                    if (button) {{
                        button.disabled = false;
                        button.textContent = button.getAttribute('title')?.includes('phantom') ? 'Delete' : 'Close';
                    }}
                }}
            }}
            
            // Initialize charts for Reports tab
            function initializeCharts() {{
                // Equity Chart
                const equityCtx = document.getElementById('equityChart');
                if (equityCtx && {json.dumps(len(equity_values))} > 0) {{
                    new Chart(equityCtx.getContext('2d'), {{
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
                            scales: {{ y: {{ beginAtZero: false }} }}
                        }}
                    }});
                }}
                
                // Drawdown Chart
                const drawdownCtx = document.getElementById('drawdownChart');
                if (drawdownCtx && {json.dumps(len(drawdown_values))} > 0) {{
                    new Chart(drawdownCtx.getContext('2d'), {{
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
                            scales: {{ y: {{ beginAtZero: true, max: 100 }} }}
                        }}
                    }});
                }}
                
                // Monthly Chart
                const monthlyCtx = document.getElementById('monthlyChart');
                if (monthlyCtx && {json.dumps(len(monthly_labels))} > 0) {{
                    new Chart(monthlyCtx.getContext('2d'), {{
                        type: 'bar',
                        data: {{
                            labels: {json.dumps(monthly_labels)},
                            datasets: [{{
                                label: 'Monthly Gain %',
                                data: {json.dumps(monthly_gains)},
                                backgroundColor: monthly_gains.map(g => g >= 0 ? '#28a745' : '#dc3545')
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {{
                                legend: {{ display: true }},
                                title: {{ display: true, text: 'Monthly Performance' }}
                            }},
                            scales: {{ y: {{ beginAtZero: false }} }}
                        }}
                    }});
                }}
            }}
        </script>
    </body>
    </html>
    """
    return html


def _generate_reports_html(report_data: Dict) -> str:
    """Generate comprehensive HTML reports page similar to MyFxBook"""
    
    general = report_data.get('general', {})
    periods = report_data.get('periods', {})
    advanced = report_data.get('advanced', {})
    equity_curve = report_data.get('equity_curve', [])
    monthly = report_data.get('monthly_analytics', [])
    
    # Prepare equity curve data for Chart.js
    equity_labels = [point['date'][:10] for point in equity_curve[-100:]] if equity_curve else []
    equity_values = [point['equity'] for point in equity_curve[-100:]] if equity_curve else []
    drawdown_values = [point['drawdown'] for point in equity_curve[-100:]] if equity_curve else []
    
    # Prepare monthly data
    monthly_labels = [m['month'] for m in monthly]
    monthly_gains = [m['gain_pct'] for m in monthly]
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SmartTrader - Performance Reports</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
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
                <h1>📊 SmartTrader Performance Reports</h1>
                <p>Comprehensive trading analytics and statistics</p>
            </div>
            
            <div class="nav">
                <a href="/">Dashboard</a>
                <a href="/reports" class="active">Reports</a>
            </div>
            
            <!-- General Account Info -->
            <div class="card">
                <h2>📈 General Account Information</h2>
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
                <h2>📊 Equity Curve</h2>
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
                <h2>📅 Trading Periods</h2>
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
                <h2>🔬 Advanced Statistics</h2>
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
                <h2>📆 Monthly Analytics</h2>
                <div class="chart-container">
                    <canvas id="monthlyChart"></canvas>
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
            
            // Monthly Chart
            const monthlyCtx = document.getElementById('monthlyChart').getContext('2d');
            new Chart(monthlyCtx, {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(monthly_labels)},
                    datasets: [{{
                        label: 'Monthly Gain %',
                        data: {json.dumps(monthly_gains)},
                        backgroundColor: monthly_gains.map(g => g >= 0 ? '#28a745' : '#dc3545')
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: true }},
                        title: {{ display: true, text: 'Monthly Performance' }}
                    }},
                    scales: {{
                        y: {{ beginAtZero: false }}
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """
    return html

