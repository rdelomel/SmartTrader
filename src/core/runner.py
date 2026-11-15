"""Process entry point for the v2 engine: brokers + live engine + dashboard."""

import logging
import os
import signal
import threading
import time

import yaml

from .live_engine import LiveEngine

log = logging.getLogger('smarttrader.v2')


RECONNECT_EVERY_SECONDS = int(os.getenv('BROKER_RECONNECT_SECONDS', '300'))
BROKER_ENV = (('oanda', 'OANDA_API_KEY'), ('alpaca', 'ALPACA_API_KEY'))


def configured_brokers() -> list:
    return [name for name, var in BROKER_ENV if os.getenv(var)]


def connect_broker(name: str):
    """Build and connect one broker from the environment; None if it fails."""
    if name == 'oanda':
        from ..data.brokers.oanda_api import OANDABroker

        account_id = os.getenv('OANDA_ACCOUNT_ID', '').strip()
        b = OANDABroker(os.getenv('OANDA_API_KEY'),
                        account_id=None if not account_id or account_id.startswith('#') else account_id,
                        testnet=os.getenv('OANDA_ENVIRONMENT', 'practice') == 'practice')
    elif name == 'alpaca':
        from ..data.brokers.alpaca_api import AlpacaBroker

        base_url = os.getenv('ALPACA_BASE_URL', '')
        b = AlpacaBroker(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_API_SECRET'),
                         testnet='paper' in base_url.lower() if base_url else True)
    else:
        return None
    return b if b.connect() else None


def build_brokers() -> dict:
    brokers = {}
    for name in configured_brokers():
        b = connect_broker(name)
        if b:
            brokers[name] = b
        else:
            log.error('%s connection failed; its markets are disabled until it reconnects '
                      '(retrying every %ss)', name, RECONNECT_EVERY_SECONDS)
    return brokers


def _reconnect_missing(engine, brokers) -> None:
    for name in configured_brokers():
        if name in brokers:
            continue
        b = connect_broker(name)
        if b:
            engine.add_broker(name, b)
            log.info('%s reconnected; markets re-enabled', name)


def _start_dashboard(storage, brokers):
    import uvicorn

    from ..monitoring.dashboard import create_dashboard_app

    trading_config = {}
    path = os.path.join('config', 'trading_config.yaml')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            trading_config = yaml.safe_load(f) or {}
    app = create_dashboard_app(storage=storage, brokers=brokers, trading_config=trading_config)
    port = int(os.getenv('PORT', 8000))
    threading.Thread(target=lambda: uvicorn.run(app, host='0.0.0.0', port=port, log_level='warning'),
                     daemon=True).start()
    return app


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    os.makedirs('data', exist_ok=True)

    brokers = build_brokers()
    if not brokers:
        raise SystemExit('No broker could be connected; set OANDA_* and/or ALPACA_* environment variables')

    storage = None
    try:
        from ..data.data_storage import DataStorage

        storage = DataStorage()
    except Exception as e:
        log.error('Trade storage unavailable (%s); dashboard history disabled', e)

    engine = LiveEngine(brokers, storage=storage)
    app = _start_dashboard(storage, brokers)
    log.info('SmartTrader v2 engine started with brokers: %s', ', '.join(brokers))

    running = {'on': True}

    def _stop(*_):
        running['on'] = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    interval = int(os.getenv('ENGINE_INTERVAL_SECONDS', '60'))
    last_reconnect = time.time()
    while running['on']:
        if time.time() - last_reconnect >= RECONNECT_EVERY_SECONDS:
            last_reconnect = time.time()
            try:
                _reconnect_missing(engine, brokers)
            except Exception:
                log.exception('broker reconnect failed')
        try:
            status = engine.run_once()
            try:
                from .integrity_sync import sync_broker_only_to_storage
                sync_broker_only_to_storage(engine)
            except Exception:
                log.exception('integrity sync failed')
            app.update_dashboard_data({'skip_counters': dict(engine.state.counters), 'engine_status': status})
        except Exception:
            log.exception('engine loop error')
        for _ in range(interval):
            if not running['on']:
                break
            time.sleep(1)
    log.info('SmartTrader v2 engine stopped')
