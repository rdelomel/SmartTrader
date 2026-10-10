"""Walk-forward research + bounded config update.

Examples:
  python scripts/research.py                       # report only
  python scripts/research.py --assets forex,metals --timeframes 1d
  python scripts/research.py --apply               # write config/live_strategy.yaml
  python scripts/research.py --source oanda        # use OANDA candles for forex/metals (needs OANDA_API_KEY)
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.core.config_manager import ConfigManager  # noqa: E402
from src.core.data import HistoryProvider  # noqa: E402


def _oanda_broker():
    from dotenv import load_dotenv

    from src.data.brokers.oanda_api import OANDABroker

    load_dotenv()
    key = os.getenv('OANDA_API_KEY')
    if not key:
        raise SystemExit('OANDA_API_KEY is not set')
    broker = OANDABroker(api_key=key, account_id=os.getenv('OANDA_ACCOUNT_ID'), testnet=True)
    if not broker.connect():
        raise SystemExit('Could not connect to OANDA')
    return broker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--assets', help='comma list: forex,metals,crypto,stocks')
    ap.add_argument('--timeframes', help='comma list: 1d,4h')
    ap.add_argument('--samples', type=int, help='parameter samples per strategy')
    ap.add_argument('--source', choices=['yahoo', 'oanda'], default='yahoo')
    ap.add_argument('--apply', action='store_true', help='write config/live_strategy.yaml')
    args = ap.parse_args()

    provider = HistoryProvider(oanda_broker=_oanda_broker() if args.source == 'oanda' else None)
    cm = ConfigManager()
    print('Running walk-forward research (out-of-sample results only)...')
    results = cm.research(
        provider,
        asset_classes=args.assets.split(',') if args.assets else None,
        timeframes=args.timeframes.split(',') if args.timeframes else None,
        samples=args.samples,
    )
    chosen, rejected = cm.select(results)
    previous = cm.load_live()
    risks, estimate = cm.allocate(chosen, previous)

    print(f'\n{len(results)} candidates tested, {len(chosen)} enabled.')
    for r in chosen:
        o = r.oos
        print(f'  ENABLE {r.symbol:8} {r.strategy:15} {r.timeframe:3} risk={risks[r.symbol]:.2f}% '
              f'trades={o["trades"]} PF={o["profit_factor"]:.2f} expR={o["expectancy_r"]:+.3f} '
              f'sharpe={o["sharpe"]:.2f}')
    print('\nPortfolio estimate (out-of-sample, after costs):')
    for k in ('avg_month_pct', 'median_month_pct', 'worst_month_pct', 'pct_positive_months',
              'cagr_pct', 'max_drawdown_pct', 'sharpe', 'expected_open_risk_pct'):
        if k in estimate:
            print(f'  {k:24} {estimate[k]:8.2f}')
    print('  (5-10%/month target is not supported by these numbers unless avg_month_pct says so.)')

    if args.apply:
        cfg = cm.build(chosen, risks, estimate, previous, data_source=args.source)
        path, changes, explanation = cm.apply(cfg, previous)
        print(f'\nWrote {path} (version {cfg["version"]}).')
        for c in changes:
            print('  ' + c)
        print('\n' + explanation)
    else:
        print('\nDry run. Re-run with --apply to write config/live_strategy.yaml.')


if __name__ == '__main__':
    main()
