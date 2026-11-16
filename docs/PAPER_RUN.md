# 2-month paper run

Goal: find out whether the 7 configured strategies hold up on paper within a 5% max drawdown.
Expect roughly 30-60 closed trades in 2 months. That is enough to catch a broken strategy or a
drawdown problem, not to prove a win rate.

## Start (once, from a flat account)
1. Build machine: `git pull && docker build --no-cache -t rdelomel/smarttrader:latest . && docker push rdelomel/smarttrader:latest`
   (deploy.sh copies `config/` out of the image, so config changes only arrive via a new image).
2. Close every open position on both paper accounts (OANDA and Alpaca). Unmanaged positions have no
   stops and each one eats 0.5% of the open-risk budget.
3. On the NAS, archive old state so peak equity and stats start fresh:
   ```
   cd ~/SmartTrader && docker stop smarttrader
   mkdir -p data/archive_pre_run
   mv data/live_engine_events.jsonl data/live_engine_state.json data/governor_state_*.json data/equity_curve.jsonl data/archive_pre_run/ 2>/dev/null
   git pull && bash deploy.sh
   ```
4. Check `docker logs smarttrader --tail 30`: brokers `oanda, alpaca`, `config_loaded` with 7 strategies.
   Write down the start date and both account equities.

## Freeze
Do not edit `config/*.yaml`, re-run the config manager, or redeploy during the run, except for a bug fix
(note the date, and use `--since` after it). Changing settings mid-run makes the results uninterpretable.

## Check in
Weekly: `python3 scripts/paper_report.py --since <start date>` and share the output.
Read per-symbol rows for sanity only; with a handful of trades each they say nothing about edge.

## Pass criteria (aggregate, at 2 months)
| Check | Bar |
|---|---|
| Closed trades | >= 30 |
| Max drawdown (equity curve, each account) | <= 5% (governor halts at 5%) |
| Profit factor | >= 1.15 |
| Expectancy | >= +0.05R |
| Win rate | >= 50% (backtest OOS was about 62%) |
| Return | > 0 |
| Engine health | no repeated `order_failed`, no orphan positions, OANDA connected |

## Decide
- All pass: scale per-strategy risk up by at most 1.5x, run another month, then consider tiny live size.
- Drawdown over 5% or PF under 1: stop, re-tune, restart the 2 months.
- Mixed or too few trades: extend a month before changing anything.

Realistic return at a 5% drawdown is about 0.4-0.8% a month. 3-5% a month is not expected from this
book; raising risk to chase it only raises drawdown.
