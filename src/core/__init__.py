"""SmartTrader v2 core.

Research-first trading core:
- data: cached historical OHLCV (Yahoo Finance or OANDA)
- strategies: small set of causal, parameterised rule-based strategies
- backtest: next-bar-open execution with per-market costs and risk-based sizing
- walk_forward: out-of-sample validation with plateau-seeking parameter choice
- risk_governor: hard, human-owned account limits
- config_manager: bounded optimiser that writes the live strategy config
"""
