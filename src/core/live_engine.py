"""Live engine for the v2 core.

Trades config/live_strategy.yaml with the same rules as core.backtest:
- Signals are evaluated once per newly closed bar, then executed at the next
  opportunity the market is open (the live equivalent of "next bar open").
- Initial stop = entry -/+ stop_dist; trailing stops ratchet once per closed bar.
- Every entry goes through the RiskGovernor of the broker account it trades in.
The broker is the source of truth for open positions. State survives restarts.
"""

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from ..data.brokers.base_broker import OrderSide, OrderType
from .config_manager import LIVE_PATH, load_space, validate_live_config
from .data import HistoryProvider, latest_closed_bars
from .risk_governor import OpenPosition, RiskGovernor, TradeProposal, load_limits
from .strategies import generate

log = logging.getLogger('smarttrader.v2')

BAR_SPAN = {'1d': pd.Timedelta(days=1), '4h': pd.Timedelta(hours=4), '1h': pd.Timedelta(hours=1)}
REFETCH_EVERY = {'1d': pd.Timedelta(minutes=60), '4h': pd.Timedelta(minutes=15), '1h': pd.Timedelta(minutes=5)}
LIVE_HISTORY_YEARS = {'1d': 6.0, '4h': 1.9, '1h': 0.5}
BROKER_FOR = {'forex': 'oanda', 'metals': 'oanda', 'crypto': 'alpaca', 'stocks': 'alpaca'}
FOREIGN_POSITION_RISK_PCT = 0.5
MISSING_CHECKS_BEFORE_CLOSED = 2
MAX_ORDER_ATTEMPTS = 3
STATE_PATH = os.path.join('data', 'live_engine_state.json')
EVENT_LOG = os.path.join('data', 'live_engine_events.jsonl')


def symbol_key(symbol: str) -> str:
    return symbol.upper().replace('/', '').replace('_', '')
