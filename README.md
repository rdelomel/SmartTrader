# SmartTrader - AI Trading Agent System

A fully automated AI trading agent that manages trades, market analysis, and risk across multiple asset classes (Cryptocurrency, Forex, Stocks).

## Features

- **Multi-Asset Support**: Trade cryptocurrency, forex, and stocks through a unified interface
- **Modular Broker Integration**: Support for OANDA (forex) and Alpaca (stocks & crypto) with easy extension to other brokers
- **Full Agentic System**: Combines multiple AI models (XGBoost, LSTM, Sentiment Analysis) with intelligent decision-making
- **Comprehensive Risk Management**: Dynamic position sizing, ATR-based stop-loss, drawdown protection, and kill switch
- **Adaptive Strategies**: Automatically switches strategies based on market conditions and model performance
- **Paper & Live Trading**: Start with paper trading and seamlessly switch to live trading
- **Real-time Monitoring**: Web-based dashboard for real-time performance tracking
- **Backtesting Framework**: Validate strategies on historical data before live deployment

## Architecture

```
SmartTrader/
├── config/              # Configuration files
├── src/
│   ├── data/           # Data acquisition and storage
│   ├── indicators/      # Technical indicators
│   ├── strategies/      # Trading strategies
│   ├── ai/             # AI models and training
│   ├── agent/          # Decision engine and strategy selection
│   ├── risk/           # Risk management
│   ├── backtesting/    # Backtesting framework
│   ├── execution/       # Order execution
│   └── monitoring/      # Dashboard and logging
└── tests/              # Unit and integration tests
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd SmartTrader
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# On Windows:
copy .env.example .env

# On Linux/Mac:
cp .env.example .env
```
Then edit `.env` with your API keys and configuration. You only need to fill in the API keys for the brokers you plan to use.

5. Configure trading parameters:
```bash
# Edit config/trading_config.yaml to set risk limits and asset classes
# Edit config/broker_config.yaml for broker-specific settings
# Edit config/model_config.yaml for AI model parameters
```

## Configuration

### Trading Configuration (`config/trading_config.yaml`)

- Risk management parameters (max drawdown, position sizing)
- Trading hours for each asset class
- Symbols to trade
- Strategy weights and parameters

### Broker Configuration (`config/broker_config.yaml`)

- Broker-specific settings
- Rate limits and order types
- Minimum order sizes

### Model Configuration (`config/model_config.yaml`)

- AI model hyperparameters
- Feature engineering settings
- Training and retraining schedules

## Usage

### Paper Trading (Recommended First Step)

1. Ensure `TRADING_MODE=paper` in `.env`
2. Run the agent:
```bash
python src/main.py
```

3. Monitor performance via the dashboard (default: http://localhost:8000)

### Backtesting

Test strategies on historical data:
```bash
python -m src.backtesting.engine --start-date 2023-01-01 --end-date 2023-12-31
```

### Live Trading

**WARNING**: Only enable live trading after extensive paper trading validation.

1. Set `TRADING_MODE=live` and `ENABLE_LIVE_TRADING=true` in `.env`
2. Review all risk parameters in `config/trading_config.yaml`
3. Start with minimal position sizes
4. Run the agent:
```bash
python src/main.py
```

## Risk Management

The system includes multiple layers of risk protection:

- **Position Sizing**: Fixed fractional or volatility-scaled sizing
- **Stop-Loss**: Dynamic ATR-based stop-loss levels
- **Drawdown Protection**: Automatic kill switch at max drawdown
- **Anomaly Detection**: Pauses trading during unusual market conditions
- **Model Monitoring**: Automatic retraining when performance degrades

## Monitoring Dashboard

Access the real-time dashboard at `http://localhost:8000` to monitor:
- Current P&L and open positions
- Model performance metrics
- Risk parameters (drawdown, position sizes)
- Trade history and statistics
- System health and alerts

## Testing

Run the test suite:
```bash
pytest tests/
```

## Security Best Practices

- Never commit `.env` file with real API keys
- Use testnet/demo accounts for initial testing
- Start with paper trading mode
- Set conservative risk limits initially
- Monitor the system closely during initial live trading
- Keep API keys with minimal required permissions

## Disclaimer

Trading involves substantial risk of loss. This software is provided for educational purposes. Past performance does not guarantee future results. Always test thoroughly in paper trading mode before risking real capital.

## License

[Specify your license here]

## Contributing

[Contributing guidelines if applicable]

