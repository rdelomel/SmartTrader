"""Main orchestration loop for AI trading agent"""

import os
import yaml
import time
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from dotenv import load_dotenv
import pandas as pd

# Patch gymnasium as gym before stable-baselines3 imports it
try:
    import gymnasium as gym
    import sys
    sys.modules['gym'] = gym  # Make gymnasium available as 'gym' for stable-baselines3
except ImportError:
    pass  # If gymnasium not available, let stable-baselines3 use gym

# Load environment variables
load_dotenv()

# Import components
try:
    # Try relative imports first (when used as module)
    from .data.brokers.oanda_api import OANDABroker
    from .data.brokers.alpaca_api import AlpacaBroker
    from .data.data_storage import DataStorage
    from .data.data_fetcher import DataFetcher
    from .data.data_preprocessor import DataPreprocessor
    from .indicators.technical import TechnicalIndicators
    from .indicators.feature_engineering import FeatureEngineer
    from .strategies.trend_following import TrendFollowingStrategy
    from .strategies.mean_reversion import MeanReversionStrategy
    from .strategies.breakout import BreakoutStrategy
    from .strategies.momentum import MomentumStrategy
    from .strategies.volatility import VolatilityStrategy
    from .strategies.news_trading import NewsTradingStrategy
    from .strategies.end_of_day import EndOfDayStrategy
    from .strategies.swing_trading import SwingTradingStrategy
    from .strategies.day_trading import DayTradingStrategy
    from .strategies.scalping import ScalpingStrategy
    from .data.news_fetcher import NewsFetcher
    from .ai.models.ml_models import MLModel
    try:
        from .ai.models.lstm_model import LSTMModel
    except ImportError:
        LSTMModel = None  # LSTM model optional if TensorFlow not available
    from .ai.models.sentiment_analyzer import SentimentAnalyzer
    from .ai.training.trainer import ModelTrainer
    from .ai.anomaly_detector import AnomalyDetector
    from .agent.decision_engine import DecisionEngine  # Keep for backward compatibility
    from .agent.strategy_selector import StrategySelector
    from .agent.analytical_agents import TechnicalAnalystAgent, SentimentAgent, FundamentalAgent
    from .agent.quantitative_agent import QuantitativeAgent
    from .agent.pattern_forecaster_agent import PatternForecasterAgent
    from .agent.regime_switching_agent import RegimeSwitchingAgent
    from .agent.orchestrator_agent import OrchestratorAgent
    from .agent.risk_manager_agent import RiskManagerAgent
    from .ai.drl.drl_agent import DRLAgent
    from .ai.drl.trainer import DRLTrainer
    from .risk.circuit_breaker import CircuitBreaker
    from .monitoring.performance_tracker import PerformanceTracker
    from .risk.position_sizer import PositionSizer
    from .risk.stop_loss import StopLossManager
    from .risk.drawdown_manager import DrawdownManager
    from .risk.risk_calculator import RiskCalculator
    from .execution.order_manager import OrderManager
    from .monitoring.logger import TradingLogger
    from .monitoring.dashboard import create_dashboard_app
    from .data.brokers.base_broker import OrderSide, OrderType, OrderStatus
    from .strategies.base_strategy import Signal
except ImportError:
    # Fall back to absolute imports (when run directly)
    from src.data.brokers.oanda_api import OANDABroker
    from src.data.brokers.alpaca_api import AlpacaBroker
    from src.data.data_storage import DataStorage
    from src.data.data_fetcher import DataFetcher
    from src.data.data_preprocessor import DataPreprocessor
    from src.indicators.technical import TechnicalIndicators
    from src.indicators.feature_engineering import FeatureEngineer
    from src.strategies.trend_following import TrendFollowingStrategy
    from src.strategies.mean_reversion import MeanReversionStrategy
    from src.strategies.breakout import BreakoutStrategy
    from src.strategies.momentum import MomentumStrategy
    from src.strategies.volatility import VolatilityStrategy
    from src.strategies.news_trading import NewsTradingStrategy
    from src.strategies.end_of_day import EndOfDayStrategy
    from src.strategies.swing_trading import SwingTradingStrategy
    from src.strategies.day_trading import DayTradingStrategy
    from src.strategies.scalping import ScalpingStrategy
    from src.data.news_fetcher import NewsFetcher
    from src.ai.models.ml_models import MLModel
    try:
        from src.ai.models.lstm_model import LSTMModel
    except ImportError:
        LSTMModel = None  # LSTM model optional if TensorFlow not available
    from src.ai.models.sentiment_analyzer import SentimentAnalyzer
    from src.ai.training.trainer import ModelTrainer
    from src.ai.anomaly_detector import AnomalyDetector
    from src.agent.decision_engine import DecisionEngine  # Keep for backward compatibility
    from src.agent.strategy_selector import StrategySelector
    from src.agent.analytical_agents import TechnicalAnalystAgent, SentimentAgent, FundamentalAgent
    from src.agent.regime_switching_agent import RegimeSwitchingAgent
    from src.agent.orchestrator_agent import OrchestratorAgent
    from src.agent.risk_manager_agent import RiskManagerAgent
    from src.ai.drl.drl_agent import DRLAgent
    from src.ai.drl.trainer import DRLTrainer
    from src.risk.circuit_breaker import CircuitBreaker
    from src.monitoring.performance_tracker import PerformanceTracker
    from src.risk.position_sizer import PositionSizer
    from src.risk.stop_loss import StopLossManager
    from src.risk.drawdown_manager import DrawdownManager
    from src.risk.risk_calculator import RiskCalculator
    from src.execution.order_manager import OrderManager
    from src.monitoring.logger import TradingLogger
    from src.monitoring.dashboard import create_dashboard_app
    from src.data.brokers.base_broker import OrderSide, OrderType, OrderStatus
    from src.strategies.base_strategy import Signal


class TradingAgent:
    """Main trading agent orchestration"""
    
    def __init__(self, config_dir: str = "config"):
        """Initialize trading agent"""
        try:
            print(f"[INIT] Starting TradingAgent initialization...")
            self.config_dir = config_dir
            self.running = False
            self.initial_equity: Optional[float] = None
            
            # Load configurations
            print(f"[INIT] Loading configuration files...")
            self.trading_config = self._load_config(f"{config_dir}/trading_config.yaml")
            self.broker_config = self._load_config(f"{config_dir}/broker_config.yaml")
            self.model_config = self._load_config(f"{config_dir}/model_config.yaml")
            print(f"[INIT] Configuration files loaded")
            
            # Initialize components
            print(f"[INIT] Initializing DataStorage...")
            self.storage = DataStorage()
            print(f"[INIT] DataStorage initialized")
            
            print(f"[INIT] Initializing Logger...")
            self.logger = TradingLogger(self.trading_config.get('logging', {}))
            print(f"[INIT] Logger initialized")
            
            print(f"[INIT] Initializing preprocessor, indicators, feature engineer...")
            self.preprocessor = DataPreprocessor()
            self.indicators = TechnicalIndicators()
            self.feature_engineer = FeatureEngineer()
            print(f"[INIT] Preprocessing components initialized")
            
            # Initialize brokers
            print(f"[INIT] Initializing brokers...")
            self.brokers = self._initialize_brokers()
            print(f"[INIT] Brokers initialized: {list(self.brokers.keys())}")
            
            # Update default broker mapping - use Alpaca for crypto
            if 'crypto' not in self.brokers and 'stocks' in self.brokers:
                # Use Alpaca for crypto if Binance not available
                self.brokers['crypto'] = self.brokers['stocks']
            
            # Initialize dictionaries for per-broker tracking (before creating managers)
            self.drawdown_managers = {}  # broker_name -> DrawdownManager
            self.circuit_breakers = {}  # broker_name -> CircuitBreaker
            self.performance_trackers = {}  # broker_name -> PerformanceTracker
            
            # Initialize per-broker drawdown managers, circuit breakers, and performance trackers
            # This allows separate drawdown tracking for each account
            risk_config = self.trading_config.get('risk', {})
            aggressive_config = self.trading_config.get('aggressive_mode', {})
            circuit_breaker_config = self.trading_config.get('agents', {}).get('risk_manager', {}).get('circuit_breaker', {})
            
            for broker_name, broker in self.brokers.items():
                # Create drawdown manager for this broker
                self.drawdown_managers[broker_name] = DrawdownManager(risk_config)
                
                # Create circuit breaker for this broker
                self.circuit_breakers[broker_name] = CircuitBreaker(circuit_breaker_config)
                
                # Create performance tracker for this broker (if aggressive mode enabled)
                if aggressive_config.get('enabled', False):
                    self.performance_trackers[broker_name] = PerformanceTracker({
                        'target_weekly_return': aggressive_config.get('target_weekly_return', 3.0),
                        'max_drawdown': aggressive_config.get('max_drawdown', 4.5)
                    })
                else:
                    self.performance_trackers[broker_name] = None
            
            # Initialize strategies
            print(f"[INIT] Initializing strategies...")
            self.strategies = self._initialize_strategies()
            print(f"[INIT] Strategies initialized: {len(self.strategies)} strategies")
            
            # Initialize AI models
            print(f"[INIT] Initializing AI models...")
            self.model_trainer = ModelTrainer(self.model_config.get('training', {}))
            self.ml_model, self.lstm_model = self._initialize_ai_models()
            self.sentiment_analyzer = SentimentAnalyzer(self.model_config.get('sentiment', {}))
            self.anomaly_detector = AnomalyDetector(self.model_config.get('anomaly_detection', {}))
            self.news_fetcher = NewsFetcher(self.model_config.get('sentiment', {}))
            self.last_model_retrain = {}  # Track last retrain time for each model
            print(f"[INIT] AI models initialized")
            
            # Initialize stock discovery service for dynamic stock selection
            from .data.stock_discovery import StockDiscoveryService
            stocks_config = self.trading_config.get('assets', {}).get('stocks', {})
            self.stock_discovery = StockDiscoveryService(
                self.news_fetcher,
                stocks_config
            ) if stocks_config.get('dynamic_discovery', False) else None
            
            # Initialize risk management components
            print(f"[INIT] Initializing risk management...")
            self.position_sizer = PositionSizer(self.trading_config.get('risk', {}))
            self.stop_loss_manager = StopLossManager(self.trading_config.get('risk', {}))
            
            # Create main drawdown manager (for backward compatibility and aggregated tracking)
            self.drawdown_manager = DrawdownManager(self.trading_config.get('risk', {}))
            
            self.risk_calculator = RiskCalculator()
            from .risk.portfolio_risk_manager import PortfolioRiskManager
            # Get portfolio risk config - check both locations for compatibility
            portfolio_risk_config = (
                self.trading_config.get('portfolio_risk', {}) or
                self.trading_config.get('agents', {}).get('risk_manager', {}).get('portfolio_risk', {})
            )
            self.portfolio_risk_manager = PortfolioRiskManager(portfolio_risk_config)
            print(f"[INIT] Risk management initialized")
            
            # Initialize Multi-Agent Framework
            print(f"[INIT] Initializing agents...")
            
            # Initialize Performance Tracker first (needed by technical agent)
            aggressive_config = self.trading_config.get('aggressive_mode', {})
            if aggressive_config.get('enabled', False):
                self.performance_tracker = PerformanceTracker({
                    'target_weekly_return': aggressive_config.get('target_weekly_return', 3.0),
                    'max_drawdown': aggressive_config.get('max_drawdown', 4.5)
                })
            else:
                self.performance_tracker = None
            
            # 1. Analytical Agents
            self.technical_agent = TechnicalAnalystAgent(
                config=self.trading_config.get('agents', {}).get('technical', {}),
                strategies=self.strategies,
                ml_model=self.ml_model,
                lstm_model=self.lstm_model,
                storage=self.storage,
                performance_tracker=self.performance_tracker
            )
            
            self.sentiment_agent = SentimentAgent(
                config=self.trading_config.get('agents', {}).get('sentiment', {}),
                sentiment_analyzer=self.sentiment_analyzer,
                news_fetcher=self.news_fetcher
            )
            
            self.fundamental_agent = FundamentalAgent(
                config=self.trading_config.get('agents', {}).get('fundamental', {})
            )
            
            self.quantitative_agent = QuantitativeAgent(
                config=self.trading_config.get('agents', {}).get('quantitative', {})
            )
            
            self.pattern_forecaster_agent = PatternForecasterAgent(
                config=self.trading_config.get('agents', {}).get('pattern_forecaster', {}),
                storage=self.storage
            )
            
            # 2. Regime-Switching Agent
            self.regime_agent = RegimeSwitchingAgent(
                config=self.trading_config.get('agents', {}).get('regime_switching', {})
            )
            # Train regime agent on historical data if available
            self._train_regime_agent()
            
            # 3. Risk Manager Agent (with Circuit Breaker)
            circuit_breaker = CircuitBreaker(
                self.trading_config.get('agents', {}).get('risk_manager', {}).get('circuit_breaker', {})
            )
            self.risk_agent = RiskManagerAgent(
                config=self.trading_config.get('agents', {}).get('risk_manager', {}),
                position_sizer=self.position_sizer,
                drawdown_manager=self.drawdown_manager,
                stop_loss_manager=self.stop_loss_manager,
                circuit_breaker=circuit_breaker
            )
            
            # 4. DRL Agent (if aggressive mode enabled)
            self.drl_agent = None
            self.drl_trainer = None
            aggressive_config = self.trading_config.get('aggressive_mode', {})
            if aggressive_config.get('enabled', False) and aggressive_config.get('use_drl', True):
                drl_config = self.model_config.get('drl', {})
                if drl_config.get('enabled', False):
                    self.drl_agent = DRLAgent(drl_config)
                    self.drl_trainer = DRLTrainer(drl_config)
                    
                    # Try to load existing model
                    model_path = os.path.join(self.model_trainer.models_dir, f"drl_{self.drl_agent.algorithm}.zip")
                    if os.path.exists(model_path):
                        try:
                            self.drl_agent.load(model_path)
                            print(f"Loaded DRL model from {model_path}")
                        except Exception as e:
                            print(f"Error loading DRL model: {e}")
            
            # 6. Orchestrator Agent (performance_tracker already initialized above)
            orchestrator_config = self.trading_config.get('agents', {}).get('orchestrator', {})
            orchestrator_config['use_drl'] = aggressive_config.get('enabled', False) and aggressive_config.get('use_drl', True)
            self.orchestrator_agent = OrchestratorAgent(
                config=orchestrator_config,
                technical_agent=self.technical_agent,
                sentiment_agent=self.sentiment_agent,
                fundamental_agent=self.fundamental_agent,
                quantitative_agent=self.quantitative_agent,
                regime_agent=self.regime_agent,
                risk_agent=self.risk_agent,
                drl_agent=self.drl_agent,
                performance_tracker=self.performance_tracker,
                sentiment_analyzer=self.sentiment_analyzer  # For LLM conflict resolution
            )
            
            # Keep DecisionEngine for backward compatibility (wrapper)
            self.decision_engine = DecisionEngine(self.trading_config.get('strategies', {}))
            for strategy in self.strategies:
                self.decision_engine.add_strategy(strategy)
            self.decision_engine.set_sentiment_analyzer(self.sentiment_analyzer)
            if self.ml_model:
                self.decision_engine.set_ml_model(self.ml_model)
            if self.lstm_model:
                self.decision_engine.set_lstm_model(self.lstm_model)
            
            # Initialize execution
            paper_trading = os.getenv('TRADING_MODE', 'paper') == 'paper'
            # Use Alpaca as default (supports both stocks and crypto)
            default_broker = self.brokers.get('stocks') or self.brokers.get('crypto') or list(self.brokers.values())[0] if self.brokers else None
            if default_broker:
                self.order_manager = OrderManager(
                    default_broker,
                    self.storage,
                    paper_trading=paper_trading
                )
            else:
                print("Warning: No broker available for order management")
            
            # Initialize strategy selector
            self.strategy_selector = StrategySelector(
                self.trading_config.get('model_monitoring', {})
            )
            
            # Initialize dashboard
            print(f"[INIT] Initializing dashboard...")
            self.dashboard_app = create_dashboard_app(
                storage=self.storage, 
                brokers=self.brokers,
                initial_equity=self.initial_equity
            )
            print(f"[INIT] Dashboard initialized")
            
            # Setup signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            print(f"[INIT] TradingAgent initialization complete!")
        except Exception as e:
            print(f"\n[INIT ERROR] Failed during TradingAgent initialization:")
            print(f"Error: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            raise  # Re-raise so main() can catch it
    
    def _load_config(self, filepath: str) -> Dict:
        """Load YAML configuration file"""
        try:
            with open(filepath, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Error loading config {filepath}: {e}")
            return {}
    
    def _initialize_brokers(self) -> Dict:
        """Initialize broker connections"""
        brokers = {}
        
        # OANDA (Forex)
        if os.getenv('OANDA_API_KEY'):
            try:
                # OANDA only needs API key (Bearer token), account_id is optional
                # Parse account_id, handling empty values and comments
                account_id = os.getenv('OANDA_ACCOUNT_ID', '').strip()
                if not account_id or account_id.startswith('#'):
                    account_id = None
                
                brokers['forex'] = OANDABroker(
                    os.getenv('OANDA_API_KEY'),
                    api_secret=None,  # OANDA doesn't use secret
                    account_id=account_id,
                    testnet=os.getenv('OANDA_ENVIRONMENT', 'practice') == 'practice'
                )
                brokers['forex'].connect()
            except Exception as e:
                self.logger.log_error(e, {'broker': 'oanda'})
        
        # Alpaca (Stocks and Crypto)
        if os.getenv('ALPACA_API_KEY'):
            try:
                # Check if using paper trading (paper-api or base URL contains 'paper')
                base_url = os.getenv('ALPACA_BASE_URL', '')
                is_testnet = 'paper' in base_url.lower() if base_url else True  # Default to testnet
                
                brokers['stocks'] = AlpacaBroker(
                    os.getenv('ALPACA_API_KEY'),
                    os.getenv('ALPACA_API_SECRET'),
                    testnet=is_testnet
                )
                brokers['stocks'].connect()
                # Alpaca handles both stocks and crypto
                brokers['crypto'] = brokers['stocks']
            except Exception as e:
                self.logger.log_error(e, {'broker': 'alpaca'})
        
        return brokers
    
    def _initialize_strategies(self) -> list:
        """Initialize trading strategies"""
        strategies = []
        strategy_configs = self.trading_config.get('strategies', {})
        
        # Trend following
        if strategy_configs.get('trend_following', {}).get('enabled', True):
            strategies.append(TrendFollowingStrategy(
                strategy_configs.get('trend_following', {})
            ))
        
        # Mean reversion
        if strategy_configs.get('mean_reversion', {}).get('enabled', True):
            strategies.append(MeanReversionStrategy(
                strategy_configs.get('mean_reversion', {})
            ))
        
        # Breakout
        if strategy_configs.get('breakout', {}).get('enabled', False):
            strategies.append(BreakoutStrategy(
                strategy_configs.get('breakout', {})
            ))
        
        # Momentum
        if strategy_configs.get('momentum', {}).get('enabled', False):
            strategies.append(MomentumStrategy(
                strategy_configs.get('momentum', {})
            ))
        
        # Volatility
        if strategy_configs.get('volatility', {}).get('enabled', False):
            strategies.append(VolatilityStrategy(
                strategy_configs.get('volatility', {})
            ))
        
        # News Trading
        if strategy_configs.get('news_trading', {}).get('enabled', False):
            strategies.append(NewsTradingStrategy(
                strategy_configs.get('news_trading', {})
            ))
        
        # End of Day
        if strategy_configs.get('end_of_day', {}).get('enabled', False):
            strategies.append(EndOfDayStrategy(
                strategy_configs.get('end_of_day', {})
            ))
        
        # Swing Trading
        if strategy_configs.get('swing_trading', {}).get('enabled', False):
            strategies.append(SwingTradingStrategy(
                strategy_configs.get('swing_trading', {})
            ))
        
        # Day Trading
        if strategy_configs.get('day_trading', {}).get('enabled', False):
            strategies.append(DayTradingStrategy(
                strategy_configs.get('day_trading', {})
            ))
        
        # Scalping
        if strategy_configs.get('scalping', {}).get('enabled', False):
            strategies.append(ScalpingStrategy(
                strategy_configs.get('scalping', {})
            ))
        
        return strategies
    
    def _initialize_ai_models(self):
        """Initialize, load, or train AI models"""
        ml_model = None
        lstm_model = None
        
        strategy_configs = self.trading_config.get('strategies', {})
        ml_config = strategy_configs.get('ml_model', {})
        lstm_config = strategy_configs.get('lstm_model', {})
        
        # Try to get historical data for training
        training_data = None
        if ml_config.get('enabled', False) or lstm_config.get('enabled', False):
            # Get data from first available broker/symbol
            for asset_class, config in self.trading_config.get('assets', {}).items():
                if config.get('enabled', False):
                    broker_name = config.get('broker', asset_class)
                    broker = self.brokers.get(broker_name) or self.brokers.get(asset_class)
                    if broker:
                        symbols = config.get('symbols', [])
                        if symbols:
                            try:
                                data_fetcher = DataFetcher(broker, self.storage)
                                data_list = data_fetcher.get_latest_data(
                                    symbols[0], '1h', days=90  # Get 90 days for training
                                )
                                if data_list:
                                    df = pd.DataFrame(data_list)
                                    df.set_index('timestamp', inplace=True)
                                    df = self.indicators.add_all_indicators(df)
                                    # Remove rows with NaN from indicators (lookback periods)
                                    df = df.dropna()
                                    if len(df) > 100:  # Ensure we have enough data after indicator calculation
                                        training_data = df
                                        print(f"Loaded {len(df)} data points for training from {symbols[0]}")
                                        break
                                    else:
                                        print(f"Warning: Only {len(df)} data points after indicators for {symbols[0]}, need at least 100")
                            except Exception as e:
                                print(f"Warning: Could not fetch training data: {e}")
        
        # Initialize ML Model (XGBoost)
        if ml_config.get('enabled', False):
            model_path = os.path.join(self.model_trainer.models_dir, 'xgboost.pkl')
            try:
                if os.path.exists(model_path):
                    ml_model = MLModel()
                    ml_model.load(model_path)
                    print(f"Loaded existing ML model from {model_path}")
                elif training_data is not None and len(training_data) > 100:
                    print(f"Training ML model with {len(training_data)} data points...")
                    try:
                        X, y, _ = self.model_trainer.prepare_data(
                            training_data, target_method='direction', horizon=1
                        )
                        print(f"  Prepared {len(X)} samples with {len(X.columns)} features")
                        if len(X) > 0 and len(y) > 0:
                            ml_model = self.model_trainer.train_ml_model(
                                X, y,
                                model_config=self.model_config.get('xgboost', {}),
                                model_name='xgboost'
                            )
                        else:
                            print("Warning: No valid features after data preparation. ML model will train later when more data is available.")
                    except ValueError as e:
                        print(f"Warning: Insufficient data for ML model training: {e}. Model will train later when more data is available.")
                else:
                    print("Info: Insufficient data to train ML model. Model will train automatically once enough historical data is collected.")
            except Exception as e:
                print(f"Warning: Error initializing ML model: {e}. Model will train later when data is available.")
        
        # Initialize LSTM Model
        if lstm_config.get('enabled', False) and LSTMModel is not None:
            model_path = os.path.join(self.model_trainer.models_dir, 'lstm.pkl')
            lstm_model = None
            try:
                if os.path.exists(model_path):
                    try:
                        lstm_model = LSTMModel(lstm_config)
                        lstm_model.load(model_path)
                        print(f"Loaded existing LSTM model from {model_path}")
                    except (ValueError, Exception) as load_error:
                        # Model file is incompatible or corrupted, delete and retrain
                        print(f"Warning: Could not load existing LSTM model ({load_error}). Will retrain.")
                        if os.path.exists(model_path):
                            try:
                                os.remove(model_path)
                            except:
                                pass
                        # Fall through to training
                        lstm_model = None
                
                if lstm_model is None or not (hasattr(lstm_model, 'is_trained') and lstm_model.is_trained):
                    if training_data is not None and len(training_data) > 200:
                        print("Training LSTM model...")
                        try:
                            X, y, _ = self.model_trainer.prepare_data(
                                training_data, target_method='regression', horizon=1
                            )
                            if len(X) > 0 and len(y) > 0:
                                lstm_model = self.model_trainer.train_lstm_model(
                                    X, y,
                                    model_config=self.model_config.get('lstm', {}),
                                    model_name='lstm'
                                )
                            else:
                                print("Warning: No valid features after data preparation. LSTM model will train later when more data is available.")
                        except ValueError as e:
                            print(f"Warning: Insufficient data for LSTM model training: {e}. Model will train later when more data is available.")
                    else:
                        print("Info: Insufficient data to train LSTM model. Model will train automatically once enough historical data is collected (needs 200+ records).")
            except Exception as e:
                print(f"Warning: Error initializing LSTM model: {e}. Model will train later when data is available.")
        
        return ml_model, lstm_model
    
    def _get_sentiment_for_symbol(self, symbol: str) -> Dict:
        """Fetch and analyze sentiment for a symbol"""
        try:
            # Fetch news for symbol
            news_items = self.news_fetcher.fetch_news_for_symbol(symbol)
            
            if news_items:
                # Analyze sentiment using sentiment analyzer
                sentiment_result = self.sentiment_analyzer.analyze_news(news_items)
                return {
                    'sentiment': sentiment_result.get('sentiment', 0.0),
                    'confidence': sentiment_result.get('confidence', 0.0),
                    'news_count': sentiment_result.get('count', 0)
                }
            else:
                return {'sentiment': 0.0, 'confidence': 0.0, 'news_count': 0}
        except Exception as e:
            self.logger.log_error(e, {'component': 'sentiment_analysis', 'symbol': symbol})
            return {'sentiment': 0.0, 'confidence': 0.0, 'news_count': 0}
    
    def _check_and_retrain_models(self):
        """Check if models need retraining and retrain if necessary"""
        try:
            strategy_configs = self.trading_config.get('strategies', {})
            ml_config = strategy_configs.get('ml_model', {})
            lstm_config = strategy_configs.get('lstm_model', {})
            
            # Check ML model retraining
            if ml_config.get('enabled', False) and self.ml_model:
                retrain_days = ml_config.get('retrain_frequency_days', 7)
                last_retrain = self.last_model_retrain.get('ml_model')
                
                if not last_retrain or (datetime.now() - last_retrain).days >= retrain_days:
                    print(f"Retraining ML model (last trained: {last_retrain})...")
                    # Get fresh training data
                    training_data = self._get_training_data()
                    if training_data is not None and len(training_data) > 100:
                        X, y, _ = self.model_trainer.prepare_data(
                            training_data, target_method='direction', horizon=1
                        )
                        self.ml_model = self.model_trainer.train_ml_model(
                            X, y,
                            model_config=self.model_config.get('xgboost', {}),
                            model_name='xgboost'
                        )
                        self.decision_engine.set_ml_model(self.ml_model)
                        self.last_model_retrain['ml_model'] = datetime.now()
                        print("ML model retrained successfully")
            
            # Check LSTM model retraining
            if lstm_config.get('enabled', False) and self.lstm_model and LSTMModel is not None:
                retrain_days = lstm_config.get('retrain_frequency_days', 14)
                last_retrain = self.last_model_retrain.get('lstm_model')
                
                if not last_retrain or (datetime.now() - last_retrain).days >= retrain_days:
                    print(f"Retraining LSTM model (last trained: {last_retrain})...")
                    training_data = self._get_training_data()
                    if training_data is not None and len(training_data) > 200:
                        X, y, _ = self.model_trainer.prepare_data(
                            training_data, target_method='regression', horizon=1
                        )
                        self.lstm_model = self.model_trainer.train_lstm_model(
                            X, y,
                            model_config=self.model_config.get('lstm', {}),
                            model_name='lstm'
                        )
                        self.decision_engine.set_lstm_model(self.lstm_model)
                        self.last_model_retrain['lstm_model'] = datetime.now()
                        print("LSTM model retrained successfully")
        
        except Exception as e:
            self.logger.log_error(e, {'component': 'model_retraining'})
    
    def _get_training_data(self) -> Optional[pd.DataFrame]:
        """Get training data from first available broker/symbol"""
        for asset_class, config in self.trading_config.get('assets', {}).items():
            if config.get('enabled', False):
                broker_name = config.get('broker', asset_class)
                broker = self.brokers.get(broker_name) or self.brokers.get(asset_class)
                if broker:
                    symbols = config.get('symbols', [])
                    if symbols:
                        try:
                            data_fetcher = DataFetcher(broker, self.storage)
                            data_list = data_fetcher.get_latest_data(
                                symbols[0], '1h', days=90
                            )
                            if data_list:
                                df = pd.DataFrame(data_list)
                                df.set_index('timestamp', inplace=True)
                                df = self.indicators.add_all_indicators(df)
                                return df
                        except Exception as e:
                            print(f"Error fetching training data: {e}")
        return None
    
    def _train_regime_agent(self):
        """Train regime-switching agent on historical data"""
        try:
            # Get training data
            training_data = self._get_training_data()
            if training_data is not None and len(training_data) > 100:
                self.regime_agent.train(training_data)
                print("Regime-switching agent trained successfully")
        except Exception as e:
            print(f"Error training regime agent: {e}")
    
    def _is_trading_hours(self, asset_class: str) -> bool:
        """
        Check if current time is within trading hours for asset class
        
        Args:
            asset_class: Asset class (crypto, forex, stocks, commodities)
        
        Returns:
            True if within trading hours, False otherwise
        """
        trading_hours_config = self.trading_config.get('trading_hours', {})
        asset_config = trading_hours_config.get(asset_class, {})
        
        if not asset_config.get('enabled', True):
            return False
        
        now = datetime.now()
        current_day = now.strftime('%A').lower()
        current_hour = now.hour
        current_minute = now.minute
        
        # Commodities (precious metals) follow forex hours (24/5)
        if asset_class == 'commodities' and not asset_config:
            # Fallback to forex hours if commodities hours not configured
            asset_config = trading_hours_config.get('forex', {})
        
        # Check if today is a trading day
        allowed_days = [d.lower() if isinstance(d, str) else d for d in asset_config.get('days', [])]
        print(f"  Trading hours check for {asset_class}: day={current_day}, allowed_days={allowed_days}")
        if current_day not in allowed_days:
            print(f"  ❌ Day {current_day} not in allowed days")
            return False
        
        # Check if within trading hours
        start_hour = asset_config.get('start_hour', 0)
        end_hour = asset_config.get('end_hour', 24)
        
        # For stocks, market opens at 13:30 UTC (9:30 AM EST), not 13:00
        # Config says start_hour: 13, but we need to check for 13:30
        if asset_class == 'stocks':
            # Stocks: 13:30-20:00 UTC (9:30 AM - 4:00 PM EST)
            current_time_minutes = current_hour * 60 + current_minute
            market_open_minutes = 13 * 60 + 30  # 13:30 UTC
            market_close_minutes = 20 * 60  # 20:00 UTC
            return market_open_minutes <= current_time_minutes < market_close_minutes
        
        # For forex and crypto, use hour-based check
        if start_hour <= end_hour:
            # Normal case: e.g., 0 to 24 (forex/crypto)
            return start_hour <= current_hour < end_hour
        else:
            # Overnight case: e.g., 22 to 6 (not applicable here but handle it)
            return current_hour >= start_hour or current_hour < end_hour
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\nShutting down trading agent...")
        self.running = False
    
    def run(self):
        """Main trading loop"""
        self.running = True
        self.logger.logger.info("Trading agent started")
        
        # Start dashboard in background
        import uvicorn
        import threading
        port = int(os.getenv("PORT", 8000))  # Use Railway's PORT or default to 8000
        dashboard_thread = threading.Thread(
            target=lambda: uvicorn.run(self.dashboard_app, host="0.0.0.0", port=port),
            daemon=True
        )
        dashboard_thread.start()
        
        # Initial sync: Load existing positions from brokers and database on startup
        print("\n" + "="*60)
        print("STARTUP: Syncing existing positions from brokers...")
        print("="*60)
        try:
            # Step 1: Sync positions FROM brokers TO database
            self._sync_positions_from_brokers()
            
            # Step 2: Clean up phantom trades (trades in database that don't exist in brokers)
            print("\n[STARTUP] Cleaning up phantom trades (database trades not found in brokers)...")
            self._cleanup_phantom_trades()
            
            open_trades = self.storage.get_open_trades() if self.storage else []
            print(f"✅ Startup sync complete: Found {len(open_trades)} open positions in database")
            if open_trades:
                positions_list = [f"{t.get('symbol')} {t.get('side')}" for t in open_trades]
                print(f"  Positions to manage: {positions_list}")
            
            # Calculate and update take profit for existing positions that don't have it
            self._update_missing_take_profits(open_trades)
        except Exception as e:
            print(f"⚠️  Warning: Error during startup sync: {e}")
            self.logger.log_error(e, {'component': 'startup_sync'})
        
        # Main loop
        while self.running:
            try:
                # Check kill switch - check all brokers (any broker can trigger kill switch)
                kill_switch_active = False
                kill_switch_brokers = []
                for broker_name, dd_manager in self.drawdown_managers.items():
                    if not dd_manager.is_trading_allowed():
                        kill_switch_active = True
                        drawdown = dd_manager.calculate_drawdown()
                        kill_switch_brokers.append(f"{broker_name} ({drawdown:.2f}%)")
                
                # Also check the main drawdown manager for backward compatibility
                if not self.drawdown_manager.is_trading_allowed():
                    kill_switch_active = True
                    drawdown = self.drawdown_manager.calculate_drawdown()
                    kill_switch_brokers.append(f"aggregated ({drawdown:.2f}%)")
                
                if kill_switch_active:
                    print(f"⚠️  Kill switch ACTIVE: Drawdown exceeds limit on: {', '.join(kill_switch_brokers)}")
                    self.logger.log_risk_event({
                        'event': 'kill_switch_active',
                        'affected_brokers': kill_switch_brokers
                    })
                    time.sleep(60)  # Wait 1 minute before checking again
                    continue
                
                # Check and retrain models if needed
                self._check_and_retrain_models()
                
                # Process each asset class
                assets_config = self.trading_config.get('assets', {})
                print(f"\n{'='*60}")
                print(f"MAIN LOOP: Processing {len(assets_config)} asset classes")
                print(f"{'='*60}")
                
                for asset_class, config in assets_config.items():
                    print(f"\n[Asset Class: {asset_class}]")
                    print(f"  Enabled: {config.get('enabled', False)}")
                    
                    if not config.get('enabled', False):
                        print(f"  ⏭️  Skipping {asset_class} - disabled in config")
                        continue
                    
                    # Check trading hours from config before processing asset class
                    is_trading_hours = self._is_trading_hours(asset_class)
                    print(f"  Trading hours check: {is_trading_hours}")
                    
                    if not is_trading_hours:
                        current_day = datetime.now().strftime('%A')
                        current_time = datetime.now().strftime('%H:%M:%S UTC')
                        print(f"  ⏸️  Skipping {asset_class} - not in trading hours (Day: {current_day}, Time: {current_time})")
                        continue
                    
                    # Get broker - handle different asset classes
                    broker_name = config.get('broker', asset_class)
                    print(f"  Broker name: {broker_name}")
                    
                    # Map broker names from config to actual broker keys
                    broker_key_map = {
                        'oanda': 'forex',
                        'alpaca': 'stocks',
                        'forex': 'forex',
                        'stocks': 'stocks',
                        'crypto': 'crypto'
                    }
                    
                    # Get the actual broker key
                    broker_key = broker_key_map.get(broker_name, asset_class)
                    
                    if asset_class == 'crypto' and broker_name == 'alpaca':
                        broker = self.brokers.get('stocks')  # Alpaca handles both stocks and crypto
                        print(f"  Using Alpaca broker (stocks) for crypto")
                    elif asset_class == 'forex':
                        # Forex must use OANDA (forex broker), not Alpaca
                        broker = self.brokers.get('forex') or self.brokers.get('oanda')
                        if not broker:
                            print(f"  ❌ No OANDA broker available for forex trading")
                            continue
                        print(f"  Using OANDA broker for forex")
                    elif asset_class == 'commodities' and broker_name == 'oanda':
                        broker = self.brokers.get('forex') or self.brokers.get('oanda')  # OANDA handles commodities
                        print(f"  Using OANDA broker for commodities")
                    else:
                        # Try mapped key first, then broker_name, then asset_class
                        broker = self.brokers.get(broker_key) or self.brokers.get(broker_name) or self.brokers.get(asset_class)
                    
                    if not broker:
                        print(f"  ⚠️  No broker available for {asset_class}")
                        continue
                    
                    print(f"  ✓ Broker found: {type(broker).__name__}")
                    
                    # For stocks: use dynamic discovery if enabled, otherwise use fixed list
                    if asset_class == 'stocks' and config.get('dynamic_discovery', False) and self.stock_discovery:
                        # Discover stocks from news
                        discovered_stocks = self.stock_discovery.discover_stocks_from_news()
                        symbols = discovered_stocks
                        print(f"  📊 Dynamic discovery: Found {len(symbols)} stocks from news: {symbols}")
                    else:
                        # Use fixed symbol list
                        symbols = config.get('symbols', [])
                        print(f"  📊 Processing {len(symbols)} symbols: {symbols}")
                    
                    # Process each symbol for this asset class
                    for symbol in symbols:
                        print(f"  → Processing {symbol}...")
                        self._process_symbol(symbol, broker, asset_class)
                
                # Monitor and manage all open positions (including external ones)
                self._manage_open_positions()
                
                # Update dashboard
                self._update_dashboard()
                
                # Sleep before next iteration
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.log_error(e, {'component': 'main_loop'})
                time.sleep(60)
    
    def _process_symbol(self, symbol: str, broker, asset_class: str):
        """Process trading for a single symbol"""
        try:
            # Check if market is open before analyzing
            if hasattr(broker, 'is_market_open'):
                if not broker.is_market_open(symbol):
                    current_day = datetime.now().strftime('%A')
                    current_time = datetime.now().strftime('%H:%M:%S UTC')
                    print(f"⏸️  Skipping {symbol} ({asset_class}) - Market is closed (Day: {current_day}, Time: {current_time})")
                    return
            
            # Fetch latest data
            data_fetcher = DataFetcher(broker, self.storage)
            data_list = data_fetcher.get_latest_data(symbol, '1h', days=30)
            
            if not data_list:
                return
            
            # Convert to DataFrame
            df = pd.DataFrame(data_list)
            df.set_index('timestamp', inplace=True)
            
            # Add indicators
            df = self.indicators.add_all_indicators(df)
            
            # Check for anomalies
            if self.anomaly_detector.is_trained:
                anomalies = self.anomaly_detector.detect(df)
                if anomalies.iloc[-1] if not anomalies.empty else False:
                    self.logger.log_risk_event({
                        'event': 'anomaly_detected',
                        'symbol': symbol
                    })
                    return  # Skip trading on anomalies
            
            # Fetch news for sentiment analysis
            news_items = self.news_fetcher.fetch_news_for_symbol(symbol) if self.news_fetcher else None
            
            # Get account balance for risk management - use broker-specific balance for this trade
            # Determine which broker will handle this symbol
            trade_broker_name = None
            for broker_name, b in self.brokers.items():
                if b == broker:
                    trade_broker_name = broker_name
                    break
            
            # Get broker-specific balance for this trade
            broker_account_balance = 0.0
            broker_available_balance = 0.0
            if trade_broker_name and broker:
                try:
                    balance_info = broker.get_account_balance()
                    broker_account_balance = balance_info.get('total', 0.0)
                    broker_available_balance = balance_info.get('available', balance_info.get('buying_power', broker_account_balance))
                except:
                    pass
            
            # Also get aggregated balance for overall risk checks
            account_balance = 0.0
            available_balance = 0.0
            for broker_name, b in self.brokers.items():
                try:
                    balance_info = b.get_account_balance()
                    total_bal = balance_info.get('total', 0.0)
                    avail_bal = balance_info.get('available', balance_info.get('buying_power', total_bal))
                    
                    # Validate: available should not exceed total (unless leverage is intentional)
                    if avail_bal > total_bal * 1.1:  # More than 10% over (likely leverage)
                        print(f"  ⚠️  WARNING [{broker_name}]: Available balance (${avail_bal:.2f}) > Total (${total_bal:.2f}) - likely leverage")
                        print(f"     Using conservative available balance: ${total_bal * 0.95:.2f}")
                        avail_bal = total_bal * 0.95  # Use 95% of total as conservative estimate
                    
                    account_balance += total_bal
                    available_balance += avail_bal
                except:
                    pass
            
            # Fallback to default if no brokers returned balance
            if broker_account_balance == 0.0:
                broker_account_balance = 10000.0
                broker_available_balance = 10000.0
            if account_balance == 0.0:
                account_balance = 10000.0
                available_balance = 10000.0
            
            # Check broker-specific drawdown before trading
            if trade_broker_name and trade_broker_name in self.drawdown_managers:
                broker_dd_manager = self.drawdown_managers[trade_broker_name]
                if not broker_dd_manager.is_trading_allowed():
                    drawdown = broker_dd_manager.calculate_drawdown()
                    print(f"  ⚠️  Skipping {symbol}: Broker {trade_broker_name} kill switch active (drawdown: {drawdown:.2f}%)")
                    return
            
            # Use broker-specific balance for position sizing, but aggregated for overall risk checks
            # Use Multi-Agent Orchestrator
            decision = self.orchestrator_agent.orchestrate(
                data=df,
                symbol=symbol,
                asset_class=asset_class,
                news_items=news_items,
                account_balance=broker_account_balance  # Use broker-specific balance for position sizing
            )
            
            # Log decision
            self.logger.log_decision({
                'symbol': symbol,
                'signal': decision['signal'].value if hasattr(decision['signal'], 'value') else str(decision['signal']),
                'confidence': decision.get('confidence', 0),
                'market_regime': decision.get('regime', 'unknown'),
                'regime_confidence': decision.get('regime_confidence', 0),
                'consensus': decision.get('consensus', {}).get('has_consensus', False),
                'agent_signals': {k: v.get('signal', 'HOLD') for k, v in decision.get('agent_signals', {}).items()}
            })
            
            # Execute trade if signal is strong enough and not vetoed
            signal_value = decision['signal']
            confidence = decision.get('confidence', 0)
            is_vetoed = decision.get('veto', False)
            min_conf = self.orchestrator_agent.min_confidence
            
            # Show agent signals for debugging (especially for forex/stocks)
            agent_signals = decision.get('agent_signals', {})
            if agent_signals:
                print(f"\n[AGENT SIGNALS for {symbol} ({asset_class})]")
                for agent_name, agent_data in agent_signals.items():
                    agent_signal = agent_data.get('signal', 'HOLD')
                    agent_conf = agent_data.get('confidence', 0)
                    print(f"  {agent_name}: {agent_signal}, confidence={agent_conf:.3f}")
            
            print(f"\n[TRADE EXECUTION CHECK for {symbol}]")
            print(f"  Signal: {signal_value.name if hasattr(signal_value, 'name') else signal_value}")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Min Confidence Required: {min_conf:.3f}")
            print(f"  Is Vetoed: {is_vetoed}")
            if is_vetoed:
                veto_reason = decision.get('veto_reason', decision.get('reason', 'Unknown veto reason'))
                print(f"  Veto Reason: {veto_reason}")
            print(f"  Position Size Info: {decision.get('position_size', {})}")
            
            # Check execution conditions
            will_execute = (signal_value != Signal.HOLD and 
                          confidence > min_conf and
                          not is_vetoed)
            
            if not will_execute:
                if signal_value == Signal.HOLD:
                    reason = f'Signal is HOLD'
                elif confidence <= min_conf:
                    reason = f'Confidence too low: {confidence:.3f} <= {min_conf:.3f}'
                elif is_vetoed:
                    reason = f'Vetoed: {decision.get("reason", "Risk manager veto")}'
                else:
                    reason = 'Unknown reason'
                
                print(f"  ❌ Trade NOT executed: {reason}")
                self.logger.log_decision({
                    'symbol': symbol,
                    'reason': reason,
                    'confidence': confidence,
                    'min_confidence': min_conf,
                    'signal': signal_value.name if hasattr(signal_value, 'name') else str(signal_value)
                }, event='trade_skipped')
            else:
                print(f"  ✅ Trade WILL be executed - calling _execute_trade()...")
                self._execute_trade(symbol, decision, df, broker)
            
        except Exception as e:
            self.logger.log_error(e, {'symbol': symbol, 'asset_class': asset_class})
    
    def _execute_trade(self, symbol: str, decision: Dict, data: pd.DataFrame, broker):
        """Execute a trade based on decision"""
        try:
            print(f"\n[EXECUTING TRADE for {symbol}]")
            
            # Validate broker is correct for symbol type - MUST happen before OrderManager
            if broker:
                broker_type = type(broker).__name__
                # Check if forex symbol is being sent to Alpaca (which doesn't support forex)
                if '/' in symbol:
                    parts = symbol.split('/')
                    if len(parts) == 2:
                        base = parts[0].upper()
                        quote = parts[1].upper()
                        # Forex pairs
                        forex_bases = ['EUR', 'GBP', 'USD', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']
                        forex_quotes = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']
                        # Crypto pairs
                        crypto_bases = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'LINK', 'MATIC', 'AVAX', 'UNI', 'ATOM', 'BNB', 'XRP', 'LTC']
                        crypto_quotes = ['USD', 'EUR', 'GBP']
                        
                        is_forex = (base in forex_bases and quote in forex_quotes)
                        is_crypto = (base in crypto_bases and quote in crypto_quotes)
                        
                        if is_forex and broker_type == 'AlpacaBroker':
                            # Forex trade sent to Alpaca - find OANDA broker instead
                            print(f"  ⚠️  Warning: Forex symbol {symbol} sent to Alpaca (which doesn't support forex)")
                            print(f"  🔄 Attempting to find OANDA broker...")
                            broker = self.brokers.get('forex') or self.brokers.get('oanda')
                            if not broker:
                                raise Exception(f"No OANDA broker available for forex trading. Forex symbol {symbol} cannot be traded with Alpaca.")
                            print(f"  ✅ Found OANDA broker for forex trading")
                            # Update OrderManager to use the correct broker
                            if self.order_manager:
                                self.order_manager.broker = broker
                                print(f"  ✅ Updated OrderManager to use OANDA broker")
            
            # Get account balance - use AVAILABLE balance for position sizing, not total
            # Also update per-broker drawdown managers with their individual balances
            account_balance = 0.0
            available_balance = 0.0
            broker_balance_info = {}
            broker_for_trade = None  # Track which broker will execute this trade
            
            for broker_name, b in self.brokers.items():
                try:
                    balance_info = b.get_account_balance()
                    total_bal = balance_info.get('total', 0.0)
                    avail_bal = balance_info.get('available', balance_info.get('buying_power', total_bal))
                    account_balance += total_bal
                    available_balance += avail_bal
                    broker_balance_info[broker_name] = balance_info
                    print(f"  Account balance from {broker_name}: Total=${total_bal:.2f}, Available=${avail_bal:.2f}")
                    
                    # Update per-broker drawdown manager with this broker's balance
                    if broker_name in self.drawdown_managers:
                        self.drawdown_managers[broker_name].update_equity(total_bal)
                    
                    # Update per-broker circuit breaker
                    if broker_name in self.circuit_breakers:
                        self.circuit_breakers[broker_name].update_equity(total_bal)
                    
                    # Update per-broker performance tracker
                    if broker_name in self.performance_trackers and self.performance_trackers[broker_name]:
                        self.performance_trackers[broker_name].update_equity(total_bal)
                    
                    # Track which broker will execute this trade (for later use)
                    if b == broker:
                        broker_for_trade = broker_name
                except Exception as e:
                    print(f"  Error getting balance from {broker_name}: {e}")
            
            # Fallback to default if no brokers returned balance
            if account_balance == 0.0:
                account_balance = 10000.0
                available_balance = 10000.0
                print(f"  Using fallback balance: ${account_balance:.2f}")
            
            print(f"  Total account balance: ${account_balance:.2f}")
            print(f"  Available balance: ${available_balance:.2f}")
            
            # Update aggregated drawdown manager (for backward compatibility)
            self.drawdown_manager.update_equity(account_balance)
            
            # Update aggregated performance tracker and circuit breaker (for backward compatibility)
            if self.performance_tracker:
                self.performance_tracker.update_equity(account_balance)
            
            # Update circuit breaker with current equity
            # IMPORTANT: Initialize peak equity if not set (first run or after reset)
            if self.risk_agent and self.risk_agent.circuit_breaker:
                cb = self.risk_agent.circuit_breaker
                cb.update_equity(account_balance)
                # If peak equity is None or very different from current, initialize it
                if cb.peak_equity is None:
                    cb.reset_peak_equity(account_balance)
                    print(f"  Circuit Breaker: Initialized peak equity to ${account_balance:,.2f}")
                elif cb.peak_equity > 0 and abs(cb.peak_equity - account_balance) / cb.peak_equity > 0.5:  # More than 50% difference
                    # Peak equity seems incorrect (likely from previous run), reset it
                    print(f"  Circuit Breaker: Peak equity (${cb.peak_equity:,.2f}) differs significantly from current (${account_balance:,.2f}). Resetting...")
                    cb.reset_peak_equity(account_balance)
            
            # Also update per-broker circuit breakers
            for broker_name, cb in self.circuit_breakers.items():
                if cb:
                    cb.update_equity(account_balance)
                    # Initialize peak equity if not set
                    if cb.peak_equity is None:
                        cb.reset_peak_equity(account_balance)
                    elif cb.peak_equity > 0 and abs(cb.peak_equity - account_balance) / cb.peak_equity > 0.5:
                        cb.reset_peak_equity(account_balance)
            
            # Use position size from orchestrator (already calculated with regime adjustments)
            position_info = decision.get('position_size', {})
            print(f"  Position info from orchestrator: {position_info}")
            
            # Determine order side and entry price first (needed for position sizing)
            signal = decision['signal']
            side = OrderSide.BUY if signal == Signal.BUY or (hasattr(signal, 'value') and signal.value == 1) else OrderSide.SELL
            entry_price = decision.get('entry_price', data['close'].iloc[-1])
            stop_loss = decision.get('stop_loss')
            take_profit = decision.get('take_profit')
            
            if not position_info or position_info.get('quantity', 0) == 0:
                print(f"  ⚠️  No position size from orchestrator, calculating manually...")
                # Fallback to manual calculation - use AVAILABLE balance, not total
                if stop_loss is None:
                    stop_loss = entry_price * 0.98
                position_info = self.position_sizer.calculate_position_size(
                    available_balance, entry_price, stop_loss, data  # Use available, not total
                )
                print(f"  Calculated position size: {position_info}")
            
            quantity = position_info.get('quantity', 0)
            if quantity == 0:
                print(f"  ❌ Trade NOT executed: Position quantity is 0")
                print(f"     Position info: {position_info}")
                return
            
            # Calculate order value and check available balance BEFORE placing order
            order_value = quantity * entry_price
            print(f"  Order details:")
            print(f"    Symbol: {symbol}")
            print(f"    Side: {side.value if hasattr(side, 'value') else side}")
            print(f"    Quantity: {quantity}")
            print(f"    Entry Price: ${entry_price:.2f}")
            print(f"    Order Value: ${order_value:.2f}")
            print(f"    Available Balance: ${available_balance:.2f}")
            print(f"    Stop Loss: ${stop_loss:.2f}" if stop_loss is not None else "    Stop Loss: None")
            print(f"    Take Profit: ${take_profit:.2f}" if take_profit is not None else "    Take Profit: None")
            
            # CRITICAL: Check position correlation BEFORE portfolio exposure check
            # Prevent over-concentration in correlated assets (e.g., multiple crypto or forex pairs)
            max_correlated_positions = self.config.get('max_correlated_positions', 3)  # Default: 3 for same asset class
            max_correlated_pairs = self.config.get('max_correlated_pairs', 2)  # Default: 2 for correlated pairs
            
            # Define correlated forex pairs
            correlated_forex_groups = [
                ['EUR/USD', 'GBP/USD', 'EUR/GBP'],  # EUR/GBP correlated
                ['USD/JPY', 'EUR/JPY', 'GBP/JPY'],  # JPY pairs correlated
                ['AUD/USD', 'NZD/USD'],  # Commodity currencies
            ]
            
            # Get current open positions from database
            open_positions = self.storage.get_open_trades()
            
            if open_positions:
                # Group positions by asset class
                positions_by_class = {}
                for pos in open_positions:
                    pos_symbol = pos.get('symbol', '')
                    asset_class = 'crypto' if any(c in pos_symbol.upper() for c in ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'LINK', 'MATIC', 'AVAX', 'UNI', 'ATOM']) else \
                                 'forex' if '/' in pos_symbol and any(c in pos_symbol.upper() for c in ['EUR', 'GBP', 'USD', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD', 'XAU', 'XAG']) else \
                                 'stocks'
                    
                    if asset_class not in positions_by_class:
                        positions_by_class[asset_class] = []
                    positions_by_class[asset_class].append(pos_symbol)
                
                # Check asset class correlation (all crypto positions are highly correlated)
                current_asset_class = 'crypto' if any(c in symbol.upper() for c in ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'LINK', 'MATIC', 'AVAX', 'UNI', 'ATOM']) else \
                                    'forex' if '/' in symbol and any(c in symbol.upper() for c in ['EUR', 'GBP', 'USD', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD', 'XAU', 'XAG']) else \
                                    'stocks'
                
                current_class_count = len(positions_by_class.get(current_asset_class, []))
                if current_class_count >= max_correlated_positions:
                    print(f"  ❌ REJECTED: Correlation limit exceeded for {current_asset_class}")
                    print(f"     Current {current_asset_class} positions: {current_class_count}, Limit: {max_correlated_positions}")
                    print(f"     Existing positions: {', '.join(positions_by_class.get(current_asset_class, []))}")
                    return
                
                # Check forex pair correlation (correlated pairs)
                if current_asset_class == 'forex':
                    for group in correlated_forex_groups:
                        existing_in_group = [p for p in positions_by_class.get('forex', []) if p in group]
                        if symbol in group and len(existing_in_group) >= max_correlated_pairs:
                            print(f"  ❌ REJECTED: Correlated forex pair limit exceeded")
                            print(f"     Symbol {symbol} is correlated with: {', '.join(group)}")
                            print(f"     Existing correlated positions: {', '.join(existing_in_group)}")
                            print(f"     Limit: {max_correlated_pairs} correlated pairs")
                            return
                
                print(f"  ✅ Correlation check passed: {current_asset_class} positions: {current_class_count + 1}/{max_correlated_positions}")
            
            # CRITICAL: Check portfolio-level exposure limits BEFORE placing order
            # This prevents opening positions that exceed total portfolio exposure limits
            if hasattr(self, 'portfolio_risk_manager') and self.portfolio_risk_manager:
                # Use actual cash balance (not leveraged buying power) for exposure calculations
                # account_balance should already be validated, but ensure we're using cash
                exposure_base_balance = account_balance
                
                # Validate: If available_balance > account_balance, we might have leverage
                # Use the smaller of the two for conservative exposure calculation
                if available_balance > account_balance * 1.1:
                    print(f"  ⚠️  WARNING: Available balance (${available_balance:.2f}) > Total (${account_balance:.2f}) - using conservative balance for exposure")
                    exposure_base_balance = account_balance  # Use total, not available (which includes leverage)
                
                # Get current open positions from brokers to calculate total exposure
                # Use a set to deduplicate positions (Alpaca is stored as both 'stocks' and 'crypto')
                current_exposure = 0.0
                position_details = []
                seen_positions = {}  # Track seen positions: (symbol, side, entry_price) -> value
                
                # Track which broker instances we've already queried (to avoid duplicate queries)
                queried_brokers = set()
                
                for broker_name, b in self.brokers.items():
                    # Skip if we've already queried this broker instance (Alpaca appears as both 'stocks' and 'crypto')
                    broker_id = id(b)
                    if broker_id in queried_brokers:
                        continue
                    queried_brokers.add(broker_id)
                    
                    if hasattr(b, 'get_open_positions'):
                        try:
                            broker_positions = b.get_open_positions()
                            for pos in broker_positions:
                                # Normalize symbol format
                                raw_symbol = pos.get('symbol', '')
                                pos_symbol = raw_symbol.replace('_', '/')
                                # Handle Alpaca crypto format: ETHUSD -> ETH/USD
                                if '/' not in pos_symbol and len(pos_symbol) >= 6:
                                    for base_len in [3, 4]:
                                        if len(pos_symbol) > base_len:
                                            base = pos_symbol[:base_len]
                                            quote = pos_symbol[base_len:]
                                            if quote in ['USD', 'EUR', 'GBP', 'JPY']:
                                                pos_symbol = f"{base}/{quote}"
                                                break
                                
                                pos_side = pos.get('side', 'buy').lower()
                                pos_entry_price = pos.get('entry_price', 0)
                                
                                # Create unique key for deduplication
                                position_key = (pos_symbol, pos_side, round(pos_entry_price, 2))
                                
                                # Skip if we've already seen this position
                                if position_key in seen_positions:
                                    continue
                                
                                pos_value = pos.get('market_value', 0)
                                if pos_value == 0:
                                    # Calculate from quantity and current price
                                    pos_qty = pos.get('quantity', 0)
                                    pos_price = pos.get('current_price', pos.get('entry_price', 0))
                                    pos_value = pos_qty * pos_price
                                
                                if pos_value > 0:
                                    current_exposure += pos_value
                                    seen_positions[position_key] = pos_value
                                    position_details.append({
                                        'symbol': pos_symbol,
                                        'value': pos_value
                                    })
                        except Exception as e:
                            print(f"  ⚠️  Error getting positions from {broker_name} for portfolio check: {e}")
                
                current_exposure_pct = (current_exposure / exposure_base_balance * 100) if exposure_base_balance > 0 else 0.0
                proposed_total_exposure = current_exposure + order_value
                proposed_exposure_pct = (proposed_total_exposure / exposure_base_balance * 100) if exposure_base_balance > 0 else 0.0
                
                # Validation: If calculated exposure > base balance, something is wrong
                if proposed_total_exposure > exposure_base_balance * 1.5:
                    print(f"  ⚠️  WARNING: Calculated exposure (${proposed_total_exposure:.2f}) > Base balance (${exposure_base_balance:.2f}) - using conservative estimate")
                    proposed_exposure_pct = min(proposed_exposure_pct, 100.0)  # Cap at 100%
                max_exposure_pct = self.portfolio_risk_manager.max_portfolio_exposure
                
                print(f"\n  [PORTFOLIO RISK CHECK]")
                print(f"    Current Positions: {len(position_details)}")
                for pos_detail in position_details:
                    print(f"      - {pos_detail['symbol']}: ${pos_detail['value']:.2f}")
                print(f"    Current Total Exposure: ${current_exposure:.2f} ({current_exposure_pct:.1f}% of account)")
                print(f"    Proposed Position Value: ${order_value:.2f}")
                print(f"    Total After Trade: ${proposed_total_exposure:.2f} ({proposed_exposure_pct:.1f}% of account)")
                print(f"    Portfolio Limit: {max_exposure_pct:.1f}%")
                
                # Check if adding this position would exceed portfolio exposure limit
                if proposed_exposure_pct > max_exposure_pct:
                    # Calculate maximum allowed position value using exposure_base_balance (cash, not leveraged)
                    max_exposure_value = exposure_base_balance * (max_exposure_pct / 100)
                    remaining_capacity = max(0, max_exposure_value - current_exposure)
                    
                    print(f"    ❌ PORTFOLIO EXPOSURE LIMIT EXCEEDED!")
                    print(f"       Would be {proposed_exposure_pct:.1f}%, limit is {max_exposure_pct:.1f}%")
                    print(f"       Remaining capacity: ${remaining_capacity:.2f}")
                    
                    if remaining_capacity < order_value * 0.1:  # Less than 10% of proposed, reject
                        error_msg = f"Portfolio exposure limit exceeded: Would be {proposed_exposure_pct:.1f}%, limit is {max_exposure_pct:.1f}%. Current exposure: {current_exposure_pct:.1f}%"
                        print(f"  ❌ Trade REJECTED: {error_msg}")
                        self.logger.log_error(Exception(error_msg), {
                            'symbol': symbol,
                            'action': 'execute_trade',
                            'current_exposure': current_exposure,
                            'current_exposure_pct': current_exposure_pct,
                            'proposed_value': order_value,
                            'proposed_exposure_pct': proposed_exposure_pct,
                            'portfolio_limit': max_exposure_pct
                        })
                        return
                    else:
                        # Reduce position size to fit within portfolio limits
                        print(f"  ⚠️  Reducing position size to fit portfolio limits:")
                        print(f"     Original value: ${order_value:.2f}")
                        print(f"     Remaining capacity: ${remaining_capacity:.2f}")
                        quantity = remaining_capacity / entry_price
                        order_value = quantity * entry_price
                        position_info['quantity'] = quantity
                        position_info['value'] = order_value
                        print(f"     Adjusted quantity: {quantity:.4f}")
                        print(f"     Adjusted value: ${order_value:.2f}")
                else:
                    print(f"    ✅ Portfolio exposure check passed ({proposed_exposure_pct:.1f}% <= {max_exposure_pct:.1f}%)")
            
            # Check if we have enough available balance (for BUY orders)
            # If not, reduce position size to fit available balance
            if side == OrderSide.BUY and order_value > available_balance:
                # Reduce quantity to fit available balance (with 2% buffer for fees/slippage)
                max_order_value = available_balance * 0.98
                adjusted_quantity = max_order_value / entry_price
                
                if adjusted_quantity < quantity * 0.5:  # If we need to reduce by more than 50%, reject
                    error_msg = f"Insufficient available balance: Need ${order_value:.2f}, have ${available_balance:.2f} (would require >50% reduction)"
                    print(f"  ❌ Trade NOT executed: {error_msg}")
                    self.logger.log_error(Exception(error_msg), {
                        'symbol': symbol,
                        'action': 'execute_trade',
                        'order_value': order_value,
                        'available_balance': available_balance
                    })
                    return
                else:
                    print(f"  ⚠️  Adjusting position size to fit available balance:")
                    print(f"     Original quantity: {quantity:.4f}")
                    print(f"     Adjusted quantity: {adjusted_quantity:.4f}")
                    quantity = adjusted_quantity
                    position_info['quantity'] = quantity
                    position_info['value'] = order_value = quantity * entry_price
                    print(f"     Adjusted order value: ${order_value:.2f}")
            
            # Final safety check: Ensure OrderManager is using the correct broker
            if self.order_manager and self.order_manager.broker != broker:
                print(f"  🔄 Updating OrderManager broker from {type(self.order_manager.broker).__name__} to {type(broker).__name__}")
                self.order_manager.broker = broker
            
            # Place order
            print(f"  📤 Placing order via OrderManager...")
            print(f"  Using broker: {type(self.order_manager.broker).__name__}")
            order_result = self.order_manager.place_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=OrderType.MARKET,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            print(f"  Order result: {order_result}")
            
            # Store trade attempt in database (whether successful or failed)
            order_id = order_result.get('order_id')
            error_msg = order_result.get('error')
            order_status = order_result.get('status', OrderStatus.REJECTED if error_msg else OrderStatus.PENDING)
            
            if order_id:
                # Get executed quantity from broker (may differ from calculated due to rounding)
                filled_quantity = order_result.get('filled_quantity', quantity)
                executed_price = order_result.get('price', entry_price)
                
                # Log quantity mismatch if significant
                if abs(filled_quantity - quantity) > 0.01:
                    print(f"  ⚠️  Quantity mismatch: Calculated={quantity:.4f}, Executed={filled_quantity:.4f} (difference: {abs(filled_quantity - quantity):.4f})")
                    quantity = filled_quantity  # Use executed quantity for consistency
                
                # Use executed price if available (may differ from expected due to slippage)
                if executed_price and abs(executed_price - entry_price) > 0.01:
                    print(f"  ⚠️  Price difference: Expected=${entry_price:.2f}, Executed=${executed_price:.2f} (slippage: {abs(executed_price - entry_price):.2f})")
                    entry_price = executed_price
                
                # CRITICAL: Ensure stop_loss and take_profit are never None before storing
                # If they're None, calculate emergency fallbacks
                if stop_loss is None:
                    print(f"  ❌ CRITICAL: Stop loss is None! Calculating emergency fallback...")
                    if side == OrderSide.BUY:
                        stop_loss = entry_price * 0.98  # 2% stop loss for longs
                    else:
                        stop_loss = entry_price * 1.02  # 2% stop loss for shorts
                    print(f"  ⚠️  EMERGENCY: Set stop loss to ${stop_loss:.2f} (2% default)")
                
                if take_profit is None and stop_loss is not None:
                    print(f"  ⚠️  WARNING: Take profit is None! Calculating emergency fallback...")
                    risk = abs(entry_price - stop_loss)
                    if risk > 0:
                        reward = risk * 2.0  # Default 1:2 risk/reward
                        if side == OrderSide.BUY:
                            take_profit = entry_price + reward
                        else:
                            take_profit = entry_price - reward
                        print(f"  ⚠️  EMERGENCY: Set take profit to ${take_profit:.2f} (1:2 R/R)")
                
                # Successful order - store as open trade with EXECUTED quantity and price
                trade_data = {
                    'trade_id': order_id,
                    'symbol': symbol,
                    'side': side.value if hasattr(side, 'value') else str(side),
                    'quantity': filled_quantity,  # Use executed quantity, not calculated
                    'entry_price': entry_price,  # Use executed price if available
                    'entry_time': datetime.now(),
                    'status': 'open',
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'strategy': decision.get('reason', 'smarttrader')
                }
                
                stored_id = self.storage.store_trade(trade_data)
                if stored_id:
                    print(f"  ✅ Trade stored in database with ID: {stored_id}")
                    if stop_loss:
                        print(f"  📊 Stop Loss: ${stop_loss:.2f} ({abs((entry_price - stop_loss) / entry_price * 100):.2f}% risk)")
                    else:
                        print(f"  ⚠️  WARNING: Stop Loss is NULL - this is a critical issue!")
                    if take_profit:
                        risk = abs(entry_price - stop_loss) if stop_loss else 0
                        reward = abs(take_profit - entry_price) if take_profit else 0
                        rr_ratio = reward / risk if risk > 0 else 0
                        print(f"  📊 Take Profit: ${take_profit:.2f} (Risk/Reward: 1:{rr_ratio:.2f})")
                    else:
                        print(f"  ⚠️  WARNING: Take Profit is NULL")
                    
                    # Verify storage
                    try:
                        stored_trade = self.storage.get_open_trades(symbol=symbol)
                        stored_trade = next((t for t in stored_trade if t.get('trade_id') == order_id), None)
                        if stored_trade:
                            if stored_trade.get('stop_loss') != stop_loss:
                                print(f"  ❌ CRITICAL: Stop loss mismatch! Stored: {stored_trade.get('stop_loss')}, Expected: {stop_loss}")
                            if stored_trade.get('take_profit') != take_profit:
                                print(f"  ❌ CRITICAL: Take profit mismatch! Stored: {stored_trade.get('take_profit')}, Expected: {take_profit}")
                    except Exception as verify_error:
                        print(f"  ⚠️  Could not verify storage: {verify_error}")
                
                trade_log = {
                    'symbol': symbol,
                    'side': side.value if hasattr(side, 'value') else str(side),
                    'quantity': position_info.get('quantity', 0),
                    'price': entry_price,
                    'order_id': order_id,
                    'regime': decision.get('regime', 'Unknown'),
                    'regime_adjustment': position_info.get('regime_adjustment', 1.0),
                    'source': 'smarttrader',  # Mark as created by our system
                    'strategy': decision.get('reason', 'Unknown')
                }
                
                # Add DRL info if available
                if decision.get('drl_action') is not None:
                    trade_log['drl_action'] = decision.get('drl_action')
                    trade_log['drl_confidence'] = decision.get('drl_confidence')
                
                self.logger.log_trade(trade_log)
                print(f"  ✅✅✅ Trade EXECUTED successfully!")
                print(f"     Order ID: {order_id}")
                print(f"     {symbol} {side.value if hasattr(side, 'value') else side} {quantity} @ ${entry_price:.2f}")
                
                # Record trade for performance tracking
                if self.performance_tracker:
                    self.performance_tracker.record_trade(trade_log)
            else:
                # Failed order - still store for tracking (marked as rejected)
                print(f"  ❌ Trade FAILED: No order_id returned")
                print(f"     Order result: {order_result}")
                error_msg = error_msg or order_result.get('error', 'Unknown error')
                print(f"     Error: {error_msg}")
                
                # Store failed trade attempt for tracking
                failed_trade_id = f"failed_{symbol}_{datetime.now().timestamp()}"
                failed_trade_data = {
                    'trade_id': failed_trade_id,
                    'symbol': symbol,
                    'side': side.value if hasattr(side, 'value') else str(side),
                    'quantity': quantity,
                    'entry_price': entry_price,
                    'entry_time': datetime.now(),
                    'status': 'rejected',  # Mark as rejected
                    'strategy': f"smarttrader_failed_{error_msg[:50]}"  # Include error in strategy field
                }
                
                stored_id = self.storage.store_trade(failed_trade_data)
                if stored_id:
                    print(f"  📝 Failed trade attempt stored in database for tracking")
                
                self.logger.log_error(Exception(f"Trade execution failed: {error_msg}"), {
                    'symbol': symbol,
                    'action': 'execute_trade',
                    'order_result': order_result,
                    'error_code': order_result.get('error_code'),
                    'available_balance': available_balance,
                    'order_value': order_value
                })
        
        except Exception as e:
            print(f"  ❌❌❌ EXCEPTION in _execute_trade: {e}")
            import traceback
            print(f"  Traceback: {traceback.format_exc()}")
            self.logger.log_error(e, {'symbol': symbol, 'action': 'execute_trade'})
    
    def _manage_open_positions(self):
        """
        Actively manage all open positions (including external synced trades)
        - Check stop losses and take profits
        - Update trailing stops
        - Close positions when needed
        """
        try:
            # Get all open positions from database
            open_positions = self.storage.get_open_trades()
            
            if not open_positions:
                return
            
            # Cleanup: Verify positions actually exist in brokers
            # This handles cases where positions were closed externally or broker sync missed them
            cleaned_positions = []
            for position in open_positions:
                symbol = position.get('symbol')
                trade_id = position.get('id')
                side = position.get('side', 'buy')
                entry_price = position.get('entry_price', 0)
                quantity = position.get('quantity', 0)
                
                # Determine which broker should have this position
                broker_to_check = None
                if '/' in symbol:
                    base = symbol.split('/')[0].upper()
                    if base in ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'LINK', 'MATIC', 'AVAX', 'UNI', 'ATOM']:
                        broker_to_check = self.brokers.get('stocks') or self.brokers.get('alpaca') or self.brokers.get('crypto')
                    elif base in ['EUR', 'GBP', 'USD', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']:
                        broker_to_check = self.brokers.get('forex') or self.brokers.get('oanda')
                    elif base in ['XAU', 'XAG']:
                        broker_to_check = self.brokers.get('forex') or self.brokers.get('oanda')
                else:
                    broker_to_check = self.brokers.get('stocks') or self.brokers.get('alpaca')
                
                # Check if position exists in broker
                position_exists = False
                if broker_to_check and hasattr(broker_to_check, 'get_open_positions'):
                    try:
                        broker_positions = broker_to_check.get_open_positions()
                        # Normalize symbol for comparison
                        normalized_symbol = symbol.replace('/', '').replace('_', '').upper()
                        for bp in broker_positions:
                            bp_symbol = bp.get('symbol', '').replace('/', '').replace('_', '').upper()
                            if bp_symbol == normalized_symbol:
                                position_exists = True
                                break
                    except Exception as e:
                        # If we can't check, assume position exists to be safe
                        position_exists = True
                
                if not position_exists and broker_to_check:
                    # Position doesn't exist in broker - mark as closed
                    try:
                        # Get current price for P&L calculation
                        current_price = 0
                        try:
                            current_price = broker_to_check.get_current_price(symbol)
                        except:
                            # Fallback to entry price if we can't get current price
                            current_price = entry_price
                        
                        # Calculate final P&L
                        if side == 'buy' or (hasattr(side, 'value') and side.value == 1):
                            final_pnl = (current_price - entry_price) * quantity
                        else:
                            final_pnl = (entry_price - current_price) * quantity
                        
                        # Mark as closed in database
                        self.storage.update_trade(trade_id, {
                            'status': 'closed',
                            'exit_price': current_price if current_price > 0 else entry_price,
                            'exit_time': datetime.now(),
                            'pnl': final_pnl
                        })
                        print(f"🧹 Cleaned up stale position: {symbol} (not found in broker, marked as closed, P&L: ${final_pnl:.2f})")
                    except Exception as cleanup_error:
                        print(f"⚠️  Error cleaning up position {symbol}: {cleanup_error}")
                        # Keep position in list if cleanup failed
                        cleaned_positions.append(position)
                else:
                    # Position exists or we couldn't verify - keep it
                    cleaned_positions.append(position)
            
            # Update open_positions to only include positions that actually exist
            open_positions = cleaned_positions
            
            if not open_positions:
                return
            
            # Count positions with stop-loss/take-profit for monitoring
            positions_with_sl_tp = sum(1 for p in open_positions if p.get('stop_loss') or p.get('take_profit'))
            if positions_with_sl_tp > 0:
                print(f"\n[Position Management] Monitoring {len(open_positions)} open positions ({positions_with_sl_tp} with stop-loss/take-profit)")
            
            # Get broker for price checks - prefer Alpaca for crypto, OANDA for forex
            broker_for_prices = None
            for broker_name, broker in self.brokers.items():
                # Prefer Alpaca for crypto symbols, OANDA for forex
                if 'alpaca' in broker_name.lower() or 'crypto' in broker_name.lower():
                    broker_for_prices = broker
                    break
            
            # Fallback to any broker if no crypto broker found
            if not broker_for_prices:
                for broker_name, broker in self.brokers.items():
                    try:
                        # Test if broker can get prices (skip OANDA for crypto)
                        test_price = broker.get_current_price('BTC/USD')
                        if test_price > 0:  # Only use if it actually returns a price
                            broker_for_prices = broker
                            break
                    except:
                        continue
            
            if not broker_for_prices:
                return  # Can't manage positions without price data
            
            # Process each position
            for position in open_positions:
                try:
                    symbol = position.get('symbol')
                    side_raw = position.get('side', 'buy')
                    # Normalize side to lowercase string
                    if hasattr(side_raw, 'value'):
                        side = side_raw.value.lower()
                    else:
                        side = str(side_raw).lower()
                    entry_price = position.get('entry_price', 0)
                    quantity = position.get('quantity', 0)
                    trade_id = position.get('trade_id')
                    stop_loss = position.get('stop_loss')
                    take_profit = position.get('take_profit')
                    entry_time_str = position.get('entry_time')
                    strategy = position.get('strategy', '')
                    
                    if not symbol or quantity == 0 or entry_price == 0:
                        continue
                    
                    # Validate stop_loss and take_profit are numbers (not None, not empty string)
                    if stop_loss is not None:
                        try:
                            stop_loss = float(stop_loss)
                        except (ValueError, TypeError):
                            stop_loss = None
                    else:
                        stop_loss = None
                    
                    if take_profit is not None:
                        try:
                            take_profit = float(take_profit)
                        except (ValueError, TypeError):
                            take_profit = None
                    else:
                        take_profit = None
                    
                    # EMERGENCY FIX: Set default stop loss for trades without one
                    # This prevents unlimited losses on existing trades
                    if not stop_loss:
                        # Use 2% stop loss as emergency measure
                        if side == 'buy':
                            stop_loss = entry_price * 0.98  # 2% stop loss
                        else:  # sell
                            stop_loss = entry_price * 1.02  # 2% stop loss for shorts
                        
                        # Update database with emergency stop loss
                        try:
                            self.storage.update_trade(trade_id, {'stop_loss': stop_loss})
                            print(f"  ⚠️  EMERGENCY: Set default stop loss for {symbol}: ${stop_loss:.2f} (2% default)")
                        except Exception as e:
                            print(f"  ❌ Error setting emergency stop loss: {e}")
                    
                    # EMERGENCY FIX: Set default take profit if missing (1:3 risk/reward - updated from 1:2)
                    if not take_profit and stop_loss:
                        risk = abs(entry_price - stop_loss)
                        if risk > 0:
                            reward = risk * 3.0  # 1:3 risk/reward (updated from 2.0 to match config)
                            if side == 'buy':
                                take_profit = entry_price + reward
                            else:  # sell
                                take_profit = entry_price - reward
                            
                            # Update database with emergency take profit
                            try:
                                self.storage.update_trade(trade_id, {'take_profit': take_profit})
                                print(f"  ⚠️  EMERGENCY: Set default take profit for {symbol}: ${take_profit:.2f} (1:3 R/R)")
                            except Exception as e:
                                print(f"  ❌ Error setting emergency take profit: {e}")
                    
                    # Get current price - use appropriate broker for symbol type
                    current_price = entry_price  # Default fallback
                    try:
                        # Determine which broker to use based on symbol
                        price_broker = None
                        if '/' in symbol:
                            base = symbol.split('/')[0].upper()
                            # Use Alpaca for crypto
                            if base in ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'LINK', 'MATIC', 'AVAX', 'UNI', 'ATOM']:
                                for broker_name, broker in self.brokers.items():
                                    if 'alpaca' in broker_name.lower():
                                        price_broker = broker
                                        break
                            else:
                                # Forex - try OANDA first, then Alpaca
                                for broker_name, broker in self.brokers.items():
                                    if 'oanda' in broker_name.lower():
                                        price_broker = broker
                                        break
                        
                        # Fallback to broker_for_prices if no specific broker found
                        if not price_broker:
                            price_broker = broker_for_prices
                        
                        if price_broker:
                            fetched_price = price_broker.get_current_price(symbol)
                            if fetched_price > 0:
                                current_price = fetched_price
                    except Exception as e:
                        # Silently continue - use entry_price as fallback
                        pass
                    
                    # Check stop loss and take profit
                    should_close = False
                    close_reason = None
                    
                    # Debug logging for crypto positions (since they don't have bracket orders)
                    is_crypto = '/' in symbol and symbol.split('/')[0].upper() in ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'LINK', 'MATIC', 'AVAX', 'UNI', 'ATOM']
                    
                    # Calculate current P&L for logging
                    if side == 'buy':
                        current_pnl = (current_price - entry_price) * quantity
                        pnl_percent = ((current_price - entry_price) / entry_price) * 100
                    else:  # sell
                        current_pnl = (entry_price - current_price) * quantity
                        pnl_percent = ((entry_price - current_price) / entry_price) * 100
                    
                    # Log position status (every check for monitoring)
                    sl_str = f"${stop_loss:.2f}" if stop_loss else 'None'
                    tp_str = f"${take_profit:.2f}" if take_profit else 'None'
                    print(f"  📊 Position {symbol}: Price ${current_price:.2f} | Entry ${entry_price:.2f} | P&L ${current_pnl:.2f} ({pnl_percent:+.2f}%) | SL {sl_str} | TP {tp_str}")
                    
                    if stop_loss:
                        if self.stop_loss_manager.check_stop_loss(current_price, stop_loss, side):
                            should_close = True
                            close_reason = 'stop_loss'
                            print(f"  🛑 STOP LOSS TRIGGERED for {symbol}:")
                            print(f"     Current Price: ${current_price:.2f}")
                            print(f"     Stop Loss: ${stop_loss:.2f}")
                            print(f"     Entry Price: ${entry_price:.2f}")
                            print(f"     Side: {side.upper()}")
                            print(f"     Loss: ${current_pnl:.2f} ({pnl_percent:.2f}%)")
                            print(f"     Stop loss was properly monitored and triggered")
                    
                    # Check take profit
                    if take_profit and not should_close:
                        if side == 'buy' and current_price >= take_profit:
                            should_close = True
                            close_reason = 'take_profit'
                            print(f"  🎯 TAKE PROFIT TRIGGERED for {symbol}:")
                            print(f"     Current Price: ${current_price:.2f}")
                            print(f"     Take Profit: ${take_profit:.2f}")
                            print(f"     Entry Price: ${entry_price:.2f}")
                            print(f"     Side: {side.upper()}")
                            print(f"     Profit: ${current_pnl:.2f} ({pnl_percent:.2f}%)")
                            print(f"     Take profit was properly monitored and triggered")
                        elif side == 'sell' and current_price <= take_profit:
                            should_close = True
                            close_reason = 'take_profit'
                            print(f"  🎯 TAKE PROFIT TRIGGERED for {symbol}:")
                            print(f"     Current Price: ${current_price:.2f}")
                            print(f"     Take Profit: ${take_profit:.2f}")
                            print(f"     Entry Price: ${entry_price:.2f}")
                            print(f"     Side: {side.upper()}")
                            print(f"     Profit: ${current_pnl:.2f} ({pnl_percent:.2f}%)")
                            print(f"     Take profit was properly monitored and triggered")
                    
                    # Debug logging for crypto positions (show current status)
                    if is_crypto and (stop_loss or take_profit):
                        if not should_close:
                            status_msg = f"  📊 Monitoring {symbol}: Price ${current_price:.2f}"
                            if stop_loss:
                                sl_diff = ((current_price - stop_loss) / stop_loss * 100) if side == 'buy' else ((stop_loss - current_price) / stop_loss * 100)
                                status_msg += f", Stop-loss ${stop_loss:.2f} ({sl_diff:+.2f}%)"
                            if take_profit:
                                tp_diff = ((current_price - take_profit) / take_profit * 100) if side == 'buy' else ((take_profit - current_price) / take_profit * 100)
                                status_msg += f", Take-profit ${take_profit:.2f} ({tp_diff:+.2f}%)"
                            print(status_msg)
                    
                    # Check time-based exit (if configured)
                    if not should_close and entry_time_str:
                        try:
                            from dateutil import parser
                            entry_time = parser.isoparse(entry_time_str) if isinstance(entry_time_str, str) else entry_time_str
                            if self.stop_loss_manager.check_time_based_exit(entry_time, current_price, entry_price):
                                should_close = True
                                close_reason = 'time_based_exit'
                        except:
                            pass
                    
                    # Close position if needed
                    if should_close:
                        # Determine which broker to use for closing
                        broker_to_use = None
                        
                        # Try to identify broker from strategy field
                        if '_sync' in strategy:
                            # External trade - find the broker it came from
                            broker_name = strategy.replace('_sync', '')
                            broker_to_use = self.brokers.get(broker_name)
                        elif 'smarttrader' in strategy.lower():
                            # SmartTrader trade - determine broker from symbol type
                            if '/' in symbol:
                                base = symbol.split('/')[0].upper()
                                # Crypto symbols
                                if base in ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'LINK', 'MATIC', 'AVAX', 'UNI', 'ATOM']:
                                    broker_to_use = self.brokers.get('stocks') or self.brokers.get('alpaca') or self.brokers.get('crypto')
                                # Forex symbols
                                elif base in ['EUR', 'GBP', 'USD', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']:
                                    broker_to_use = self.brokers.get('forex') or self.brokers.get('oanda')
                                # Commodities
                                elif base in ['XAU', 'XAG']:
                                    broker_to_use = self.brokers.get('forex') or self.brokers.get('oanda')
                            else:
                                # Stock symbol
                                broker_to_use = self.brokers.get('stocks') or self.brokers.get('alpaca')
                        
                        # Fallback to any broker that supports the symbol
                        if not broker_to_use:
                            for broker_name, broker in self.brokers.items():
                                try:
                                    # Check if broker supports this symbol
                                    test_price = broker.get_current_price(symbol)
                                    if test_price > 0:
                                        broker_to_use = broker
                                        break
                                except:
                                    continue
                        
                        if broker_to_use:
                            try:
                                # Close the position
                                order_side = OrderSide.SELL if side == 'buy' else OrderSide.BUY
                                close_result = broker_to_use.close_position(symbol, order_side)
                                
                                if close_result:
                                    # Calculate final P&L
                                    if side == 'buy':
                                        final_pnl = (current_price - entry_price) * quantity
                                    else:
                                        final_pnl = (entry_price - current_price) * quantity
                                    
                                    # Update trade in database
                                    self.storage.update_trade(trade_id, {
                                        'status': 'closed',
                                        'exit_price': current_price,
                                        'exit_time': datetime.now(),
                                        'pnl': final_pnl
                                    })
                                    
                                    # Remove from portfolio risk manager if it exists
                                    if hasattr(self, 'portfolio_risk_manager'):
                                        self.portfolio_risk_manager.remove_position(symbol)
                                    
                                    self.logger.log_decision({
                                        'symbol': symbol,
                                        'action': 'position_closed',
                                        'side': side,
                                        'price': current_price,
                                        'reason': close_reason,
                                        'pnl': final_pnl
                                    }, event='position_closed')
                                    print(f"✅ Position closed: {symbol} {side} @ ${current_price:.2f} - {close_reason} (P&L: ${final_pnl:.2f})")
                                else:
                                    # Position not found in broker - likely already closed
                                    # Mark as closed in database to keep it in sync
                                    print(f"⚠️  Position {symbol} not found in broker - marking as closed in database (likely already closed)")
                                    try:
                                        # Calculate final P&L based on current price
                                        if side == 'buy':
                                            final_pnl = (current_price - entry_price) * quantity
                                        else:
                                            final_pnl = (entry_price - current_price) * quantity
                                        
                                        self.storage.update_trade(trade_id, {
                                            'status': 'closed',
                                            'exit_price': current_price,
                                            'exit_time': datetime.now(),
                                            'pnl': final_pnl
                                        })
                                        print(f"✅ Marked position {symbol} as closed in database (P&L: ${final_pnl:.2f})")
                                    except Exception as db_error:
                                        print(f"❌ Failed to update database for {symbol}: {db_error}")
                                    print(f"❌ Failed to close position {symbol} via broker API (reason: {close_reason})")
                            except Exception as e:
                                error_msg = str(e)
                                print(f"❌ Error closing position {symbol} (reason: {close_reason}): {error_msg}")
                                self.logger.log_error(e, {'symbol': symbol, 'action': 'close_position', 'reason': close_reason})
                        else:
                            print(f"⚠️  No broker found to close position {symbol} (reason: {close_reason})")
                    
                    # Update trailing stop if enabled and position is profitable
                    elif stop_loss and self.stop_loss_manager.trailing_stop_enabled:
                        # Only update trailing stop for SmartTrader trades or if explicitly managing external trades
                        manage_external = self.trading_config.get('manage_external_positions', True)  # Default to True
                        if strategy == 'smarttrader' or (manage_external and '_sync' in strategy):
                            # Get market data for ATR calculation
                            try:
                                data_fetcher = DataFetcher(broker_for_prices, self.storage)
                                data_list = data_fetcher.get_latest_data(symbol, '1h', days=7)
                                if data_list:
                                    df = pd.DataFrame(data_list)
                                    df.set_index('timestamp', inplace=True)
                                    df = self.indicators.add_all_indicators(df)
                                    
                                    # Update trailing stop
                                    new_stop_loss = self.stop_loss_manager.update_trailing_stop(
                                        current_price, entry_price, stop_loss, side, df
                                    )
                                    
                                    # Update if stop loss changed
                                    if abs(new_stop_loss - stop_loss) > 0.01:
                                        self.storage.update_trade(trade_id, {'stop_loss': new_stop_loss})
                                        self.logger.log_decision({
                                            'symbol': symbol,
                                            'action': 'trailing_stop_updated',
                                            'old_stop_loss': stop_loss,
                                            'new_stop_loss': new_stop_loss
                                        }, event='trailing_stop_updated')
                            except Exception as e:
                                # Silently fail - trailing stop update is not critical
                                pass
                
                except Exception as e:
                    self.logger.log_error(e, {'symbol': position.get('symbol'), 'action': 'manage_position'})
        
        except Exception as e:
            self.logger.log_error(e, {'component': 'position_management'})
    
    def _update_missing_take_profits(self, open_trades: List[Dict]):
        """
        Calculate and update stop_loss and take_profit for existing positions that don't have them
        Uses proper ATR-based calculations (same as orchestrator) instead of emergency fallbacks
        
        Args:
            open_trades: List of open trade dictionaries from database
        """
        if not open_trades:
            return
        
        # Get risk/reward ratio from config
        orchestrator_config = self.trading_config.get('agents', {}).get('orchestrator', {})
        risk_reward_ratio = orchestrator_config.get('risk_reward_ratio', 2.0)  # Default 1:2
        
        updated_sl_count = 0
        updated_tp_count = 0
        print(f"\n[STARTUP] Checking {len(open_trades)} open positions for missing stop_loss/take_profit...")
        print(f"  Using ATR-based calculations with {risk_reward_ratio:.1f}:1 risk/reward ratio")
        
        for trade in open_trades:
            trade_id = trade.get('trade_id') or trade.get('id')  # Try both fields
            symbol = trade.get('symbol')
            side = trade.get('side', 'buy').lower()
            entry_price = trade.get('entry_price')
            stop_loss = trade.get('stop_loss')
            take_profit = trade.get('take_profit')
            
            if not entry_price or entry_price <= 0:
                print(f"  ⚠️  {symbol}: Skipping (invalid entry_price: {entry_price})")
                continue
            
            # Determine asset class from symbol or broker
            asset_class = 'crypto'  # Default
            if '/' in symbol:
                # Try to determine from symbol format
                if any(fx in symbol for fx in ['EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']):
                    asset_class = 'forex'
                elif any(metal in symbol for metal in ['XAU', 'XAG']):
                    asset_class = 'commodities'
            
            # CRITICAL: Calculate proper stop_loss using ATR (same as orchestrator)
            if not stop_loss or stop_loss <= 0:
                try:
                    # Fetch market data for ATR calculation
                    print(f"  📊 {symbol}: Fetching market data for ATR-based stop loss calculation...")
                    # Get broker for asset class
                    if asset_class == 'forex':
                        broker = self.brokers.get('forex') or self.brokers.get('oanda')
                    elif asset_class == 'commodities':
                        broker = self.brokers.get('forex') or self.brokers.get('oanda')  # OANDA handles commodities
                    elif asset_class == 'crypto':
                        broker = self.brokers.get('crypto') or self.brokers.get('stocks')  # Alpaca handles crypto
                    else:
                        broker = self.brokers.get(asset_class) or self.brokers.get('stocks')
                    
                    if not broker:
                        print(f"  ⚠️  {symbol}: No broker available for {asset_class}, using emergency fallback")
                        # Emergency fallback
                        if side == 'buy':
                            stop_loss = entry_price * 0.98
                        else:
                            stop_loss = entry_price * 1.02
                    else:
                        # Fetch recent data for ATR calculation
                        try:
                            data = broker.get_historical_data(symbol, timeframe='1h', limit=50)
                            if data is None or data.empty or len(data) < 14:
                                raise ValueError("Insufficient data for ATR")
                            
                            # Preprocess data
                            data = self.preprocessor.preprocess(data)
                            
                            # Calculate ATR-based stop loss (same logic as orchestrator)
                            atr = self.indicators.atr(data)
                            atr_value = atr.iloc[-1] if not atr.empty else entry_price * 0.02
                            
                            # Ensure ATR value is reasonable (at least 0.5% of price)
                            min_stop_distance = entry_price * 0.005  # 0.5% minimum
                            atr_value = max(atr_value, min_stop_distance)
                            
                            stop_distance = atr_value * 2.0  # 2x ATR
                            
                            is_buy = side == 'buy'
                            if is_buy:
                                stop_loss = entry_price - stop_distance
                            else:  # sell
                                stop_loss = entry_price + stop_distance
                            
                            # Ensure stop loss is valid (not negative for buy, reasonable for sell)
                            if is_buy:
                                stop_loss = max(stop_loss, entry_price * 0.95)  # Max 5% loss
                            else:
                                stop_loss = min(stop_loss, entry_price * 1.05)  # Max 5% loss for shorts
                            
                            risk_pct = abs((entry_price - stop_loss) / entry_price * 100)
                            print(f"  ✅ {symbol}: Calculated ATR-based stop loss ${stop_loss:.2f} ({risk_pct:.2f}% risk)")
                            
                        except Exception as data_error:
                            print(f"  ⚠️  {symbol}: Could not fetch data for ATR calculation: {data_error}")
                            print(f"  ⚠️  {symbol}: Using emergency fallback (2% stop loss)")
                            # Emergency fallback if data fetch fails
                            if side == 'buy':
                                stop_loss = entry_price * 0.98
                            else:
                                stop_loss = entry_price * 1.02
                    
                    # Update database
                    try:
                        success = self.storage.update_trade(trade_id, {'stop_loss': stop_loss})
                        if success:
                            updated_sl_count += 1
                            if stop_loss == entry_price * 0.98 or stop_loss == entry_price * 1.02:
                                print(f"  ⚠️  {symbol}: Set emergency stop loss ${stop_loss:.2f} (2% default)")
                            else:
                                print(f"  ✅ {symbol}: Updated stop loss to ${stop_loss:.2f}")
                        else:
                            print(f"  ⚠️  {symbol}: Failed to update stop loss in database")
                    except Exception as e:
                        print(f"  ❌ {symbol}: Error updating stop loss: {e}")
                        
                except Exception as e:
                    print(f"  ❌ {symbol}: Error calculating stop loss: {e}")
                    # Final emergency fallback
                    if side == 'buy':
                        stop_loss = entry_price * 0.98
                    else:
                        stop_loss = entry_price * 1.02
                    try:
                        self.storage.update_trade(trade_id, {'stop_loss': stop_loss})
                        updated_sl_count += 1
                        print(f"  ⚠️  {symbol}: Set emergency stop loss ${stop_loss:.2f} (2% default)")
                    except:
                        pass
            
            # Now calculate take profit if missing (using proper risk/reward ratio)
            if (take_profit is None or take_profit <= 0) and stop_loss and stop_loss > 0:
                # Calculate risk (distance from entry to stop loss)
                risk = abs(entry_price - stop_loss)
                
                # Only calculate if we have meaningful risk
                if risk > 0:
                    # Calculate reward based on risk/reward ratio
                    reward = risk * risk_reward_ratio
                    
                    # Set take profit
                    if side == 'buy':
                        take_profit = entry_price + reward
                    else:  # sell
                        take_profit = entry_price - reward
                    
                    # Update database
                    try:
                        success = self.storage.update_trade(trade_id, {'take_profit': take_profit})
                        if success:
                            updated_tp_count += 1
                            print(f"  ✅ {symbol}: Set take profit ${take_profit:.2f} (1:{risk_reward_ratio:.1f} R/R)")
                        else:
                            print(f"  ⚠️  {symbol}: Failed to update take profit")
                    except Exception as e:
                        print(f"  ❌ {symbol}: Error updating take profit: {e}")
                else:
                    print(f"  ⚠️  {symbol}: Cannot calculate take profit (stop loss equals entry price)")
        
        print(f"\n[STARTUP] ✅ Updated {updated_sl_count} positions with stop_loss, {updated_tp_count} positions with take_profit")
    
    def _sync_positions_from_brokers(self):
        """Sync open positions and orders from all brokers to local database"""
        try:
            for broker_name, broker in self.brokers.items():
                # Sync orders from Alpaca (if method exists)
                if hasattr(broker, 'get_orders'):
                    try:
                        # Get filled orders from the last 7 days
                        orders = broker.get_orders(status='filled', limit=100)
                        print(f"  Syncing {len(orders)} filled orders from {broker_name}...")
                        
                        for order in orders:
                            order_id = order.get('order_id')
                            symbol = order.get('symbol', '')
                            side = order.get('side', 'buy')
                            filled_qty = order.get('filled_quantity', 0)
                            price = order.get('price', 0)
                            filled_at = order.get('filled_at')
                            
                            if not order_id or filled_qty == 0 or price == 0:
                                continue
                            
                            # Normalize symbol format (Alpaca returns ETHUSD, we need ETH/USD)
                            normalized_symbol = symbol.replace('_', '/')
                            if '/' not in normalized_symbol and len(normalized_symbol) >= 6:
                                for base_len in [3, 4]:
                                    if len(normalized_symbol) > base_len:
                                        base = normalized_symbol[:base_len]
                                        quote = normalized_symbol[base_len:]
                                        if quote in ['USD', 'EUR', 'GBP', 'JPY']:
                                            normalized_symbol = f"{base}/{quote}"
                                            break
                            
                            # Check if trade already exists by order_id (most reliable check)
                            existing_trades = self.storage.get_all_trades(limit=1000)
                            trade_exists = False
                            is_smarttrader_trade = False
                            
                            for existing in existing_trades:
                                # Match by order_id (trade_id)
                                if existing.get('trade_id') == order_id:
                                    trade_exists = True
                                    existing_strategy = (existing.get('strategy') or '').lower()
                                    is_smarttrader_trade = ('smarttrader' in existing_strategy or 
                                                           not existing_strategy or
                                                           existing_strategy in ['unknown', 'none', ''])
                                    break
                            
                            # Also check if we have a trade for this symbol/side/price that matches
                            # (This helps sync trades that were placed but order_id wasn't stored)
                            if not trade_exists:
                                for existing in existing_trades:
                                    existing_symbol = existing.get('symbol', '').replace('_', '/')
                                    if '/' not in existing_symbol and len(existing_symbol) >= 6:
                                        for base_len in [3, 4]:
                                            if len(existing_symbol) > base_len:
                                                base = existing_symbol[:base_len]
                                                quote = existing_symbol[base_len:]
                                                if quote in ['USD', 'EUR', 'GBP', 'JPY']:
                                                    existing_symbol = f"{base}/{quote}"
                                                    break
                                    
                                    # Match by normalized symbol, side, and similar price (within 1%)
                                    if (existing_symbol == normalized_symbol and 
                                        existing.get('side', '').lower() == side.lower() and
                                        abs(existing.get('entry_price', 0) - price) / max(price, 0.01) < 0.01):
                                        trade_exists = True
                                        # Update the existing trade with the order_id if it doesn't have one
                                        if not existing.get('trade_id') or existing.get('trade_id', '').startswith('failed_'):
                                            self.storage.update_trade(existing.get('trade_id'), {'trade_id': order_id})
                                        existing_strategy = (existing.get('strategy') or '').lower()
                                        is_smarttrader_trade = ('smarttrader' in existing_strategy or 
                                                               not existing_strategy or
                                                               existing_strategy in ['unknown', 'none', ''])
                                        break
                            
                            if not trade_exists:
                                # Parse filled_at timestamp
                                entry_time = datetime.now()
                                if filled_at:
                                    try:
                                        from dateutil import parser
                                        entry_time = parser.parse(filled_at)
                                    except:
                                        pass
                                
                                # Check if position is still open (if not, mark as closed)
                                open_positions = broker.get_open_positions()
                                position_still_open = False
                                for pos in open_positions:
                                    if (pos.get('symbol') == symbol and 
                                        pos.get('side') == side and
                                        abs(pos.get('entry_price', 0) - price) / max(price, 0.01) < 0.01):
                                        position_still_open = True
                                        break
                                
                                # Determine if this is a SmartTrader trade
                                # If strategy wasn't determined from existing trade, check if it looks like ours
                                if not is_smarttrader_trade:
                                    # Check recent trades to see if this matches a SmartTrader pattern
                                    for recent in existing_trades[:50]:  # Check last 50 trades
                                        recent_strategy = (recent.get('strategy') or '').lower()
                                        if ('smarttrader' in recent_strategy or 
                                            recent_strategy in ['unknown', 'weighted voting', 'trend following', 'mean reversion']):
                                            is_smarttrader_trade = True
                                            break
                                
                                strategy_to_use = 'smarttrader' if is_smarttrader_trade else f'{broker_name}_order_sync'
                                
                                trade_data = {
                                    'trade_id': order_id,
                                    'symbol': normalized_symbol,  # Use normalized symbol
                                    'side': side,
                                    'quantity': filled_qty,
                                    'entry_price': price,
                                    'entry_time': entry_time,
                                    'status': 'open' if position_still_open else 'closed',
                                    'strategy': strategy_to_use
                                }
                                
                                stored_id = self.storage.store_trade(trade_data)
                                if stored_id:
                                    status_str = 'open' if position_still_open else 'closed'
                                    trade_type = "SmartTrader" if is_smarttrader_trade else "external"
                                    print(f"    ✅ Synced {trade_type} order {order_id}: {normalized_symbol} {side} {filled_qty} @ ${price:.2f} (status: {status_str})")
                    except Exception as e:
                        print(f"  Error syncing orders from {broker_name}: {e}")
                        self.logger.log_error(e, {'broker': broker_name, 'action': 'sync_orders'})
                
                # Sync open positions
                if not hasattr(broker, 'get_open_positions'):
                    continue
                
                try:
                    broker_positions = broker.get_open_positions()
                    if not broker_positions:
                        continue
                    
                    # Log position sync (using log_decision as info logging)
                    self.logger.log_decision({
                        'event': 'position_sync',
                        'broker': broker_name,
                        'position_count': len(broker_positions)
                    })
                    
                    for pos in broker_positions:
                        # Normalize symbol format properly
                        # Alpaca returns ETHUSD, we need ETH/USD
                        # OANDA uses BTC_USD, we use BTC/USD
                        raw_symbol = pos.get('symbol', '')
                        symbol = raw_symbol.replace('_', '/')
                        
                        # Handle Alpaca crypto format: ETHUSD -> ETH/USD
                        if '/' not in symbol and len(symbol) >= 6:
                            # Try to split common crypto pairs (3-4 char base, 3 char quote)
                            for base_len in [3, 4]:
                                if len(symbol) > base_len:
                                    base = symbol[:base_len]
                                    quote = symbol[base_len:]
                                    if quote in ['USD', 'EUR', 'GBP', 'JPY']:
                                        symbol = f"{base}/{quote}"
                                        break
                        
                        side = pos.get('side', 'buy')
                        quantity = pos.get('quantity', 0)
                        entry_price = pos.get('entry_price', 0)
                        current_price = pos.get('current_price', entry_price)  # Use broker's current_price if available
                        unrealized_pnl = pos.get('unrealized_pnl', 0)  # Use broker's P&L (most accurate)
                        
                        if quantity == 0 or entry_price == 0:
                            continue
                        
                        # Generate trade_id from broker position
                        # Use a consistent format: broker_symbol_side_entryprice
                        trade_id = f"{broker_name}_{symbol.replace('/', '_')}_{side}_{int(entry_price * 1000)}"
                        
                        # Check if trade already exists - check ALL trades (not just open ones)
                        # Also check with both symbol formats (ETHUSD and ETH/USD)
                        existing_trades = self.storage.get_all_trades(limit=500)  # Get more trades to check
                        trade_exists = False
                        is_smarttrader_trade = False
                        
                        for existing in existing_trades:
                            existing_symbol = existing.get('symbol', '')
                            existing_side = existing.get('side', '')
                            existing_price = existing.get('entry_price', 0)
                            existing_strategy = existing.get('strategy', '')
                            
                            # Normalize existing symbol for comparison
                            existing_symbol_normalized = existing_symbol.replace('_', '/')
                            if '/' not in existing_symbol_normalized and len(existing_symbol_normalized) >= 6:
                                for base_len in [3, 4]:
                                    if len(existing_symbol_normalized) > base_len:
                                        base = existing_symbol_normalized[:base_len]
                                        quote = existing_symbol_normalized[base_len:]
                                        if quote in ['USD', 'EUR', 'GBP', 'JPY']:
                                            existing_symbol_normalized = f"{base}/{quote}"
                                            break
                            
                            # Match by symbol (normalized), side, and entry price (within 1% tolerance for better matching)
                            symbol_match = (existing_symbol_normalized == symbol or 
                                          existing_symbol == raw_symbol or
                                          existing_symbol == symbol)
                            
                            side_match = (existing_side.lower() == side.lower() if isinstance(side, str) else 
                                         existing_side == side.value if hasattr(side, 'value') else existing_side == str(side))
                            
                            price_match = (abs(existing_price - entry_price) / max(entry_price, 0.01) < 0.01)  # 1% tolerance
                            
                            if symbol_match and side_match and price_match:
                                trade_exists = True
                                
                                # Check if this is a SmartTrader trade
                                # Default to SmartTrader unless explicitly marked as external (_sync)
                                strategy_lower = (existing_strategy or '').lower()
                                is_smarttrader_trade = (
                                    'smarttrader' in strategy_lower or
                                    strategy_lower in ['unknown', 'weighted voting', 'trend following', 'mean reversion', 'momentum', 'breakout'] or
                                    not strategy_lower or
                                    strategy_lower == 'none' or
                                    '_sync' not in strategy_lower  # If it doesn't have _sync, assume it's SmartTrader
                                )
                                
                                # Update P&L, entry_price, and other fields from broker data
                                # This ensures we use Alpaca's accurate avg_entry_price and unrealized_pl
                                update_data = {}
                                
                                # Always update P&L with broker's value (most accurate)
                                if abs(unrealized_pnl - existing.get('pnl', 0)) > 0.01:
                                    update_data['pnl'] = unrealized_pnl
                                
                                # Update entry_price if it differs significantly (broker's avg_entry_price is authoritative)
                                existing_entry = existing.get('entry_price', 0)
                                if abs(entry_price - existing_entry) / max(entry_price, 0.01) > 0.001:  # 0.1% difference
                                    update_data['entry_price'] = entry_price
                                
                                # Update quantity if it changed
                                existing_qty = existing.get('quantity', 0)
                                if abs(quantity - existing_qty) > 0.0001:
                                    update_data['quantity'] = quantity
                                
                                # Update status to 'open' if it was closed but position still exists
                                if existing.get('status') != 'open' and quantity > 0:
                                    update_data['status'] = 'open'
                                
                                # Only update strategy if it's truly unknown - don't overwrite SmartTrader trades
                                if not existing_strategy or existing_strategy in ['unknown', 'None', '']:
                                    if is_smarttrader_trade:
                                        update_data['strategy'] = 'smarttrader'  # Mark as SmartTrader trade
                                    else:
                                        update_data['strategy'] = f'{broker_name}_sync'
                                
                                if update_data:
                                    self.storage.update_trade(
                                        existing.get('trade_id'),
                                        update_data
                                    )
                                break
                        
                        # Only create new trade if it doesn't exist AND it's not a SmartTrader trade
                        if not trade_exists:
                            # Check if this might be a SmartTrader trade
                            might_be_smarttrader = False
                            
                            try:
                                # Method 1: Check if we have any recent trades with similar characteristics
                                recent_trades = self.storage.get_all_trades(limit=200)  # Check more trades
                                for recent in recent_trades:
                                    recent_symbol = recent.get('symbol', '').replace('_', '/')
                                    if '/' not in recent_symbol and len(recent_symbol) >= 6:
                                        for base_len in [3, 4]:
                                            if len(recent_symbol) > base_len:
                                                base = recent_symbol[:base_len]
                                                quote = recent_symbol[base_len:]
                                                if quote in ['USD', 'EUR', 'GBP', 'JPY']:
                                                    recent_symbol = f"{base}/{quote}"
                                                    break
                                    
                                    if (recent_symbol == symbol and 
                                        recent.get('side', '').lower() == side.lower() and
                                        abs(recent.get('entry_price', 0) - entry_price) / max(entry_price, 0.01) < 0.05):  # 5% tolerance
                                        recent_strategy = (recent.get('strategy') or '').lower()
                                        # If recent trade is SmartTrader or doesn't have _sync, assume this is too
                                        if 'smarttrader' in recent_strategy or not recent_strategy or '_sync' not in recent_strategy:
                                            might_be_smarttrader = True
                                            break
                                
                                # Method 2: If no match found, check if symbol is in our typical trading list
                                # Default to SmartTrader for symbols we typically trade (crypto, forex, stocks)
                                if not might_be_smarttrader:
                                    # Check if symbol matches our trading patterns
                                    base_symbol = symbol.split('/')[0].upper() if '/' in symbol else symbol.upper()
                                    
                                    # Crypto symbols we trade
                                    crypto_bases = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'LINK', 'MATIC', 'AVAX', 'UNI', 'ATOM', 'BNB', 'XRP', 'LTC']
                                    # Forex symbols we trade
                                    forex_bases = ['EUR', 'GBP', 'USD', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']
                                    # Commodities
                                    commodity_bases = ['XAU', 'XAG']
                                    
                                    if (base_symbol in crypto_bases or 
                                        base_symbol in forex_bases or 
                                        base_symbol in commodity_bases or
                                        len(symbol) <= 5):  # Likely a stock symbol
                                        # Default to SmartTrader for symbols we typically trade
                                        # Only mark as external if we're very confident it's not ours
                                        might_be_smarttrader = True
                                        
                            except Exception as e:
                                # On error, default to SmartTrader to avoid false negatives
                                might_be_smarttrader = True
                            
                            strategy_to_use = 'smarttrader' if might_be_smarttrader else f'{broker_name}_sync'
                            message = "SmartTrader trade" if might_be_smarttrader else "external trade (not created by SmartTrader)"
                            
                            trade_data = {
                                'trade_id': trade_id,
                                'symbol': symbol,
                                'side': side,
                                'quantity': quantity,
                                'entry_price': entry_price,
                                'entry_time': datetime.now(),
                                'status': 'open',
                                'pnl': unrealized_pnl,
                                'strategy': strategy_to_use
                            }
                            stored_id = self.storage.store_trade(trade_data)
                            if stored_id:
                                self.logger.log_decision({
                                    'action': 'position_synced',
                                    'broker': broker_name,
                                    'symbol': symbol,
                                    'side': side.value if hasattr(side, 'value') else str(side),
                                    'quantity': quantity,
                                    'entry_price': entry_price,
                                    'is_smarttrader': might_be_smarttrader
                                }, event='position_synced')
                                if might_be_smarttrader:
                                    print(f"✅ Synced {message} in {broker_name}: {symbol} {side} {quantity} @ {entry_price}")
                                else:
                                    print(f"⚠️  Found {message} in {broker_name}: {symbol} {side} {quantity} @ {entry_price}")
                except Exception as e:
                    self.logger.log_error(e, {'broker': broker_name, 'action': 'sync_positions'})
        except Exception as e:
            self.logger.log_error(e, {'action': 'sync_positions_from_brokers'})
    
    def _cleanup_phantom_trades(self):
        """
        Clean up phantom trades - trades in database that don't exist in brokers.
        This ensures database and broker APIs are in sync.
        """
        try:
            # Get all open trades from database
            open_trades = self.storage.get_open_trades()
            if not open_trades:
                print("  No open trades in database to check")
                return
            
            # Get all open positions from all brokers
            broker_positions_by_symbol = {}  # {symbol: [positions]}
            for broker_name, broker in self.brokers.items():
                try:
                    if hasattr(broker, 'get_open_positions'):
                        broker_positions = broker.get_open_positions()
                        for pos in broker_positions:
                            raw_symbol = pos.get('symbol', '')
                            symbol = raw_symbol.replace('_', '/')
                            
                            # Handle Alpaca crypto format: ETHUSD -> ETH/USD
                            if '/' not in symbol and len(symbol) >= 6:
                                for base_len in [3, 4]:
                                    if len(symbol) > base_len:
                                        base = symbol[:base_len]
                                        quote = symbol[base_len:]
                                        if quote in ['USD', 'EUR', 'GBP', 'JPY']:
                                            symbol = f"{base}/{quote}"
                                            break
                            
                            if symbol not in broker_positions_by_symbol:
                                broker_positions_by_symbol[symbol] = []
                            broker_positions_by_symbol[symbol].append({
                                'broker': broker_name,
                                'side': pos.get('side', 'buy'),
                                'quantity': pos.get('quantity', 0),
                                'entry_price': pos.get('entry_price', 0),
                                'position': pos
                            })
                except Exception as e:
                    print(f"  ⚠️  Error getting positions from {broker_name}: {e}")
                    continue
            
            # Check each database trade against broker positions
            phantom_trades = []
            for trade in open_trades:
                trade_id = trade.get('trade_id')
                symbol = trade.get('symbol', '').replace('_', '/')
                side = trade.get('side', 'buy').lower()
                entry_price = trade.get('entry_price', 0)
                quantity = trade.get('quantity', 0)
                
                # Normalize symbol for comparison
                if '/' not in symbol and len(symbol) >= 6:
                    for base_len in [3, 4]:
                        if len(symbol) > base_len:
                            base = symbol[:base_len]
                            quote = symbol[base_len:]
                            if quote in ['USD', 'EUR', 'GBP', 'JPY']:
                                symbol = f"{base}/{quote}"
                                break
                
                # Check if this trade exists in any broker
                found_in_broker = False
                matching_broker = None
                
                # Check all broker positions for this symbol
                for check_symbol, broker_positions in broker_positions_by_symbol.items():
                    # Normalize check_symbol
                    check_symbol_normalized = check_symbol.replace('_', '/')
                    if '/' not in check_symbol_normalized and len(check_symbol_normalized) >= 6:
                        for base_len in [3, 4]:
                            if len(check_symbol_normalized) > base_len:
                                base = check_symbol_normalized[:base_len]
                                quote = check_symbol_normalized[base_len:]
                                if quote in ['USD', 'EUR', 'GBP', 'JPY']:
                                    check_symbol_normalized = f"{base}/{quote}"
                                    break
                    
                    # Check if symbols match
                    if check_symbol_normalized == symbol or check_symbol == symbol:
                        for broker_pos in broker_positions:
                            broker_side = broker_pos['side'].lower() if isinstance(broker_pos['side'], str) else str(broker_pos['side']).lower()
                            broker_price = broker_pos['entry_price']
                            broker_qty = broker_pos['quantity']
                            
                            # Match by side and similar price (within 2% tolerance)
                            side_match = broker_side == side
                            price_match = abs(broker_price - entry_price) / max(entry_price, 0.01) < 0.02 if entry_price > 0 else False
                            
                            if side_match and price_match:
                                found_in_broker = True
                                matching_broker = broker_pos['broker']
                                break
                        
                        if found_in_broker:
                            break
                
                if not found_in_broker:
                    phantom_trades.append(trade)
            
            # Handle phantom trades
            if phantom_trades:
                print(f"  ⚠️  Found {len(phantom_trades)} phantom trade(s) (not found in any broker):")
                for trade in phantom_trades:
                    symbol = trade.get('symbol')
                    side = trade.get('side')
                    trade_id = trade.get('trade_id')
                    entry_time = trade.get('entry_time')
                    
                    # Check how old the trade is
                    age_days = 0
                    if entry_time:
                        if isinstance(entry_time, str):
                            try:
                                from dateutil import parser
                                entry_time = parser.parse(entry_time)
                            except:
                                pass
                        if isinstance(entry_time, datetime):
                            age_days = (datetime.now() - entry_time).total_seconds() / 86400
                    
                    print(f"    - {symbol} {side} (ID: {trade_id}, Age: {age_days:.1f} days)")
                    
                    # If trade is more than 1 day old, mark as closed (likely was closed externally)
                    # If trade is recent (< 1 day), keep it open but log a warning (might be a timing issue)
                    if age_days > 1:
                        # Mark as closed - trade was likely closed externally
                        try:
                            self.storage.update_trade(trade_id, {
                                'status': 'closed',
                                'exit_time': datetime.now(),
                                'exit_price': trade.get('entry_price', 0),  # Use entry price as exit (no better data)
                                'pnl': trade.get('pnl', 0)  # Keep existing P&L
                            })
                            print(f"      ✅ Marked as closed (age: {age_days:.1f} days)")
                        except Exception as e:
                            print(f"      ❌ Error marking trade as closed: {e}")
                    else:
                        print(f"      ⚠️  Keeping open (recent trade, might be timing issue)")
            else:
                print("  ✅ All database trades verified in brokers - no phantom trades found")
                
        except Exception as e:
            print(f"  ❌ Error during phantom trade cleanup: {e}")
            self.logger.log_error(e, {'component': 'phantom_trade_cleanup'})
    
    def _update_dashboard(self):
        """Update dashboard data"""
        try:
            # Sync positions from brokers first
            self._sync_positions_from_brokers()
            
            # Get account balance and positions directly from brokers (most accurate)
            balance = 0.0
            broker_for_prices = None
            broker_balances = {}
            all_broker_positions = []  # Collect positions from all brokers
            seen_positions = {}  # Track seen positions to prevent duplicates: (symbol, side, entry_price) -> position
            
            for broker_name, broker in self.brokers.items():
                try:
                    balance_info = broker.get_account_balance()
                    broker_balance = balance_info.get('total', 0.0)
                    if broker_balance > 0:
                        balance += broker_balance
                        broker_balances[broker_name] = broker_balance
                        if broker_for_prices is None:
                            broker_for_prices = broker
                    
                    # Get positions directly from broker (most accurate data)
                    if hasattr(broker, 'get_open_positions'):
                        broker_positions = broker.get_open_positions()
                        for pos in broker_positions:
                            # Normalize symbol format
                            raw_symbol = pos.get('symbol', '')
                            symbol = raw_symbol.replace('_', '/')
                            if '/' not in symbol and len(symbol) >= 6:
                                for base_len in [3, 4]:
                                    if len(symbol) > base_len:
                                        base = symbol[:base_len]
                                        quote = symbol[base_len:]
                                        if quote in ['USD', 'EUR', 'GBP', 'JPY']:
                                            symbol = f"{base}/{quote}"
                                            break
                            
                            side = pos.get('side', 'buy').lower()
                            entry_price = pos.get('entry_price', 0)
                            
                            # Create unique key for position deduplication
                            # Use symbol, side, and entry_price (within 0.1% tolerance) as unique identifier
                            position_key = (symbol, side, round(entry_price, 2))  # Round to 2 decimals for matching
                            
                            # Check if we've already seen this position
                            if position_key in seen_positions:
                                # Position already exists, skip duplicate
                                continue
                            
                            # Get trade_id from database if it exists (for tracking)
                            db_trades = self.storage.get_open_trades(symbol=symbol)
                            trade_id = None
                            for db_trade in db_trades:
                                db_symbol = db_trade.get('symbol', '').replace('_', '/')
                                if '/' not in db_symbol and len(db_symbol) >= 6:
                                    for base_len in [3, 4]:
                                        if len(db_symbol) > base_len:
                                            base = db_symbol[:base_len]
                                            quote = db_symbol[base_len:]
                                            if quote in ['USD', 'EUR', 'GBP', 'JPY']:
                                                db_symbol = f"{base}/{quote}"
                                                break
                                
                                if (db_symbol == symbol and 
                                    db_trade.get('side', '').lower() == side and
                                    abs(db_trade.get('entry_price', 0) - entry_price) / max(entry_price, 0.01) < 0.05):
                                    trade_id = db_trade.get('trade_id')
                                    break
                            
                            pos['symbol'] = symbol  # Use normalized symbol
                            pos['trade_id'] = trade_id  # Add trade_id for dashboard
                            all_broker_positions.append(pos)
                            seen_positions[position_key] = pos  # Mark as seen
                except Exception as e:
                    self.logger.log_error(e, {'broker': broker_name, 'action': 'get_balance_or_positions'})
            
            # Fallback to default if no brokers returned balance
            if balance == 0.0:
                balance = 10000.0

            # Initialize starting equity once from the first successful balance
            if self.initial_equity is None:
                self.initial_equity = balance

            # Deduplicate positions before processing (same symbol, side, entry_price = same position)
            unique_positions = {}
            for position in all_broker_positions:
                symbol = position.get('symbol', '')
                side = position.get('side', 'buy').lower()
                entry_price = position.get('entry_price', 0)
                
                # Create unique key: (symbol, side, rounded_entry_price)
                position_key = (symbol, side, round(entry_price, 2))
                
                # If we haven't seen this position, or if this one has more complete data, use it
                if position_key not in unique_positions:
                    unique_positions[position_key] = position
                else:
                    # Merge data if this position has more complete information
                    existing = unique_positions[position_key]
                    # Prefer position with trade_id, or better price data
                    if not existing.get('trade_id') and position.get('trade_id'):
                        unique_positions[position_key] = position
                    elif position.get('current_price', 0) > 0 and existing.get('current_price', 0) == 0:
                        unique_positions[position_key] = position
            
            # Use broker positions directly (they have accurate entry_price, current_price, and unrealized_pnl)
            total_unrealized_pnl = 0.0
            positions_with_pnl = []
            
            for position in unique_positions.values():
                # Broker provides: entry_price (avg_entry_price), current_price (calculated), unrealized_pnl
                entry_price = position.get('entry_price', 0)
                current_price = position.get('current_price', entry_price)  # Use broker's current_price
                unrealized_pnl = position.get('unrealized_pnl', 0)  # Use broker's P&L (most accurate)
                
                # If current_price is same as entry_price, try to fetch fresh price from appropriate broker
                if current_price == entry_price or current_price == 0:
                    symbol = position.get('symbol', '')
                    # Determine which broker to use based on symbol type
                    price_broker = None
                    if '/' in symbol:
                        base = symbol.split('/')[0].upper()
                        # Use Alpaca for crypto
                        if base in ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'LINK', 'MATIC', 'AVAX', 'UNI', 'ATOM']:
                            for broker_name, broker in self.brokers.items():
                                if 'alpaca' in broker_name.lower():
                                    price_broker = broker
                                    break
                        else:
                            # Forex - try OANDA first
                            for broker_name, broker in self.brokers.items():
                                if 'oanda' in broker_name.lower():
                                    price_broker = broker
                                    break
                    
                    # Fallback to broker_for_prices
                    if not price_broker:
                        price_broker = broker_for_prices
                    
                    if price_broker:
                        try:
                            fetched_price = price_broker.get_current_price(symbol)
                            if fetched_price > 0:
                                current_price = fetched_price
                        except:
                            pass  # Use broker's provided price as fallback
                
                total_unrealized_pnl += unrealized_pnl
                
                # Prepare position data for dashboard
                position_with_pnl = {
                    'symbol': position.get('symbol'),
                    'side': position.get('side', 'buy'),
                    'quantity': position.get('quantity', 0),
                    'entry_price': entry_price,  # Broker's avg_entry_price (accurate)
                    'current_price': current_price,  # Broker's calculated current price
                    'pnl': unrealized_pnl,  # Broker's unrealized_pl (accurate)
                    'trade_id': position.get('trade_id')
                }
                positions_with_pnl.append(position_with_pnl)

            # Calculate equity (balance + unrealized P&L)
            equity = balance + total_unrealized_pnl
            # Also update drawdown manager with current equity
            if hasattr(self, 'drawdown_manager'):
                self.drawdown_manager.update_equity(equity)

            # Calculate total return based on real starting equity
            if self.initial_equity and self.initial_equity > 0:
                total_return = ((equity - self.initial_equity) / self.initial_equity) * 100
            else:
                total_return = 0.0

            # Get all recent trades for dashboard (both open and closed)
            all_recent_trades = []
            closed_trades_for_display = []  # For displaying recent trades in table
            all_closed_trades = []  # For calculating metrics (all closed trades)
            try:
                # Get all trades (not just closed ones) - last 50 trades for display
                all_trades = self.storage.get_all_trades(limit=50)
                
                for trade in all_trades:
                    trade_status = trade.get('status', 'open')
                    trade_time = trade.get('exit_time') or trade.get('entry_time')
                    
                    trade_info = {
                        'time': trade_time,
                            'symbol': trade.get('symbol', 'N/A'),
                            'side': trade.get('side', 'N/A'),
                            'quantity': trade.get('quantity', 0),
                            'price': trade.get('exit_price') or trade.get('entry_price', 0),
                        'pnl': trade.get('pnl', 0),
                        'status': trade_status
                    }
                    
                    all_recent_trades.append(trade_info)
                    
                    # Also collect closed trades separately for realized P&L calculation
                    if trade_status == 'closed':
                        closed_trades_for_display.append(trade_info)
                
                # Sort by time, most recent first
                all_recent_trades.sort(key=lambda x: x.get('time', '') or '', reverse=True)
                all_recent_trades = all_recent_trades[:30]  # Show last 30 trades (open + closed)
                
                closed_trades_for_display.sort(key=lambda x: x.get('time', '') or '', reverse=True)
                closed_trades_for_display = closed_trades_for_display[:20]  # Show last 20 closed trades for display
                
                # Get ALL closed trades for accurate metrics calculation (same as reports)
                all_trades_for_metrics = self.storage.get_all_trades()  # No limit - get all trades
                for trade in all_trades_for_metrics:
                    if trade.get('status') == 'closed':
                        all_closed_trades.append({
                            'pnl': trade.get('pnl', 0),
                            'status': 'closed'
                        })
            except Exception as e:
                self.logger.log_error(e, {'component': 'dashboard_trades'})
            
            # Calculate realized P&L from ALL closed trades (for consistency with reports)
            total_realized_pnl = sum(t.get('pnl', 0) for t in all_closed_trades)
            
            # Calculate win rate and loss rate from ALL closed trades (same as reports)
            # Note: Break-even trades (P&L = 0) are counted as losses so win_rate + loss_rate = 100%
            total_closed_trades = len(all_closed_trades)
            if total_closed_trades > 0:
                winning_trades = [t for t in all_closed_trades if t.get('pnl', 0) > 0]
                losing_trades = [t for t in all_closed_trades if t.get('pnl', 0) <= 0]  # Include break-even (P&L = 0) in losses
                win_rate = (len(winning_trades) / total_closed_trades) * 100
                loss_rate = (len(losing_trades) / total_closed_trades) * 100
            else:
                win_rate = 0.0
                loss_rate = 0.0

            # Risk/performance alerts
            alerts = []
            drawdown_status = self.drawdown_manager.get_status() if hasattr(self, 'drawdown_manager') else {}
            current_dd = drawdown_status.get('drawdown_percent', 0.0)
            if current_dd and current_dd >= 3.0:
                alert_msg = f"Drawdown alert: {current_dd:.2f}%"
                alerts.append(alert_msg)
                self.logger.log_risk_event({'event': 'drawdown_alert', 'drawdown_percent': current_dd})

            performance_metrics = {}
            if self.performance_tracker:
                performance_metrics = self.performance_tracker.calculate_metrics()
                sharpe = performance_metrics.get('sharpe_ratio', 0.0)
                weekly_return = performance_metrics.get('weekly_return', 0.0)
                if sharpe < 0.5:
                    alert_msg = f"Sharpe alert: {sharpe:.2f} < 0.50"
                    alerts.append(alert_msg)
                    self.logger.log_risk_event({'event': 'sharpe_alert', 'sharpe_ratio': sharpe})
                # Weekly return shortfall vs 3% target
                if weekly_return < 0.5:  # lightweight threshold to flag underperformance early
                    alert_msg = f"Weekly return alert: {weekly_return:.2f}%"
                    alerts.append(alert_msg)
                    self.logger.log_risk_event({'event': 'weekly_return_alert', 'weekly_return': weekly_return})
            else:
                performance_metrics = {}
            
            # Update dashboard
            dashboard_data = {
                'equity': equity,
                'balance': balance,
                'positions': positions_with_pnl,
                'trades': all_recent_trades,  # Show all recent trades (open + closed), not just closed
                'performance': {
                    'total_return': total_return,
                    'sharpe_ratio': performance_metrics.get('sharpe_ratio', 0.0) if performance_metrics else 0.0,
                    'realized_pnl': total_realized_pnl,  # Total realized P&L from closed trades
                    'unrealized_pnl': total_unrealized_pnl,  # Total unrealized P&L from open positions
                    'initial_equity': self.initial_equity or balance,  # Show initial equity for reference
                    'win_rate': win_rate,  # Percentage of winning trades
                    'loss_rate': loss_rate,  # Percentage of losing trades
                    'total_trades': total_closed_trades  # Total number of closed trades
                },
                'risk_metrics': {
                    **(self.drawdown_manager.get_status() if hasattr(self, 'drawdown_manager') else {}),
                    'alerts': alerts
                }
            }
            
            self.dashboard_app.update_dashboard_data(dashboard_data)
        
        except Exception as e:
            self.logger.log_error(e, {'component': 'dashboard_update'})


def main():
    """Main entry point"""
    try:
        print("=" * 60)
        print("Starting SmartTrader Trading Agent...")
        print("=" * 60)
        agent = TradingAgent()
        print("Trading agent initialized successfully")
        agent.run()
    except KeyboardInterrupt:
        print("\nShutting down trading agent...")
        sys.exit(0)
    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"FATAL ERROR: Failed to start trading agent")
        print(f"{'=' * 60}")
        print(f"Error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        print(f"{'=' * 60}")
        sys.exit(1)


if __name__ == "__main__":
    main()

