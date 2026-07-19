"""Orchestrator agent with DRL-based decision making"""

from typing import Dict, Optional, List
from datetime import datetime
import pandas as pd
import numpy as np
from .base_agent import BaseAgent
from .analytical_agents import TechnicalAnalystAgent, SentimentAgent, FundamentalAgent
from .quantitative_agent import QuantitativeAgent
from .pattern_forecaster_agent import PatternForecasterAgent
from .regime_switching_agent import RegimeSwitchingAgent
from .risk_manager_agent import RiskManagerAgent
from ..strategies.base_strategy import Signal
from ..ai.drl.drl_agent import DRLAgent
from ..ai.models.sentiment_analyzer import SentimentAnalyzer


class OrchestratorAgent(BaseAgent):
    """Central orchestrator that coordinates all agents"""
    
    def __init__(self, config: Optional[Dict] = None,
                 technical_agent: Optional[TechnicalAnalystAgent] = None,
                 sentiment_agent: Optional[SentimentAgent] = None,
                 fundamental_agent: Optional[FundamentalAgent] = None,
                 quantitative_agent: Optional[QuantitativeAgent] = None,
                 pattern_forecaster_agent: Optional[PatternForecasterAgent] = None,
                 regime_agent: Optional[RegimeSwitchingAgent] = None,
                 risk_agent: Optional[RiskManagerAgent] = None,
                 drl_agent: Optional[DRLAgent] = None,
                 performance_tracker: Optional[object] = None,
                 sentiment_analyzer: Optional[SentimentAnalyzer] = None):
        """
        Initialize orchestrator agent
        
        Args:
            config: Orchestrator configuration
            technical_agent: Technical analyst agent
            sentiment_agent: Sentiment agent
            fundamental_agent: Fundamental agent
            quantitative_agent: Quantitative analysis agent
            pattern_forecaster_agent: Pattern forecaster agent (historical pattern matching)
            regime_agent: Regime-switching agent
            risk_agent: Risk manager agent
            drl_agent: DRL agent for decision making (optional, falls back to weighted voting)
            performance_tracker: Performance tracker for dynamic position sizing (optional)
            sentiment_analyzer: SentimentAnalyzer instance for LLM-based conflict resolution (optional)
        """
        super().__init__("OrchestratorAgent", config)
        self.technical_agent = technical_agent
        self.sentiment_agent = sentiment_agent
        self.fundamental_agent = fundamental_agent
        self.quantitative_agent = quantitative_agent
        self.pattern_forecaster_agent = pattern_forecaster_agent
        self.regime_agent = regime_agent
        self.risk_agent = risk_agent
        self.drl_agent = drl_agent
        self.performance_tracker = performance_tracker
        self.sentiment_analyzer = sentiment_analyzer  # For LLM conflict resolution
        
        # Decision tree thresholds (used as fallback)
        # Lowered from 0.70 to 0.50 to allow more trades while maintaining quality
        self.min_confidence = self.config.get('min_confidence', 0.35)  # Lowered fallback default
        self.consensus_threshold = self.config.get('consensus_threshold', 0.35)  # Lowered fallback default
        self.veto_enabled = self.config.get('veto_enabled', True)
        # Changed default to False to allow single strong signals (more trades)
        self.require_agent_agreement = self.config.get('require_agent_agreement', False)  # Allow single strong signals
        self.min_agents_agreeing = self.config.get('min_agents_agreeing', 2)  # Now actually read from config (was hardcoded to 2 below)
        self.disable_short_trades = self.config.get('disable_short_trades', False)  # Enabled by default; set true in config to re-disable
        
        # DRL mode
        self.use_drl = self.config.get('use_drl', True) and drl_agent is not None and drl_agent.is_trained
        
        # LLM conflict resolution
        self.use_llm_conflict_resolution = self.config.get('use_llm_conflict_resolution', True) and sentiment_analyzer is not None
    
    def analyze(self, data, symbol: str, asset_class: str,
                news_items: Optional[List[Dict]] = None,
                account_balance: float = 10000.0) -> Dict:
        """
        Analyze and orchestrate decision-making (implements BaseAgent interface)
        
        This is an alias for orchestrate() to satisfy the BaseAgent abstract method
        
        Args:
            data: Market data DataFrame
            symbol: Trading symbol
            asset_class: Asset class (crypto, forex, stocks)
            news_items: Optional news items
            account_balance: Current account balance
        
        Returns:
            Final trading decision dictionary
        """
        return self.orchestrate(data, symbol, asset_class, news_items, account_balance)
    
    def orchestrate(self, data, symbol: str, asset_class: str, 
                   news_items: Optional[List[Dict]] = None,
                   account_balance: float = 10000.0) -> Dict:
        """
        Orchestrate decision-making using hierarchical decision tree
        
        Args:
            data: Market data DataFrame
            symbol: Trading symbol
            asset_class: Asset class (crypto, forex, stocks)
            news_items: Optional news items
            account_balance: Current account balance
        
        Returns:
            Final trading decision dictionary
        """
        print(f"\n{'='*60}")
        print(f"ORCHESTRATOR: Processing {symbol} ({asset_class})")
        print(f"{'='*60}")
        
        if not self.enabled:
            print("ORCHESTRATOR: Disabled, returning HOLD")
            return {
                'signal': Signal.HOLD,
                'confidence': 0.0,
                'reason': 'Orchestrator disabled'
            }
        
        # Rule 1: Risk Manager Veto Check
        print("\n[Step 1] Risk Manager Veto Check...")
        if self.veto_enabled and self.risk_agent:
            veto_result = self.risk_agent.check_veto(symbol, data, account_balance)
            if veto_result.get('veto', False):
                print(f"  ❌ VETOED: {veto_result.get('reason', 'Risk limit exceeded')}")
                return {
                    'signal': Signal.HOLD,
                    'confidence': 0.0,
                    'reason': f"Risk Manager Veto: {veto_result.get('reason', 'Risk limit exceeded')}",
                    'veto': True
                }
        print("  ✓ No veto, proceeding...")
        
        # Rule 2: Regime Detection and Weight Assignment
        print("\n[Step 2] Regime Detection...")
        regime_result = None
        agent_weights = {}
        if self.regime_agent and self.regime_agent.is_enabled():
            regime_result = self.regime_agent.analyze(data)
            agent_weights = regime_result.get('agent_weights', {})
            current_regime = regime_result.get('regime', 'Neutral')
            print(f"  Regime: {current_regime}, confidence={regime_result.get('confidence', 0):.2f}")
            print(f"  Agent weights: {agent_weights}")
        else:
            current_regime = 'Neutral'
            agent_weights = {}
            print("  Regime agent disabled, using default weights")
        
        # Rule 3: Get Signals from Analytical Agents (Parallel)
        print("\n[Step 3] Collecting Agent Signals...")
        agent_signals = {}
        
        # Technical Analyst
        if self.technical_agent and self.technical_agent.is_enabled():
            print("  Querying Technical Analyst...")
            tech_result = self.technical_agent.analyze(data, symbol)
            agent_signals['technical'] = tech_result
            print(f"  Technical: {tech_result['signal'].name}, confidence={tech_result.get('confidence', 0):.3f}")
        else:
            print("  Technical Analyst: DISABLED")
        
        # Sentiment Agent
        if self.sentiment_agent and self.sentiment_agent.is_enabled():
            print("  Querying Sentiment Agent...")
            sent_result = self.sentiment_agent.analyze(symbol, news_items)
            agent_signals['sentiment'] = sent_result
            print(f"  Sentiment: {sent_result['signal'].name}, confidence={sent_result.get('confidence', 0):.3f}")
        else:
            print("  Sentiment Agent: DISABLED")
        
        # Fundamental Agent
        if self.fundamental_agent and self.fundamental_agent.is_enabled():
            print("  Querying Fundamental Agent...")
            fund_result = self.fundamental_agent.analyze(symbol, asset_class, news_items=news_items)
            agent_signals['fundamental'] = fund_result
            print(f"  Fundamental: {fund_result['signal'].name}, confidence={fund_result.get('confidence', 0):.3f}")
        else:
            print("  Fundamental Agent: DISABLED")
        
        # Quantitative Agent
        if self.quantitative_agent and self.quantitative_agent.is_enabled():
            print("  Querying Quantitative Agent...")
            quant_result = self.quantitative_agent.analyze(data, symbol)
            agent_signals['quantitative'] = quant_result
            print(f"  Quantitative: {quant_result['signal'].name}, confidence={quant_result.get('confidence', 0):.3f}")
        else:
            print("  Quantitative Agent: DISABLED")
        
        # Pattern Forecaster Agent (Historical Pattern Matching)
        if self.pattern_forecaster_agent and self.pattern_forecaster_agent.is_enabled():
            print("  Querying Pattern Forecaster Agent...")
            try:
                pattern_result = self.pattern_forecaster_agent.analyze(data, symbol)
                agent_signals['pattern_forecaster'] = pattern_result
                print(f"  Pattern Forecaster: {pattern_result['signal'].name}, confidence={pattern_result.get('confidence', 0):.3f}")
                if pattern_result.get('matches_found'):
                    print(f"    Matches Found: {pattern_result['matches_found']}, Forecast: {pattern_result.get('forecast_change_pct', 0):.2f}%")
            except Exception as e:
                print(f"  Pattern Forecaster Error: {e}")
                # Don't block trading on pattern forecaster errors - it's optional analysis
                # Continue with other agents' signals
        else:
            print("  Pattern Forecaster Agent: DISABLED")
        
        # DIAGNOSTIC: Log all agent signals before decision making
        print("\n[DIAGNOSTIC] Agent Signals Summary:")
        if not agent_signals:
            print("  ⚠️  WARNING: No agent signals collected!")
        else:
            for agent_name, signal_data in agent_signals.items():
                sig = signal_data.get('signal', Signal.HOLD)
                conf = signal_data.get('confidence', 0.0)
                print(f"  - {agent_name}: {sig.name} (confidence: {conf:.3f})")
        
        # Rule 4: DRL-Based Decision or Weighted Voting Fallback
        print("\n[Step 4] Making Final Decision...")
        if self.use_drl:
            print("  Using DRL-based decision making...")
            final_signal = self._drl_decision(data, agent_signals, regime_result)
        else:
            print("  Using weighted voting...")
            final_signal = self._weighted_voting(agent_signals, agent_weights, regime_result=regime_result, market_data=data)
        
        print(f"  Final Signal: {final_signal['signal'].name}")
        print(f"  Confidence: {final_signal.get('confidence', 0):.3f}")
        print(f"  Weighted Score: {final_signal.get('weighted_score', 0):.3f}")
        print(f"  Reason: {final_signal.get('reason', 'N/A')}")
        
        # Rule 5: Consensus Check
        print("\n[Step 5] Consensus Check...")
        consensus = self._check_consensus(agent_signals, final_signal)
        print(f"  Has consensus: {consensus.get('has_consensus', False)}")
        print(f"  Consensus strength: {consensus.get('consensus_strength', 0):.2f}")
        print(f"  Agreeing agents: {consensus.get('agreeing_agents', [])}")
        print(f"  Disagreeing agents: {consensus.get('disagreeing_agents', [])}")
        
        # Check for conflicting signals
        if len(agent_signals) >= 2:
            signals_list = [(name, sig.get('signal'), sig.get('confidence', 0)) 
                          for name, sig in agent_signals.items()]
            buy_signals = [s for s in signals_list if s[1] == Signal.BUY]
            sell_signals = [s for s in signals_list if s[1] == Signal.SELL]
            
            if buy_signals and sell_signals:
                print(f"\n  ⚠️  CONFLICT DETECTED:")
                print(f"     BUY signals: {[(n, f'{c:.2f}') for n, _, c in buy_signals]}")
                print(f"     SELL signals: {[(n, f'{c:.2f}') for n, _, c in sell_signals]}")
                print(f"     Weighted voting will resolve based on confidence and weights")
        
        # Rule 6: Position Sizing Recommendation
        position_size_info = None
        stop_loss = None
        take_profit = None
        
        # CRITICAL FIX: Always calculate stop_loss and take_profit for non-HOLD signals
        # Even if risk_agent is None, we must set these values for risk management
        if final_signal['signal'] != Signal.HOLD:
            try:
                # Get entry price and stop loss from technical signal
                entry_price = data['close'].iloc[-1]
                stop_loss = agent_signals.get('technical', {}).get('stop_loss')
                take_profit = agent_signals.get('technical', {}).get('take_profit')
                
                # CRITICAL FIX: Always validate and recalculate SL/TP if they don't match signal direction
                # Even if technical agent provided values, we must ensure they're correct for final signal
                is_buy = final_signal['signal'] == Signal.BUY
                
                # Track where inverted SL/TP came from for debugging
                sl_source = "technical_agent" if stop_loss else None
                tp_source = "technical_agent" if take_profit else None
                
                if stop_loss:
                    # Validate stop_loss is correct for the signal direction
                    if is_buy and stop_loss > entry_price:
                        # Stop loss is above entry for a BUY - this is wrong!
                        print(f"  ❌ CRITICAL: Stop loss ${stop_loss:.2f} is above entry ${entry_price:.2f} for BUY signal (source: {sl_source})")
                        print(f"     Recalculating stop loss based on signal direction...")
                        stop_loss = None  # Force recalculation
                    elif not is_buy and stop_loss < entry_price:
                        # Stop loss is below entry for a SELL - this is wrong!
                        print(f"  ❌ CRITICAL: Stop loss ${stop_loss:.2f} is below entry ${entry_price:.2f} for SELL signal (source: {sl_source})")
    