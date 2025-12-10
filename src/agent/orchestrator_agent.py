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
from ..ai.drl.drl_agent import DRLAgent


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
                 performance_tracker: Optional[object] = None):
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
        
        # Decision tree thresholds (used as fallback)
        self.min_confidence = self.config.get('min_confidence', 0.5)
        self.consensus_threshold = self.config.get('consensus_threshold', 0.6)
        self.veto_enabled = self.config.get('veto_enabled', True)
        
        # DRL mode
        self.use_drl = self.config.get('use_drl', True) and drl_agent is not None and drl_agent.is_trained
    
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
        else:
            print("  Pattern Forecaster Agent: DISABLED")
        
        # Rule 4: DRL-Based Decision or Weighted Voting Fallback
        print("\n[Step 4] Making Final Decision...")
        if self.use_drl:
            print("  Using DRL-based decision making...")
            final_signal = self._drl_decision(data, agent_signals, regime_result)
        else:
            print("  Using weighted voting...")
            final_signal = self._weighted_voting(agent_signals, agent_weights)
        
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
        if self.risk_agent and final_signal['signal'] != Signal.HOLD:
            # Get entry price and stop loss from technical signal
            entry_price = data['close'].iloc[-1]
            stop_loss = agent_signals.get('technical', {}).get('stop_loss')
            if not stop_loss:
                # Default stop loss - ensure it's always set
                from ..indicators.technical import TechnicalIndicators
                indicators = TechnicalIndicators()
                atr = indicators.atr(data)
                atr_value = atr.iloc[-1] if not atr.empty else entry_price * 0.02
                
                # Ensure ATR value is reasonable (at least 0.5% of price)
                min_stop_distance = entry_price * 0.005  # 0.5% minimum
                atr_value = max(atr_value, min_stop_distance)
                
                stop_distance = atr_value * 2.0  # 2x ATR
                
                if final_signal['signal'] == Signal.BUY:
                    stop_loss = entry_price - stop_distance
                else:  # SELL
                    stop_loss = entry_price + stop_distance
                
                # Ensure stop loss is valid (not negative for buy, reasonable for sell)
                if final_signal['signal'] == Signal.BUY:
                    stop_loss = max(stop_loss, entry_price * 0.95)  # Max 5% loss
                else:
                    stop_loss = min(stop_loss, entry_price * 1.05)  # Max 5% loss for shorts
                
                print(f"  ✅ Calculated default stop loss: ${stop_loss:.2f} (ATR-based, {abs((entry_price - stop_loss) / entry_price * 100):.2f}% risk)")
            
            # Calculate take profit if not provided by technical agent
            take_profit = agent_signals.get('technical', {}).get('take_profit')
            if not take_profit and stop_loss is not None:
                # Calculate take profit based on risk/reward ratio
                # Default 2.0 (1:2) - minimum for most profitable strategies
                # 1:2 is the practical minimum, 1:3 is often the sweet spot
                # Higher ratios (1:5+) require very precise entries and lower win rates
                side_str = 'buy' if final_signal['signal'] == Signal.BUY else 'sell'
                risk_reward_ratio = self.config.get('risk_reward_ratio', 2.0)  # Default 2.0:1 (1:2)
                
                # Calculate risk (distance from entry to stop loss)
                risk = abs(entry_price - stop_loss)
                
                # Only calculate take profit if we have meaningful risk
                if risk > 0:
                    # Calculate reward based on risk/reward ratio
                    reward = risk * risk_reward_ratio
                    
                    # Set take profit
                    if side_str == 'buy':
                        take_profit = entry_price + reward
                    else:  # sell
                        take_profit = entry_price - reward
                    
                    print(f"  Calculated Take Profit: ${take_profit:.2f} (Risk/Reward: 1:{risk_reward_ratio:.1f})")
                else:
                    print(f"  ⚠️  Warning: Stop loss equals entry price (${entry_price:.2f}), cannot calculate take profit")
            
            position_size_info = self.risk_agent.calculate_position_size(
                account_balance, entry_price, stop_loss, data, current_regime,
                performance_tracker=self.performance_tracker
            )
            
            # Adjust position size based on DRL action if available
            drl_action = final_signal.get('drl_action', None)
            drl_confidence = final_signal.get('drl_confidence', 0.5)
            if drl_action is not None and self.use_drl:
                # Enhanced DRL position adjustment:
                # - Minimum 50% position even with low action
                # - Scales to 100% with high confidence and action
                # Formula: quantity * (0.5 + 0.5 * abs(action)) * (0.7 + 0.3 * confidence)
                action_multiplier = 0.5 + 0.5 * abs(drl_action)
                confidence_multiplier = 0.7 + 0.3 * drl_confidence
                drl_adjustment = action_multiplier * confidence_multiplier
                
                position_size_info['drl_adjustment'] = drl_adjustment
                position_size_info['quantity'] = position_size_info.get('quantity', 0) * drl_adjustment
                position_size_info['value'] = position_size_info.get('quantity', 0) * entry_price
        
        self._update_timestamp()
        
        # Final decision summary
        print("\n[FINAL DECISION SUMMARY]")
        print(f"  Signal: {final_signal['signal'].name}")
        print(f"  Confidence: {final_signal['confidence']:.3f}")
        print(f"  Min Confidence Required: {self.min_confidence:.3f}")
        
        will_trade = (final_signal['signal'] != Signal.HOLD and 
                     final_signal['confidence'] > self.min_confidence)
        print(f"  Will Execute Trade: {'YES ✓' if will_trade else 'NO ✗'}")
        
        if not will_trade:
            if final_signal['signal'] == Signal.HOLD:
                print(f"    Reason: Signal is HOLD")
            else:
                print(f"    Reason: Confidence too low ({final_signal['confidence']:.3f} <= {self.min_confidence:.3f})")
        
        if position_size_info:
            print(f"  Position Size: {position_size_info.get('quantity', 0):.4f} units")
            print(f"  Position Value: ${position_size_info.get('value', 0):.2f}")
        
        print(f"{'='*60}\n")
        
        return {
            'signal': final_signal['signal'],
            'confidence': final_signal['confidence'],
            'weighted_score': final_signal.get('weighted_score', 0.0),
            'regime': current_regime,
            'regime_confidence': regime_result.get('confidence', 0.0) if regime_result else 0.0,
            'agent_signals': agent_signals,
            'consensus': consensus,
            'position_size': position_size_info,
            'reason': final_signal.get('reason', 'Orchestrated decision'),
            'entry_price': data['close'].iloc[-1],
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
    
    def _drl_decision(self, data: pd.DataFrame, agent_signals: Dict, 
                      regime_result: Optional[Dict]) -> Dict:
        """
        Make decision using DRL agent
        
        Args:
            data: Market data DataFrame
            agent_signals: Signals from analytical agents
            regime_result: Regime switching result
        
        Returns:
            Decision dictionary with signal and position size
        """
        if not self.drl_agent or not self.drl_agent.is_trained:
            # Fallback to weighted voting
            return self._weighted_voting(agent_signals, regime_result.get('agent_weights', {}) if regime_result else {})
        
        # Extract market data features
        market_data = self._extract_market_features(data)
        
        # Get performance metrics for expanded state space
        performance_metrics = None
        if self.performance_tracker:
            try:
                metrics = self.performance_tracker.calculate_metrics()
                performance_metrics = {
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
                    'win_rate': metrics.get('win_rate', 0.0),
                    'last_7d_return': metrics.get('weekly_return', 0.0)  # Approximate with weekly
                }
            except:
                pass
        
        # Get position information for expanded state space
        position_info = None
        # This would need to be passed from main or retrieved from storage
        # For now, we'll leave it as None and it will use defaults
        
        # Build state vector with expanded features
        state = self.drl_agent.build_state_vector(
            agent_signals, 
            regime_result or {'regime': 'Neutral', 'confidence': 0.0},
            market_data,
            performance_metrics=performance_metrics,
            position_info=position_info
        )
        
        # Get DRL action
        action, confidence = self.drl_agent.predict(state, deterministic=True)
        
        # Adaptive signal thresholds based on volatility, performance, and market regime
        aggressive = self.config.get('aggressive_mode', False) or self.config.get('use_drl', False)
        
        # Base thresholds (aggressive mode)
        base_buy_threshold = 0.2 if aggressive else 0.3
        base_sell_threshold = -0.2 if aggressive else -0.3
        
        # Volatility-adjusted thresholds (lower in low volatility for more trades)
        volatility = market_data.get('volatility', 0.0)
        if volatility < 0.01:  # Low volatility
            vol_adjustment = 0.15  # Lower thresholds by 15%
        elif volatility > 0.03:  # High volatility
            vol_adjustment = 0.25  # Raise thresholds by 25% (more conservative)
        else:
            vol_adjustment = 0.0  # No adjustment
        
        # Performance-adjusted thresholds (lower when performing well)
        performance_adjustment = 0.0
        if self.performance_tracker:
            try:
                metrics = self.performance_tracker.calculate_metrics()
                sharpe = metrics.get('sharpe_ratio', 0.0)
                weekly_return = metrics.get('weekly_return', 0.0)
                
                # Lower thresholds when performing well
                if sharpe > 1.0 and weekly_return > 0:
                    performance_adjustment = -0.1  # Lower thresholds by 10%
                elif sharpe < 0.5 or weekly_return < -1.0:
                    performance_adjustment = 0.15  # Raise thresholds by 15% (more conservative)
            except:
                pass
        
        # Regime-adjusted thresholds
        regime_adjustment = 0.0
        current_regime = regime_result.get('regime', 'Neutral') if regime_result else 'Neutral'
        if current_regime == 'Bullish Trend':
            regime_adjustment = -0.05  # Slightly lower thresholds (more aggressive)
        elif current_regime == 'Bearish Trend':
            regime_adjustment = 0.1  # Higher thresholds (more conservative)
        elif current_regime == 'High Volatility':
            regime_adjustment = 0.15  # Higher thresholds (more conservative)
        
        # Apply all adjustments
        buy_threshold = base_buy_threshold + vol_adjustment + performance_adjustment + regime_adjustment
        sell_threshold = base_sell_threshold - vol_adjustment - performance_adjustment - regime_adjustment
        
        # Ensure thresholds stay within reasonable bounds
        buy_threshold = max(0.1, min(0.4, buy_threshold))
        sell_threshold = min(-0.1, max(-0.4, sell_threshold))
        
        if action > buy_threshold:
            signal = Signal.BUY
        elif action < sell_threshold:
            signal = Signal.SELL
        else:
            signal = Signal.HOLD
        
        return {
            'signal': signal,
            'confidence': confidence,
            'weighted_score': action,
            'reason': f'DRL decision: action={action:.3f}, confidence={confidence:.3f}',
            'drl_action': action,
            'drl_confidence': confidence
        }
    
    def _extract_market_features(self, data: pd.DataFrame) -> Dict:
        """Extract market features for state vector"""
        if len(data) == 0:
            return {
                'volatility': 0.0,
                'volume_ratio': 1.0,
                'price_momentum': 0.0,
                'rsi': 50.0,
                'adx': 0.0
            }
        
        latest = data.iloc[-1]
        
        # Calculate volatility (rolling std of returns)
        returns = data['close'].pct_change()
        volatility = returns.tail(20).std() if len(returns) >= 20 else returns.std()
        
        # Volume ratio
        volume_ma = data['volume'].rolling(window=20).mean()
        volume_ratio = (latest['volume'] / volume_ma.iloc[-1]) if len(volume_ma) > 0 and volume_ma.iloc[-1] > 0 else 1.0
        
        # Price momentum
        if len(data) >= 10:
            price_momentum = (latest['close'] - data['close'].iloc[-10]) / data['close'].iloc[-10]
        else:
            price_momentum = 0.0
        
        return {
            'volatility': float(volatility) if not pd.isna(volatility) else 0.0,
            'volume_ratio': float(volume_ratio) if not pd.isna(volume_ratio) else 1.0,
            'price_momentum': float(price_momentum) if not pd.isna(price_momentum) else 0.0,
            'rsi': float(latest.get('rsi', 50.0)) if 'rsi' in latest else 50.0,
            'adx': float(latest.get('adx', 0.0)) if 'adx' in latest else 0.0
        }
    
    def _weighted_voting(self, agent_signals: Dict, agent_weights: Dict) -> Dict:
        """
        Perform weighted voting on agent signals
        
        Formula: Final Signal = W_Tech * S_Tech + W_Sent * S_Sent + W_Fund * S_Fund
        """
        if not agent_signals:
            return {
                'signal': Signal.HOLD,
                'confidence': 0.0,
                'weighted_score': 0.0,
                'reason': 'No agent signals available'
            }
        
        # Base weights (if not provided by regime agent)
        base_weights = {
            'technical': 0.30,
            'sentiment': 0.20,
            'fundamental': 0.20,
            'quantitative': 0.15,
            'pattern_forecaster': 0.15
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        signal_details = []
        
        print(f"  Weighted Voting Calculation:")
        for agent_name, signal in agent_signals.items():
            # Get weight (regime-adjusted or base)
            weight = agent_weights.get(agent_name, base_weights.get(agent_name, 0.33))
            
            # Convert signal to numeric (direction: BUY=+1, SELL=-1, HOLD=0)
            signal_val = 0
            if signal['signal'] == Signal.BUY:
                signal_val = 1
            elif signal['signal'] == Signal.SELL:
                signal_val = -1
            
            # Get confidence (strength of signal)
            confidence = signal.get('confidence', 0.5)
            
            # Get score (may be directional, but we use signal_val for direction)
            score = signal.get('score', 0.0)
            
            # FIXED: Use signal_val for direction, confidence for strength
            # Don't multiply by score if it's already directional - use absolute value for magnitude
            # If score is provided and non-zero, use it to scale confidence
            # Otherwise, use confidence directly
            if abs(score) > 0.01:
                # Score is provided - use it to scale, but signal_val determines direction
                # Take absolute value of score for magnitude, apply signal_val for direction
                contribution = signal_val * abs(score) * confidence * weight
            else:
                # No score or score is 0 - use confidence directly with signal direction
                contribution = signal_val * confidence * weight
            
            weighted_score += contribution
            total_weight += weight
            
            print(f"    {agent_name}: {signal['signal'].name}, signal_val={signal_val}, score={score:.3f}, confidence={confidence:.3f}, weight={weight:.2f}, contribution={contribution:.4f}")
            
            signal_details.append({
                'agent': agent_name,
                'signal': signal['signal'],
                'score': score,
                'confidence': confidence,
                'weight': weight,
                'contribution': contribution
            })
        
        # Normalize by total weight
        if total_weight > 0:
            weighted_score /= total_weight
        print(f"  Total weighted_score (before normalization): {weighted_score * total_weight:.4f}")
        print(f"  Normalized weighted_score: {weighted_score:.4f}")
        
        # Determine final signal
        # In aggressive mode, use lower threshold for weighted voting too
        aggressive = self.config.get('aggressive_mode', False) or self.config.get('use_drl', False)
        threshold = self.min_confidence * 0.7 if aggressive else self.min_confidence
        
        # Calculate confidence based on agreeing agents (not weighted_score which is directional)
        # Count signals
        buy_count = sum(1 for s in agent_signals.values() if s['signal'] == Signal.BUY)
        sell_count = sum(1 for s in agent_signals.values() if s['signal'] == Signal.SELL)
        hold_count = sum(1 for s in agent_signals.values() if s['signal'] == Signal.HOLD)
        
        # Determine final signal direction
        if weighted_score > threshold:
            final_signal = Signal.BUY
        elif weighted_score < -threshold:
            final_signal = Signal.SELL
        else:
            final_signal = Signal.HOLD
        
        # Calculate confidence based on agreeing agents' confidences
        # For BUY/SELL: use weighted average of agreeing agents' confidences
        # For HOLD: use lower confidence
        if final_signal == Signal.BUY:
            # Get confidences of agents that agree (BUY) or are neutral (HOLD with low weight)
            agreeing_agents = [s for name, s in agent_signals.items() 
                              if s['signal'] == Signal.BUY]
            if agreeing_agents:
                # Weighted average confidence of agreeing agents
                agreeing_weight = sum(agent_weights.get(name, base_weights.get(name, 0.33))
                                    for name, s in agent_signals.items() 
                                    if s['signal'] == Signal.BUY)
                agreeing_confidence = sum(s['confidence'] * agent_weights.get(name, base_weights.get(name, 0.33))
                                         for name, s in agent_signals.items() 
                                         if s['signal'] == Signal.BUY) / agreeing_weight if agreeing_weight > 0 else 0.0
                # Boost confidence if multiple agents agree
                consensus_boost = min(1.0, 1.0 + (buy_count - 1) * 0.1)
                confidence = min(agreeing_confidence * consensus_boost, 1.0)
            else:
                confidence = 0.3  # Low confidence if no agents agree
        elif final_signal == Signal.SELL:
            # Get confidences of agents that agree (SELL) or are neutral (HOLD with low weight)
            agreeing_agents = [s for name, s in agent_signals.items() 
                              if s['signal'] == Signal.SELL]
            if agreeing_agents:
                # Weighted average confidence of agreeing agents
                agreeing_weight = sum(agent_weights.get(name, base_weights.get(name, 0.33))
                                    for name, s in agent_signals.items() 
                                    if s['signal'] == Signal.SELL)
                agreeing_confidence = sum(s['confidence'] * agent_weights.get(name, base_weights.get(name, 0.33))
                                         for name, s in agent_signals.items() 
                                         if s['signal'] == Signal.SELL) / agreeing_weight if agreeing_weight > 0 else 0.0
                # Boost confidence if multiple agents agree
                consensus_boost = min(1.0, 1.0 + (sell_count - 1) * 0.1)
                confidence = min(agreeing_confidence * consensus_boost, 1.0)
            else:
                confidence = 0.3  # Low confidence if no agents agree
        else:
            # HOLD signal - use lower confidence
            avg_confidence = sum(s.get('confidence', 0) * agent_weights.get(name, base_weights.get(name, 0.33))
                                for name, s in agent_signals.items()) / total_weight if total_weight > 0 else 0.0
            confidence = min(avg_confidence, 0.5)  # Cap HOLD confidence at 0.5
        
        # Safety check: If all agents agree on SELL but we got BUY, something is wrong
        # (buy_count, sell_count, hold_count already calculated above)
        
        if sell_count >= 2 and buy_count == 0 and final_signal == Signal.BUY:
            print(f"  ⚠️  WARNING: All agents say SELL/HOLD but final signal is BUY! Overriding to SELL.")
            print(f"     Agent signals: BUY={buy_count}, SELL={sell_count}, HOLD={hold_count}")
            final_signal = Signal.SELL
            confidence = min(avg_confidence, 0.5)  # Moderate confidence due to conflict
            weighted_score = -abs(weighted_score)  # Force negative
        elif buy_count >= 2 and sell_count == 0 and final_signal == Signal.SELL:
            print(f"  ⚠️  WARNING: All agents say BUY/HOLD but final signal is SELL! Overriding to BUY.")
            print(f"     Agent signals: BUY={buy_count}, SELL={sell_count}, HOLD={hold_count}")
            final_signal = Signal.BUY
            confidence = min(avg_confidence, 0.5)
            weighted_score = abs(weighted_score)  # Force positive
        
        return {
            'signal': final_signal,
            'confidence': confidence,
            'weighted_score': weighted_score,
            'reason': f'Weighted voting: {weighted_score:.3f} (BUY:{buy_count}, SELL:{sell_count}, HOLD:{hold_count})',
            'signal_details': signal_details
        }
    
    def _check_consensus(self, agent_signals: Dict, final_signal: Dict) -> Dict:
        """
        Check consensus among agents
        
        Returns consensus information
        """
        if not agent_signals:
            return {
                'has_consensus': False,
                'consensus_strength': 0.0,
                'agreeing_agents': [],
                'disagreeing_agents': []
            }
        
        final_signal_val = final_signal['signal']
        agreeing = []
        disagreeing = []
        
        for agent_name, signal in agent_signals.items():
            if signal['signal'] == final_signal_val:
                agreeing.append(agent_name)
            elif signal['signal'] != Signal.HOLD:
                disagreeing.append(agent_name)
        
        consensus_strength = len(agreeing) / len(agent_signals) if agent_signals else 0.0
        has_consensus = consensus_strength >= self.consensus_threshold
        
        return {
            'has_consensus': has_consensus,
            'consensus_strength': consensus_strength,
            'agreeing_agents': agreeing,
            'disagreeing_agents': disagreeing,
            'total_agents': len(agent_signals)
        }

