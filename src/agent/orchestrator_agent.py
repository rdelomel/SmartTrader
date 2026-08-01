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
        self.sentiment_analyzer = sentiment_analyzer
        
        # Quality-first defaults (config overrides these)
        self.min_confidence = self.config.get('min_confidence', 0.48)
        self.consensus_threshold = self.config.get('consensus_threshold', 0.48)
        self.veto_enabled = self.config.get('veto_enabled', True)
        self.require_agent_agreement = self.config.get('require_agent_agreement', True)
        self.min_agents_agreeing = self.config.get('min_agents_agreeing', 2)
        # Default false so YAML disable_short_trades:false is not overridden by a True fallback
        self.disable_short_trades = self.config.get('disable_short_trades', False)
        self.risk_reward_ratio = self.config.get('risk_reward_ratio', 3.0)
        
        self.use_drl = self.config.get('use_drl', True) and drl_agent is not None and drl_agent.is_trained
        self.use_llm_conflict_resolution = self.config.get('use_llm_conflict_resolution', True) and sentiment_analyzer is not None
    
    def analyze(self, data, symbol: str, asset_class: str,
                news_items: Optional[List[Dict]] = None,
                account_balance: float = 10000.0) -> Dict:
        return self.orchestrate(data, symbol, asset_class, news_items, account_balance)
    
    def orchestrate(self, data, symbol: str, asset_class: str, 
                   news_items: Optional[List[Dict]] = None,
                   account_balance: float = 10000.0) -> Dict:
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
        
        if self.technical_agent and self.technical_agent.is_enabled():
            print("  Querying Technical Analyst...")
            tech_result = self.technical_agent.analyze(data, symbol)
            agent_signals['technical'] = tech_result
            print(f"  Technical: {tech_result['signal'].name}, confidence={tech_result.get('confidence', 0):.3f}")
        else:
            print("  Technical Analyst: DISABLED")
        
        if self.sentiment_agent and self.sentiment_agent.is_enabled():
            print("  Querying Sentiment Agent...")
            sent_result = self.sentiment_agent.analyze(symbol, news_items)
            agent_signals['sentiment'] = sent_result
            print(f"  Sentiment: {sent_result['signal'].name}, confidence={sent_result.get('confidence', 0):.3f}")
        else:
            print("  Sentiment Agent: DISABLED")
        
        if self.fundamental_agent and self.fundamental_agent.is_enabled():
            print("  Querying Fundamental Agent...")
            fund_result = self.fundamental_agent.analyze(symbol, asset_class, news_items=news_items)
            agent_signals['fundamental'] = fund_result
            print(f"  Fundamental: {fund_result['signal'].name}, confidence={fund_result.get('confidence', 0):.3f}")
        else:
            print("  Fundamental Agent: DISABLED")
        
        if self.quantitative_agent and self.quantitative_agent.is_enabled():
            print("  Querying Quantitative Agent...")
            quant_result = self.quantitative_agent.analyze(data, symbol)
            agent_signals['quantitative'] = quant_result
            print(f"  Quantitative: {quant_result['signal'].name}, confidence={quant_result.get('confidence', 0):.3f}")
        else:
            print("  Quantitative Agent: DISABLED")
        
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
        
        # DIAGNOSTIC: Log all agent signals
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
        
        # Rule 6: Position Sizing and SL/TP
        position_size_info = None
        stop_loss = None
        take_profit = None
        
        if final_signal['signal'] != Signal.HOLD:
            try:
                entry_price = data['close'].iloc[-1]
                stop_loss = agent_signals.get('technical', {}).get('stop_loss')
                take_profit = agent_signals.get('technical', {}).get('take_profit')
                
                is_buy = final_signal['signal'] == Signal.BUY
                
                # Validate and recalculate SL/TP if needed
                if stop_loss:
                    if is_buy and stop_loss > entry_price:
                        print(f"  ❌ CRITICAL: Stop loss ${stop_loss:.2f} is above entry ${entry_price:.2f} for BUY")
                        stop_loss = None
                    elif not is_buy and stop_loss < entry_price:
                        print(f"  ❌ CRITICAL: Stop loss ${stop_loss:.2f} is below entry ${entry_price:.2f} for SELL")
                        stop_loss = None
                
                if take_profit:
                    if is_buy and take_profit < entry_price:
                        print(f"  ❌ CRITICAL: Take profit ${take_profit:.2f} is below entry ${entry_price:.2f} for BUY")
                        take_profit = None
                    elif not is_buy and take_profit > entry_price:
                        print(f"  ❌ CRITICAL: Take profit ${take_profit:.2f} is above entry ${entry_price:.2f} for SELL")
                        take_profit = None
                
                if not stop_loss:
                    from ..indicators.technical import TechnicalIndicators
                    indicators = TechnicalIndicators()
                    atr = indicators.atr(data)
                    atr_value = atr.iloc[-1] if not atr.empty else entry_price * 0.02
                    min_stop_distance = entry_price * 0.005
                    atr_value = max(atr_value, min_stop_distance)
                    stop_distance = atr_value * 2.0
                    
                    if is_buy:
                        stop_loss = entry_price - stop_distance
                    else:
                        stop_loss = entry_price + stop_distance
                    
                    if is_buy:
                        stop_loss = max(stop_loss, entry_price * 0.95)
                    else:
                        stop_loss = min(stop_loss, entry_price * 1.05)
                    
                    print(f"  ✅ Calculated default stop loss: ${stop_loss:.2f}")
                
                if not take_profit and stop_loss is not None:
                    risk_reward_ratio = self.risk_reward_ratio
                    risk = abs(entry_price - stop_loss)
                    if risk > 0:
                        reward = risk * risk_reward_ratio
                        if is_buy:
                            take_profit = entry_price + reward
                        else:
                            take_profit = entry_price - reward
                        print(f"  Calculated Take Profit: ${take_profit:.2f} (R/R: 1:{risk_reward_ratio:.1f})")
                
                # Final safety checks
                if stop_loss is None:
                    stop_loss = entry_price * 0.98 if is_buy else entry_price * 1.02
                
                if take_profit is None and stop_loss is not None:
                    risk = abs(entry_price - stop_loss)
                    if risk > 0:
                        reward = risk * self.risk_reward_ratio
                        if is_buy:
                            take_profit = entry_price + reward
                        else:
                            take_profit = entry_price - reward
                
                # Final validation
                if is_buy and stop_loss >= entry_price:
                    stop_loss = entry_price * 0.98
                elif not is_buy and stop_loss <= entry_price:
                    stop_loss = entry_price * 1.02
                
                if is_buy and take_profit <= entry_price:
                    if stop_loss is not None:
                        risk = abs(entry_price - stop_loss)
                        take_profit = entry_price + (risk * self.risk_reward_ratio)
                    else:
                        take_profit = entry_price * 1.03
                elif not is_buy and take_profit >= entry_price:
                    if stop_loss is not None:
                        risk = abs(entry_price - stop_loss)
                        take_profit = entry_price - (risk * self.risk_reward_ratio)
                    else:
                        take_profit = entry_price * 0.97
                
                if stop_loss is not None and take_profit is not None:
                    risk = abs(entry_price - stop_loss)
                    reward = abs(take_profit - entry_price)
                    if risk > 0:
                        actual_rr = reward / risk
                        print(f"  ✅ FINAL: Entry=${entry_price:.2f}, SL=${stop_loss:.2f}, TP=${take_profit:.2f}, R/R=1:{actual_rr:.2f}")
                
            except Exception as e:
                print(f"  ❌ ERROR calculating SL/TP: {e}")
                entry_price = data['close'].iloc[-1]
                is_buy = final_signal['signal'] == Signal.BUY
                if stop_loss is None:
                    stop_loss = entry_price * 0.98 if is_buy else entry_price * 1.02
                if take_profit is None and stop_loss is not None:
                    risk = abs(entry_price - stop_loss)
                    if risk > 0:
                        reward = risk * 2.0
                        take_profit = entry_price + reward if is_buy else entry_price - reward
            
            if self.risk_agent:
                try:
                    entry_price = data['close'].iloc[-1]
                    position_size_info = self.risk_agent.calculate_position_size(
                        account_balance, entry_price, stop_loss, data, current_regime,
                        performance_tracker=self.performance_tracker,
                        symbol=symbol
                    )
                    
                    drl_action = final_signal.get('drl_action', None)
                    drl_confidence = final_signal.get('drl_confidence', 0.5)
                    if drl_action is not None and self.use_drl:
                        action_multiplier = 0.5 + 0.5 * abs(drl_action)
                        confidence_multiplier = 0.7 + 0.3 * drl_confidence
                        drl_adjustment = action_multiplier * confidence_multiplier
                        position_size_info['drl_adjustment'] = drl_adjustment
                        position_size_info['quantity'] = position_size_info.get('quantity', 0) * drl_adjustment
                        position_size_info['value'] = position_size_info.get('quantity', 0) * entry_price
                except Exception as e:
                    print(f"  ⚠️  Error calculating position size: {e}")
                    position_size_info = None
        
        self._update_timestamp()
        
        # Final decision summary
        print("\n[FINAL DECISION SUMMARY]")
        print(f"  Signal: {final_signal['signal'].name}")
        print(f"  Confidence: {final_signal['confidence']:.3f}")
        print(f"  Min Confidence Required: {self.min_confidence:.3f}")
        
        will_trade = (final_signal['signal'] != Signal.HOLD and 
                     final_signal['confidence'] >= self.min_confidence)
        
        # === FIX 4: Relaxed sentiment check ===
        if will_trade and final_signal['signal'] != Signal.HOLD:
            min_technical_confidence = self.config.get('min_technical_confidence_for_sentiment_trade', 0.25)  # Was 0.40
            technical_signal = agent_signals.get('technical', {})
            technical_conf = technical_signal.get('confidence', 0.0)
            sentiment_signal = agent_signals.get('sentiment', {})
            sentiment_conf = sentiment_signal.get('confidence', 0.0)
            
            is_sentiment_driven = (sentiment_conf > technical_conf * 1.5 and sentiment_conf > 0.5)  # Was 0.6
            
            if is_sentiment_driven and technical_conf < min_technical_confidence:
                technical_sig = technical_signal.get('signal', Signal.HOLD)
                sentiment_sig = sentiment_signal.get('signal', Signal.HOLD)
                has_conflict = (technical_sig != Signal.HOLD and sentiment_sig != Signal.HOLD and 
                               technical_sig != sentiment_sig)
                
                if has_conflict:
                    print(f"  ⚠️  WARNING: Sentiment-driven trade with weak technical and conflict")
                    print(f"     Technical confidence ({technical_conf:.3f}) < minimum ({min_technical_confidence:.3f})")
                    # === FIX: Reduce confidence instead of rejecting ===
                    final_signal['confidence'] *= 0.6
                    will_trade = final_signal['confidence'] >= self.min_confidence
                    print(f"     Reduced confidence to {final_signal['confidence']:.3f}, will_trade={will_trade}")
                else:
                    print(f"  ⚠️  Sentiment-driven trade with weak technical — reducing confidence")
                    final_signal['confidence'] *= 0.7
                    will_trade = final_signal['confidence'] >= self.min_confidence
        
        print(f"  Will Execute Trade: {'YES ✓' if will_trade else 'NO ✗'}")
        
        if not will_trade:
            print(f"\n  [DIAGNOSTIC] Trade Rejection Analysis:")
            if final_signal['signal'] == Signal.HOLD:
                print(f"    ❌ Reason: Signal is HOLD")
            else:
                print(f"    ❌ Reason: Confidence too low")
                print(f"       Final confidence: {final_signal['confidence']:.3f}")
                print(f"       Required minimum: {self.min_confidence:.3f}")
                print(f"       Difference: {self.min_confidence - final_signal['confidence']:.3f}")
        
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
        if not self.drl_agent or not self.drl_agent.is_trained:
            market_data = self._extract_market_features(data)
            return self._weighted_voting(agent_signals, regime_result.get('agent_weights', {}) if regime_result else {}, 
                                        regime_result, market_data)
        
        market_data = self._extract_market_features(data)
        
        performance_metrics = None
        if self.performance_tracker:
            try:
                metrics = self.performance_tracker.calculate_metrics()
                performance_metrics = {
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
                    'win_rate': metrics.get('win_rate', 0.0),
                    'last_7d_return': metrics.get('weekly_return', 0.0)
                }
            except:
                pass
        
        position_info = None
        
        state = self.drl_agent.build_state_vector(
            agent_signals, 
            regime_result or {'regime': 'Neutral', 'confidence': 0.0},
            market_data,
            performance_metrics=performance_metrics,
            position_info=position_info
        )
        
        action, confidence = self.drl_agent.predict(state, deterministic=True)
        
        aggressive = self.config.get('aggressive_mode', False) or self.config.get('use_drl', False)
        
        base_buy_threshold = 0.15 if aggressive else 0.25  # Was 0.2/0.3
        base_sell_threshold = -0.15 if aggressive else -0.25
        
        volatility = market_data.get('volatility', 0.0)
        if volatility < 0.01:
            vol_adjustment = 0.10  # Was 0.15
        elif volatility > 0.03:
            vol_adjustment = 0.20  # Was 0.25
        else:
            vol_adjustment = 0.0
        
        performance_adjustment = 0.0
        if self.performance_tracker:
            try:
                metrics = self.performance_tracker.calculate_metrics()
                sharpe = metrics.get('sharpe_ratio', 0.0)
                weekly_return = metrics.get('weekly_return', 0.0)
                
                if sharpe > 1.0 and weekly_return > 0:
                    performance_adjustment = -0.15  # Was -0.1
                elif sharpe < 0.5 or weekly_return < -1.0:
                    performance_adjustment = 0.10  # Was 0.15
            except:
                pass
        
        regime_adjustment = 0.0
        current_regime = regime_result.get('regime', 'Neutral') if regime_result else 'Neutral'
        if current_regime == 'Bullish Trend':
            regime_adjustment = -0.10  # Was -0.05
        elif current_regime == 'Bearish Trend':
            regime_adjustment = 0.05  # Was 0.1
        elif current_regime == 'High Volatility':
            regime_adjustment = 0.10  # Was 0.15
        
        buy_threshold = base_buy_threshold + vol_adjustment + performance_adjustment + regime_adjustment
        sell_threshold = base_sell_threshold - vol_adjustment - performance_adjustment - regime_adjustment
        
        buy_threshold = max(0.05, min(0.3, buy_threshold))  # Was 0.1/0.4
        sell_threshold = min(-0.05, max(-0.3, sell_threshold))
        
        if action > buy_threshold:
            signal = Signal.BUY
        elif action < sell_threshold:
            signal = Signal.SELL
        else:
            signal = Signal.HOLD
        
        market_data_for_voting = self._extract_market_features(data)
        weighted_result = self._weighted_voting(agent_signals, 
                                                regime_result.get('agent_weights', {}) if regime_result else {},
                                                regime_result, market_data_for_voting)
        weighted_confidence = weighted_result.get('confidence', 0.5)
        
        hybrid_confidence = 0.6 * confidence + 0.4 * weighted_confidence
        
        print(f"  DRL Decision: {signal.name}, DRL confidence={confidence:.3f}, Weighted confidence={weighted_confidence:.3f}, Hybrid={hybrid_confidence:.3f}")
        
        return {
            'signal': signal,
            'confidence': hybrid_confidence,
            'weighted_score': action,
            'reason': f'DRL decision: action={action:.3f}, confidence={hybrid_confidence:.3f}',
            'drl_action': action,
            'drl_confidence': confidence,
            'weighted_confidence': weighted_confidence
        }
    
    def _extract_market_features(self, data: pd.DataFrame) -> Dict:
        if len(data) == 0:
            return {
                'volatility': 0.0,
                'volume_ratio': 1.0,
                'price_momentum': 0.0,
                'rsi': 50.0,
                'adx': 0.0
            }
        
        latest = data.iloc[-1]
        
        returns = data['close'].pct_change()
        volatility = returns.tail(20).std() if len(returns) >= 20 else returns.std()
        
        volume_ma = data['volume'].rolling(window=20).mean()
        volume_ratio = (latest['volume'] / volume_ma.iloc[-1]) if len(volume_ma) > 0 and volume_ma.iloc[-1] > 0 else 1.0
        
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
    
    def _weighted_voting(self, agent_signals: Dict, agent_weights: Dict, 
                        regime_result: Optional[Dict] = None, market_data: Optional[Dict] = None) -> Dict:
        if not agent_signals:
            return {
                'signal': Signal.HOLD,
                'confidence': 0.0,
                'weighted_score': 0.0,
                'reason': 'No agent signals available'
            }
        
        base_weights = {
            'technical': self.config.get('technical_weight', 0.45),  # Was 0.50
            'sentiment': self.config.get('sentiment_weight', 0.10),   # Was 0.15
            'fundamental': self.config.get('fundamental_weight', 0.10),  # Unchanged
            'quantitative': self.config.get('quantitative_weight', 0.25),  # Was 0.20
            'pattern_forecaster': self.config.get('pattern_forecaster_weight', 0.10)  # Was 0.15
        }
        
        total_base_weight = sum(base_weights.values())
        if total_base_weight > 0:
            base_weights = {k: v / total_base_weight for k, v in base_weights.items()}
        
        # === FIX 5: Relaxed filtering — include HOLD signals with some confidence ===
        filtered_signals = {}
        for agent_name, signal in agent_signals.items():
            confidence = signal.get('confidence', 0.0)
            # Include all non-zero confidence signals, even HOLD
            if confidence > 0.005:  # Was 0.01
                filtered_signals[agent_name] = signal
            else:
                print(f"    {agent_name}: EXCLUDED (confidence={confidence:.3f} too low)")
        
        if not filtered_signals:
            print(f"  ❌ DIAGNOSTIC: All agent signals filtered out!")
            return {
                'signal': Signal.HOLD,
                'confidence': 0.0,
                'weighted_score': 0.0,
                'reason': 'No valid agent signals (all filtered out)'
            }
        
        weighted_score = 0.0
        total_weight = 0.0
        signal_details = []
        
        print(f"  Weighted Voting Calculation:")
        for agent_name, signal in filtered_signals.items():
            base_weight = base_weights.get(agent_name, 0.0)
            regime_multiplier = agent_weights.get(agent_name, 1.0)
            weight = base_weight * regime_multiplier
            
            signal_val = 0
            if signal['signal'] == Signal.BUY:
                signal_val = 1
            elif signal['signal'] == Signal.SELL:
                signal_val = -1
            
            confidence = signal.get('confidence', 0.5)
            contribution = signal_val * confidence * weight
            
            weighted_score += contribution
            total_weight += weight
            
            print(f"    {agent_name}: {signal['signal'].name}, signal_val={signal_val}, confidence={confidence:.3f}, base_weight={base_weight:.2f}, multiplier={regime_multiplier:.2f}, final_weight={weight:.3f}, contribution={contribution:.4f}")
            
            signal_details.append({
                'agent': agent_name,
                'signal': signal['signal'],
                'confidence': confidence,
                'base_weight': base_weight,
                'regime_multiplier': regime_multiplier,
                'final_weight': weight,
                'contribution': contribution
            })
        
        if total_weight > 0:
            normalized_score = weighted_score / total_weight
        else:
            normalized_score = 0.0
        
        print(f"  Total contribution: {weighted_score:.4f}")
        print(f"  Total weight: {total_weight:.4f}")
        print(f"  Normalized weighted_score: {normalized_score:.4f}")
        
        # === FIX 6: Lower threshold for weighted voting ===
        aggressive = self.config.get('aggressive_mode', False) or self.config.get('use_drl', False)
        threshold = self.min_confidence * 0.6 if aggressive else self.min_confidence  # Was 0.7
        
        buy_count = sum(1 for s in filtered_signals.values() if s['signal'] == Signal.BUY)
        sell_count = sum(1 for s in filtered_signals.values() if s['signal'] == Signal.SELL)
        hold_count = sum(1 for s in filtered_signals.values() if s['signal'] == Signal.HOLD)
        
        if normalized_score > threshold:
            final_signal = Signal.BUY
        elif normalized_score < -threshold:
            final_signal = Signal.SELL
        else:
            final_signal = Signal.HOLD
            
            # Tie-breaking logic
            max_confidence = 0.0
            max_confidence_signal = None
            max_confidence_agent = None
            
            for agent_name, signal in filtered_signals.items():
                agent_confidence = signal.get('confidence', 0.0)
                agent_signal = signal['signal']
                
                if agent_confidence > max_confidence:
                    max_confidence = agent_confidence
                    max_confidence_signal = agent_signal
                    max_confidence_agent = agent_name
            
            opposing_confidences = []
            for agent_name, signal in filtered_signals.items():
                agent_confidence = signal.get('confidence', 0.0)
                agent_signal = signal['signal']
                
                if agent_signal != max_confidence_signal and agent_signal != Signal.HOLD:
                    opposing_confidences.append(agent_confidence)
            
            if max_confidence > 0.6 and max_confidence_signal != Signal.HOLD:  # Was 0.7
                avg_opposing_confidence = sum(opposing_confidences) / len(opposing_confidences) if opposing_confidences else 0.0
                
                if max_confidence > 1.3 * avg_opposing_confidence or (max_confidence > 0.7 and avg_opposing_confidence < 0.4):  # Was 1.5/0.8/0.5
                    final_signal = max_confidence_signal
                    print(f"  🔀 Tie-breaking: Strong signal from {max_confidence_agent} ({max_confidence_signal.name}, confidence={max_confidence:.3f})")
            
            if max_confidence > 0.7 and max_confidence_signal != Signal.HOLD:  # Was 0.8
                weak_opposing = all(c < 0.4 for c in opposing_confidences)  # Was 0.5
                if weak_opposing:
                    final_signal = max_confidence_signal
                    print(f"  ⚡ Strong signal override: {max_confidence_agent} ({max_confidence_signal.name}, confidence={max_confidence:.3f})")
            
            # LLM conflict resolution
            sentiment_signal = filtered_signals.get('sentiment')
            technical_signal = filtered_signals.get('technical')
            force_llm = False
            
            if sentiment_signal and technical_signal:
                sentiment_conf = sentiment_signal.get('confidence', 0)
                technical_conf = technical_signal.get('confidence', 0)
                sentiment_sig = sentiment_signal.get('signal')
                technical_sig = technical_signal.get('signal')
                
                if (sentiment_sig != technical_sig and 
                    sentiment_sig != Signal.HOLD and 
                    technical_sig != Signal.HOLD and
                    sentiment_conf > 0.25 and technical_conf > 0.25):  # Was 0.3
                    
                    print(f"  ⚠️  STRONG CONFLICT: Sentiment {sentiment_sig.name} vs Technical {technical_sig.name}")
                    force_llm = True
            
            if force_llm or (final_signal == Signal.HOLD and abs(normalized_score) < threshold * 1.2):
                llm_result = self._llm_resolve_conflicts(
                    filtered_signals, normalized_score, threshold, 
                    regime_result=regime_result, market_data=market_data
                )
                
                if llm_result:
                    llm_signal = llm_result.get('signal')
                    llm_confidence = llm_result.get('confidence', 0.5)
                    
                    if llm_confidence > 0.35:  # Was 0.4
                        final_signal = llm_signal
                        print(f"  ✅ LLM resolution: {llm_signal.name} (confidence: {llm_confidence:.3f})")
                    elif force_llm:
                        final_signal = llm_signal
                        print(f"  ⚠️  LLM forced (low confidence {llm_confidence:.3f}): {llm_signal.name}")
        
        # === FIX 7: Reduced disagreement penalty ===
        if final_signal == Signal.BUY:
            agreeing_agents = [s for name, s in agent_signals.items() if s['signal'] == Signal.BUY]
            if agreeing_agents:
                agreeing_weight = sum(base_weights.get(name, 0.33) * agent_weights.get(name, 1.0)
                                    for name, s in agent_signals.items() if s['signal'] == Signal.BUY)
                agreeing_confidence = sum(s['confidence'] * base_weights.get(name, 0.33) * agent_weights.get(name, 1.0)
                                         for name, s in agent_signals.items() if s['signal'] == Signal.BUY) / agreeing_weight if agreeing_weight > 0 else 0.0
                consensus_boost = min(1.0, 1.0 + (buy_count - 1) * 0.08)  # Was 0.1
                confidence = min(agreeing_confidence * consensus_boost, 1.0)
                
                if sell_count > 0:
                    opposing_weight = sum(base_weights.get(name, 0.33) * agent_weights.get(name, 1.0)
                                         for name, s in agent_signals.items() if s['signal'] == Signal.SELL)
                    opposing_confidence = sum(s['confidence'] * base_weights.get(name, 0.33) * agent_weights.get(name, 1.0)
                                             for name, s in agent_signals.items() if s['signal'] == Signal.SELL) / opposing_weight if opposing_weight > 0 else 0.0
                    
                    penalty = min(0.15, opposing_confidence * 0.2)  # Was 0.3 and 0.4
                    confidence = confidence * (1.0 - penalty)
                    print(f"  ⚠️  Disagreement penalty: {penalty*100:.1f}% (opposing: {opposing_confidence:.3f})")
            else:
                confidence = 0.3
        elif final_signal == Signal.SELL:
            agreeing_agents = [s for name, s in agent_signals.items() if s['signal'] == Signal.SELL]
            if agreeing_agents:
                agreeing_weight = sum(base_weights.get(name, 0.33) * agent_weights.get(name, 1.0)
                                    for name, s in agent_signals.items() if s['signal'] == Signal.SELL)
                agreeing_confidence = sum(s['confidence'] * base_weights.get(name, 0.33) * agent_weights.get(name, 1.0)
                                         for name, s in agent_signals.items() if s['signal'] == Signal.SELL) / agreeing_weight if agreeing_weight > 0 else 0.0
                consensus_boost = min(1.0, 1.0 + (sell_count - 1) * 0.08)
                confidence = min(agreeing_confidence * consensus_boost, 1.0)
                
                if buy_count > 0:
                    opposing_weight = sum(base_weights.get(name, 0.33) * agent_weights.get(name, 1.0)
                                         for name, s in agent_signals.items() if s['signal'] == Signal.BUY)
                    opposing_confidence = sum(s['confidence'] * base_weights.get(name, 0.33) * agent_weights.get(name, 1.0)
                                             for name, s in agent_signals.items() if s['signal'] == Signal.BUY) / opposing_weight if opposing_weight > 0 else 0.0
                    
                    penalty = min(0.15, opposing_confidence * 0.2)
                    confidence = confidence * (1.0 - penalty)
                    print(f"  ⚠️  Disagreement penalty: {penalty*100:.1f}% (opposing: {opposing_confidence:.3f})")
            else:
                confidence = 0.3
        else:
            avg_confidence = sum(s.get('confidence', 0) * base_weights.get(name, 0.33) * agent_weights.get(name, 1.0)
                                for name, s in agent_signals.items()) / total_weight if total_weight > 0 else 0.0
            confidence = min(avg_confidence, 0.5)
        
        print(f"  📊 Confidence: normalized_score={normalized_score:.4f}, final_confidence={confidence:.4f}")
        if final_signal != Signal.HOLD:
            print(f"     Agreeing: {buy_count if final_signal == Signal.BUY else sell_count}, Opposing: {sell_count if final_signal == Signal.BUY else buy_count}")
        
        avg_confidence_all = sum(s.get('confidence', 0) * base_weights.get(name, 0.33) * agent_weights.get(name, 1.0)
                                for name, s in agent_signals.items()) / total_weight if total_weight > 0 else 0.0
        
        # Hard reject when fewer than min_agents_agreeing vote with the signal
        if self.require_agent_agreement and final_signal != Signal.HOLD:
            agreeing_count = buy_count if final_signal == Signal.BUY else sell_count
            if agreeing_count < self.min_agents_agreeing:
                print(f"  [SKIP] Insufficient agreement: {agreeing_count}/{self.min_agents_agreeing} agents agree")
                print(f"     Rejecting trade - require multiple agent agreement for higher quality signals")
                final_signal = Signal.HOLD
                confidence = 0.0

        # Long-only mode: convert SELL to HOLD (execution layer also blocks)
        if self.disable_short_trades and final_signal == Signal.SELL:
            print(f"  [SKIP] Short trades disabled - converting SELL to HOLD")
            final_signal = Signal.HOLD
            confidence = 0.0
        
        # Safety checks
        if sell_count >= 2 and buy_count == 0 and final_signal == Signal.BUY:
            print(f"  ⚠️  WARNING: All agents say SELL but final is BUY! Overriding.")
            final_signal = Signal.SELL
            confidence = min(avg_confidence_all, 0.5)
            normalized_score = -abs(normalized_score)
        elif buy_count >= 2 and sell_count == 0 and final_signal == Signal.SELL:
            print(f"  ⚠️  WARNING: All agents say BUY but final is SELL! Overriding.")
            final_signal = Signal.BUY
            confidence = min(avg_confidence_all, 0.5)
            normalized_score = abs(normalized_score)
        
        return {
            'signal': final_signal,
            'confidence': confidence,
            'weighted_score': normalized_score,
            'reason': f'Weighted voting: {normalized_score:.3f} (BUY:{buy_count}, SELL:{sell_count}, HOLD:{hold_count})',
            'signal_details': signal_details
        }
    
    def _check_consensus(self, agent_signals: Dict, final_signal: Dict) -> Dict:
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
        neutral = []
        
        valid_agents = {name: sig for name, sig in agent_signals.items() 
                       if sig.get('confidence', 0.0) > 0.05}
        
        for agent_name, signal in valid_agents.items():
            agent_signal = signal['signal']
            if agent_signal == Signal.HOLD:
                neutral.append(agent_name)
            elif agent_signal == final_signal_val:
                agreeing.append(agent_name)
            else:
                disagreeing.append(agent_name)
        
        total_voting = len(agreeing) + len(disagreeing)
        if total_voting > 0:
            consensus_strength = len(agreeing) / total_voting
        else:
            consensus_strength = 0.0
        
        has_consensus = consensus_strength >= self.consensus_threshold
        
        return {
            'has_consensus': has_consensus,
            'consensus_strength': consensus_strength,
            'agreeing_agents': agreeing,
            'disagreeing_agents': disagreeing,
            'neutral_agents': neutral,
            'total_agents': len(agent_signals),
            'voting_agents': total_voting
        }
    
    def _llm_resolve_conflicts(self, agent_signals: Dict, normalized_score: float, 
                               threshold: float, regime_result: Optional[Dict],
                               market_data: Optional[Dict] = None) -> Optional[Dict]:
        if not self.use_llm_conflict_resolution or not self.sentiment_analyzer:
            return None
        
        buy_signals = [(name, sig) for name, sig in agent_signals.items() 
                      if sig.get('signal') == Signal.BUY and sig.get('confidence', 0) > 0.05]
        sell_signals = [(name, sig) for name, sig in agent_signals.items() 
                       if sig.get('signal') == Signal.SELL and sig.get('confidence', 0) > 0.05]
        
        has_strong_conflict = (len(buy_signals) > 0 and len(sell_signals) > 0 and 
                              abs(normalized_score) < threshold * 1.2)
        
        if not has_strong_conflict:
            return None
        
        try:
            signal_summary = []
            for name, sig in agent_signals.items():
                signal_type = sig.get('signal', Signal.HOLD)
                confidence = sig.get('confidence', 0.0)
                reason = sig.get('reason', 'N/A')
                signal_summary.append(f"- {name}: {signal_type.name} (confidence {confidence:.2f}) - {reason}")
            
            regime = regime_result.get('regime', 'Neutral') if regime_result else 'Neutral'
            regime_conf = regime_result.get('confidence', 0.0) if regime_result else 0.0
            
            market_context = ""
            if market_data:
                volatility = market_data.get('volatility', 0.0)
                volume_ratio = market_data.get('volume_ratio', 1.0)
                market_context = f"\nMarket Context:\n- Volatility: {volatility:.4f}\n- Volume Ratio: {volume_ratio:.2f}"
            
            perf_context = ""
            if self.performance_tracker:
                try:
                    metrics = self.performance_tracker.calculate_metrics()
                    sharpe = metrics.get('sharpe_ratio', 0.0)
                    win_rate = metrics.get('win_rate', 0.0)
                    perf_context = f"\nRecent Performance:\n- Sharpe Ratio: {sharpe:.2f}\n- Win Rate: {win_rate:.1f}%"
                except:
                    pass
            
            prompt = f"""You are a trading signal aggregation expert. Analyze these conflicting signals and recommend which should dominate:

Agent Signals:
{chr(10).join(signal_summary)}

Market Regime: {regime} (confidence: {regime_conf:.2f}){market_context}{perf_context}

Current Analysis:
- Normalized Score: {normalized_score:.3f} (threshold: {threshold:.3f})
- BUY signals: {len(buy_signals)}
- SELL signals: {len(sell_signals)}

Which signal should dominate? Provide:
1. Recommended signal (BUY, SELL, or HOLD)
2. Confidence level (0.0 to 1.0)
3. Brief reasoning (2-3 sentences)

Respond in JSON format:
{{
  "signal": "BUY|SELL|HOLD",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}"""
            
            result = self.sentiment_analyzer.analyze_text(prompt)
            
            reasoning_text = result.get('reasoning', '')
            llm_signal = None
            llm_confidence = None
            llm_reasoning = reasoning_text
            
            sentiment_score = result.get('sentiment', 0.0)
            
            import json
            import re
            json_match = re.search(r'\{[^{}]*"signal"[^{}]*\}', reasoning_text, re.IGNORECASE)
            if not json_match:
                json_match = re.search(r'\{.*?"signal".*?\}', reasoning_text, re.IGNORECASE | re.DOTALL)
            
            if json_match:
                try:
                    llm_result = json.loads(json_match.group())
                    signal_str = str(llm_result.get('signal', 'HOLD')).upper()
                    llm_confidence = float(llm_result.get('confidence', 0.5))
                    llm_reasoning = llm_result.get('reasoning', reasoning_text)
                    
                    if 'BUY' in signal_str:
                        llm_signal = Signal.BUY
                    elif 'SELL' in signal_str:
                        llm_signal = Signal.SELL
                    else:
                        llm_signal = Signal.HOLD
                except Exception as e:
                    print(f"  ⚠️  LLM JSON parsing error: {e}")
            
            if llm_signal is None:
                reasoning_lower = reasoning_text.lower()
                if 'buy' in reasoning_lower and 'sell' not in reasoning_lower:
                    llm_signal = Signal.BUY
                elif 'sell' in reasoning_lower and 'buy' not in reasoning_lower:
                    llm_signal = Signal.SELL
                else:
                    if abs(sentiment_score) > 0.3:
                        llm_signal = Signal.BUY if sentiment_score > 0 else Signal.SELL
                    else:
                        llm_signal = Signal.HOLD
                
                if llm_confidence is None:
                    llm_confidence = result.get('confidence', 0.5)
                    if abs(sentiment_score) > 0.5:
                        llm_confidence = min(llm_confidence + 0.1, 1.0)
            
            if llm_signal and llm_signal != Signal.HOLD:
                print(f"  🤖 LLM Conflict Resolution:")
                print(f"     Recommended: {llm_signal.name} (confidence: {llm_confidence:.3f})")
                print(f"     Reasoning: {llm_reasoning[:200]}...")
                
                return {
                    'signal': llm_signal,
                    'confidence': llm_confidence,
                    'reasoning': llm_reasoning,
                    'source': 'LLM'
                }
            
        except Exception as e:
            print(f"  ⚠️  LLM conflict resolution error: {e}")
        
        return None
