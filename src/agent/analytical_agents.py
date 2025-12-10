"""Analytical agents: Technical, Sentiment, and Fundamental"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import requests
import yfinance as yf
from .base_agent import BaseAgent
from ..strategies.base_strategy import Signal
from ..indicators.technical import TechnicalIndicators
from ..indicators.chart_patterns import ChartPatternRecognizer, PatternType
from ..ai.models.sentiment_analyzer import SentimentAnalyzer
from ..data.news_fetcher import NewsFetcher
import os


class TechnicalAnalystAgent(BaseAgent):
    """Technical analysis agent using indicators and ML models"""
    
    def __init__(self, config: Optional[Dict] = None, strategies: Optional[List] = None,
                 ml_model=None, lstm_model=None):
        """
        Initialize technical analyst agent
        
        Args:
            config: Agent configuration
            strategies: List of trading strategies
            ml_model: ML model instance
            lstm_model: LSTM model instance
        """
        super().__init__("TechnicalAnalyst", config)
        self.strategies = strategies or []
        self.ml_model = ml_model
        self.lstm_model = lstm_model
        self.indicators = TechnicalIndicators()
        self.pattern_recognizer = ChartPatternRecognizer()
        self.feature_engineer = None  # Will be set if needed
    
    def analyze(self, data: pd.DataFrame, symbol: str = None) -> Dict:
        """
        Perform technical analysis with detailed logging
        
        Args:
            data: Market data DataFrame
            symbol: Trading symbol
        
        Returns:
            Dictionary with technical signal and score
        """
        if not self.enabled or len(data) < 20:
            print(f"TechnicalAnalyst: Skipping analysis - enabled={self.enabled}, data_len={len(data)}")
            return {
                'signal': Signal.HOLD,
                'score': 0.0,
                'confidence': 0.0,
                'source': self.name,
                'reason': 'Insufficient data or disabled'
            }
        
        print(f"\n=== TechnicalAnalyst analyzing {symbol if symbol else 'symbol'} ===")
        
        # Enhanced Technical Analysis based on Investopedia principles:
        # 1. Market discounts everything - price/volume reflect all information
        # 2. Prices move in trends - identify and follow trends
        # 3. History repeats itself - use chart patterns and historical data
        
        # Detect chart patterns (Investopedia: pattern recognition)
        patterns = self.pattern_recognizer.detect_patterns(data)
        pattern_signals = []
        if patterns:
            print(f"  📊 Chart Patterns Detected: {len(patterns)}")
            for pattern in patterns[:3]:  # Top 3 patterns
                pattern_type = pattern['type'].value if hasattr(pattern['type'], 'value') else str(pattern['type'])
                direction = pattern['direction']
                confidence = pattern.get('confidence', 0.5)
                
                signal = Signal.BUY if direction == 'bullish' else Signal.SELL
                pattern_signals.append({
                    'signal': signal,
                    'confidence': confidence,
                    'source': f'Pattern_{pattern_type}',
                    'pattern_type': pattern_type,
                    'target': pattern.get('target'),
                    'stop_loss': pattern.get('stop_loss')
                })
                print(f"    {pattern_type}: {direction.upper()}, confidence={confidence:.2f}")
        
        # Find support/resistance levels (Investopedia: key price levels)
        support_resistance = self.pattern_recognizer.find_support_resistance(data)
        if support_resistance.get('support') or support_resistance.get('resistance'):
            print(f"  📈 Support/Resistance:")
            if support_resistance.get('support'):
                print(f"    Support: ${support_resistance['support']:.2f} (strength: {support_resistance.get('support_strength', 0)})")
            if support_resistance.get('resistance'):
                print(f"    Resistance: ${support_resistance['resistance']:.2f} (strength: {support_resistance.get('resistance_strength', 0)})")
        
        signals = []
        weights = []
        strategy_details = []
        
        # Add pattern signals with weight
        pattern_weight = self.config.get('pattern_weight', 0.15)
        for pattern_signal in pattern_signals:
            signals.append(pattern_signal)
            weights.append(pattern_weight)
        
        # Get signals from strategies
        for strategy in self.strategies:
            if strategy.is_enabled():
                try:
                    signal = strategy.generate_signal(data)
                    signals.append(signal)
                    weight = strategy.get_weight()
                    weights.append(weight)
                    
                    # Log strategy details
                    signal_val = signal['signal'].name if hasattr(signal['signal'], 'name') else str(signal['signal'])
                    print(f"  Strategy {strategy.name}: {signal_val}, confidence={signal.get('confidence', 0):.3f}, weight={weight:.2f}")
                    print(f"    Reason: {signal.get('reason', 'N/A')}")
                    
                    strategy_details.append({
                        'name': strategy.name,
                        'signal': signal_val,
                        'confidence': signal.get('confidence', 0),
                        'weight': weight,
                        'reason': signal.get('reason', 'N/A')
                    })
                except Exception as e:
                    print(f"  Error in {strategy.name}: {e}")
            else:
                print(f"  Strategy {strategy.name}: DISABLED")
        
        # Get ML model signal
        if self.ml_model and self.ml_model.is_trained:
            try:
                ml_signal = self._get_ml_signal(data)
                if ml_signal:
                    signals.append(ml_signal)
                    weight = self.config.get('ml_weight', 0.3)
                    weights.append(weight)
                    print(f"  ML Model: {ml_signal['signal'].name}, confidence={ml_signal.get('confidence', 0):.3f}, weight={weight:.2f}")
            except Exception as e:
                print(f"  Error getting ML signal: {e}")
        else:
            print(f"  ML Model: {'NOT TRAINED' if self.ml_model else 'NOT ENABLED'}")
        
        # Get LSTM signal
        if self.lstm_model and self.lstm_model.is_trained:
            try:
                lstm_signal = self._get_lstm_signal(data)
                if lstm_signal:
                    signals.append(lstm_signal)
                    weight = self.config.get('lstm_weight', 0.15)
                    weights.append(weight)
                    print(f"  LSTM Model: {lstm_signal['signal'].name}, confidence={lstm_signal.get('confidence', 0):.3f}, weight={weight:.2f}")
            except Exception as e:
                print(f"  Error getting LSTM signal: {e}")
        else:
            print(f"  LSTM Model: {'NOT TRAINED' if self.lstm_model else 'NOT ENABLED'}")
        
        if not signals:
            print(f"TechnicalAnalyst: No signals generated for {symbol}")
            return {
                'signal': Signal.HOLD,
                'score': 0.0,
                'confidence': 0.0,
                'source': self.name,
                'reason': 'No signals generated'
            }
        
        # Aggregate signals
        total_weight = sum(weights)
        if total_weight == 0:
            total_weight = 1.0
        
        weighted_score = 0.0
        weighted_confidence = 0.0
        
        for signal, weight in zip(signals, weights):
            signal_val = 1 if signal['signal'] == Signal.BUY else (-1 if signal['signal'] == Signal.SELL else 0)
            confidence = signal.get('confidence', 0.5)
            
            weighted_score += signal_val * confidence * weight
            weighted_confidence += confidence * weight
        
        weighted_score /= total_weight
        weighted_confidence /= total_weight
        
        # Determine final signal (lowered threshold)
        if weighted_score > 0.2:  # Lowered from 0.3
            final_signal = Signal.BUY
        elif weighted_score < -0.2:  # Lowered from -0.3
            final_signal = Signal.SELL
        else:
            final_signal = Signal.HOLD
        
        print(f"  AGGREGATED: {final_signal.name}, weighted_score={weighted_score:.3f}, confidence={weighted_confidence:.3f}")
        print(f"=== End TechnicalAnalyst for {symbol if symbol else 'symbol'} ===\n")
        
        self._update_timestamp()
        
        # Include pattern and support/resistance info in result
        result = {
            'signal': final_signal,
            'score': weighted_score,
            'confidence': min(weighted_confidence, 1.0),
            'source': self.name,
            'reason': f'Technical analysis: {len(signals)} signals, score={weighted_score:.2f}',
            'signals_count': len(signals),
            'strategy_details': strategy_details,
            'patterns': [{'type': p['type'].value if hasattr(p['type'], 'value') else str(p['type']), 
                         'direction': p['direction'], 
                         'confidence': p.get('confidence', 0)} for p in patterns[:3]],
            'support_resistance': support_resistance,
            'stop_loss': None,
            'take_profit': None
        }
        
        # Extract stop loss and take profit from strongest pattern if available
        if patterns:
            strongest_pattern = patterns[0]
            result['stop_loss'] = strongest_pattern.get('stop_loss')
            result['take_profit'] = strongest_pattern.get('target')
        
        return result
    
    def _get_ml_signal(self, data: pd.DataFrame) -> Optional[Dict]:
        """Get signal from ML model"""
        if not self.feature_engineer:
            from ..indicators.feature_engineering import FeatureEngineer
            self.feature_engineer = FeatureEngineer()
        
        try:
            df_features = self.feature_engineer.create_features(data)
            feature_columns = self.feature_engineer.get_feature_columns(df_features)
            
            if not feature_columns or len(df_features) == 0:
                return None
            
            X = df_features[feature_columns].iloc[-1:].fillna(0)
            
            if self.ml_model.model_type == 'classification':
                prediction = self.ml_model.predict(X)[0]
                proba = self.ml_model.predict_proba(X)[0]
                confidence = max(proba)
                signal = Signal.BUY if prediction == 1 else Signal.SELL
            else:
                prediction = self.ml_model.predict(X)[0]
                signal = Signal.BUY if prediction > 0 else Signal.SELL
                confidence = min(abs(prediction) * 10, 1.0)
            
            return {
                'signal': signal,
                'confidence': confidence,
                'source': 'ML_Model'
            }
        except Exception as e:
            print(f"Error in ML signal: {e}")
            return None
    
    def _get_lstm_signal(self, data: pd.DataFrame) -> Optional[Dict]:
        """Get signal from LSTM model"""
        if not self.feature_engineer:
            from ..indicators.feature_engineering import FeatureEngineer
            self.feature_engineer = FeatureEngineer()
        
        try:
            df_features = self.feature_engineer.create_features(data)
            feature_columns = self.feature_engineer.get_feature_columns(df_features)
            
            if not feature_columns or len(df_features) < 60:
                return None
            
            X = df_features[feature_columns].tail(60).copy()
            
            # Clean data: replace infinity and NaN
            X = X.replace([np.inf, -np.inf], np.nan)
            # Fill NaN with median, then 0
            for col in X.columns:
                if X[col].isnull().any():
                    median_val = X[col].median()
                    if pd.isna(median_val):
                        X[col] = X[col].fillna(0)
                    else:
                        X[col] = X[col].fillna(median_val)
                        X[col] = X[col].fillna(0)  # Final fallback
            
            prediction = self.lstm_model.predict(X)[0]
            
            signal = Signal.BUY if prediction > 0 else Signal.SELL
            confidence = min(abs(prediction) * 10, 1.0)
            
            return {
                'signal': signal,
                'confidence': confidence,
                'source': 'LSTM_Model'
            }
        except Exception as e:
            print(f"Error in LSTM signal: {e}")
            return None


class SentimentAgent(BaseAgent):
    """Sentiment analysis agent using LLMs and news"""
    
    def __init__(self, config: Optional[Dict] = None, sentiment_analyzer: Optional[SentimentAnalyzer] = None,
                 news_fetcher: Optional[NewsFetcher] = None):
        """
        Initialize sentiment agent
        
        Args:
            config: Agent configuration
            sentiment_analyzer: SentimentAnalyzer instance
            news_fetcher: NewsFetcher instance
        """
        super().__init__("SentimentAgent", config)
        self.sentiment_analyzer = sentiment_analyzer
        self.news_fetcher = news_fetcher
    
    def analyze(self, symbol: str, news_items: Optional[List[Dict]] = None) -> Dict:
        """
        Analyze sentiment for a symbol with fallback to market-based sentiment
        
        Args:
            symbol: Trading symbol
            news_items: Optional pre-fetched news items
        
        Returns:
            Dictionary with sentiment signal and score
        """
        if not self.enabled:
            return {
                'signal': Signal.HOLD,
                'score': 0.0,
                'confidence': 0.0,
                'source': self.name,
                'reason': 'Agent disabled'
            }
        
        try:
            # Fetch news if not provided or if empty list was passed
            if (news_items is None or (isinstance(news_items, list) and len(news_items) == 0)) and self.news_fetcher:
                try:
                    news_items = self.news_fetcher.fetch_news_for_symbol(symbol)
                    print(f"SentimentAgent: Fetched {len(news_items) if news_items else 0} news items for {symbol}")
                except Exception as e:
                    print(f"SentimentAgent: Error fetching news for {symbol}: {e}")
                    news_items = None
            
            # If no news available (None or empty list), use fallback sentiment based on market behavior
            if not news_items or (isinstance(news_items, list) and len(news_items) == 0):
                print(f"SentimentAgent: No news available for {symbol}, using fallback neutral sentiment")
                # Return weak neutral signal instead of 0 confidence
                # This allows other agents to drive decisions without sentiment blocking
                return {
                    'signal': Signal.HOLD,
                    'score': 0.0,
                    'confidence': 0.3,  # Moderate confidence in neutral stance
                    'source': self.name,
                    'reason': 'No news - neutral sentiment (fallback)',
                    'news_count': 0
                }
            
            # Analyze sentiment with available news
            if self.sentiment_analyzer:
                try:
                    sentiment_result = self.sentiment_analyzer.analyze_news(news_items)
                    sentiment_score = sentiment_result.get('sentiment', 0.0)
                    confidence = sentiment_result.get('confidence', 0.0)
                    
                    # If confidence is very low (likely fallback was used), ensure minimum usable confidence
                    if confidence < 0.2:
                        confidence = 0.3  # Minimum confidence for fallback analysis
                        print(f"SentimentAgent: Low confidence from analyzer ({sentiment_result.get('confidence', 0.0):.3f}), using minimum {confidence:.3f}")
                    
                    print(f"SentimentAgent: Analyzer result for {symbol}: score={sentiment_score:.3f}, confidence={confidence:.3f}")
                    if 'reasoning' in sentiment_result and 'fallback' in sentiment_result.get('reasoning', '').lower():
                        print(f"SentimentAgent: Note - Using fallback analysis (LLM unavailable)")
                except Exception as e:
                    print(f"SentimentAgent: Error in sentiment analyzer: {e}, using simple fallback")
                    sentiment_score = self._simple_sentiment(news_items)
                    confidence = 0.4
            else:
                # Fallback: simple keyword-based sentiment
                print(f"SentimentAgent: Using simple keyword sentiment for {symbol}")
                sentiment_score = self._simple_sentiment(news_items)
                confidence = 0.4
            
            # Convert to signal with graduated confidence
            threshold = self.config.get('sentiment_threshold', 0.3)
            
            if sentiment_score > threshold:
                signal = Signal.BUY
                # Boost confidence for strong sentiment
                if sentiment_score > threshold * 2:
                    confidence = min(confidence * 1.2, 0.85)
            elif sentiment_score < -threshold:
                signal = Signal.SELL
                # Boost confidence for strong sentiment
                if sentiment_score < -threshold * 2:
                    confidence = min(confidence * 1.2, 0.85)
            else:
                signal = Signal.HOLD
                # For neutral sentiment, provide moderate confidence
                confidence = max(confidence, 0.3)
            
            self._update_timestamp()
            
            result = {
                'signal': signal,
                'score': sentiment_score,
                'confidence': confidence,
                'source': self.name,
                'reason': f'Sentiment: {len(news_items)} articles, score={sentiment_score:.2f}',
                'news_count': len(news_items)
            }
            print(f"SentimentAgent: Final result for {symbol}: {signal}, confidence={confidence:.3f}")
            return result
        
        except Exception as e:
            print(f"SentimentAgent: Unexpected error in sentiment analysis for {symbol}: {e}")
            # Return neutral with moderate confidence rather than 0
            return {
                'signal': Signal.HOLD,
                'score': 0.0,
                'confidence': 0.3,
                'source': self.name,
                'reason': f'Error (using fallback neutral): {str(e)}'
            }
    
    def _simple_sentiment(self, news_items: List[Dict]) -> float:
        """Simple keyword-based sentiment fallback"""
        bullish_keywords = ['bullish', 'up', 'rise', 'gain', 'positive', 'buy', 'strong']
        bearish_keywords = ['bearish', 'down', 'fall', 'drop', 'negative', 'sell', 'weak']
        
        total_score = 0.0
        for item in news_items:
            text = (item.get('title', '') + ' ' + item.get('content', '')).lower()
            bullish_count = sum(1 for word in bullish_keywords if word in text)
            bearish_count = sum(1 for word in bearish_keywords if word in text)
            
            if bullish_count > bearish_count:
                total_score += 0.2
            elif bearish_count > bullish_count:
                total_score -= 0.2
        
        return max(-1.0, min(1.0, total_score / len(news_items) if news_items else 0.0))


class FundamentalAgent(BaseAgent):
    """Fundamental analysis agent for macro and stock fundamentals"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize fundamental agent
        
        Args:
            config: Agent configuration
        """
        super().__init__("FundamentalAgent", config)
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.fred_api_key = os.getenv('FRED_API_KEY')
    
    def analyze(self, symbol: str, asset_class: str = 'crypto', news_items: Optional[List[Dict]] = None) -> Dict:
        """
        Perform fundamental analysis
        
        Args:
            symbol: Trading symbol
            asset_class: Asset class (crypto, forex, stocks)
            news_items: Optional pre-fetched news items (used for crypto/forex analysis)
        
        Returns:
            Dictionary with fundamental bias and score
        """
        if not self.enabled:
            return {
                'signal': Signal.HOLD,
                'score': 0.0,
                'confidence': 0.0,
                'source': self.name,
                'reason': 'Agent disabled'
            }
        
        try:
            if asset_class == 'stocks':
                return self._analyze_stock_fundamentals(symbol)
            elif asset_class in ['forex', 'crypto']:
                return self._analyze_macro_fundamentals(symbol, asset_class, news_items=news_items)
            else:
                return {
                    'signal': Signal.HOLD,
                    'score': 0.0,
                    'confidence': 0.0,
                    'source': self.name,
                    'reason': f'Unknown asset class: {asset_class}'
                }
        except Exception as e:
            print(f"Error in fundamental analysis: {e}")
            return {
                'signal': Signal.HOLD,
                'score': 0.0,
                'confidence': 0.0,
                'source': self.name,
                'reason': f'Error: {str(e)}'
            }
    
    def _analyze_stock_fundamentals(self, symbol: str) -> Dict:
        """
        Enhanced stock fundamental analysis based on Investopedia principles
        - Evaluates intrinsic value through financial metrics
        - Analyzes financial statements, earnings, sales
        - Compares to industry and market
        """
        try:
            # Clean symbol (remove /USD etc)
            clean_symbol = symbol.split('/')[0] if '/' in symbol else symbol
            
            # Use yfinance for comprehensive fundamental data
            ticker = yf.Ticker(clean_symbol)
            info = ticker.info
            
            # === VALUATION METRICS (Intrinsic Value Assessment) ===
            pe_ratio = info.get('trailingPE', None)
            forward_pe = info.get('forwardPE', None)
            peg_ratio = info.get('pegRatio', None)
            price_to_book = info.get('priceToBook', None)
            price_to_sales = info.get('priceToSalesTrailing12Months', None)
            enterprise_value = info.get('enterpriseValue', None)
            market_cap = info.get('marketCap', None)
            
            # === PROFITABILITY METRICS ===
            profit_margin = info.get('profitMargins', None)
            operating_margin = info.get('operatingMargins', None)
            return_on_equity = info.get('returnOnEquity', None)
            return_on_assets = info.get('returnOnAssets', None)
            gross_margin = info.get('grossMargins', None)
            
            # === GROWTH METRICS ===
            revenue_growth = info.get('revenueGrowth', None)
            earnings_growth = info.get('earningsGrowth', None)
            earnings_quarterly_growth = info.get('earningsQuarterlyGrowth', None)
            
            # === FINANCIAL HEALTH METRICS ===
            debt_to_equity = info.get('debtToEquity', None)
            current_ratio = info.get('currentRatio', None)
            quick_ratio = info.get('quickRatio', None)
            total_cash = info.get('totalCash', None)
            total_debt = info.get('totalDebt', None)
            free_cashflow = info.get('freeCashflow', None)
            
            # === DIVIDEND METRICS ===
            dividend_yield = info.get('dividendYield', None)
            payout_ratio = info.get('payoutRatio', None)
            
            # Calculate comprehensive fundamental score
            score = 0.0
            confidence_factors = []
            metrics_analyzed = 0
            
            # === VALUATION ANALYSIS (40% weight) ===
            valuation_score = 0.0
            
            # P/E Ratio Analysis (compare to market average ~20)
            if pe_ratio:
                metrics_analyzed += 1
                if pe_ratio < 15:
                    valuation_score += 0.4  # Undervalued
                elif pe_ratio < 20:
                    valuation_score += 0.2  # Fairly valued
                elif pe_ratio > 30:
                    valuation_score -= 0.3  # Overvalued
                confidence_factors.append(0.15)
            
            # Forward P/E (more predictive)
            if forward_pe and pe_ratio:
                metrics_analyzed += 1
                if forward_pe < pe_ratio * 0.9:  # Earnings growing
                    valuation_score += 0.2
                confidence_factors.append(0.1)
            
            # PEG Ratio (growth-adjusted valuation)
            if peg_ratio:
                metrics_analyzed += 1
                if peg_ratio < 1.0:
                    valuation_score += 0.5  # Excellent value
                elif peg_ratio < 1.5:
                    valuation_score += 0.2
                elif peg_ratio > 2.5:
                    valuation_score -= 0.4
                confidence_factors.append(0.2)
            
            # Price-to-Book
            if price_to_book:
                metrics_analyzed += 1
                if price_to_book < 1.5:
                    valuation_score += 0.3  # Undervalued
                elif price_to_book > 3.0:
                    valuation_score -= 0.2
                confidence_factors.append(0.1)
            
            # Price-to-Sales
            if price_to_sales:
                metrics_analyzed += 1
                if price_to_sales < 2.0:
                    valuation_score += 0.2
                elif price_to_sales > 5.0:
                    valuation_score -= 0.2
                confidence_factors.append(0.1)
            
            score += valuation_score * 0.4  # 40% weight
            
            # === PROFITABILITY ANALYSIS (30% weight) ===
            profitability_score = 0.0
            
            if profit_margin:
                metrics_analyzed += 1
                if profit_margin > 0.20:
                    profitability_score += 0.4  # Excellent
                elif profit_margin > 0.10:
                    profitability_score += 0.2
                elif profit_margin < 0:
                    profitability_score -= 0.5  # Losing money
                confidence_factors.append(0.15)
            
            if operating_margin:
                metrics_analyzed += 1
                if operating_margin > 0.15:
                    profitability_score += 0.3
                confidence_factors.append(0.1)
            
            if return_on_equity:
                metrics_analyzed += 1
                if return_on_equity > 0.20:
                    profitability_score += 0.3  # Excellent ROE
                elif return_on_equity > 0.15:
                    profitability_score += 0.15
                elif return_on_equity < 0.05:
                    profitability_score -= 0.3
                confidence_factors.append(0.15)
            
            if return_on_assets:
                metrics_analyzed += 1
                if return_on_assets > 0.10:
                    profitability_score += 0.2
                confidence_factors.append(0.1)
            
            score += profitability_score * 0.3  # 30% weight
            
            # === GROWTH ANALYSIS (20% weight) ===
            growth_score = 0.0
            
            if revenue_growth:
                metrics_analyzed += 1
                if revenue_growth > 0.15:
                    growth_score += 0.4  # Strong growth
                elif revenue_growth > 0.05:
                    growth_score += 0.2
                elif revenue_growth < -0.10:
                    growth_score -= 0.4  # Declining revenue
                confidence_factors.append(0.15)
            
            if earnings_growth:
                metrics_analyzed += 1
                if earnings_growth > 0.20:
                    growth_score += 0.4
                elif earnings_growth > 0.10:
                    growth_score += 0.2
                elif earnings_growth < -0.10:
                    growth_score -= 0.4
                confidence_factors.append(0.15)
            
            score += growth_score * 0.2  # 20% weight
            
            # === FINANCIAL HEALTH ANALYSIS (10% weight) ===
            health_score = 0.0
            
            if debt_to_equity:
                metrics_analyzed += 1
                if debt_to_equity < 0.5:
                    health_score += 0.3  # Low debt
                elif debt_to_equity > 2.0:
                    health_score -= 0.3  # High debt
                confidence_factors.append(0.1)
            
            if current_ratio:
                metrics_analyzed += 1
                if current_ratio > 2.0:
                    health_score += 0.2  # Strong liquidity
                elif current_ratio < 1.0:
                    health_score -= 0.3  # Liquidity concerns
                confidence_factors.append(0.1)
            
            if free_cashflow and free_cashflow > 0:
                metrics_analyzed += 1
                health_score += 0.2  # Positive cash flow
                confidence_factors.append(0.1)
            
            score += health_score * 0.1  # 10% weight
            
            # Normalize score
            score = max(-1.0, min(1.0, score))
            
            # Calculate confidence based on metrics analyzed
            base_confidence = 0.3
            if metrics_analyzed >= 8:
                confidence = min(0.7, base_confidence + (metrics_analyzed - 5) * 0.05)
            elif metrics_analyzed >= 5:
                confidence = min(0.5, base_confidence + (metrics_analyzed - 3) * 0.04)
            else:
                confidence = base_confidence
            
            # Determine signal (fundamental is more long-term)
            if score > 0.3:
                signal = Signal.BUY
            elif score < -0.3:
                signal = Signal.SELL
            else:
                signal = Signal.HOLD
            
            self._update_timestamp()
            
            return {
                'signal': signal,
                'score': score,
                'confidence': confidence,
                'source': self.name,
                'reason': f'Fundamental analysis: {metrics_analyzed} metrics, score={score:.2f}',
                'time_horizon': 'long_term',
                'metrics': {
                    'pe_ratio': pe_ratio,
                    'forward_pe': forward_pe,
                    'peg_ratio': peg_ratio,
                    'price_to_book': price_to_book,
                    'profit_margin': profit_margin,
                    'roe': return_on_equity,
                    'revenue_growth': revenue_growth,
                    'earnings_growth': earnings_growth,
                    'debt_to_equity': debt_to_equity,
                    'metrics_count': metrics_analyzed
                },
                'valuation_score': valuation_score,
                'profitability_score': profitability_score,
                'growth_score': growth_score,
                'health_score': health_score
            }
        
        except Exception as e:
            print(f"Error analyzing stock fundamentals: {e}")
            return {
                'signal': Signal.HOLD,
                'score': 0.0,
                'confidence': 0.0,
                'source': self.name,
                'reason': f'Error: {str(e)}'
            }
    
    def _analyze_macro_fundamentals(self, symbol: str, asset_class: str, news_items: Optional[List[Dict]] = None) -> Dict:
        """
        Analyze macro economic fundamentals for forex/crypto
        
        Uses news items to extract fundamental insights when available.
        NOTE: Without real-time fundamental data APIs, this agent uses conservative
        heuristics. It should NOT override strong technical signals.
        """
        try:
            print(f"FundamentalAgent: Analyzing {asset_class} fundamentals for {symbol}")
            
            # For crypto/forex without real-time macro data, use VERY conservative approach
            # Fundamental analysis is long-term, so we should be neutral unless we have real data
            score = 0.0
            confidence = 0.2  # LOW confidence - we don't have real fundamental data
            signal = Signal.HOLD
            reason = "No real-time fundamental data available - neutral stance"
            
            # === CRYPTO-SPECIFIC FUNDAMENTAL SIGNALS ===
            if asset_class == 'crypto':
                print(f"FundamentalAgent: Crypto analysis for {symbol} - using conservative approach")
                
                # Use news to extract fundamental insights for crypto
                news_analysis_used = False
                if news_items and len(news_items) > 0:
                    print(f"FundamentalAgent: Analyzing {len(news_items)} news articles for fundamental insights")
                    news_score = self._analyze_crypto_news_fundamentals(symbol, news_items)
                    if news_score is not None:
                        # News provides some fundamental insight - boost confidence
                        score = news_score
                        confidence = min(0.4, 0.2 + (len(news_items) * 0.02))  # Boost confidence based on news volume
                        if score > 0.1:
                            signal = Signal.BUY
                            reason = f"Crypto: Positive fundamental news detected ({len(news_items)} articles)"
                        elif score < -0.1:
                            signal = Signal.SELL
                            reason = f"Crypto: Negative fundamental news detected ({len(news_items)} articles)"
                        else:
                            signal = Signal.HOLD
                            reason = f"Crypto: Mixed/neutral fundamental news ({len(news_items)} articles)"
                        print(f"FundamentalAgent: News-based fundamental score: {score:.3f}, confidence: {confidence:.3f}")
                        news_analysis_used = True
                
                # Without real fundamental data (market cap trends, adoption metrics, etc.),
                # we should be very conservative. Fundamental analysis requires:
                # - Market cap trends
                # - Adoption metrics
                # - Network activity
                # - Regulatory environment
                # - Institutional flows
                #
                # Since we don't have this data, return neutral with low confidence
                # This allows technical analysis to drive decisions
                # Only use default logic if news analysis didn't provide insights
                if not news_analysis_used:
                    if 'BTC' in symbol.upper():
                        # BTC is the market leader, but without real data, stay neutral
                        # Only provide very weak bias if we had adoption/flow data
                        score = 0.0  # Changed from 0.15 - too aggressive without data
                        confidence = 0.2  # Lowered from 0.35 - we don't have real data
                        signal = Signal.HOLD  # Changed from BUY - don't override technical
                        reason = "BTC: No real-time fundamental data - neutral (technical should drive)"
                    
                    elif 'ETH' in symbol.upper():
                        score = 0.0
                        confidence = 0.2
                        signal = Signal.HOLD
                        reason = "ETH: No real-time fundamental data - neutral"
                    
                    elif 'SOL' in symbol.upper() or 'ADA' in symbol.upper():
                        score = 0.0
                        confidence = 0.2
                        signal = Signal.HOLD
                        reason = "Alt coin: No real-time fundamental data - neutral"
                    
                    else:
                        score = 0.0
                        confidence = 0.2
                        signal = Signal.HOLD
                        reason = "Unknown crypto - no fundamental data available"
            
            # === FOREX-SPECIFIC FUNDAMENTAL SIGNALS ===
            elif asset_class == 'forex':
                print(f"FundamentalAgent: Forex analysis for {symbol} - using conservative approach")
                
                # Forex fundamentals require:
                # - Central bank interest rates
                # - Economic indicators (GDP, inflation, employment)
                # - Political stability
                # - Trade balances
                #
                # Without this data, stay neutral
                
                if 'EUR/USD' in symbol.upper() or 'EURUSD' in symbol.upper():
                    score = 0.0
                    confidence = 0.2
                    signal = Signal.HOLD
                    reason = "EUR/USD: No real-time rate/economic data - neutral"
                
                elif 'GBP/USD' in symbol.upper() or 'GBPUSD' in symbol.upper():
                    score = 0.0
                    confidence = 0.2
                    signal = Signal.HOLD
                    reason = "GBP/USD: No real-time economic data - neutral"
                
                elif 'USD/JPY' in symbol.upper() or 'USDJPY' in symbol.upper():
                    score = 0.0
                    confidence = 0.2
                    signal = Signal.HOLD
                    reason = "USD/JPY: No real-time rate differential data - neutral"
                
                else:
                    score = 0.0
                    confidence = 0.2
                    signal = Signal.HOLD
                    reason = "Forex pair - no fundamental data available"
            
            # === GENERAL FALLBACK ===
            else:
                print(f"FundamentalAgent: Unknown asset class {asset_class}, using neutral")
                score = 0.0
                confidence = 0.2
                signal = Signal.HOLD
                reason = f"Unknown asset class: {asset_class} - no data"
            
            self._update_timestamp()
            
            result = {
                'signal': signal,
                'score': score,
                'confidence': confidence,
                'source': self.name,
                'reason': reason,
                'time_horizon': 'long_term'
            }
            print(f"FundamentalAgent: Result for {symbol}: {signal}, score={score:.3f}, confidence={confidence:.3f}")
            if news_items and len(news_items) > 0:
                print(f"FundamentalAgent: Used {len(news_items)} news articles for fundamental analysis")
            else:
                print(f"FundamentalAgent: NOTE - Low confidence due to lack of real-time fundamental data")
            return result
        
        except Exception as e:
            print(f"FundamentalAgent: Error analyzing macro fundamentals for {symbol}: {e}")
            return {
                'signal': Signal.HOLD,
                'score': 0.0,
                'confidence': 0.2,
                'source': self.name,
                'reason': f'Error: {str(e)}',
                'time_horizon': 'long_term'
            }
    
    def _analyze_crypto_news_fundamentals(self, symbol: str, news_items: List[Dict]) -> Optional[float]:
        """
        Extract fundamental insights from crypto news articles
        
        Looks for:
        - Regulatory news (positive/negative)
        - Adoption news (partnerships, integrations)
        - Technical upgrades (network improvements)
        - Market structure changes (ETF approvals, institutional adoption)
        - Security incidents (hacks, vulnerabilities)
        
        Returns:
            Fundamental score (-1 to 1) or None if no clear signal
        """
        if not news_items:
            return None
        
        positive_keywords = [
            'adoption', 'partnership', 'integration', 'upgrade', 'launch', 'approval',
            'etf', 'institutional', 'investment', 'growth', 'expansion', 'milestone',
            'breakthrough', 'innovation', 'success', 'record', 'surge', 'rally',
            'regulatory clarity', 'approval', 'greenlight', 'endorsement'
        ]
        
        negative_keywords = [
            'ban', 'regulation', 'crackdown', 'restriction', 'warning', 'risk',
            'hack', 'exploit', 'vulnerability', 'security breach', 'theft',
            'decline', 'drop', 'crash', 'concern', 'uncertainty', 'rejection',
            'lawsuit', 'investigation', 'fine', 'penalty'
        ]
        
        fundamental_score = 0.0
        article_count = 0
        
        for article in news_items:
            title = article.get('title', '').lower()
            content = article.get('content', '').lower()
            text = f"{title} {content}"
            
            # Count positive and negative keywords
            positive_count = sum(1 for kw in positive_keywords if kw in text)
            negative_count = sum(1 for kw in negative_keywords if kw in text)
            
            if positive_count > 0 or negative_count > 0:
                article_count += 1
                # Weight by article sentiment if available
                article_sentiment = article.get('sentiment_score', 0.0)
                if article_sentiment != 0.0:
                    # Use sentiment score if available
                    fundamental_score += article_sentiment * 0.3
                else:
                    # Use keyword-based scoring
                    if positive_count > negative_count:
                        fundamental_score += 0.2
                    elif negative_count > positive_count:
                        fundamental_score -= 0.2
        
        if article_count == 0:
            return None
        
        # Normalize score
        avg_score = fundamental_score / article_count
        # Clamp to [-1, 1] range
        return max(-1.0, min(1.0, avg_score))

