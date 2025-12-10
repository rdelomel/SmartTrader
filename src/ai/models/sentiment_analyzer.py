"""Sentiment analysis using LLM"""

import os
from typing import Dict, Optional, List
from dotenv import load_dotenv
import requests
from datetime import datetime
import time
import json

load_dotenv()


class SentimentAnalyzer:
    """Sentiment analyzer using OpenRouter, OpenAI, or other LLM"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize sentiment analyzer
        
        Config parameters:
            provider: 'openrouter', 'openai', or 'local_llm'
            model: Model name (e.g., 'openai/gpt-3.5-turbo' for OpenRouter, 'gpt-3.5-turbo' for OpenAI)
            max_tokens: Maximum tokens for response
            temperature: Temperature for generation
        """
        self.config = config or {}
        self.provider = self.config.get('provider', 'openrouter')
        self.model = self.config.get('model', 'openai/gpt-3.5-turbo')
        self.max_tokens = self.config.get('max_tokens', 150)
        self.temperature = self.config.get('temperature', 0.3)
        
        if self.provider == 'openrouter':
            self.api_key = os.getenv('OPENROUTER_API_KEY')
            self.base_url = 'https://openrouter.ai/api/v1'
            if not self.api_key:
                print("Warning: OPENROUTER_API_KEY not found in environment")
        elif self.provider == 'openai':
            self.api_key = os.getenv('OPENAI_API_KEY')
            self.base_url = 'https://api.openai.com/v1'
            if not self.api_key:
                print("Warning: OPENAI_API_KEY not found in environment")
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze sentiment of text
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with sentiment score and analysis
            {
                'sentiment': float (-1 to 1),
                'confidence': float (0 to 1),
                'reasoning': str
            }
        """
        if self.provider == 'openrouter':
            return self._analyze_openrouter(text)
        elif self.provider == 'openai':
            return self._analyze_openai(text)
        else:
            # Placeholder for local LLM implementation
            return self._analyze_placeholder(text)
    
    def _analyze_openrouter(self, text: str) -> Dict:
        """Analyze sentiment using OpenRouter API with retry logic"""
        if not self.api_key:
            print("OpenRouter: No API key, using keyword-based fallback")
            return self._analyze_placeholder(text)
        
        # Retry configuration
        max_retries = 3
        retry_delays = [1, 2, 4]  # Exponential backoff: 1s, 2s, 4s
        retryable_status_codes = [503, 502, 504, 429]  # Service unavailable, bad gateway, gateway timeout, rate limit
        
        prompt = f"""Analyze the sentiment of the following financial news/text and provide a sentiment score from -1 (very bearish) to +1 (very bullish), where 0 is neutral.

Text: {text}

Respond in JSON format with:
- "sentiment": float between -1 and 1
- "confidence": float between 0 and 1
- "reasoning": brief explanation

JSON:"""
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://github.com/SmartTrader',  # Optional but recommended
            'X-Title': 'SmartTrader AI Trading Agent'  # Optional but recommended
        }
        
        payload = {
            'model': self.model,
            'messages': [
                {"role": "system", "content": "You are a financial sentiment analyst. Provide accurate, objective sentiment analysis."},
                {"role": "user", "content": prompt}
            ],
            'max_tokens': self.max_tokens,
            'temperature': self.temperature
        }
        
        last_exception = None
        
        # Retry loop
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    delay = retry_delays[min(attempt - 1, len(retry_delays) - 1)]
                    print(f"OpenRouter: Retry attempt {attempt + 1}/{max_retries} after {delay}s delay...")
                    time.sleep(delay)
                
                response = requests.post(
                    f'{self.base_url}/chat/completions',
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                # Check for retryable errors
                if response.status_code in retryable_status_codes:
                    if attempt < max_retries - 1:
                        print(f"OpenRouter: Received {response.status_code} (retryable), will retry...")
                        last_exception = requests.exceptions.HTTPError(f"{response.status_code} {response.reason}")
                        continue
                    else:
                        # Last attempt failed
                        print(f"OpenRouter: Final attempt failed with {response.status_code}")
                        response.raise_for_status()
                
                # Check for other errors
                response.raise_for_status()
                data = response.json()
                
                # Parse response
                content = data['choices'][0]['message']['content'].strip()
                
                # Try to extract JSON
                try:
                    # Remove markdown code blocks if present
                    if '```json' in content:
                        content = content.split('```json')[1].split('```')[0]
                    elif '```' in content:
                        content = content.split('```')[1].split('```')[0]
                    
                    result = json.loads(content)
                    print(f"OpenRouter: Successfully analyzed sentiment: {result.get('sentiment', 0.0):.3f}")
                    return {
                        'sentiment': float(result.get('sentiment', 0.0)),
                        'confidence': float(result.get('confidence', 0.5)),
                        'reasoning': result.get('reasoning', '')
                    }
                except json.JSONDecodeError:
                    # Fallback: try to extract sentiment from text
                    sentiment = 0.0
                    if 'bearish' in content.lower() or 'negative' in content.lower():
                        sentiment = -0.5
                    elif 'bullish' in content.lower() or 'positive' in content.lower():
                        sentiment = 0.5
                    
                    print(f"OpenRouter: JSON parse failed, extracted sentiment from text: {sentiment:.3f}")
                    return {
                        'sentiment': sentiment,
                        'confidence': 0.5,
                        'reasoning': content
                    }
                    
            except requests.exceptions.Timeout:
                last_exception = requests.exceptions.Timeout("Request timeout")
                if attempt < max_retries - 1:
                    print(f"OpenRouter: Request timeout, will retry...")
                    continue
                else:
                    print(f"OpenRouter: All retry attempts failed due to timeout")
                    break
                    
            except requests.exceptions.HTTPError as e:
                last_exception = e
                status_code = e.response.status_code if hasattr(e, 'response') and e.response else None
                
                if status_code in retryable_status_codes and attempt < max_retries - 1:
                    # Already handled above, but catch here for safety
                    continue
                else:
                    # Non-retryable error
                    print(f"OpenRouter: HTTP error {status_code}: {e}")
                    break
                    
            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < max_retries - 1:
                    print(f"OpenRouter: Request error (retryable): {e}, will retry...")
                    continue
                else:
                    print(f"OpenRouter: All retry attempts failed: {e}")
                    break
                    
            except Exception as e:
                last_exception = e
                print(f"OpenRouter: Unexpected error: {e}")
                break
        
        # All retries exhausted or non-retryable error - use fallback
        error_msg = str(last_exception) if last_exception else "Unknown error"
        print(f"Error analyzing sentiment with OpenRouter: {error_msg}")
        print(f"OpenRouter: Falling back to keyword-based sentiment analysis")
        
        # Use keyword-based fallback with moderate confidence
        fallback_result = self._analyze_placeholder(text)
        fallback_result['reasoning'] = f'OpenRouter unavailable ({error_msg}), using keyword-based analysis'
        return fallback_result
    
    def _analyze_openai(self, text: str) -> Dict:
        """Analyze sentiment using OpenAI API (direct)"""
        try:
            if not self.api_key:
                return self._analyze_placeholder(text)
            
            prompt = f"""Analyze the sentiment of the following financial news/text and provide a sentiment score from -1 (very bearish) to +1 (very bullish), where 0 is neutral.

Text: {text}

Respond in JSON format with:
- "sentiment": float between -1 and 1
- "confidence": float between 0 and 1
- "reasoning": brief explanation

JSON:"""
            
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': self.model,
                'messages': [
                    {"role": "system", "content": "You are a financial sentiment analyst. Provide accurate, objective sentiment analysis."},
                    {"role": "user", "content": prompt}
                ],
                'max_tokens': self.max_tokens,
                'temperature': self.temperature
            }
            
            response = requests.post(
                f'{self.base_url}/chat/completions',
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            # Parse response
            content = data['choices'][0]['message']['content'].strip()
            
            # Try to extract JSON
            try:
                # Remove markdown code blocks if present
                if '```json' in content:
                    content = content.split('```json')[1].split('```')[0]
                elif '```' in content:
                    content = content.split('```')[1].split('```')[0]
                
                result = json.loads(content)
                return {
                    'sentiment': float(result.get('sentiment', 0.0)),
                    'confidence': float(result.get('confidence', 0.5)),
                    'reasoning': result.get('reasoning', '')
                }
            except json.JSONDecodeError:
                # Fallback: try to extract sentiment from text
                sentiment = 0.0
                if 'bearish' in content.lower() or 'negative' in content.lower():
                    sentiment = -0.5
                elif 'bullish' in content.lower() or 'positive' in content.lower():
                    sentiment = 0.5
                
                return {
                    'sentiment': sentiment,
                    'confidence': 0.5,
                    'reasoning': content
                }
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if hasattr(e, 'response') and e.response else None
            print(f"OpenAI: HTTP error {status_code}: {e}")
            print(f"OpenAI: Falling back to keyword-based sentiment analysis")
            fallback_result = self._analyze_placeholder(text)
            fallback_result['reasoning'] = f'OpenAI API error ({status_code}), using keyword-based analysis'
            return fallback_result
        except requests.exceptions.RequestException as e:
            print(f"OpenAI: Request error: {e}")
            print(f"OpenAI: Falling back to keyword-based sentiment analysis")
            fallback_result = self._analyze_placeholder(text)
            fallback_result['reasoning'] = f'OpenAI API unavailable ({str(e)}), using keyword-based analysis'
            return fallback_result
        except Exception as e:
            print(f"OpenAI: Unexpected error: {e}")
            print(f"OpenAI: Falling back to keyword-based sentiment analysis")
            fallback_result = self._analyze_placeholder(text)
            fallback_result['reasoning'] = f'OpenAI error ({str(e)}), using keyword-based analysis'
            return fallback_result
    
    def _analyze_placeholder(self, text: str) -> Dict:
        """Keyword-based sentiment analysis (fallback when LLM unavailable)"""
        # Enhanced keyword-based sentiment (fallback)
        text_lower = text.lower()
        
        # Strong bullish indicators
        strong_bullish = ['surge', 'rally', 'soar', 'skyrocket', 'breakout', 'bullish', 'strong buy', 'upgrade']
        # Moderate bullish indicators
        bullish_keywords = ['up', 'rise', 'gain', 'positive', 'buy', 'strong', 'growth', 'profit', 'beat', 'exceed']
        # Strong bearish indicators
        strong_bearish = ['crash', 'plunge', 'collapse', 'downgrade', 'bearish', 'strong sell', 'warning']
        # Moderate bearish indicators
        bearish_keywords = ['down', 'fall', 'drop', 'negative', 'sell', 'weak', 'loss', 'miss', 'decline', 'concern']
        
        strong_bullish_count = sum(1 for word in strong_bullish if word in text_lower)
        bullish_count = sum(1 for word in bullish_keywords if word in text_lower)
        strong_bearish_count = sum(1 for word in strong_bearish if word in text_lower)
        bearish_count = sum(1 for word in bearish_keywords if word in text_lower)
        
        # Calculate sentiment with stronger weight for strong indicators
        bullish_score = strong_bullish_count * 0.4 + bullish_count * 0.15
        bearish_score = strong_bearish_count * 0.4 + bearish_count * 0.15
        
        if bullish_score > bearish_score:
            sentiment = min(0.6, bullish_score)
            confidence = min(0.5, 0.3 + (bullish_score - bearish_score) * 0.1)
        elif bearish_score > bullish_score:
            sentiment = max(-0.6, -bearish_score)
            confidence = min(0.5, 0.3 + (bearish_score - bullish_score) * 0.1)
        else:
            sentiment = 0.0
            confidence = 0.3
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'reasoning': f'Keyword-based analysis (bullish: {bullish_score:.2f}, bearish: {bearish_score:.2f})'
        }
    
    def analyze_news(self, news_items: List[Dict]) -> Dict:
        """
        Analyze sentiment of multiple news items
        
        Args:
            news_items: List of news dictionaries with 'title' and/or 'content'
        
        Returns:
            Aggregated sentiment analysis
        """
        if not news_items:
            return {
                'sentiment': 0.0,
                'confidence': 0.0,
                'count': 0
            }
        
        sentiments = []
        confidences = []
        
        for item in news_items:
            text = item.get('title', '') + ' ' + item.get('content', '')
            if text.strip():
                result = self.analyze_text(text)
                sentiments.append(result['sentiment'])
                confidences.append(result['confidence'])
        
        if not sentiments:
            return {
                'sentiment': 0.0,
                'confidence': 0.0,
                'count': 0
            }
        
        # Weighted average by confidence
        total_confidence = sum(confidences)
        if total_confidence > 0:
            weighted_sentiment = sum(s * c for s, c in zip(sentiments, confidences)) / total_confidence
            avg_confidence = sum(confidences) / len(confidences)
        else:
            weighted_sentiment = sum(sentiments) / len(sentiments)
            avg_confidence = 0.0
        
        return {
            'sentiment': weighted_sentiment,
            'confidence': avg_confidence,
            'count': len(sentiments)
        }
    
    def get_sentiment_signal(self, sentiment_score: float, threshold: float = 0.3) -> Dict:
        """
        Convert sentiment score to trading signal
        
        Args:
            sentiment_score: Sentiment score (-1 to 1)
            threshold: Minimum absolute sentiment to generate signal
        
        Returns:
            Dictionary with signal information
        """
        if abs(sentiment_score) < threshold:
            return {
                'signal': 'HOLD',
                'strength': abs(sentiment_score) / threshold,
                'direction': 'neutral'
            }
        
        if sentiment_score > 0:
            return {
                'signal': 'BUY',
                'strength': min(sentiment_score / threshold, 1.0),
                'direction': 'bullish'
            }
        else:
            return {
                'signal': 'SELL',
                'strength': min(abs(sentiment_score) / threshold, 1.0),
                'direction': 'bearish'
            }

