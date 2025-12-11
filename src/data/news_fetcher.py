"""News fetching module for sentiment analysis with multiple sources"""

import feedparser
import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import time
import re
import os
import xml.etree.ElementTree as ET
from io import StringIO


class NewsFetcher:
    """
    Fetches financial news from multiple sources:
    - NewsAPI (professional news aggregation)
    - Alpha Vantage News & Sentiment (if API key available)
    - CryptoCompare News (crypto-specific)
    - RSS feeds (fallback)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize news fetcher
        
        Config parameters:
            rss_feeds: List of RSS feed URLs
            newsapi_enabled: Enable NewsAPI (requires NEWSAPI_API_KEY)
            alphavantage_enabled: Enable Alpha Vantage (requires ALPHA_VANTAGE_API_KEY)
            cryptocompare_enabled: Enable CryptoCompare (free, no key needed)
            update_frequency_minutes: How often to fetch news
            max_articles_per_feed: Maximum articles to fetch per feed
            keywords: Keywords to filter relevant news
        """
        self.config = config or {}
        
        # API keys from environment
        self.newsapi_key = os.getenv('NEWSAPI_API_KEY')
        self.alphavantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        
        # Enable/disable sources
        self.newsapi_enabled = self.config.get('newsapi_enabled', True) and self.newsapi_key
        self.alphavantage_enabled = self.config.get('alphavantage_enabled', False) and self.alphavantage_key
        self.cryptocompare_enabled = self.config.get('cryptocompare_enabled', True)  # Free, no key needed
        
        # Enhanced RSS feeds with crypto-specific sources
        # Updated URLs as of Dec 2025 - focusing on most reliable feeds
        # Note: Many RSS feeds have become unreliable (403, 526 errors, SSL issues)
        # Primary news sources: NewsAPI, CryptoCompare, Alpha Vantage (if configured), Reddit
        # RSS feeds are secondary/fallback sources
        self.rss_feeds = self.config.get('rss_feeds', [
            # Google News RSS (more reliable, aggregates multiple sources)
            'https://news.google.com/rss/search?q=cryptocurrency+OR+bitcoin+OR+ethereum&hl=en-US&gl=US&ceid=US:en',
            # Crypto-specific RSS feeds (most reliable for crypto news)
            'https://www.coindesk.com/arc/outboundfeeds/rss/',  # CoinDesk - usually reliable
            'https://cointelegraph.com/rss',  # CoinTelegraph - usually reliable
            'https://cryptoslate.com/feed/',  # CryptoSlate - usually reliable
            # General financial news (may have intermittent issues)
            'https://www.cnbc.com/id/100003114/device/rss/rss.html',  # CNBC Business
            # Note: Removed feeds that consistently fail:
            # - Yahoo Finance (no longer supports RSS)
            # - Reuters (frequent SSL/timeout issues)
            # - CNN (SSL issues)
            # - Bloomberg (may require authentication)
            # - Financial Times (subscription required)
            # - CoinJournal (526 errors)
            # - ForexFactory (403 forbidden)
            # - FXStreet, DailyFX (may require authentication)
        ])
        
        self.update_frequency_minutes = self.config.get('update_frequency_minutes', 15)
        self.max_articles_per_feed = self.config.get('max_articles_per_feed', 10)
        self.keywords = self.config.get('keywords', [
            'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'cryptocurrency',
            'forex', 'eur', 'gbp', 'jpy', 'usd', 'dollar', 'euro', 'pound',
            'stock', 'market', 'trading', 'invest', 'finance', 'economy'
        ])
        self.last_fetch_time = None
        self.cached_news = []
        # Per-symbol caching for better efficiency
        self.symbol_cache = {}  # symbol -> (articles, timestamp)
        self.cache_ttl_minutes = self.config.get('cache_ttl_minutes', 15)
        
        # Enable Reddit as free news source (no API key needed for public data)
        self.reddit_enabled = self.config.get('reddit_enabled', True)
        
        # Log enabled sources
        enabled_sources = []
        if self.newsapi_enabled:
            enabled_sources.append("NewsAPI")
        if self.alphavantage_enabled:
            enabled_sources.append("Alpha Vantage")
        if self.cryptocompare_enabled:
            enabled_sources.append("CryptoCompare")
        if self.reddit_enabled:
            enabled_sources.append("Reddit")
        enabled_sources.append(f"RSS ({len(self.rss_feeds)} feeds)")
        print(f"NewsFetcher: Initialized with sources: {', '.join(enabled_sources)}")
    
    def fetch_newsapi(self, symbol: str, query: Optional[str] = None) -> List[Dict]:
        """
        Fetch news from NewsAPI (https://newsapi.org/)
        Free tier: 100 requests/day, 1 request/second
        
        Args:
            symbol: Trading symbol
            query: Optional search query (defaults to symbol keywords)
        
        Returns:
            List of news articles
        """
        if not self.newsapi_enabled:
            return []
        
        try:
            symbol_keywords = self._extract_symbol_keywords(symbol)
            
            # Build query from symbol keywords
            if not query:
                # For crypto, use coin name; for forex, use currency pair
                if '/' in symbol:
                    base = symbol.split('/')[0]
                    if base.upper() in ['BTC', 'ETH', 'SOL']:
                        query = base.lower() if base.upper() == 'BTC' else symbol_keywords[0].lower()
                    else:
                        query = ' '.join(symbol_keywords[:2])
                else:
                    query = symbol
            
            # NewsAPI endpoint
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': query,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 20,
                'from': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                'apiKey': self.newsapi_key
            }
            
            print(f"NewsFetcher: Fetching from NewsAPI with query: {query}")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for article in data.get('articles', []):
                articles.append({
                    'title': article.get('title', ''),
                    'content': article.get('description', '') or article.get('content', ''),
                    'link': article.get('url', ''),
                    'published': datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00')),
                    'source': f"NewsAPI: {article.get('source', {}).get('name', 'Unknown')}"
                })
            
            print(f"NewsFetcher: NewsAPI returned {len(articles)} articles")
            return articles
            
        except Exception as e:
            print(f"NewsFetcher: Error fetching from NewsAPI: {e}")
            return []
    
    def fetch_cryptocompare(self, symbol: str) -> List[Dict]:
        """
        Fetch crypto news from CryptoCompare API (free, no key needed)
        https://min-api.cryptocompare.com/documentation
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USD')
        
        Returns:
            List of news articles
        """
        if not self.cryptocompare_enabled or '/' not in symbol:
            return []
        
        try:
            base = symbol.split('/')[0].upper()
            
            # CryptoCompare news endpoint
            url = "https://min-api.cryptocompare.com/data/v2/news/"
            params = {
                'categories': base,  # Filter by coin
                'lang': 'EN'
            }
            
            print(f"NewsFetcher: Fetching from CryptoCompare for {base}")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for article in data.get('Data', [])[:10]:  # Limit to recent
                # Parse published time (Unix timestamp)
                published = datetime.fromtimestamp(article.get('published_on', 0))
                
                articles.append({
                    'title': article.get('title', ''),
                    'content': article.get('body', ''),
                    'link': article.get('url', ''),
                    'published': published,
                    'source': f"CryptoCompare: {article.get('source', 'Unknown')}"
                })
            
            print(f"NewsFetcher: CryptoCompare returned {len(articles)} articles")
            return articles
            
        except Exception as e:
            print(f"NewsFetcher: Error fetching from CryptoCompare: {e}")
            return []
    
    def fetch_alphavantage_news(self, symbol: str) -> List[Dict]:
        """
        Fetch news from Alpha Vantage News & Sentiment API
        Requires ALPHA_VANTAGE_API_KEY
        
        Args:
            symbol: Trading symbol
        
        Returns:
            List of news articles
        """
        if not self.alphavantage_enabled:
            return []
        
        try:
            # Clean symbol for Alpha Vantage (remove /USD etc for stocks)
            clean_symbol = symbol.split('/')[0] if '/' in symbol else symbol
            
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': clean_symbol,
                'apikey': self.alphavantage_key,
                'limit': 20
            }
            
            print(f"NewsFetcher: Fetching from Alpha Vantage for {clean_symbol}")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for article in data.get('feed', []):
                # Parse published time
                published_str = article.get('time_published', '')
                try:
                    published = datetime.strptime(published_str, '%Y%m%dT%H%M%S')
                except:
                    published = datetime.now()
                
                articles.append({
                    'title': article.get('title', ''),
                    'content': article.get('summary', ''),
                    'link': article.get('url', ''),
                    'published': published,
                    'source': f"Alpha Vantage: {article.get('source', 'Unknown')}",
                    'sentiment_score': article.get('overall_sentiment_score', 0.0)  # Bonus: includes sentiment!
                })
            
            print(f"NewsFetcher: Alpha Vantage returned {len(articles)} articles")
            return articles
            
        except Exception as e:
            print(f"NewsFetcher: Error fetching from Alpha Vantage: {e}")
            return []
    
    def _sanitize_xml(self, xml_content: str) -> str:
        """
        Sanitize XML content to handle common malformed XML issues.
        Removes invalid characters and fixes common encoding issues.
        """
        if not xml_content:
            return ""
        
        # Remove invalid XML characters (control characters except \n, \r, \t)
        xml_content = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', xml_content)
        
        # Fix common encoding issues
        xml_content = xml_content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        # But preserve valid entities
        xml_content = re.sub(r'&amp;([a-z]+|#[0-9]+);', r'&\1;', xml_content)
        xml_content = xml_content.replace('&amp;lt;', '<').replace('&amp;gt;', '>')
        
        # Remove null bytes
        xml_content = xml_content.replace('\x00', '')
        
        return xml_content
    
    def fetch_rss_feeds(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Fetch news from RSS feeds with improved error handling and XML sanitization
        
        Args:
            symbol: Optional symbol to filter news (e.g., 'BTC/USD', 'EUR/USD')
        
        Returns:
            List of news dictionaries with 'title', 'content', 'link', 'published'
        """
        all_news = []
        successful_feeds = 0
        failed_feeds = []
        
        # Headers to mimic a browser request (helps with some feeds)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/rss+xml, application/xml, text/xml, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        }
        
        for feed_url in self.rss_feeds:
            try:
                print(f"NewsFetcher: Fetching from {feed_url}...")
                
                # Retry logic for transient network errors
                feed = None
                raw_content = None
                max_retries = 2
                
                for attempt in range(max_retries):
                    try:
                        # Use requests with timeout and headers for better reliability
                        response = requests.get(
                            feed_url,
                            headers=headers,
                            timeout=10,
                            allow_redirects=True
                        )
                        response.raise_for_status()
                        raw_content = response.text
                        
                        # Sanitize XML before parsing
                        sanitized_content = self._sanitize_xml(raw_content)
                        
                        # Parse with feedparser
                        feed = feedparser.parse(sanitized_content)
                        break  # Success, exit retry loop
                        
                    except requests.exceptions.Timeout:
                        if attempt < max_retries - 1:
                            time.sleep(1)
                            continue
                        else:
                            failed_feeds.append(feed_url)
                            continue
                    except requests.exceptions.RequestException as e:
                        if attempt < max_retries - 1:
                            time.sleep(0.5)
                            continue
                        else:
                            error_msg = str(e)
                            # Suppress common expected errors (feeds down, auth required, etc.)
                            # Only log unexpected errors
                            status_code = None
                            if hasattr(e, 'response') and e.response is not None:
                                status_code = e.response.status_code
                            
                            # Suppress common HTTP errors (400, 403, 404, 526 are common for RSS feeds)
                            if status_code in [400, 403, 404, 526]:
                                # Silently fail - these are expected for many RSS feeds
                                pass
                            elif 'getaddrinfo' not in error_msg and 'SSL' not in error_msg and 'timeout' not in error_msg.lower():
                                # Log other unexpected errors
                                print(f"NewsFetcher: Network error fetching {feed_url}: {error_msg[:100]}")
                            failed_feeds.append(feed_url)
                            continue
                    except Exception as e:
                        # Try fallback: parse directly from URL (feedparser handles some errors better)
                        if attempt == 0:
                            try:
                                feed = feedparser.parse(feed_url)
                                break
                            except:
                                pass
                        
                        if attempt < max_retries - 1:
                            time.sleep(0.5)
                            continue
                        else:
                            error_msg = str(e)
                            if 'getaddrinfo' not in error_msg and 'SSL' not in error_msg and 'timeout' not in error_msg.lower():
                                print(f"NewsFetcher: Error parsing RSS feed {feed_url}: {error_msg[:100]}")
                            failed_feeds.append(feed_url)
                            continue
                
                if feed is None:
                    continue
                
                # Check if feed parsing was successful (feedparser is lenient, so bozo might be set even if entries exist)
                if hasattr(feed, 'bozo') and feed.bozo:
                    error = str(feed.bozo_exception) if hasattr(feed, 'bozo_exception') else 'Unknown error'
                    # Only log non-trivial parsing errors (suppress common network/SSL errors)
                    if 'getaddrinfo' not in error and 'SSL' not in error and 'timeout' not in error.lower() and 'not well-formed' not in error:
                        # Suppress common XML parsing warnings that don't prevent reading entries
                        if 'syntax error' not in error.lower():
                            print(f"NewsFetcher: Warning - Feed {feed_url} has parsing issues: {error[:100]}")
                
                if not hasattr(feed, 'entries') or len(feed.entries) == 0:
                    # Don't log empty feeds - this is common and not an error
                    failed_feeds.append(feed_url)
                    continue
                
                feed_count = 0
                for entry in feed.entries[:self.max_articles_per_feed]:
                    # Extract content
                    title = entry.get('title', '')
                    summary = entry.get('summary', '')
                    link = entry.get('link', '')
                    
                    # Parse published date
                    published = datetime.now()
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        try:
                            published = datetime(*entry.published_parsed[:6])
                        except:
                            pass  # Use current time if parsing fails
                    
                    # Filter by symbol if provided (but be less strict)
                    if symbol:
                        symbol_keywords = self._extract_symbol_keywords(symbol)
                        # For major assets like BTC, be more lenient
                        if 'BTC' in symbol.upper() or 'BITCOIN' in symbol.upper():
                            # Accept if it mentions crypto/bitcoin even if not exact match
                            text = (title + ' ' + summary).lower()
                            if not any(kw.lower() in text for kw in symbol_keywords + ['bitcoin', 'btc', 'crypto', 'cryptocurrency']):
                                continue
                        else:
                            if not any(kw.lower() in (title + ' ' + summary).lower() 
                                       for kw in symbol_keywords + self.keywords):
                                continue
                    
                    # Filter by keywords (but skip for symbol-specific searches)
                    if not symbol:
                        if not any(kw.lower() in (title + ' ' + summary).lower() 
                                   for kw in self.keywords):
                            continue
                    
                    # Only include recent news (last 24 hours) - but be lenient
                    age_hours = (datetime.now() - published).total_seconds() / 3600
                    if age_hours > 48:  # Extended to 48 hours for better coverage
                        continue
                    
                    all_news.append({
                        'title': title,
                        'content': summary,
                        'link': link,
                        'published': published,
                        'source': feed_url
                    })
                    feed_count += 1
                
                print(f"NewsFetcher: Fetched {feed_count} articles from {feed_url}")
                successful_feeds += 1
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"NewsFetcher: Error fetching RSS feed {feed_url}: {e}")
                failed_feeds.append(feed_url)
                continue
        
        print(f"NewsFetcher: Successfully fetched from {successful_feeds}/{len(self.rss_feeds)} RSS feeds")
        # Only log failed feeds if most feeds failed (indicates a potential issue)
        # RSS feeds are secondary sources - failures are common and expected
        if failed_feeds and len(failed_feeds) > len(self.rss_feeds) * 0.7:
            print(f"NewsFetcher: Note - {len(failed_feeds)} RSS feeds unavailable (this is normal - using primary news sources)")
        
        # Sort by published date (newest first)
        all_news.sort(key=lambda x: x['published'], reverse=True)
        
        return all_news
    
    def _extract_symbol_keywords(self, symbol: str) -> List[str]:
        """Extract keywords from trading symbol"""
        keywords = []
        
        # Remove separators
        clean_symbol = symbol.replace('/', ' ').replace('_', ' ')
        
        # Add full symbol and parts
        keywords.append(symbol)
        keywords.append(clean_symbol)
        
        # Add individual parts
        parts = clean_symbol.split()
        keywords.extend(parts)
        
        # Common mappings
        symbol_map = {
            'BTC': ['bitcoin', 'btc'],
            'ETH': ['ethereum', 'eth'],
            'SOL': ['solana', 'sol'],
            'EUR': ['euro', 'eur'],
            'GBP': ['pound', 'sterling', 'gbp'],
            'JPY': ['yen', 'jpy'],
            'USD': ['dollar', 'usd', 'us dollar']
        }
        
        for part in parts:
            if part.upper() in symbol_map:
                keywords.extend(symbol_map[part.upper()])
        
        return keywords
    
    def fetch_reddit_news(self, symbol: str) -> List[Dict]:
        """
        Fetch news from Reddit (free, no API key needed for public data)
        Uses Reddit's public JSON API
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USD')
        
        Returns:
            List of news articles from Reddit
        """
        if not self.reddit_enabled:
            return []
        
        try:
            symbol_keywords = self._extract_symbol_keywords(symbol)
            base = symbol.split('/')[0].upper() if '/' in symbol else symbol.upper()
            
            # Reddit subreddits for crypto/forex news
            subreddits = []
            if base in ['BTC', 'ETH', 'SOL'] or 'CRYPTO' in symbol.upper():
                subreddits = ['cryptocurrency', 'CryptoCurrency', 'Bitcoin', 'ethereum', 'solana']
            elif '/' in symbol:
                subreddits = ['Forex', 'forex', 'investing', 'stocks']
            else:
                subreddits = ['investing', 'stocks', 'StockMarket']
            
            all_articles = []
            
            for subreddit in subreddits[:3]:  # Limit to 3 subreddits to avoid rate limits
                try:
                    # Reddit JSON API (no auth needed for public data)
                    url = f"https://www.reddit.com/r/{subreddit}/hot.json"
                    params = {'limit': 10}
                    headers = {'User-Agent': 'SmartTrader/1.0 (News Fetcher)'}
                    
                    response = requests.get(url, params=params, headers=headers, timeout=5)
                    response.raise_for_status()
                    data = response.json()
                    
                    for post in data.get('data', {}).get('children', [])[:5]:  # Top 5 posts
                        post_data = post.get('data', {})
                        title = post_data.get('title', '')
                        selftext = post_data.get('selftext', '')
                        score = post_data.get('score', 0)
                        
                        # Filter by relevance
                        text = (title + ' ' + selftext).lower()
                        if any(kw.lower() in text for kw in symbol_keywords[:3]):  # Check top 3 keywords
                            # Parse timestamp
                            created_utc = post_data.get('created_utc', 0)
                            published = datetime.fromtimestamp(created_utc) if created_utc else datetime.now()
                            
                            all_articles.append({
                                'title': title,
                                'content': selftext[:500],  # Limit content length
                                'link': f"https://www.reddit.com{post_data.get('permalink', '')}",
                                'published': published,
                                'source': f"Reddit: r/{subreddit}",
                                'score': score  # Reddit upvotes as relevance indicator
                            })
                    
                    time.sleep(0.5)  # Rate limiting
                    
                except Exception as e:
                    # Silently fail - Reddit is optional
                    continue
            
            if all_articles:
                print(f"NewsFetcher: Reddit returned {len(all_articles)} articles")
            return all_articles
            
        except Exception as e:
            # Reddit is optional, don't log errors
            return []
    
    def fetch_news_for_symbol(self, symbol: str, force_refresh: bool = False) -> List[Dict]:
        """
        Fetch news for a specific trading symbol with improved caching
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USD', 'EUR/USD')
            force_refresh: Force refresh even if recently fetched
        
        Returns:
            List of relevant news articles
        """
        symbol_keywords = self._extract_symbol_keywords(symbol)
        print(f"NewsFetcher: Fetching news for {symbol} with keywords: {symbol_keywords[:5]}...")
        
        # Check per-symbol cache first (more efficient)
        if not force_refresh and symbol in self.symbol_cache:
            cached_articles, cache_time = self.symbol_cache[symbol]
            time_since_cache = (datetime.now() - cache_time).total_seconds() / 60
            if time_since_cache < self.cache_ttl_minutes:
                print(f"NewsFetcher: Returning {len(cached_articles)} cached articles for {symbol} (cached {time_since_cache:.1f} min ago)")
                return cached_articles
        
        # Check global cache as fallback
        if not force_refresh and self.last_fetch_time:
            time_since_fetch = (datetime.now() - self.last_fetch_time).total_seconds()
            if time_since_fetch < (self.update_frequency_minutes * 60):
                # Return cached news filtered by symbol
                cached_filtered = [n for n in self.cached_news 
                       if any(kw.lower() in (n['title'] + ' ' + n.get('content', '')).lower()
                            for kw in symbol_keywords)]
                if cached_filtered:
                    print(f"NewsFetcher: Returning {len(cached_filtered)} cached articles for {symbol}")
                    return cached_filtered
        
        # Fetch fresh news from multiple sources (prioritize free sources)
        print(f"NewsFetcher: Fetching fresh news from multiple sources...")
        all_news = []
        
        # Priority 1: Free sources (no API key needed)
        # 1a. CryptoCompare for crypto symbols (free, crypto-specific, reliable)
        if '/' in symbol and self.cryptocompare_enabled:
            try:
                crypto_articles = self.fetch_cryptocompare(symbol)
                all_news.extend(crypto_articles)
                time.sleep(0.3)  # Rate limiting
            except Exception as e:
                print(f"NewsFetcher: CryptoCompare error (non-critical): {e}")
        
        # 1b. Reddit (free, good for sentiment, no API key)
        if self.reddit_enabled:
            try:
                reddit_articles = self.fetch_reddit_news(symbol)
                all_news.extend(reddit_articles)
                time.sleep(0.3)
            except Exception as e:
                # Reddit is optional, don't log
                pass
        
        # Priority 2: API sources (if configured)
        # 2a. NewsAPI (requires API key, free tier: 100 requests/day)
        if self.newsapi_enabled:
            try:
                newsapi_articles = self.fetch_newsapi(symbol)
                all_news.extend(newsapi_articles)
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                print(f"NewsFetcher: NewsAPI error (non-critical): {e}")
        
        # 2b. Alpha Vantage (if enabled and has API key)
        if self.alphavantage_enabled:
            try:
                av_articles = self.fetch_alphavantage_news(symbol)
                all_news.extend(av_articles)
                time.sleep(0.5)
            except Exception as e:
                print(f"NewsFetcher: Alpha Vantage error (non-critical): {e}")
        
        # Priority 3: RSS feeds (fallback, less reliable)
        # Only fetch RSS if we don't have enough articles yet
        if len(all_news) < 5:
            try:
                rss_articles = self.fetch_rss_feeds(symbol)
                all_news.extend(rss_articles)
            except Exception as e:
                print(f"NewsFetcher: RSS feeds error (non-critical): {e}")
        
        # Remove duplicates (by title similarity)
        unique_news = self._deduplicate_articles(all_news)
        
        print(f"NewsFetcher: Fetched {len(unique_news)} total unique articles from all sources")
        self.cached_news = unique_news
        self.last_fetch_time = datetime.now()
        
        # Filter by symbol - be less strict for major assets
        filtered_news = []
        for n in all_news:
            title_content = (n['title'] + ' ' + n.get('content', '')).lower()
            # Check if any keyword matches
            if any(kw.lower() in title_content for kw in symbol_keywords):
                filtered_news.append(n)
        
        # For major cryptos, if no direct matches, include general crypto news
        if len(filtered_news) == 0 and any(x in symbol.upper() for x in ['BTC', 'ETH', 'SOL']):
            print(f"NewsFetcher: No direct matches for {symbol}, checking general crypto news...")
            crypto_keywords = ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'cryptocurrency', 'blockchain']
            for n in all_news:
                title_content = (n['title'] + ' ' + n.get('content', '')).lower()
                if any(kw in title_content for kw in crypto_keywords):
                    filtered_news.append(n)
                    if len(filtered_news) >= 5:  # Limit to 5 general crypto articles
                        break
        
        print(f"NewsFetcher: Found {len(filtered_news)} relevant articles for {symbol}")
        
        # Update per-symbol cache
        self.symbol_cache[symbol] = (filtered_news, datetime.now())
        
        # Clean old cache entries (older than 1 hour)
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.symbol_cache = {k: v for k, v in self.symbol_cache.items() 
                            if v[1] > cutoff_time}
        
        return filtered_news
    
    def extract_stock_symbols_from_news(self, news_articles: List[Dict]) -> Dict[str, int]:
        """
        Extract stock ticker symbols from news articles
        
        Args:
            news_articles: List of news article dictionaries
            
        Returns:
            Dictionary mapping stock symbols to occurrence count
        """
        import re
        from collections import Counter
        
        # Common stock ticker pattern: 1-5 uppercase letters, optionally followed by exchange suffix
        # Also check for patterns like "AAPL stock", "MSFT shares", etc.
        ticker_pattern = re.compile(r'\b([A-Z]{1,5})\b')
        
        # Common words to exclude (not stock tickers)
        excluded_words = {
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE',
            'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW',
            'OLD', 'SEE', 'TWO', 'WHO', 'WAY', 'USE', 'HER', 'SHE', 'MAN', 'HAS', 'HAD', 'DID',
            'USD', 'EUR', 'GBP', 'JPY', 'CAD', 'CHF', 'AUD', 'NZD', 'BTC', 'ETH', 'SOL', 'API',
            'CEO', 'CFO', 'IPO', 'SEC', 'FDA', 'GDP', 'CPI', 'ETF', 'AI', 'IT', 'US', 'UK', 'EU'
        }
        
        # Known stock exchanges for validation (common ticker patterns)
        # Most stock tickers are 1-4 characters, some are 5
        found_symbols = []
        
        for article in news_articles:
            title = article.get('title', '')
            content = article.get('content', '')
            text = f"{title} {content}".upper()
            
            # Extract potential tickers
            matches = ticker_pattern.findall(text)
            
            for match in matches:
                ticker = match.upper()
                # Filter out common words and very short/long tickers
                if (len(ticker) >= 1 and len(ticker) <= 5 and 
                    ticker not in excluded_words and
                    ticker.isalpha()):
                    found_symbols.append(ticker)
        
        # Count occurrences
        symbol_counts = Counter(found_symbols)
        
        # Also check Alpha Vantage news which includes ticker metadata
        for article in news_articles:
            # Alpha Vantage includes ticker information in metadata
            if 'source' in article and 'Alpha Vantage' in article.get('source', ''):
                # Try to extract from link or other metadata
                link = article.get('link', '')
                if link:
                    # Some news sources include ticker in URL
                    ticker_match = re.search(r'/([A-Z]{1,5})/', link.upper())
                    if ticker_match:
                        ticker = ticker_match.group(1)
                        if ticker not in excluded_words and len(ticker) >= 1 and len(ticker) <= 5:
                            symbol_counts[ticker] += 1
        
        return dict(symbol_counts)
    
    def is_significant_news(self, article: Dict, min_sentiment_score: float = 0.5) -> bool:
        """
        Determine if a news article represents significant corporate news
        
        Args:
            article: News article dictionary
            min_sentiment_score: Minimum absolute sentiment score for significance
            
        Returns:
            True if article represents significant news
        """
        title = article.get('title', '').lower()
        content = article.get('content', '').lower()
        text = f"{title} {content}"
        
        # Keywords indicating significant corporate events
        significant_keywords = [
            'earnings', 'quarterly results', 'q1', 'q2', 'q3', 'q4', 'revenue', 'profit', 'loss',
            'merger', 'acquisition', 'takeover', 'buyout', 'deal',
            'fda approval', 'regulatory approval', 'clinical trial', 'drug approval',
            'product launch', 'new product', 'announcement',
            'lawsuit', 'legal action', 'settlement',
            'regulatory', 'sec', 'investigation', 'fine', 'penalty',
            'ceo', 'cfo', 'executive', 'leadership change', 'resignation',
            'bankruptcy', 'chapter 11', 'restructuring',
            'ipo', 'initial public offering', 'going public',
            'dividend', 'stock split', 'buyback', 'share repurchase',
            'guidance', 'forecast', 'outlook', 'expectations',
            'partnership', 'strategic alliance', 'joint venture'
        ]
        
        # Check for significant keywords
        has_significant_keyword = any(keyword in text for keyword in significant_keywords)
        
        # Check sentiment score if available
        sentiment_score = article.get('sentiment_score', 0.0)
        if isinstance(sentiment_score, str):
            try:
                sentiment_score = float(sentiment_score)
            except:
                sentiment_score = 0.0
        
        has_strong_sentiment = abs(sentiment_score) >= min_sentiment_score
        
        # Check recency (within last 48 hours is more significant)
        published = article.get('published', datetime.now())
        if isinstance(published, str):
            try:
                from dateutil import parser
                published = parser.parse(published)
            except:
                published = datetime.now()
        
        hours_ago = (datetime.now() - published.replace(tzinfo=None) if published.tzinfo else datetime.now() - published).total_seconds() / 3600
        is_recent = hours_ago <= 48
        
        # Article is significant if it has significant keywords OR strong sentiment AND is recent
        return (has_significant_keyword or has_strong_sentiment) and is_recent
    
    def _deduplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate articles based on title similarity"""
        if not articles:
            return []
        
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            title = article.get('title', '').lower().strip()
            # Simple deduplication: check if title is very similar
            title_key = title[:50]  # Use first 50 chars as key
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_articles.append(article)
        
        return unique_articles
    
    def get_latest_news(self, hours: int = 24) -> List[Dict]:
        """
        Get latest news from all feeds
        
        Args:
            hours: Number of hours to look back
        
        Returns:
            List of news articles
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        if not self.cached_news or not self.last_fetch_time:
            self.cached_news = self.fetch_rss_feeds()
            self.last_fetch_time = datetime.now()
        
        return [
            n for n in self.cached_news
            if n['published'] >= cutoff_time
        ]

