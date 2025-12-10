"""Stock discovery service for dynamic stock selection based on news."""

import re
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict


class StockDiscoveryService:
    """
    Discovers stocks to trade based on significant corporate news.

    Features:
    - Extracts stock symbols from news articles
    - Identifies significant news (earnings, major announcements)
    - Filters by sentiment and news volume
    - Maintains active watchlist with expiration
    """

    def __init__(self, news_fetcher, config: Optional[Dict] = None):
        """Initialize stock discovery service."""
        self.news_fetcher = news_fetcher
        self.config = config or {}

        # Configuration parameters
        self.max_concurrent_stocks = self.config.get("max_concurrent_stocks", 15)
        self.min_sentiment_score = self.config.get("min_sentiment_score", 0.5)
        self.discovery_refresh_hours = self.config.get("discovery_refresh_hours", 1)
        self.news_lookback_hours = self.config.get("news_lookback_hours", 48)
        self.min_news_articles = self.config.get("min_news_articles", 2)

        # Cache for discovered stocks
        self.discovered_stocks: Dict[str, Dict] = {}
        self.last_discovery_time: Optional[datetime] = None

        # Common words to filter out (not stock tickers)
        self.common_words = {
            "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HER", "WAS",
            "ONE", "OUR", "OUT", "DAY", "GET", "HAS", "HIM", "HIS", "HOW", "ITS", "MAY",
            "NEW", "NOW", "OLD", "SEE", "TWO", "WHO", "WAY", "USE", "SHE", "MAN", "THIS",
            "WHAT", "SAID", "EACH", "WHICH", "THEIR", "TIME", "WILL", "ABOUT", "IF", "UP",
            "MANY", "THEN", "THEM", "THESE", "SO", "SOME", "WOULD", "MAKE", "LIKE", "INTO",
            "TAKE", "THAN", "OIL", "GAS", "USD", "EUR", "GBP", "JPY", "CAD", "CHF", "AUD",
            "NZD", "BTC", "ETH", "SOL", "API", "CEO", "CFO", "IPO", "SEC", "FDA", "GDP",
            "CPI", "ETF", "AI", "IT", "USA", "UK", "EU"
        }

        # Keywords indicating significant corporate news
        self.significant_keywords = [
            "earnings", "quarterly results", "revenue", "profit", "loss", "guidance",
            "merger", "acquisition", "takeover", "deal", "buyout",
            "fda approval", "regulatory approval", "clinical trial",
            "product launch", "new product", "innovation",
            "lawsuit", "legal action", "settlement",
            "regulatory", "sec", "investigation", "fine",
            "ceo", "executive", "leadership change", "resignation",
            "partnership", "strategic alliance", "joint venture",
            "bankruptcy", "restructuring", "layoffs", "job cuts",
            "dividend", "stock split", "buyback", "share repurchase",
            "upgrade", "downgrade", "analyst", "price target",
            "breakthrough", "milestone", "record", "beat", "miss"
        ]

    def extract_stock_symbols_from_news(self, news_articles: List[Dict]) -> Dict[str, List[Dict]]:
        """Extract stock symbols from news articles."""
        symbol_articles: Dict[str, List[Dict]] = defaultdict(list)

        for article in news_articles:
            title = article.get("title", "")
            content = article.get("content", "")
            text = f"{title} {content}".upper()

            # Regex for potential stock tickers: 1-5 uppercase letters
            ticker_pattern = r"\b([A-Z]{1,5})\b"
            matches = re.findall(ticker_pattern, text)

            for match in matches:
                ticker = match.upper()
                if len(ticker) < 1 or len(ticker) > 5:
                    continue
                if ticker in self.common_words:
                    continue
                if ticker.isdigit():
                    continue

                context_keywords = ["stock", "shares", "ticker", "trading", "company", "corp", "inc", "ltd"]
                text_lower = text.lower()
                ticker_pos = text_lower.find(ticker.lower())
                if ticker_pos >= 0:
                    context_start = max(0, ticker_pos - 50)
                    context_end = min(len(text_lower), ticker_pos + len(ticker) + 50)
                    context = text_lower[context_start:context_end]
                    if any(keyword in context for keyword in context_keywords) or len(ticker) >= 2:
                        symbol_articles[ticker].append(article)

            # Also check Alpha Vantage metadata
            if "tickers" in article or "ticker" in article:
                tickers = article.get("tickers", []) or [article.get("ticker")]
                for ticker in tickers:
                    if ticker and isinstance(ticker, str):
                        ticker = ticker.upper().strip()
                        if ticker and len(ticker) <= 5 and ticker not in self.common_words:
                            symbol_articles[ticker].append(article)

        return dict(symbol_articles)

    def is_significant_news(self, article: Dict, symbol: str) -> bool:
        """Determine if news article represents a significant corporate event."""
        title = article.get("title", "").lower()
        content = article.get("content", "").lower()
        text = f"{title} {content}"

        if any(keyword in text for keyword in self.significant_keywords):
            return True

        sentiment_score = article.get("sentiment_score", 0.0)
        if abs(sentiment_score) >= self.min_sentiment_score:
            return True

        symbol_lower = symbol.lower()
        if symbol_lower in title or symbol_lower in content[:200]:
            return True

        return False

    def discover_stocks_from_news(self, force_refresh: bool = False) -> List[str]:
        """Discover stocks to trade based on recent significant news."""
        if not force_refresh and self.last_discovery_time:
            time_since_discovery = (datetime.now() - self.last_discovery_time).total_seconds()
            if time_since_discovery < (self.discovery_refresh_hours * 3600):
                return list(self.discovered_stocks.keys())

        print("\n[Stock Discovery] Scanning news for significant corporate events...")
        cutoff_time = datetime.now() - timedelta(hours=self.news_lookback_hours)

        all_news: List[Dict] = []

        if getattr(self.news_fetcher, "newsapi_enabled", False):
            try:
                newsapi_articles = self.news_fetcher.fetch_newsapi("stocks OR earnings OR corporate")
                all_news.extend(newsapi_articles)
            except Exception:
                pass

        if getattr(self.news_fetcher, "alphavantage_enabled", False):
            try:
                av_articles = self.news_fetcher.fetch_alphavantage_news("MARKET")
                all_news.extend(av_articles)
            except Exception:
                pass

        try:
            rss_articles = self.news_fetcher.fetch_rss_feeds()
            all_news.extend(rss_articles)
        except Exception:
            pass

        recent_news = [n for n in all_news if n.get("published", datetime.now()) >= cutoff_time]
        print(f"  Found {len(recent_news)} recent news articles")

        symbol_articles = self.extract_stock_symbols_from_news(recent_news)
        print(f"  Extracted {len(symbol_articles)} unique stock symbols from news")

        significant_stocks: Dict[str, Dict] = {}
        for symbol, articles in symbol_articles.items():
            significant_articles = [a for a in articles if self.is_significant_news(a, symbol)]
            if len(significant_articles) < self.min_news_articles:
                continue

            sentiment_scores = [
                a.get("sentiment_score", 0.0) for a in significant_articles if a.get("sentiment_score") is not None
            ]
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0

            if sentiment_scores and abs(avg_sentiment) < self.min_sentiment_score:
                continue

            significant_stocks[symbol] = {
                "discovered_at": datetime.now(),
                "news_count": len(significant_articles),
                "avg_sentiment": avg_sentiment,
                "articles": significant_articles[:5],
            }

        sorted_stocks = sorted(
            significant_stocks.items(),
            key=lambda x: (x[1]["news_count"], abs(x[1]["avg_sentiment"])),
            reverse=True,
        )
        top_stocks = sorted_stocks[: self.max_concurrent_stocks]

        self.discovered_stocks = {symbol: info for symbol, info in top_stocks}
        self.last_discovery_time = datetime.now()

        discovered_symbols = list(self.discovered_stocks.keys())
        print(f"  ✅ Discovered {len(discovered_symbols)} significant stocks: {discovered_symbols}")
        return discovered_symbols

    def get_discovered_stocks(self) -> List[str]:
        """Get currently discovered stocks, removing expired ones."""
        cutoff_time = datetime.now() - timedelta(hours=self.news_lookback_hours)
        expired_symbols = [s for s, info in self.discovered_stocks.items() if info["discovered_at"] < cutoff_time]
        for s in expired_symbols:
            del self.discovered_stocks[s]
        return list(self.discovered_stocks.keys())

    def should_trade_stock(self, symbol: str) -> bool:
        """Check if stock should be traded based on discovery criteria."""
        if symbol not in self.discovered_stocks:
            return False
        info = self.discovered_stocks[symbol]
        cutoff_time = datetime.now() - timedelta(hours=self.news_lookback_hours)
        if info["discovered_at"] < cutoff_time:
            return False
        if info["news_count"] < self.min_news_articles:
            return False
        if info["avg_sentiment"] != 0.0 and abs(info["avg_sentiment"]) < self.min_sentiment_score:
            return False
        return True

