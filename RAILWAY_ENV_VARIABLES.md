# Environment Variables for Railway Deployment

## Good News! 🎉

**No new environment variables are required** for the new news APIs we just added:
- ✅ **Reddit**: FREE, no API key needed (uses public API)
- ✅ **CryptoCompare**: Already working, no API key needed

## Optional Environment Variables (For Enhanced Coverage)

These are **optional** but recommended if you want more news sources:

### 1. NewsAPI (Optional)
- **Variable Name**: `NEWSAPI_API_KEY`
- **Free Tier**: 100 requests/day
- **Get Key**: https://newsapi.org/register
- **Cost**: FREE (up to 100 requests/day)
- **Status**: Optional - Reddit + CryptoCompare provide good coverage

### 2. Alpha Vantage (Optional)
- **Variable Name**: `ALPHA_VANTAGE_API_KEY`
- **Free Tier**: 500 requests/day
- **Get Key**: https://www.alphavantage.co/support/#api-key
- **Cost**: FREE (up to 500 requests/day)
- **Status**: Optional - Good for stock fundamentals

## Required Environment Variables (For Trading)

These are **required** for the trading system to work:

### Broker APIs
```
OANDA_API_KEY=your_oanda_key
OANDA_ACCOUNT_ID=your_account_id  # Optional, auto-detected
OANDA_ENVIRONMENT=practice  # or 'live'

ALPACA_API_KEY=your_alpaca_key
ALPACA_API_SECRET=your_alpaca_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2
```

### Sentiment Analysis
```
OPENROUTER_API_KEY=your_openrouter_key
# OR
OPENAI_API_KEY=your_openai_key  # Alternative to OpenRouter
```

### Database
```
DATABASE_URL=postgresql://user:pass@host:port/dbname
# For Railway PostgreSQL, this is usually auto-provided
```

## Railway Setup Instructions

### Step 1: Add Required Variables
1. Go to your Railway project
2. Click on your service
3. Go to **Variables** tab
4. Add these **required** variables:
   - `OANDA_API_KEY`
   - `ALPACA_API_KEY`
   - `ALPACA_API_SECRET`
   - `OPENROUTER_API_KEY`
   - `DATABASE_URL` (usually auto-provided by Railway PostgreSQL)

### Step 2: Add Optional News API Variables (Recommended)
For better news coverage, add:
- `NEWSAPI_API_KEY` (optional but recommended)
- `ALPHA_VANTAGE_API_KEY` (optional)

### Step 3: Add Trading Mode Variables
```
TRADING_MODE=paper
ENABLE_LIVE_TRADING=false
LOG_LEVEL=INFO
```

## Quick Reference: All Environment Variables

### Required
```bash
# Brokers
OANDA_API_KEY=...
ALPACA_API_KEY=...
ALPACA_API_SECRET=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2

# Sentiment Analysis
OPENROUTER_API_KEY=...

# Database (usually auto-provided by Railway)
DATABASE_URL=postgresql://...
```

### Optional (But Recommended)
```bash
# News APIs (for better coverage)
NEWSAPI_API_KEY=...  # Free: 100 requests/day
ALPHA_VANTAGE_API_KEY=...  # Free: 500 requests/day

# Trading Mode
TRADING_MODE=paper
ENABLE_LIVE_TRADING=false
LOG_LEVEL=INFO
```

### Not Needed (New APIs Work Without Keys)
```bash
# These work without API keys - no variables needed:
# - Reddit (free public API)
# - CryptoCompare (free, no key needed)
```

## Summary

**For the new news APIs we added:**
- ✅ **No new environment variables needed**
- ✅ Reddit works without any API key
- ✅ CryptoCompare works without any API key

**Optional enhancements:**
- Add `NEWSAPI_API_KEY` for more news sources (100 free requests/day)
- Add `ALPHA_VANTAGE_API_KEY` for stock fundamentals (500 free requests/day)

The system will work fine with just Reddit + CryptoCompare (both free, no keys needed), but adding NewsAPI will give you more coverage.

