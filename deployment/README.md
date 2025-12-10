# SmartTrader Deployment Guide

This directory contains deployment configurations for hosting SmartTrader on various platforms.

## Railway Deployment (Recommended for Cost)

Railway is a cost-effective platform ($5-20/month) with easy GitHub integration.

### Setup Steps:

1. **Create Railway Account**
   - Go to https://railway.app
   - Sign up with GitHub

2. **Deploy from GitHub**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your SmartTrader repository

3. **Configure Environment Variables**
   In Railway dashboard, add these environment variables:
   ```
   OANDA_API_KEY=your_key
   OANDA_ACCOUNT_ID=your_account_id
   ALPACA_API_KEY=your_key
   ALPACA_API_SECRET=your_secret
   ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2
   OPENROUTER_API_KEY=your_key
   TRADING_MODE=paper
   DATABASE_URL=sqlite:///./data/trading.db
   ```

4. **Deploy**
   - Railway will automatically detect the Python project
   - It will use `railway.json` or `railway.toml` if present
   - Deployment happens automatically on git push

5. **Monitor**
   - Check logs in Railway dashboard
   - Set up health checks if needed

### Cost Optimization:
- Use smallest instance size ($5/month)
- Enable sleep on inactivity (if available)
- Use Railway's free tier for testing

## DigitalOcean Deployment

DigitalOcean offers more control and lower costs ($6-12/month for basic VPS).

### Setup Steps:

1. **Create Droplet**
   - Go to https://digitalocean.com
   - Create Ubuntu 22.04 droplet
   - Choose $6/month basic plan (1GB RAM) or $12/month (2GB RAM)

2. **SSH into Droplet**
   ```bash
   ssh root@your_droplet_ip
   ```

3. **Run Setup Script**
   ```bash
   chmod +x deployment/digitalocean-setup.sh
   ./deployment/digitalocean-setup.sh
   ```

4. **Configure Environment Variables**
   Create `/opt/smarttrader/.env` file with your API keys

5. **Start Service**
   ```bash
   sudo systemctl start smarttrader
   sudo systemctl status smarttrader
   ```

### Manual Setup (Alternative):

```bash
# Install Python
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3-pip

# Clone repository
cd /opt
git clone <your-repo> smarttrader
cd smarttrader

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
nano .env  # Add your API keys

# Run as systemd service (see digitalocean-setup.sh)
```

## Latency Optimization

### For Low Latency:

1. **Choose Region Closest to Broker APIs**
   - Alpaca: US East (New York)
   - OANDA: US East or Europe (depending on account)

2. **Enable Connection Pooling**
   - Already implemented in broker classes
   - Reuses HTTP connections

3. **Minimize Dependencies**
   - Use lightweight Python base image
   - Only install required packages

4. **Use Keep-Alive Connections**
   - Configured in broker implementations

## Monitoring

### Health Checks:
- Dashboard endpoint: `http://your-server:8000/`
- Health check endpoint: `http://your-server:8000/health` (if implemented)

### Logs:
- Railway: View in dashboard
- DigitalOcean: `sudo journalctl -u smarttrader -f`

### Alerts:
- Set up uptime monitoring (UptimeRobot, Pingdom)
- Monitor error logs
- Track trading performance metrics

## Cost Comparison

| Platform | Monthly Cost | Pros | Cons |
|----------|-------------|------|------|
| Railway | $5-20 | Easy setup, GitHub integration | Less control |
| DigitalOcean | $6-12 | Full control, good performance | Manual setup |
| AWS EC2 | $0-15 | Free tier available, scalable | Complex setup |
| Hetzner | €4-8 | Very cheap, good performance | Limited US regions |

## Recommended Setup

For **cost optimization**: Railway or DigitalOcean
For **low latency**: DigitalOcean (choose region closest to broker)
For **ease of use**: Railway
For **maximum control**: DigitalOcean

