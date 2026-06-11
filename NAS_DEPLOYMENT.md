# SmartTrader — NAS Deployment Guide
## Synology / QNAP / TrueNAS / Any Docker-compatible NAS

---

## Why NAS is a great fit for a trading bot

SmartTrader only makes **outbound** API calls to OANDA, Alpaca/Binance, and
OpenRouter. It needs no open inbound ports (unless you want the dashboard).
This means:

- ✅ Works fine behind your home router's NAT
- ✅ No static IP needed for trading (only for dashboard access)
- ✅ Runs 24/7 on a device you already own (essentially free)
- ✅ Data stays on your NAS — no cloud storage cost
- ✅ Survives ISP blips (the broker APIs have their own reconnect logic now)

---

## Minimum NAS Requirements

| Resource | Minimum | Recommended |
|---|---|---|
| RAM | 1.5 GB free | 2–4 GB free |
| CPU | Any dual-core | ARM64 or x86_64 |
| Disk | 5 GB | 20 GB (for logs + DB) |
| Docker | Container Manager / Station | Portainer also works |
| Internet | Any stable broadband | Wired preferred |

**Tested on:** Synology DS923+, DS720+, QNAP TS-464

---

## Step-by-Step: Synology DSM

### 1. Enable Container Manager
- Open **Package Center** → Install **Container Manager**
- Go to **Container Manager** → **Registry** → Add if needed

### 2. Clone the repo onto your NAS
SSH into your NAS:
```bash
ssh admin@YOUR-NAS-IP

# Install git if not present
sudo apt-get install git -y

# Clone to your docker volume
cd /volume1/docker   # adjust to your volume name
git clone https://github.com/rdelomel/SmartTrader.git smarttrader
cd smarttrader
```

### 3. Create your .env file
```bash
cp env.example .env
nano .env   # Fill in your API keys
```

### 4. Build and start
```bash
docker-compose -f docker-compose.yml up -d --build
```

### 5. Check it's running
```bash
docker logs smarttrader -f
# You should see: SmartTrader started | paper trading | signal_log_mode=ON
```

### 6. Access the dashboard (optional)
Open a browser: **http://YOUR-NAS-IP:8000**

---

## Step-by-Step: QNAP Container Station

1. Open **Container Station** → **Create**
2. Choose **Create Application** (docker-compose support)
3. Paste the contents of `docker-compose.yml`
4. Set environment variables via the UI or upload `.env`
5. Click **Create**

---

## Step-by-Step: Portainer (any NAS)

1. Open Portainer → **Stacks** → **Add Stack**
2. Paste `docker-compose.yml` content
3. Add environment variables from `.env`
4. Click **Deploy the stack**

---

## Keeping it updated

The `watchtower` service in docker-compose checks for new images daily.
For a manual update:
```bash
cd /volume1/docker/smarttrader
git pull
docker-compose down
docker-compose up -d --build
```

---

## Power outage protection

- Enable **UPS support** in your NAS settings (most Synology/QNAP support APC)
- The bot uses `restart: unless-stopped` — it auto-restarts after power loss
- The SQLite database persists in `./data/` (mapped volume)

---

## Internet reliability

Trading only needs outbound HTTPS — any stable broadband works.
If your home internet is unreliable, consider:
- **NBN with a 4G backup router** (Synology RT2600ac has this built-in)
- Or use **Oracle Cloud Free Tier** (see HOSTING_OPTIONS.md) as cloud fallback

---

## What to do with Railway

Keep your **AussiePetSitting backend** on Railway — it's a web app that
needs inbound connections from users and has a clear business purpose
(cleaner for tax purposes).

SmartTrader moves to your NAS (personal investment tool, separate from business).
