# SmartTrader on TerraMaster TOS 6 — Docker Manager Guide

This guide covers deploying SmartTrader using **TOS 6's built-in Docker Manager UI**.  
No SSH. No Portainer. No docker-compose CLI. Pure GUI.

---

## Overview

```
GitHub push → GitHub Actions builds image → pushes to Docker Hub
                                                     ↓
                              TOS 6 Docker Manager pulls image
                                                     ↓
                         Configure volumes + env vars via GUI
                                                     ↓
                                           Container running ✅
```

---

## Part 1 — One-Time Setup (GitHub → Docker Hub)

### 1.1 Create a Docker Hub account & access token

1. Go to [hub.docker.com](https://hub.docker.com) and create a free account  
   (use your username — e.g. `rdelomel`)
2. Go to **Account Settings → Security → New Access Token**
   - Description: `github-actions`
   - Permissions: **Read & Write**
   - Copy the token (shown only once)

### 1.2 Add secrets to GitHub repo

1. Open [github.com/rdelomel/SmartTrader](https://github.com/rdelomel/SmartTrader)
2. Go to **Settings → Secrets and variables → Actions → New repository secret**
3. Add these two secrets:

   | Name | Value |
   |------|-------|
   | `DOCKERHUB_USERNAME` | your Docker Hub username (e.g. `rdelomel`) |
   | `DOCKERHUB_TOKEN` | the token you just copied |

### 1.3 Trigger your first build

The workflow file is already committed. To trigger it:
- Push any code change to `main`, **OR**
- Go to **Actions → Build & Push Docker Image → Run workflow**

The build takes ~10–15 minutes the first time (downloading PyTorch CPU wheels).  
Subsequent builds are ~3–5 minutes thanks to layer caching.

When complete you'll see the image at:  
`https://hub.docker.com/r/rdelomel/smarttrader`

---

## Part 2 — Create Folders on Your TerraMaster

Before pulling the image, create the persistent data folders.

### Option A — TOS 6 File Manager (GUI)

1. Open **File Manager** in TOS 6
2. Navigate to your main storage pool (e.g. `main`)
3. Create the following folder structure:

```
/mnt/main/docker/smarttrader/
├── data/       ← SQLite trading database
├── logs/       ← Application logs
├── models/     ← Trained ML models
├── config/     ← YAML config files (editable without rebuild)
└── .env        ← Your API keys (create this file — see Part 3)
```

### Option B — SSH (faster)

```bash
ssh admin@YOUR-NAS-IP
mkdir -p /mnt/main/docker/smarttrader/{data,logs,models,config}
```

> **Not sure of your pool name?**  
> Run `ls /mnt/` via SSH — common names: `main`, `sda`, `Pool1`, `HDD_Pool`, `md0`

---

## Part 3 — Create the .env File

This file holds your API keys. It must exist **before** starting the container.

Create `/mnt/main/docker/smarttrader/.env` with this content  
(replace placeholder values with your real keys):

```env
# ── Broker Credentials ──────────────────────────────────────────────────────
OANDA_API_KEY=your_oanda_api_key_here
OANDA_ACCOUNT_ID=your_oanda_account_id_here
OANDA_API_URL=https://api-fxtrade.oanda.com

ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_api_secret_here
BINANCE_TESTNET=true

# ── AI / LLM ─────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY=your_openrouter_api_key_here

# ── Trading Mode ─────────────────────────────────────────────────────────────
TRADING_MODE=paper
ENABLE_LIVE_TRADING=false

# ── App Settings ─────────────────────────────────────────────────────────────
LOG_LEVEL=INFO
TZ=Australia/Melbourne
```

> ⚠️ **Security**: This file never gets committed to GitHub.  
> It lives only on your NAS. Back it up to your password manager.

---

## Part 4 — Pull Image via Docker Manager

1. Open **Docker Manager** in TOS 6
2. Click the **Images** tab (left sidebar)
3. Click **Add** or the **Search / Pull** button
4. In the image search box type:
   ```
   rdelomel/smarttrader
   ```
5. Select tag **`latest`**
6. Click **Pull** — wait for the download to complete (~800 MB)

You'll see `rdelomel/smarttrader:latest` appear in your image list when done.

---

## Part 5 — Create & Configure the Container

### 5.1 Start the container wizard

1. In Docker Manager → **Containers** tab → Click **Create**
2. Select image: `rdelomel/smarttrader:latest`
3. Click **Next** / **Configure**

### 5.2 Basic settings

| Setting | Value |
|---------|-------|
| Container name | `smarttrader` |
| Restart policy | **Unless stopped** |
| Network mode | Bridge (default) |

### 5.3 Port mapping

Click **Add port mapping**:

| Host port | Container port | Protocol |
|-----------|---------------|----------|
| `8000` | `8000` | TCP |

This lets you access the dashboard at `http://YOUR-NAS-IP:8000`

### 5.4 Volume mappings (bind mounts)

Click **Add volume** for each row:

| Host path (on your NAS) | Container path | Mode |
|------------------------|----------------|------|
| `/mnt/main/docker/smarttrader/data` | `/app/data` | Read/Write |
| `/mnt/main/docker/smarttrader/logs` | `/app/logs` | Read/Write |
| `/mnt/main/docker/smarttrader/models` | `/app/models` | Read/Write |
| `/mnt/main/docker/smarttrader/config` | `/app/config` | Read/Write |

> Replace `/mnt/main` with your actual pool path (check Part 2 above)

### 5.5 Environment variables

You have two options:

**Option A (recommended) — env_file**  
Some versions of TOS 6 Docker Manager have an **Env file** field.  
Enter: `/mnt/main/docker/smarttrader/.env`

**Option B — Manual entry**  
Click **Add environment variable** for each key.  
Copy all key=value pairs from your `.env` file.

Required variables to add manually if Option A isn't available:

```
OANDA_API_KEY          = your_key
OANDA_ACCOUNT_ID       = your_account_id
OANDA_API_URL          = https://api-fxtrade.oanda.com
ALPACA_API_KEY         = your_key
ALPACA_SECRET_KEY      = your_secret
ALPACA_BASE_URL        = https://paper-api.alpaca.markets
BINANCE_API_KEY        = your_key
BINANCE_API_SECRET     = your_secret
BINANCE_TESTNET        = true
OPENROUTER_API_KEY     = your_key
TRADING_MODE           = paper
ENABLE_LIVE_TRADING    = false
LOG_LEVEL              = INFO
TZ                     = Australia/Melbourne
```

### 5.6 Resource limits (optional but recommended)

If Docker Manager shows a **Resources** tab:

| Setting | Value |
|---------|-------|
| Memory limit | `1536` MB (1.5 GB) |
| CPU limit | `1.0` |
| Memory reservation | `512` MB |

### 5.7 Launch

Click **Create** or **Apply** → then click **Start** on the container.

---

## Part 6 — Verify It's Running

### Check container status
Docker Manager → Containers → `smarttrader` should show **Running** (green)

### View logs
1. Click on the `smarttrader` container
2. Click **Logs** tab
3. Look for lines like:
   ```
   INFO  SmartTrader starting...
   INFO  Connected to OANDA ✓
   INFO  Connected to Alpaca ✓
   INFO  Signal log mode: ON (dry run)
   ```

### Access dashboard
Open a browser on your local network:  
`http://YOUR-NAS-IP:8000`

### Check from NAS terminal (optional)
```bash
ssh admin@YOUR-NAS-IP
docker logs smarttrader --tail 50 -f
```

---

## Part 7 — Config Files (edit without rebuilding)

Because `/mnt/main/docker/smarttrader/config` is mounted to `/app/config`,  
you can edit config files directly on your NAS and restart the container —  
no image rebuild needed.

### Copy default configs to NAS

First time only — SSH in and copy the bundled configs:

```bash
ssh admin@YOUR-NAS-IP

# Copy bundled configs out to the host mount
docker cp smarttrader:/app/config/trading_config.yaml  /mnt/main/docker/smarttrader/config/
docker cp smarttrader:/app/config/broker_config.yaml   /mnt/main/docker/smarttrader/config/
docker cp smarttrader:/app/config/model_config.yaml    /mnt/main/docker/smarttrader/config/
```

Now you can edit them via **File Manager** in TOS 6 (or any text editor over SMB).

### Key config values to know

| File | Setting | Current | When to change |
|------|---------|---------|----------------|
| `trading_config.yaml` | `signal_log_mode` | `true` | Set to `false` after ~2 weeks of calibration |
| `trading_config.yaml` | `min_confidence` | `0.42` | Tune up if too many bad signals |
| `broker_config.yaml` | `default_brokers.crypto` | `binance` | Leave as-is |
| `trading_config.yaml` | `position_sizing_method` | `kelly` | Leave as-is |

---

## Part 8 — Updating to a New Version

When you push code changes to GitHub, the Action auto-builds a new image.  
To update your running container:

1. Docker Manager → **Images** → select `rdelomel/smarttrader:latest` → **Pull** (re-pull)
2. Docker Manager → **Containers** → `smarttrader` → **Stop** → **Delete**
3. Re-create the container with the same settings (Part 5 above)
4. Start

> **Tip**: TOS 6 Docker Manager may have a **Recreate** button that handles steps 2–4 automatically after a pull.

---

## Troubleshooting

### Container exits immediately
```bash
docker logs smarttrader
```
Most common causes:
- Missing env var (API key typo) → check `.env` file
- Volume path wrong → verify `/mnt/main/` is correct for your pool

### No trades happening
- Check `signal_log_mode: true` in `trading_config.yaml` — this is intentional for first 2 weeks
- After calibration, edit the config file and restart container

### Cannot access dashboard on port 8000
- Verify port 8000 is not blocked by TerraMaster firewall  
  (TOS 6 → Control Panel → Security → Firewall)
- Try `curl http://localhost:8000/health` from SSH to confirm container is serving

### Image pull fails (authentication)
- The image is public on Docker Hub — no auth needed to pull
- If you made your Docker Hub repo private, set it back to public or log in:  
  `docker login` in SSH then pull

### Wrong timezone
- Edit `.env` → change `TZ=Australia/Melbourne` to your timezone
- Restart container

---

## Quick Reference Card

```
📦 Image:     rdelomel/smarttrader:latest
🔌 Port:      8000 → 8000
📁 Data:      /mnt/main/docker/smarttrader/data   → /app/data
📋 Logs:      /mnt/main/docker/smarttrader/logs   → /app/logs
🧠 Models:    /mnt/main/docker/smarttrader/models → /app/models
⚙️  Config:    /mnt/main/docker/smarttrader/config → /app/config
🔐 Env file:  /mnt/main/docker/smarttrader/.env
🔄 Restart:   unless-stopped
💾 RAM limit: 1.5 GB
```

---

## Calibration Roadmap

```
Week 1-2:  signal_log_mode = true  ← you are here
           Watch logs — are signals being generated?
           Review /app/logs/ for signal quality

Week 3+:   Set signal_log_mode = false
           Restart container
           Monitor paper trades at http://NAS-IP:8000

Month 1-3: Win rate > 52%, Sharpe > 1.5?
           Consider enabling DRL in model_config.yaml

Month 3+:  Change ALPACA_BASE_URL to https://api.alpaca.markets
           Set TRADING_MODE=live, ENABLE_LIVE_TRADING=true
           🚀 Live trading!
```
