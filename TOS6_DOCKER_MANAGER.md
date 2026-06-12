# SmartTrader on TerraMaster TOS 6 — Docker Manager Guide

Deploy SmartTrader using **TOS 6's built-in Docker Manager UI** — no SSH required
after initial folder setup. All data lives in `~/SmartTrader/` (your home folder).

---

## Overview

```
GitHub push → GitHub Actions builds image → pushes to Docker Hub (private)
                                                        ↓
                   TOS 6 Docker Manager authenticates + pulls image
                                                        ↓
                           Volumes → ~/SmartTrader/{data,logs,models,config}
                                                        ↓
                                              Container running ✅
```

---

## Part 1 — One-Time Setup: GitHub → Docker Hub

### 1.1 Create Docker Hub account & access token

1. Go to [hub.docker.com](https://hub.docker.com) — create a free account
2. **Account Settings → Security → New Access Token**
   - Name: `github-actions`
   - Permissions: **Read & Write**
   - **Copy the token** (shown only once)

### 1.2 Add secrets to your GitHub repo

Go to [github.com/rdelomel/SmartTrader → Settings → Secrets → Actions](https://github.com/rdelomel/SmartTrader/settings/secrets/actions):

| Secret name | Value |
|------------|-------|
| `DOCKERHUB_USERNAME` | your Docker Hub username (e.g. `rdelomel`) |
| `DOCKERHUB_TOKEN` | the token you copied above |

### 1.3 Trigger the first build

- Go to **Actions → Build & Push Docker Image → Run workflow**
- Takes ~12 min first time; ~3-5 min after that (layer cache)
- When done: `hub.docker.com/r/rdelomel/smarttrader` shows the image

---

## Part 2 — Create SmartTrader Folder (Home Folder)

All persistent data lives in your home folder at `/root/SmartTrader/`.
This is easy to find via TOS 6 File Manager — it's right in the root home.

### Option A — TOS 6 File Manager (no SSH)

1. Open **File Manager** in TOS 6
2. Navigate to **Home** (the home folder, usually shown as `Home` or `/root`)
3. Create a folder named `SmartTrader`
4. Inside it, create these subfolders:

```
~/SmartTrader/          (i.e. /root/SmartTrader/)
├── data/               ← SQLite trading database
├── logs/               ← Application logs  
├── models/             ← Trained ML models
├── config/             ← YAML config files (editable without rebuild)
└── .env                ← Your API keys (create this — see Part 3)
```

### Option B — SSH (30 seconds)

```bash
ssh admin@YOUR-NAS-IP
mkdir -p ~/SmartTrader/{data,logs,models,config}
echo "Folders created at: $(ls ~/SmartTrader/)"
```

---

## Part 3 — Create Your .env File

This file holds your API keys and **never gets committed to GitHub**.

Create `/root/SmartTrader/.env`:

**Via SSH:**
```bash
nano ~/SmartTrader/.env
```

**Via TOS 6 File Manager:**  
File Manager → Home → SmartTrader → New File → name it `.env` → Open with Text Editor

**Paste this template and fill in your real keys:**

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

---

## Part 4 — Authenticate Docker Hub in Docker Manager

> ⚠️ Your Docker Hub repo is **private** — TOS 6 must be logged in before pulling.

### 4.1 Add Docker Hub credentials to Docker Manager

1. Open **Docker Manager** in TOS 6
2. Go to **Settings** (or the gear icon) → **Registry** (or **Repositories**)
3. Click **Add** registry
4. Fill in:

   | Field | Value |
   |-------|-------|
   | Registry URL | `https://registry-1.docker.io` |
   | Username | your Docker Hub username (e.g. `rdelomel`) |
   | Password | your Docker Hub **access token** (same one from Part 1) |

5. Click **Save** / **Test** — should show ✅ Connected

> **Note:** If Docker Manager doesn't have a registry settings screen,  
> use SSH to authenticate once and Docker Manager will reuse it:
> ```bash
> ssh admin@YOUR-NAS-IP
> docker login
> # Enter your Docker Hub username and access token when prompted
> ```

---

## Part 5 — Pull Image via Docker Manager

1. Open **Docker Manager** → **Images** tab
2. Click **Add** or **Pull**
3. Enter image name:
   ```
   rdelomel/smarttrader:latest
   ```
4. Click **Pull** — wait for download (~800 MB)

When done, `rdelomel/smarttrader:latest` appears in your image list.

---

## Part 6 — Create & Configure the Container

### 6.1 Start the wizard

Docker Manager → **Containers** tab → **Create**  
Select image: `rdelomel/smarttrader:latest` → **Next**

### 6.2 Basic settings

| Setting | Value |
|---------|-------|
| Container name | `smarttrader` |
| Restart policy | **Unless stopped** |
| Network mode | Bridge (default) |

### 6.3 Port mapping

| Host port | Container port | Protocol |
|-----------|---------------|----------|
| `8000` | `8000` | TCP |

### 6.4 Volume mappings

| Host path (NAS) | Container path | Mode |
|----------------|----------------|------|
| `/root/SmartTrader/data` | `/app/data` | Read/Write |
| `/root/SmartTrader/logs` | `/app/logs` | Read/Write |
| `/root/SmartTrader/models` | `/app/models` | Read/Write |
| `/root/SmartTrader/config` | `/app/config` | Read/Write |

### 6.5 Environment variables

**Option A (best) — Env file field** (if visible in Docker Manager):  
Enter: `/root/SmartTrader/.env`

**Option B — Add each var manually:**

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

### 6.6 Resource limits (optional)

| Setting | Value |
|---------|-------|
| Memory limit | `1536` MB (1.5 GB) |
| CPU limit | `1.0` |

### 6.7 Launch

Click **Create** → **Start**

---

## Part 7 — Verify It's Running

**In Docker Manager:**  
Containers → `smarttrader` → should show **Running** (green dot)

**View logs:**  
Click container → **Logs** → look for:
```
INFO  SmartTrader starting...
INFO  Connected to OANDA ✓
INFO  Connected to Alpaca ✓
INFO  Signal log mode: ON (signals logged, no trades executed)
```

**Dashboard:**  
`http://YOUR-NAS-IP:8000`

---

## Part 8 — Copy Config Files (First Time Only)

Since `/root/SmartTrader/config` is mounted to `/app/config`, you can edit
config files directly from TOS 6 File Manager without touching the container.

But first, copy the bundled defaults out:

```bash
ssh admin@YOUR-NAS-IP
docker cp smarttrader:/app/config/trading_config.yaml  ~/SmartTrader/config/
docker cp smarttrader:/app/config/broker_config.yaml   ~/SmartTrader/config/
docker cp smarttrader:/app/config/model_config.yaml    ~/SmartTrader/config/
```

Now they appear in **File Manager → Home → SmartTrader → config** and you
can open and edit them directly in TOS 6 without SSH.

### Key settings

| File | Setting | Current | When to change |
|------|---------|---------|----------------|
| `trading_config.yaml` | `signal_log_mode` | `true` | Set `false` after ~2 weeks |
| `trading_config.yaml` | `min_confidence` | `0.42` | Tune up if too many bad signals |
| `broker_config.yaml` | `default_brokers.crypto` | `binance` | Leave as-is |

---

## Part 9 — Updating to a New Version

When you push code to GitHub, Actions auto-builds a new image.

1. Docker Manager → **Images** → `rdelomel/smarttrader:latest` → **Pull** (re-pull)
2. Containers → `smarttrader` → **Stop** → **Delete**
3. Re-create using the same settings (Part 6)
4. **Start**

> TOS 6 may have a **Recreate** button that handles steps 2-4 automatically.

---

## Troubleshooting

### Pull fails — "authentication required" or "unauthorized"
→ Complete Part 4 (add Docker Hub credentials to Docker Manager registry settings)

### Container exits immediately
```bash
docker logs smarttrader
```
Common causes:
- Typo in API key in `.env` file
- Volume path wrong — confirm `/root/SmartTrader/` exists: `ls ~/SmartTrader/`

### No trades happening after weeks
- Check `signal_log_mode: true` in `~/SmartTrader/config/trading_config.yaml`
- This is **intentional** for the first 2 weeks
- After calibration: set to `false`, restart container

### Can't reach dashboard on port 8000
- TOS 6 → Control Panel → Security → Firewall → allow port 8000 on LAN
- Test: `curl http://localhost:8000/health` from SSH

---

## Quick Reference Card

```
📦 Image:     rdelomel/smarttrader:latest   (private Hub)
🔌 Port:      8000 → 8000
📁 Data:      /root/SmartTrader/data    → /app/data
📋 Logs:      /root/SmartTrader/logs    → /app/logs
🧠 Models:    /root/SmartTrader/models  → /app/models
⚙️  Config:    /root/SmartTrader/config  → /app/config
🔐 Env file:  /root/SmartTrader/.env
🔄 Restart:   unless-stopped
💾 RAM limit: 1.5 GB
```

---

## Calibration Roadmap

```
Now (Week 1-2):  signal_log_mode = true
                 Signals logged, zero trades executed
                 Watch logs at ~/SmartTrader/logs/

Week 3+:         Set signal_log_mode = false → restart
                 Paper trading starts
                 Monitor at http://NAS-IP:8000

Month 1-3:       Win rate > 52%, Sharpe > 1.5?
                 Enable DRL in model_config.yaml

Month 3+:        Change ALPACA_BASE_URL to https://api.alpaca.markets
                 Set TRADING_MODE=live, ENABLE_LIVE_TRADING=true
                 🚀 Live trading!
```
