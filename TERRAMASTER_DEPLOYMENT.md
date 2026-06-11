# SmartTrader — TerraMaster NAS Deployment Guide
## Works on TOS 5, TOS 6, and TOS 7

---

## What you need to know about TerraMaster + Docker

- ✅ Docker is available in **TOS App Center** on all recent TOS versions
- ✅ Docker Compose is **built-in** (TOS 6+) or available as standalone (TOS 5)
- ✅ TOS 7 has a **native "Projects" tab** in Docker Manager — paste your `docker-compose.yml` directly
- ✅ **Portainer** (also in App Center) is the easiest way to manage stacks on TOS 5/6
- ✅ TerraMaster NAS models (F2-/F4-/F6-) use **Intel Celeron x86_64** — the existing `Dockerfile.nas` works as-is

---

## Step 0 — Find your TOS version

In TOS web UI: top-right menu → **About** → note your TOS version (5.x, 6.x, or 7.x)

---

## Method A — TOS 7: Docker Manager Projects (easiest, no SSH needed)

TOS 7 added a **Projects** tab that accepts a docker-compose YAML directly.

1. Open **TOS → Docker Manager → Projects**
2. Click **Create Project**
3. Name it: `smarttrader`
4. In the YAML editor, paste the full contents of `docker-compose.yml` from the repo
5. Scroll down to **Environment Variables** and add each line from your `.env` file
6. Click **Create** → the container starts automatically

**Volumes on TOS 7:**
Edit the volume paths in the YAML to match your TerraMaster pool. To find yours:
```bash
# Open Docker Manager → Terminal tab, or SSH in:
ls /mnt/
# You'll see something like: main  usb-1  ...
# Your data lives at /mnt/main/ (or whatever your pool is named)
```

Change the volume section in docker-compose.yml to:
```yaml
volumes:
  - /mnt/main/docker/smarttrader/data:/app/data
  - /mnt/main/docker/smarttrader/logs:/app/logs
  - /mnt/main/docker/smarttrader/models:/app/models
  - /mnt/main/docker/smarttrader/config:/app/config
```

---

## Method B — TOS 5/6: Portainer (recommended, visual interface)

Portainer is the best tool for managing docker-compose stacks on TOS 5 and 6.

### Step 1: Install Portainer
- Open **TOS App Center** → search **Portainer** → Install
- Open Portainer at `http://YOUR-NAS-IP:9000`
- Create admin account on first launch

### Step 2: Create a Stack
1. Left sidebar → **Stacks** → **+ Add Stack**
2. Name: `smarttrader`
3. Select **Web editor**
4. Paste the contents of `docker-compose.yml`
5. Update the volume paths (see TOS 7 section above for path discovery)
6. Scroll to **Environment variables** → click **Add an environment variable** for each key in your `.env`
7. Click **Deploy the stack**

### Step 3: Check logs
- Portainer → **Containers** → `smarttrader` → **Logs**
- You should see: `SmartTrader started | mode=paper | signal_log_mode=ON`

---

## Method C — SSH + docker-compose (works on all TOS versions)

### Step 1: Enable SSH
- TOS web UI → **Control Panel** → **Network Services** → **SSH** → Enable
- Default port: 22
- Default user: `admin`

### Step 2: SSH in
```bash
ssh admin@YOUR-NAS-IP
```

### Step 3: Find your volume path
```bash
ls /mnt/
# Your NAS pool is usually 'main' - confirm with:
ls /mnt/main/
```

### Step 4: Create the SmartTrader folder
```bash
mkdir -p /mnt/main/docker/smarttrader
cd /mnt/main/docker/smarttrader
```

### Step 5: Get the code
```bash
# Option A: git clone
git clone https://github.com/rdelomel/SmartTrader.git .

# Option B: if git is not available, download zip:
wget https://github.com/rdelomel/SmartTrader/archive/refs/heads/main.zip
unzip main.zip && mv SmartTrader-main/* . && rm -rf SmartTrader-main main.zip
```

### Step 6: Create your .env file
```bash
cp env.example .env
vi .env
```

Fill in your API keys (use the NEW rotated keys):
```
OANDA_API_KEY=your_new_rotated_key
ALPACA_API_KEY=your_new_rotated_key
ALPACA_API_SECRET=your_new_rotated_secret
BINANCE_API_KEY=your_binance_key
BINANCE_API_SECRET=your_binance_secret
OPENROUTER_API_KEY=your_new_rotated_key
TRADING_MODE=paper
ENABLE_LIVE_TRADING=false
```

### Step 7: Update volume paths in docker-compose.yml
```bash
vi docker-compose.yml
# Change the volumes section to use your pool path:
```
```yaml
volumes:
  - /mnt/main/docker/smarttrader/data:/app/data
  - /mnt/main/docker/smarttrader/logs:/app/logs
  - /mnt/main/docker/smarttrader/models:/app/models
  - /mnt/main/docker/smarttrader/config:/app/config
```

### Step 8: Build and start
```bash
# TOS 5 (standalone docker-compose):
docker-compose up -d --build

# TOS 6+ (plugin - try both if one fails):
docker-compose up -d --build
# OR:
docker compose up -d --build
```

### Step 9: Verify
```bash
docker ps
# Should show: smarttrader   Up X minutes   0.0.0.0:8000->8000/tcp

docker logs smarttrader -f
```

---

## Volume path cheat sheet

| What you see in `ls /mnt/` | Use this in docker-compose volumes |
|---|---|
| `main` | `/mnt/main/docker/smarttrader/` |
| `sda` | `/mnt/sda/docker/smarttrader/` |
| `Pool1` | `/mnt/Pool1/docker/smarttrader/` |
| `HDD_Pool` | `/mnt/HDD_Pool/docker/smarttrader/` |
| `md0` | `/mnt/md0/docker/smarttrader/` |

---

## Keeping it updated

```bash
cd /mnt/main/docker/smarttrader
git pull
docker-compose down
docker-compose up -d --build
```

---

## Auto-start on NAS reboot

The `restart: unless-stopped` in docker-compose.yml handles this automatically.
When your TerraMaster boots up, Docker starts and SmartTrader restarts with it.

---

## Resource usage (TerraMaster Intel Celeron models)

| Resource | Idle | Active trading |
|---|---|---|
| RAM | ~400 MB | ~700 MB |
| CPU | <1% | 2-5% |
| Disk/month | ~50 MB | ~200 MB (logs + DB) |

Leaves plenty for Plex, Backup, and other NAS services running simultaneously.

---

## Troubleshooting

### "docker-compose: command not found" on TOS 5
```bash
curl -L "https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-linux-x86_64" \
  -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
docker-compose --version
```

### Container exits immediately
```bash
docker logs smarttrader
# Look for: "Broker not connected" or "Missing API key"
# Fix: check your .env file
```

### Permission denied on volume paths
```bash
mkdir -p /mnt/main/docker/smarttrader/{data,logs,models,config}
chmod 777 /mnt/main/docker/smarttrader/{data,logs,models,config}
```

### Can't reach dashboard at port 8000
```bash
# Check TOS firewall:
# TOS UI → Control Panel → Security → Firewall → allow port 8000
# Or check what's running:
docker ps | grep smarttrader
```
