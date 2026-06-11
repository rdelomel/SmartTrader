# SmartTrader — Hosting Options (Personal Use, Tax-Separate from Business)

## TL;DR Recommendation

| Option | Cost/month | Setup | Reliability | Best For |
|---|---|---|---|---|
| 🏆 **Your NAS** | ~$0 (electricity) | Medium | ★★★★☆ | If you have a NAS already |
| 🥈 **Oracle Cloud Free** | **Free forever** | Easy | ★★★★★ | If you don't have a NAS |
| 🥉 **Hetzner CX22** | ~AUD $5.50 | Easy | ★★★★★ | Cheapest reliable VPS |
| Fly.io | Free / $3 | Easy | ★★★☆☆ | Dev/testing only |
| DigitalOcean | $6 USD | Easy | ★★★★★ | Good but pricier |
| Vultr | $6 USD | Easy | ★★★★☆ | Like DigitalOcean |

---

## Option 1 — Your NAS (Recommended if you already own one) 🏆

**Cost:** ~$0.50–$2/month in electricity  
**Setup:** See `NAS_DEPLOYMENT.md`

Pros:
- Essentially free
- Your data stays local (no cloud storage costs)
- Already running 24/7 if it's your media server
- Full control

Cons:
- Home internet outage = trading pause (mitigated by broker reconnect logic)
- No SLA — if NAS dies, bot stops

**Best for:** You already have a Synology/QNAP/TrueNAS running

---

## Option 2 — Oracle Cloud Free Tier 🥈

**Cost:** Completely free forever (not a trial)  
**URL:** https://www.oracle.com/cloud/free/

Oracle gives you permanently free:
- 2× AMD VMs (1 OCPU, 1GB RAM each)
- **OR** 1× ARM VM (4 OCPU, 24GB RAM) — overkill-good
- 200GB block storage

Setup:
```bash
sudo apt-get update && sudo apt-get install -y docker.io docker-compose git
git clone https://github.com/rdelomel/SmartTrader.git
cd SmartTrader
cp env.example .env && nano .env
docker-compose up -d --build
```

**Tax note:** Oracle Cloud personal account is separate from your Railway
business account. Keep them on different email addresses.

---

## Option 3 — Hetzner VPS 🥉

**Cost:** CX22 = €3.29/month (~AUD $5.50)  
**URL:** https://www.hetzner.com/cloud

Specs: 2 vCPU AMD, 4GB RAM, 40GB SSD, 20TB traffic — massive overkill for a trading bot.

```bash
sudo apt update && sudo apt install -y docker.io docker-compose git
git clone https://github.com/rdelomel/SmartTrader.git
cd SmartTrader && cp env.example .env && nano .env
docker-compose up -d --build
```

**Tax tip:** Pay with a personal card (not business). Keep invoices.
Trading bot costs are generally deductible as investment expenses in Australia.

---

## Option 4 — Fly.io (Dev/testing only)

**Cost:** Free tier — but machines can be paused after inactivity  
**URL:** https://fly.io  
**Warning:** Not reliable for a 24/7 trading bot — use Hetzner or Oracle instead.

---

## Keeping Railway clean for AussiePetSitting

1. Move SmartTrader **off Railway** (use NAS or Oracle/Hetzner personal account)
2. Keep Railway **only** for AussiePetSitting (business account, deductible)
3. Railway bills to your business card → AussiePetSitting infrastructure expense
4. NAS/Oracle/Hetzner → Personal investment expense (or $0 on NAS)

---

## Summary

- **Have a Synology/QNAP NAS?** → Use it. Follow `NAS_DEPLOYMENT.md`.
- **No NAS, want free?** → Oracle Cloud Free Tier. 10-min signup.
- **No NAS, want simplest paid?** → Hetzner CX22. ~AUD $5.50/month.
