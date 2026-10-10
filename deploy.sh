#!/bin/bash
# =============================================================================
# SmartTrader — NAS Deploy Script
# =============================================================================
# Usage (first time or any update):
#   bash ~/SmartTrader/deploy.sh
#
# Configs are ALWAYS refreshed from the image on each deploy so that
# changes committed to the repo (trading_config.yaml etc.) take effect.
# If you need local overrides, edit the files in $BASE_DIR/config/ AFTER
# running this script and then do: docker restart smarttrader
# =============================================================================

set -e

BASE_DIR="/home/rdelomel/SmartTrader"
IMAGE="rdelomel/smarttrader:latest"
CONTAINER="smarttrader"

echo ""
echo "🚀 SmartTrader Deploy"
echo "================================"

# ── 1. Check .env exists ───────────────────────────────────────────────
if [ ! -f "$BASE_DIR/.env" ]; then
  echo "❌ ERROR: $BASE_DIR/.env not found."
  echo "   Create it with your API keys before deploying."
  exit 1
fi
echo "✅ .env found"

# ── 2. Create data folders if missing ─────────────────────────────────
mkdir -p "$BASE_DIR"/{data,logs,models,config}
echo "✅ Data folders ready"

# ── 3. Stop + remove old container ────────────────────────────────────
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
  echo "🔄 Stopping old container..."
  docker stop "$CONTAINER" 2>/dev/null || true
  docker rm   "$CONTAINER" 2>/dev/null || true
  echo "✅ Old container removed"
else
  echo "ℹ️  No existing container found"
fi

# ── 4. Pull latest image ───────────────────────────────────────────────
echo "📥 Pulling latest image..."
docker pull "$IMAGE"
echo "✅ Image up to date"

# ── 5. Extract configs from image (ALWAYS — applies repo config changes) ──
echo "📄 Refreshing configs from image..."
docker run --rm \
  -v "$BASE_DIR/config:/output" \
  --entrypoint sh "$IMAGE" \
  -c "cp /app/config/*.yaml /output/ && echo ok" \
  && echo "✅ Configs refreshed" \
  || { echo "❌ Config extraction failed — check image and retry"; exit 1; }

# ── 6. Start container ─────────────────────────────────────────────────
echo "📦 Starting container..."
docker run -d \
  --name "$CONTAINER" \
  --restart unless-stopped \
  --env-file "$BASE_DIR/.env" \
  -p 8000:8000 \
  -v "$BASE_DIR/data:/app/data" \
  -v "$BASE_DIR/logs:/app/logs" \
  -v "$BASE_DIR/models:/app/models" \
  -v "$BASE_DIR/config:/app/config:ro" \
  --memory=1536m \
  --cpus=1.0 \
  "$IMAGE"
echo "✅ Container started"

# ── 7. Done ────────────────────────────────────────────────────────────
echo ""
echo "✅ SmartTrader is running!"
echo "   Dashboard : http://$(hostname -I | awk '{print $1}'):8000"
echo "   Logs      : docker logs $CONTAINER -f"
echo "   Config    : $BASE_DIR/config/"
echo ""
