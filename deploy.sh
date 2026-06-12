#!/bin/bash
# =============================================================================
# SmartTrader — NAS Deploy Script
# =============================================================================
# Usage (first time or any update):
#   bash ~/SmartTrader/deploy.sh
#
# What it does:
#   1. Stops + removes the old container (if running)
#   2. Pulls the latest image from Docker Hub
#   3. Copies bundled configs to host if config folder is empty
#   4. Starts the container with all settings — reads .env automatically
#
# No need to re-enter env vars in Docker Manager ever again.
# =============================================================================

set -e

BASE_DIR="$HOME/SmartTrader"
IMAGE="rdelomel/smarttrader:latest"
CONTAINER="smarttrader"

echo ""
echo "🚀 SmartTrader Deploy"
echo "================================"

# ── 1. Check .env exists ───────────────────────────────────────────────
if [ ! -f "$BASE_DIR/.env" ]; then
  echo "❌ ERROR: $BASE_DIR/.env not found."
  echo "   Create it with your API keys before deploying."
  echo "   See TOS6_DOCKER_MANAGER.md Part 3 for the template."
  exit 1
fi
echo "✅ .env found"

# ── 2. Create data folders if missing ───────────────────────────────────────
mkdir -p "$BASE_DIR"/{data,logs,models,config}
echo "✅ Data folders ready"

# ── 3. Stop + remove old container ─────────────────────────────────────────
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
  echo "🔄 Stopping old container..."
  docker stop "$CONTAINER" 2>/dev/null || true
  docker rm   "$CONTAINER" 2>/dev/null || true
  echo "✅ Old container removed"
else
  echo "ℹ️  No existing container found"
fi

# ── 4. Pull latest image ─────────────────────────────────────────────────────
echo "📥 Pulling latest image..."
docker pull "$IMAGE"
echo "✅ Image up to date"

# ── 5. Start container ────────────────────────────────────────────────────────────
echo "📦 Starting container..."
docker run -d \
  --name "$CONTAINER" \
  --restart unless-stopped \
  --env-file "$BASE_DIR/.env" \
  -p 8000:8000 \
  -v "$BASE_DIR/data:/app/data" \
  -v "$BASE_DIR/logs:/app/logs" \
  -v "$BASE_DIR/models:/app/models" \
  -v "$BASE_DIR/config:/app/config" \
  --memory=1536m \
  --cpus=1.0 \
  "$IMAGE"
echo "✅ Container started"

# ── 6. Copy bundled configs if config folder is empty ───────────────────────
if [ -z "$(ls -A $BASE_DIR/config 2>/dev/null)" ]; then
  echo "📄 Config folder empty — copying bundled configs..."
  sleep 3  # give container a moment to start
  docker cp "$CONTAINER:/app/config/trading_config.yaml" "$BASE_DIR/config/" 2>/dev/null && echo "   ✅ trading_config.yaml" || echo "   ⚠️  trading_config.yaml not found in image"
  docker cp "$CONTAINER:/app/config/broker_config.yaml"  "$BASE_DIR/config/" 2>/dev/null && echo "   ✅ broker_config.yaml"  || echo "   ⚠️  broker_config.yaml not found in image"
  docker cp "$CONTAINER:/app/config/model_config.yaml"   "$BASE_DIR/config/" 2>/dev/null && echo "   ✅ model_config.yaml"   || echo "   ⚠️  model_config.yaml not found in image"
  echo "🔄 Restarting to pick up configs..."
  docker restart "$CONTAINER"
  echo "✅ Restarted with configs"
else
  echo "✅ Config files already present — no copy needed"
fi

# ── 7. Done ──────────────────────────────────────────────────────────────────
echo ""
echo "✅ SmartTrader is running!"
echo "   Dashboard : http://$(hostname -I | awk '{print $1}'):8000"
echo "   Logs      : docker logs $CONTAINER -f"
echo "   Config    : $BASE_DIR/config/"
echo ""
