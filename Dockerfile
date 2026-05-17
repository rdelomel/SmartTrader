# Dockerfile for SmartTrader on Railway

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for TA-Lib, psycopg2, and TF build tools)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Layer 1: base runtime deps (cached unless requirements.txt changes) ────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Layer 1b: enforce numpy<2.0 BEFORE torch/TF so they don't upgrade it ──────
RUN pip install --no-cache-dir "numpy>=1.24.0,<2.0.0"

# ── Layer 2: PyTorch CPU-only (slim wheel - avoids pulling CUDA ~5GB) ─────────
# Pinned to torch 2.2.2 CPU which is compatible with stable-baselines3 >=2.0
RUN pip install --no-cache-dir \
    torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

# ── Layer 3: AI/ML optional deps (TF CPU, SB3, gymnasium) ────────────────────
COPY requirements-optional-ai.txt .
RUN pip install --no-cache-dir -r requirements-optional-ai.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# Create necessary runtime directories
RUN mkdir -p logs models data

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV TF_CPP_MIN_LOG_LEVEL=2

# Expose dashboard port
EXPOSE 8000

# Health check - generous start period for ML model loading
HEALTHCHECK --interval=60s --timeout=30s --start-period=300s --retries=3 \
  CMD curl -f http://localhost:8000/ || exit 1

# Run the application
CMD ["python", "run_agent.py"]
