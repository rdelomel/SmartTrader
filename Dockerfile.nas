# Dockerfile.nas — Optimised for NAS (Synology/QNAP/TrueNAS)
# Stripped of Railway-specific layers. No CUDA. Smaller image (~800MB vs 2GB).

FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencies in layers for Docker cache efficiency ────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pin numpy <2.0 before optional heavy deps
RUN pip install --no-cache-dir "numpy>=1.24.0,<2.0.0"

# PyTorch CPU-only (no CUDA — NAS has no GPU)
RUN pip install --no-cache-dir \
    torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

# Optional AI/RL packages
COPY requirements-optional-ai.txt .
RUN pip install --no-cache-dir -r requirements-optional-ai.txt

# ── Application ───────────────────────────────────────────────────────────────
COPY . .

RUN mkdir -p logs models data

ENV PYTHONUNBUFFERED=1
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV TF_CPP_MIN_LOG_LEVEL=2

EXPOSE 8000

HEALTHCHECK --interval=60s --timeout=20s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "run_agent.py"]
