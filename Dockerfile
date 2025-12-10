# Dockerfile for SmartTrader on Railway

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs models data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TF_ENABLE_ONEDNN_OPTS=0

# Expose dashboard port
EXPOSE 8000

# Health check - Railway maps PORT env var to container's port 8000
# Increased start period to allow app to fully initialize (ML models take time to load)
HEALTHCHECK --interval=60s --timeout=30s --start-period=180s --retries=3 \
  CMD curl -f http://localhost:8000/ || exit 1

# Run the application
CMD ["python", "run_agent.py"]

