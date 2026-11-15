# ==============================================================================
# Animal Metrology Computer Vision - Production Dockerfile
# Base Image: Python 3.10-slim
# ==============================================================================

FROM python:3.10-slim

# Set environment variables for non-interactive installs and Python optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000

# Install required system dependencies for OpenCV and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-privileged user and group for security compliance
RUN groupadd -r appuser && useradd -r -g appuser -u 1000 -d /app appuser

# Set working directory
WORKDIR /app

# Copy dependency definition
COPY requirements.txt .

# Install Python dependencies without caching to keep image lean
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source only; images and model weights are mounted at runtime.
COPY src/ ./src/
COPY web/ ./web/
RUN mkdir -p /app/outputs /models && chown -R appuser:appuser /app /models

ENV SEG_MODEL_WEIGHTS_PATH=/models/yolov8n-seg.pt \
    POSE_MODEL_WEIGHTS_PATH=/models/animal_eye_pose.pt \
    PIXEL_TO_MM_SCALE=0.15

# Set ownership to non-root appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose HTTP port for FastAPI
EXPOSE 8000

# Healthcheck definition
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

# Launch application via Uvicorn ASGI server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
