FROM python:3.10-slim

WORKDIR /app

# Install minimal system deps needed for OpenCV and general runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    # common lightweight X11/OpenGL runtime libs used by OpenCV in containers
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libglib2.0-0 \
    wget \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# gdown helps download from Google Drive links that require confirmation tokens
RUN pip install --no-cache-dir gdown

# Allow optional build-time model download. Set BUILD_MODELS=true and provide
# U2NET_DOWNLOAD_URL / SAM_DOWNLOAD_URL as build args to bake weights into image.
ARG BUILD_MODELS=false
ARG U2NET_DOWNLOAD_URL=""
ARG SAM_DOWNLOAD_URL=""
ENV MODEL_CACHE_DIR=/models

RUN if [ "${BUILD_MODELS}" = "true" ]; then \
            echo "[dockerfile] build-time model download enabled (placing weights into /app/models/weights)"; \
            mkdir -p /app/models/weights ${MODEL_CACHE_DIR}/weights ${MODEL_CACHE_DIR}; \
            # U2Net -> place into package weights dir (u2netp.pth)
            if [ -n "${U2NET_DOWNLOAD_URL}" ]; then \
                if command -v gdown >/dev/null 2>&1; then \
                    gdown -O /app/models/weights/u2netp.pth "${U2NET_DOWNLOAD_URL}" || true; \
                else \
                    wget -q -O /app/models/weights/u2netp.pth "${U2NET_DOWNLOAD_URL}" || true; \
                fi; \
            fi; \
            # SAM -> place into package weights dir (sam_vit_b_01ec64.pth)
            if [ -n "${SAM_DOWNLOAD_URL}" ]; then \
                if command -v wget >/dev/null 2>&1; then \
                    wget -q --show-progress -O /app/models/weights/sam_vit_b_01ec64.pth "${SAM_DOWNLOAD_URL}" || true; \
                elif command -v curl >/dev/null 2>&1; then \
                    curl -sSL -o /app/models/weights/sam_vit_b_01ec64.pth "${SAM_DOWNLOAD_URL}" || true; \
                fi; \
            fi; \
        else \
            echo "[dockerfile] skipping build-time model download"; \
        fi

# Copy application
COPY . /app

# Create models directory (optional: mount a volume here to persist weights)
RUN mkdir -p ${MODEL_CACHE_DIR}/weights /app/models/weights

# Add entrypoint helper (downloads models if env vars are provided)
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Expose port
EXPOSE 8000

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["/docker-entrypoint.sh"]

# Default command (overridden by entrypoint)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
