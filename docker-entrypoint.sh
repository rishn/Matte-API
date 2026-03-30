#!/usr/bin/env bash
set -e

# Simple entrypoint: optionally download model weights, then exec the passed command (or uvicorn)

echo "[entrypoint] starting..."

# Default download locations (if you prefer, set env vars to override):
# U2NET_DOWNLOAD_URL, SAM_DOWNLOAD_URL

download_if_needed() {
  local url="$1"; shift
  local dest="$1"; shift
  if [ -z "$url" ]; then
    return 0
  fi
  if [ -f "$dest" ]; then
    echo "[entrypoint] file exists at $dest, skipping download"
    return 0
  fi
  echo "[entrypoint] downloading model from $url to $dest"
  mkdir -p "$(dirname "$dest")"
  # If this looks like a Google Drive link, prefer gdown which handles confirmation tokens
  if echo "$url" | grep -qi "drive.google.com"; then
    if command -v gdown >/dev/null 2>&1; then
      echo "[entrypoint] using gdown for Google Drive URL"
      # gdown supports both full view links and uc?export=download links
      gdown -O "$dest" "$url"
      return $?
    else
      echo "[entrypoint] gdown not found; falling back to curl/wget (may fail on large Drive files)"
    fi
  fi

  # try wget, fallback to curl
  if command -v wget >/dev/null 2>&1; then
    wget -q --show-progress -O "$dest" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -sSL -o "$dest" "$url"
  else
    echo "No downloader available (wget/curl/gdown)"
    return 1
  fi
}

# Allow configurable model cache directory (default /models)
MODEL_CACHE_DIR=${MODEL_CACHE_DIR:-/models}

# Download U2Net if env var provided
if [ ! -z "$U2NET_DOWNLOAD_URL" ]; then
  # default path inside container
  U2NET_DEST=${U2NET_DEST:-${MODEL_CACHE_DIR}/weights/u2netp.pth}
  download_if_needed "$U2NET_DOWNLOAD_URL" "$U2NET_DEST"
fi

# Download SAM weights if env var provided
if [ ! -z "$SAM_DOWNLOAD_URL" ]; then
  SAM_DEST=${SAM_DEST:-${MODEL_CACHE_DIR}/weights/sam_vit_b_01ec64.pth}
  download_if_needed "$SAM_DOWNLOAD_URL" "$SAM_DEST"
fi

echo "[entrypoint] models directory listing:"
ls -lah "$MODEL_CACHE_DIR" || true

# Ensure weights dir exists under model cache
mkdir -p "${MODEL_CACHE_DIR}/weights"

# Download U2Net if env var provided
if [ ! -z "$U2NET_DOWNLOAD_URL" ]; then
  # default path inside container (single weights folder)
  U2NET_DEST=${U2NET_DEST:-${MODEL_CACHE_DIR}/weights/u2netp.pth}
  download_if_needed "$U2NET_DOWNLOAD_URL" "$U2NET_DEST"
fi

# Download SAM weights if env var provided
if [ ! -z "$SAM_DOWNLOAD_URL" ]; then
  SAM_DEST=${SAM_DEST:-${MODEL_CACHE_DIR}/weights/sam_vit_b_01ec64.pth}
  download_if_needed "$SAM_DOWNLOAD_URL" "$SAM_DEST"
fi

echo "[entrypoint] model cache listing (${MODEL_CACHE_DIR}/weights):"
ls -lah "${MODEL_CACHE_DIR}/weights" || true

# If U2NET_WEIGHTS or SAM_WEIGHTS are not set, prefer files under MODEL_CACHE_DIR/weights,
# falling back to package-local /app/models/weights if present.
if [ -z "$U2NET_WEIGHTS" ]; then
  if [ -f "${MODEL_CACHE_DIR}/weights/u2netp.pth" ]; then
    export U2NET_WEIGHTS="${MODEL_CACHE_DIR}/weights/u2netp.pth"
  elif [ -f "/app/models/weights/u2netp.pth" ]; then
    export U2NET_WEIGHTS="/app/models/weights/u2netp.pth"
  fi
fi

if [ -z "$SAM_WEIGHTS" ]; then
  if [ -f "${MODEL_CACHE_DIR}/weights/sam_vit_b_01ec64.pth" ]; then
    export SAM_WEIGHTS="${MODEL_CACHE_DIR}/weights/sam_vit_b_01ec64.pth"
  elif [ -f "/app/models/weights/sam_vit_b_01ec64.pth" ]; then
    export SAM_WEIGHTS="/app/models/weights/sam_vit_b_01ec64.pth"
  fi
fi

echo "[entrypoint] models directory listing:"
ls -lah "$MODEL_CACHE_DIR" || true

# Ensure we're in the app directory so imports like `models.*` resolve
cd /app || true
export PYTHONPATH="/app:${PYTHONPATH:-}"

echo "[entrypoint] debug: /app listing:"
ls -lah /app || true
echo "[entrypoint] debug: /app/models listing:"
ls -lah /app/models || true
echo "[entrypoint] debug: check for /app/models/__init__.py"
if [ -f /app/models/__init__.py ]; then echo "/app/models/__init__.py exists"; else echo "/app/models/__init__.py MISSING"; fi
echo "[entrypoint] debug: print python sys.path"
python - <<'PY'
import sys, json
print('\n'.join(sys.path))
print(json.dumps({'cwd': __import__('os').getcwd(), 'exists_app_models': __import__('os').path.isdir('/app/models'), 'init_py': __import__('os').path.exists('/app/models/__init__.py')}))
PY

# If FIREBASE_SERVICE_ACCOUNT_JSON is a path to a file and file exists, nothing to do.
# If it's a JSON string, the app's initialization helper already handles parsing.

if [ "$#" -eq 0 ]; then
  echo "No command supplied, launching uvicorn via python -m uvicorn"
  # Use the Python interpreter to run uvicorn so PYTHONPATH and current working
  # directory are respected in the same process environment.
  exec python -m uvicorn app:app --host 0.0.0.0 --port 8000
else
  exec "$@"
fi
