# Ensure weights dir exists under model cache
mkdir -p "${MODEL_CACHE_DIR}/weights"
#!/usr/bin/env bash
set -e

# Simple entrypoint: optionally download model weights, then exec the passed command (or uvicorn)

echo "[entrypoint] starting..."

# Default download locations (if you prefer, set env vars to override):
# U2NET_DOWNLOAD_URL, SAM_DOWNLOAD_URL

download_to_tmp() {
  # Lightweight helper that downloads $1 -> $2.part then moves into place.
  local url="$1"; local out="$2"
  mkdir -p "$(dirname "$out")"
  tmp_dest="${out}.part"
  if echo "$url" | grep -qi "drive.google.com" && command -v gdown >/dev/null 2>&1; then
    gdown -O "$tmp_dest" "$url" || true
  elif command -v wget >/dev/null 2>&1; then
    wget -q --show-progress -O "$tmp_dest" "$url" || true
  elif command -v curl >/dev/null 2>&1; then
    curl -sSL -o "$tmp_dest" "$url" || true
  else
    return 1
  fi
  if [ -s "$tmp_dest" ]; then
    mv "$tmp_dest" "$out"
    return 0
  else
    [ -f "$tmp_dest" ] && rm -f "$tmp_dest" || true
    return 1
  fi
}

# Allow configurable model cache directory (default /models)
MODEL_CACHE_DIR=${MODEL_CACHE_DIR:-/models}

# Ensure weights dir exists under model cache
mkdir -p "${MODEL_CACHE_DIR}/weights"

# Normalize USE_SAM
USE_SAM=${USE_SAM:-true}
use_sam_lc=$(echo "$USE_SAM" | tr '[:upper:]' '[:lower:]')

# prefer_or_download: prefer baked-in path or existing model cache file; otherwise download
prefer_or_download() {
  local url="$1"; local dest="$2"; local baked="$3"
  if [ -n "$baked" ] && [ -f "$baked" ]; then
    echo "[entrypoint] found baked-in weights at $baked"
    echo "$baked"
    return 0
  fi
  if [ -f "$dest" ]; then
    echo "[entrypoint] found existing weights at $dest"
    echo "$dest"
    return 0
  fi
  if [ -z "$url" ]; then
    return 1
  fi
  echo "[entrypoint] downloading model from $url to $dest"
  download_to_tmp "$url" "$dest" && { echo "$dest"; return 0; } || return 1
}

# Handle U2Net: prefer baked-in (/app/models/weights) then model cache, otherwise download
U2NET_DEST=${U2NET_DEST:-${MODEL_CACHE_DIR}/weights/u2netp.pth}
BAKED_U2NET=/app/models/weights/u2netp.pth
chosen_u2=$(prefer_or_download "$U2NET_DOWNLOAD_URL" "$U2NET_DEST" "$BAKED_U2NET") || true
if [ -n "$chosen_u2" ]; then
  export U2NET_WEIGHTS="$chosen_u2"
  echo "[entrypoint] U2NET_WEIGHTS set to $U2NET_WEIGHTS"
else
  echo "[entrypoint] no U2Net weights found or configured"
fi

# Handle SAM if enabled. We avoid baking SAM by default but allow download.
if [ "$use_sam_lc" = "1" ] || [ "$use_sam_lc" = "true" ] || [ "$use_sam_lc" = "yes" ]; then
  SAM_DEST=${SAM_DEST:-${MODEL_CACHE_DIR}/weights/sam_vit_b_01ec64.pth}
  BAKED_SAM=/app/models/weights/sam_vit_b_01ec64.pth
  chosen_sam=$(prefer_or_download "$SAM_DOWNLOAD_URL" "$SAM_DEST" "$BAKED_SAM") || true
  if [ -n "$chosen_sam" ]; then
    export SAM_WEIGHTS="$chosen_sam"
    echo "[entrypoint] SAM_WEIGHTS set to $SAM_WEIGHTS"
  else
    echo "[entrypoint] no SAM weights found or configured"
  fi
else
  echo "[entrypoint] USE_SAM is false; skipping SAM download"
fi

echo "[entrypoint] model cache listing (${MODEL_CACHE_DIR}/weights):"
ls -lah "${MODEL_CACHE_DIR}/weights" || true

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
import sys, json, os
print('\n'.join(sys.path))
print(json.dumps({'cwd': os.getcwd(), 'exists_app_models': os.path.isdir('/app/models'), 'init_py': os.path.exists('/app/models/__init__.py')}))
PY

# Launch the app
if [ "$#" -eq 0 ]; then
  echo "No command supplied, launching uvicorn via python -m uvicorn"
  PORT=${PORT:-8000}
  exec python -m uvicorn app:app --host 0.0.0.0 --port "$PORT" --proxy-headers
else
  exec "$@"
fi
