#!/usr/bin/env bash
# Start the RF-DETR segmentation server on GPU. Uses the inference image (CUDA torch); rfdetr is
# pip-installed at container start (it reuses the image's existing CUDA torch, no reinstall). The GUI
# polls this at VPI_SEG_URL (default http://localhost:8100). First start takes ~1-2 min (install + load).
#
#   ./seg_server.sh            # start on :8100
#   ./seg_server.sh 8100 stop  # stop
set -uo pipefail

PORT="${1:-8100}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 2-class (hand + hook) RF-DETR. Override with VPI_SEG_WEIGHTS / VPI_SEG_NUM_CLASSES.
WEIGHTS="${VPI_SEG_WEIGHTS:-$REPO/models/segmentation/best_2cls_ver1.pth}"
NUM_CLASSES="${VPI_SEG_NUM_CLASSES:-2}"
IMG="visuomotor-policy-inference:latest"
NAME="vpi_seg"

if [ "${2:-}" = stop ]; then docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped $NAME"; exit 0; fi
[ -f "$WEIGHTS" ] || { echo "ERROR: seg weights not found: $WEIGHTS"; exit 1; }

docker rm -f "$NAME" >/dev/null 2>&1 || true
echo "[seg] starting $NAME on :$PORT (GPU) — first run installs rfdetr + loads model (~1-2 min)"
WNAME="$(basename "$WEIGHTS")"
docker run -d --name "$NAME" --gpus all --network host \
  -e SEG_WEIGHTS="/weights/$WNAME" -e SEG_NUM_CLASSES="$NUM_CLASSES" -e SEG_PORT="$PORT" \
  -v "$WEIGHTS:/weights/$WNAME:ro" \
  -v "$REPO/app/seg_server.py:/app/seg_server.py:ro" \
  "$IMG" bash -lc "pip install -q rfdetr 'numpy==2.1.3' 2>&1 | tail -1; python3 /app/seg_server.py"

echo "[seg] waiting for /health ..."
for i in $(seq 1 60); do
  if curl -s --max-time 3 "http://localhost:$PORT/health" 2>/dev/null | grep -q '"ok":true'; then
    echo "[seg] ready:  $(curl -s http://localhost:$PORT/health)"; exit 0
  fi
  sleep 3
done
echo "[seg] not ready yet — check: docker logs $NAME"
