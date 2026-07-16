#!/usr/bin/env bash
# Convert the GUI's RAW teleop recording (PNG images + per-episode parquet) into a real LeRobot v3.0
# dataset (image mode, no video), using the actual lerobot API inside the inference image. Run this
# AFTER recording episodes in the GUI (the GUI only writes raw/; this builds the loadable dataset).
#
#   ./to_lerobot.sh [raw_dir] [out_dir] [repo_id]
#
# Defaults match teleop_gui.sh's VPI_DATASET_DIR (=.teleop_state/lerobot_dataset):
#   raw = .teleop_state/lerobot_dataset/raw     out = .teleop_state/lerobot_dataset/lerobot
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DS="$REPO/.teleop_state/lerobot_dataset"
RAW="${1:-$DS/raw}"
OUT="${2:-$DS/lerobot}"
REPO_ID="${3:-rec/teleop}"
IMG="visuomotor-policy-inference:latest"

[ -d "$RAW" ] || { echo "[to_lerobot] no raw dir: $RAW  (record some episodes in the GUI first)"; exit 1; }
NEPS=$(find "$RAW" -maxdepth 2 -name frames.parquet 2>/dev/null | wc -l)
[ "$NEPS" -gt 0 ] || { echo "[to_lerobot] no finished episodes under $RAW"; exit 1; }

# INCREMENTAL: the converter appends only NEW raw episodes to $OUT (tracked via a manifest). For a
# full rebuild from scratch, delete $OUT first:  sudo rm -rf "$OUT"
OUTP="$(dirname "$OUT")"; OUTB="$(basename "$OUT")"
mkdir -p "$OUTP"

echo "[to_lerobot] $NEPS episode(s)"
echo "[to_lerobot] raw = $RAW"
echo "[to_lerobot] out = $OUT   repo_id=$REPO_ID"

docker run --rm --network host \
  -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 -e VPI_SEG_URL="${VPI_SEG_URL:-http://localhost:8100}" \
  -v "$RAW:/raw:ro" \
  -v "$OUTP:/outp" \
  -v "$REPO/examples/convert_raw_to_lerobot.py:/convert.py:ro" \
  "$IMG" \
  python3 /convert.py --raw /raw --out "/outp/$OUTB" --repo-id "$REPO_ID"

rc=$?
if [ $rc -eq 0 ]; then
  echo "[to_lerobot] ✅ done -> $OUT"
  echo "             load:  HF_HUB_OFFLINE=1 python -c \"from lerobot.datasets.lerobot_dataset import LeRobotDataset as D; d=D(repo_id='$REPO_ID', root='$OUT'); print(d.num_episodes,'eps',d.num_frames,'frames')\""
else
  echo "[to_lerobot] ❌ conversion failed (rc=$rc)"
fi
exit $rc
