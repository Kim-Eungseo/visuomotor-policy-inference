#!/usr/bin/env bash
# Retroactively mask raw episodes to hand+hook via the GPU seg server (the GUI does this automatically
# on STOP; this is for episodes recorded before that, or to re-mask). Overwrites the raw PNGs.
#   ./mask_raw.sh                       # all episodes
#   ./mask_raw.sh episode_000001        # just this one
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW="$REPO/.teleop_state/lerobot_dataset/raw"
[ -d "$RAW" ] || { echo "no raw dir: $RAW"; exit 1; }
curl -s --max-time 3 http://localhost:8100/health 2>/dev/null | grep -q '"ok":true' \
  || { echo "seg server not up — run ./seg_server.sh first"; exit 1; }

docker run --rm --network host \
  -v "$RAW:/raw" \
  -v "$REPO/examples/mask_raw_episodes.py:/m.py:ro" \
  ros2-teleop:latest python3 /m.py --raw /raw ${@:+--episodes "$@"}
