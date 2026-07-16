#!/usr/bin/env bash
# Launch the keyboard teleop GUI for the RIGHT arm + Inspire hand.
# Runs in a fresh teleop container (PyQt5 + rclpy + hdr_stream + host network + X11).
#
#   ./teleop_gui.sh                 # right arm + hand
#   ./teleop_gui.sh right --no-hand # arm only
#   ./teleop_gui.sh right --no-arm  # hand only
#
# Prereq: robot network up (enp4s0 link) for the arm; the hand driver (teleop_hand_right)
# running for the hand. Do NOT run while run_closed_loop is driving the arm (one control
# session at a time).
set -uo pipefail

SIDE="right"
case "${1:-}" in left|right) SIDE="$1"; shift;; esac
PASS="$*"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TELEOP="$HOME/Tesollo/1.Project/1.hyundae/seoul_uiwang/system_Teleop"   # holds src/Robot_/.../hdr_stream
IMG="ros2-teleop:latest"

# robot-network IP on enp4s0 (arm + hand reachability); harmless if already set
sudo ip addr add 192.168.4.55/24 dev enp4s0 2>/dev/null || true
# The GUI must show on the ROBOT PC's physical monitor (:0), NOT an SSH-forwarded display
# (localhost:10.0). Force :0 regardless of how you connected.
DISPLAY=":0"
# Allow the root container to use the desktop X server. The graphical session is owned by
# user 'pin', so authorize via pin's X auth (we usually run this as a different user).
sudo -u pin DISPLAY="$DISPLAY" XAUTHORITY=/home/pin/.Xauthority xhost +local: >/dev/null 2>&1 \
  || xhost +local: >/dev/null 2>&1 || true

mkdir -p "$REPO/.teleop_state"     # persists the Home pose across restarts

# SINGLE INSTANCE per side: two GUIs on the same side would both drive the arm (OpenStream) and
# publish to the hand → they fight. Replace any existing GUI for THIS side (and the legacy name).
NAME="teleop_gui_${SIDE}"
if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "[teleop_gui] a $SIDE GUI is already running — stopping it first (avoids a control conflict)."
fi
docker rm -f "$NAME" teleop_gui >/dev/null 2>&1 || true

echo "[teleop_gui] side=$SIDE  ${PASS}"
TTYFLAG=""; [ -t 0 ] && TTYFLAG="-it"   # interactive TTY only when run from a terminal (not headless)
docker run --rm $TTYFLAG --name "$NAME" \
  --network host \
  -e DISPLAY="${DISPLAY:-:0}" -e ROS_DOMAIN_ID=0 -e QT_X11_NO_MITSHM=1 -e VPI_STATE_DIR=/state \
  -e VPI_HOST_REPO="$REPO" -e VPI_HOST_STATE="$REPO/.teleop_state" \
  -e VPI_INFER_URL="http://localhost:8000" -e VPI_SEG_URL="http://localhost:8100" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /dev/shm:/dev/shm:rw \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$TELEOP:/workspace" \
  -v "$REPO/.teleop_state:/state" \
  -v "$REPO/examples/teleop_gui.py:/tmp/teleop_gui.py:ro" \
  -v "$REPO/examples/hdr35_20.urdf:/tmp/hdr35_20.urdf:ro" \
  -v "${VPI_HF_CACHE:-$HOME/.cache/huggingface}:/hf:ro" \
  "$IMG" \
  bash -lc "source /opt/ros/humble/setup.bash; pip install -q ikpy pyarrow 2>/dev/null; \
            python3 -u /tmp/teleop_gui.py --side $SIDE $PASS"
