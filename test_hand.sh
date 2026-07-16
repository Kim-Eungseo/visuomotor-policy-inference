#!/usr/bin/env bash
# Quick hand test: GRIP then OPEN, and report whether the hand actually moved.
#   ./test_hand.sh right    # Inspire (/inspire/right/target, 6 normalized)
#   ./test_hand.sh left     # DG5F   (/dg5f_left/lj_dg_pospid/reference, 20 joints rad)
#
# Prereq: that side's hand driver up (./up.sh <side>).
#   right (Inspire): if it doesn't move it's a serial freeze -> ./fix_hand.sh right
#   left  (DG5F):    needs the hand powered + Modbus reachable (192.168.4.73 or 169.254.186.73 :502). If the
#                    ros2_control controller isn't active this script says so instead of
#                    silently doing nothing.
set -uo pipefail

SIDE="${1:-right}"
[ "$SIDE" = left ] || [ "$SIDE" = right ] || { echo "usage: $0 {left|right}"; exit 1; }
CONTAINER="ros2_teleop_system"

if [ "$SIDE" = right ]; then
  TOPIC=/inspire/right/target
  TYPE=std_msgs/msg/Float64MultiArray
  STATE=/inspire/joint_states
  # Inspire: low normalized -> fingers CLOSED (grip), high -> OPEN (verified on hardware)
  GRIP='{data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}'
  OPEN='{data: [0.9, 0.9, 0.9, 0.9, 0.6, 0.6]}'
  MOVE_MIN=15          # inspire state ~[89..174]; a real grip<->open swings ~80
else
  TOPIC=/dg5f_left/lj_dg_pospid/reference
  TYPE=control_msgs/msg/MultiDOFCommand
  STATE=/dg5f_left/joint_states
  # DG5F: 20 joints in RADIANS, canonical sequential order lj_dg_<finger>_<joint> (finger 1..5,
  # joint 1..4). OPEN = all 0. GRIP curls joints _2/_3/_4 of each finger (values within URDF limits).
  N='[lj_dg_1_1,lj_dg_1_2,lj_dg_1_3,lj_dg_1_4,lj_dg_2_1,lj_dg_2_2,lj_dg_2_3,lj_dg_2_4,lj_dg_3_1,lj_dg_3_2,lj_dg_3_3,lj_dg_3_4,lj_dg_4_1,lj_dg_4_2,lj_dg_4_3,lj_dg_4_4,lj_dg_5_1,lj_dg_5_2,lj_dg_5_3,lj_dg_5_4]'
  Z='[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'
  G='[0,0.6,0.8,0.6, 0,0.6,0.8,0.6, 0,0.6,0.8,0.6, 0,0.6,0.8,0.6, 0,0.6,0.8,0.6]'
  GRIP="{dof_names: $N, values: $G, values_dot: $Z}"
  OPEN="{dof_names: $N, values: $Z, values_dot: $Z}"
  MOVE_MIN=0.2         # dg5f state in rad; a real grip<->open swings ~0.6+ on curled joints
fi

echo "[test_hand] side=$SIDE  topic=$TOPIC"
docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$" \
  || { echo "ERROR: '$CONTAINER' not running -> ./up.sh $SIDE"; exit 1; }

# Preflight: the hand controller/driver must be subscribed to the command topic, else the
# publish lands nowhere (DG5F: controller not spawned because hardware/Modbus not ready).
subs=$(docker exec "$CONTAINER" bash -lc "source /opt/ros/humble/setup.bash; export ROS_DOMAIN_ID=0;
  ros2 topic info $TOPIC 2>/dev/null | sed -n 's/.*Subscription count: //p'" 2>/dev/null)
if [ "${subs:-0}" = "0" ]; then
  echo "ERROR: no subscriber on $TOPIC — the hand controller/driver is not active."
  if [ "$SIDE" = left ]; then
    # resolve which hand IP is actually live (.4.73 or .186.73) and report it
    source "$(dirname "$0")/dg5f_ip.sh"
    if resolve_dg5f_ip; then
      echo "       DG5F Modbus IS up at $DG5F_IP:$DG5F_PORT but no controller — driver not spawned."
    else
      echo "       DG5F not reachable on any known IP (tried .4.73 / .186.73). Check hand power + ethernet."
    fi
    echo "       docker restart teleop_hand_left   (controller spawns once Modbus connects)"
  else
    echo "       ./fix_hand.sh right"
  fi
  exit 1
fi

# Grip + open in ONE container session (warm ROS discovery; -w 1 waits for the subscriber).
docker exec -e TOPIC="$TOPIC" -e TYPE="$TYPE" -e STATE="$STATE" -e GRIP="$GRIP" -e OPEN="$OPEN" -e MOVE_MIN="$MOVE_MIN" \
  "$CONTAINER" bash -lc '
  source /opt/ros/humble/setup.bash; export ROS_DOMAIN_ID=0
  pub()   { timeout 8 ros2 topic pub --once -w 1 "$TOPIC" "$TYPE" "$1" >/dev/null 2>&1; }
  read6() { timeout 4 ros2 topic echo --once "$STATE" 2>/dev/null | sed -n "/position:/,/]/p" | grep -E "^- " | head -6 | sed "s/^- *//" | tr "\n" " "; }
  echo ">> 잡기 (GRIP)"; pub "$GRIP"; sleep 2; A="$(read6)"; echo "   state: $A"
  echo ">> 펴기 (OPEN)"; pub "$OPEN"; sleep 2; B="$(read6)"; echo "   state: $B"
  python3 -c "
import os
a=[float(x) for x in \"\"\"$A\"\"\".split()]; b=[float(x) for x in \"\"\"$B\"\"\".split()]
d=max((abs(x-y) for x,y in zip(a,b)), default=0.0)
print(\"   max move=%.2f  -> %s\" % (d, \"MOVES\" if d>float(os.environ[\"MOVE_MIN\"]) else \"NO MOVEMENT (check hand power/serial)\"))
" 2>/dev/null || true
'
echo "[test_hand] done"
