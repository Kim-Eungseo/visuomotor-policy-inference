#!/usr/bin/env bash
# Mindlessly fix a stuck hand. Does EVERYTHING it can in software, then verifies motion.
#   ./fix_hand.sh right    # Inspire (serial)
#   ./fix_hand.sh left     # DG5F   (Modbus TCP @ 192.168.4.73 or 169.254.186.73 : 502)
#
# LEFT (DG5F): fixes the host IP conflict, waits for the gripper's Modbus port, restarts the
# driver, waits for the controllers + joint_states, retries, then does a grip->open motion test.
# If the DG5F's Modbus (502) never comes up, that's a HARDWARE freeze — power-cycle the hand.
# RIGHT (Inspire): restarts the serial driver (auto-detects the FTDI port) and verifies motion.
set -uo pipefail

SIDE="${1:-right}"
[ "$SIDE" = left ] || [ "$SIDE" = right ] || { echo "usage: $0 {left|right}"; exit 1; }
CONTAINER="ros2_teleop_system"
HAND="teleop_hand_${SIDE}"
MAX=4

docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$" \
  || { echo "ERROR: '$CONTAINER' not running -> ./up.sh $SIDE"; exit 1; }

# ---- topic / command config per side ----
if [ "$SIDE" = right ]; then
  TOPIC=/inspire/right/target;  TYPE=std_msgs/msg/Float64MultiArray;  STATE=/inspire/joint_states
  GRIP='{data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}'
  OPEN='{data: [0.9, 0.9, 0.9, 0.9, 0.6, 0.6]}'
  MOVE_MIN=15
else
  TOPIC=/dg5f_left/lj_dg_pospid/reference;  TYPE=control_msgs/msg/MultiDOFCommand;  STATE=/dg5f_left/joint_states
  N='[lj_dg_1_1,lj_dg_1_2,lj_dg_1_3,lj_dg_1_4,lj_dg_2_1,lj_dg_2_2,lj_dg_2_3,lj_dg_2_4,lj_dg_3_1,lj_dg_3_2,lj_dg_3_3,lj_dg_3_4,lj_dg_4_1,lj_dg_4_2,lj_dg_4_3,lj_dg_4_4,lj_dg_5_1,lj_dg_5_2,lj_dg_5_3,lj_dg_5_4]'
  Z='[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'
  G='[0,0.6,0.8,0.6, 0,0.6,0.8,0.6, 0,0.6,0.8,0.6, 0,0.6,0.8,0.6, 0,0.6,0.8,0.6]'
  GRIP="{dof_names: $N, values: $G, values_dot: $Z}"
  OPEN="{dof_names: $N, values: $Z, values_dot: $Z}"
  MOVE_MIN=0.2
fi

echo "[fix_hand] side=$SIDE"

# ---- LEFT preflight: resolve the hand IP (.4.73 or .186.73) + wait for Modbus ----
if [ "$SIDE" = left ]; then
  # shellcheck source=dg5f_ip.sh
  source "$(dirname "$0")/dg5f_ip.sh"
  echo -n "[fix_hand] locating DG5F Modbus (tries .4.73 / .186.73 : $DG5F_PORT) "
  up=0
  for i in $(seq 1 20); do
    if resolve_dg5f_ip; then up=1; echo " OPEN @ $DG5F_IP:$DG5F_PORT"; break; fi
    echo -n "."; sleep 2
  done
  if [ "$up" -eq 0 ]; then
    echo " STILL CLOSED"
    echo "[fix_hand] ❌ DG5F Modbus (502) is down on all known IPs = HARDWARE freeze."
    echo "           Power-cycle the DG5F hand (off, wait ~5s, on), then re-run: ./fix_hand.sh left"
    echo "           (verify with: ./dg5f_probe.py)"
    exit 1
  fi
fi

# ---- returns 0 if a grip<->open command produces real motion ----
test_move() {
  docker exec -e TOPIC="$TOPIC" -e TYPE="$TYPE" -e STATE="$STATE" -e GRIP="$GRIP" -e OPEN="$OPEN" -e MOVE_MIN="$MOVE_MIN" \
    "$CONTAINER" bash -lc '
    source /opt/ros/humble/setup.bash; export ROS_DOMAIN_ID=0
    read6() { timeout 4 ros2 topic echo --once "$STATE" 2>/dev/null | sed -n "/position:/,/]/p" | grep -E "^- " | head -6 | sed "s/^- *//" | tr "\n" " "; }
    timeout 8 ros2 topic pub --once -w 1 "$TOPIC" "$TYPE" "$GRIP" >/dev/null 2>&1; sleep 2; A="$(read6)"
    timeout 8 ros2 topic pub --once -w 1 "$TOPIC" "$TYPE" "$OPEN" >/dev/null 2>&1; sleep 2; B="$(read6)"
    python3 -c "
import os,sys
a=[float(x) for x in \"\"\"$A\"\"\".split()]; b=[float(x) for x in \"\"\"$B\"\"\".split()]
d=max((abs(x-y) for x,y in zip(a,b)), default=0.0)
print(\"   grip=\", [round(x,2) for x in a]); print(\"   open=\", [round(x,2) for x in b])
print(\"   max move=%.2f -> %s\" % (d, \"MOVES\" if d>float(os.environ[\"MOVE_MIN\"]) else \"STUCK\"))
sys.exit(0 if d>float(os.environ[\"MOVE_MIN\"]) else 1)
"'
}

# ---- wait until joint_states actually streams (controllers spawned + hardware connected) ----
wait_state() {
  for _ in $(seq 1 15); do
    r=$(docker exec "$CONTAINER" bash -lc "source /opt/ros/humble/setup.bash; export ROS_DOMAIN_ID=0;
      timeout 3 ros2 topic hz $STATE 2>/dev/null | grep -m1 'average rate'" 2>/dev/null)
    [ -n "$r" ] && return 0
    sleep 2
  done
  return 1
}

for a in $(seq 1 $MAX); do
  echo ">> attempt $a/$MAX: docker restart $HAND ..."
  docker restart "$HAND" >/dev/null 2>&1
  echo -n "   waiting for $STATE ... "
  if wait_state; then echo "streaming"; else echo "no state yet"; continue; fi
  if test_move; then
    echo "[fix_hand] ✅ FIXED — hand moves (grip->open verified)."
    exit 0
  fi
  echo "   still stuck, retrying ..."
done

echo "[fix_hand] ❌ FAILED after $MAX attempts."
docker logs "$HAND" 2>&1 | tail -4 | sed 's/^/   /'
[ "$SIDE" = left ]  && echo "   -> DG5F may need a power-cycle. Check ./dg5f_probe.py"
[ "$SIDE" = right ] && echo "   -> check the FTDI serial: ls -l /dev/serial/by-id/"
exit 1
