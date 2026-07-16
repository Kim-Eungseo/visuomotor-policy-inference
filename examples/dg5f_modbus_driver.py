#!/usr/bin/env python3
"""Standalone Modbus driver for the DG5F hand — OUR node, independent of the vendor
(Tesollo delto_hardware) ros2_control driver.

After the 2026-07 repair the hand's firmware speaks **Modbus TCP** (input/holding registers),
not the old Tesollo developer-TCP blob the vendor driver expects — so that driver hangs. This
node talks Modbus directly and drops into the exact topics the teleop GUI / policy already use,
so nothing downstream changes:

  publishes   /dg5f_<side>/joint_states            sensor_msgs/JointState     (20 joints, rad)
  subscribes  /dg5f_<side>/lj_dg_pospid/reference  control_msgs/MultiDOFCommand (rad targets)

Behaviour: passive (read-only, publishes joint_states) until the FIRST command arrives; then it
enables the hand (SYSTEM_STOP_START=1) and position-controls it, with per-tick slew limiting and
a global range clamp so a bad command can't cause a violent motion.

  python3 dg5f_modbus_driver.py --side left                 # auto-pick the hand IP
  python3 dg5f_modbus_driver.py --side left --ip 169.254.186.73
"""
import argparse
import math
import socket
import struct
import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from control_msgs.msg import MultiDOFCommand

RAD_SCALE = math.pi / 1800.0        # register unit -> rad (matches vendor dg5f_operator_TCP.hpp)
MOTOR_NUM = 20
REG_STARTSTOP = 0                   # holding: SYSTEM_STOP_START (1 = enable/hold)
REG_TARGET_POS = 7                  # holding: MOTOR1_TARGET_POSITION
REG_CUR_POS = 6                     # input:   MOTOR1_CURRENT_POSITION
# Safety envelope (rad). The DG5F's deep-curl joints (_3) reach ~2.75 in a full grip, so a tight
# ±1.75 clamp would cut off the finger close (and break the staged sequential curl). Cover the real
# recorded pose range (-1.599 .. 2.747) with margin; the hand's own hardware stops are the last line.
POS_MIN, POS_MAX = -1.8, 3.0
# The hand answers on the main robot subnet OR its native link-local; try both.
IP_CANDIDATES = ["192.168.4.73", "169.254.186.73"]


class Modbus:
    """Minimal Modbus/TCP client (input-register read + holding-register write). No deps."""

    def __init__(self, ip, port, timeout=1.0):
        self.ip, self.port, self.timeout = ip, port, timeout
        self.s = None
        self.tid = 0

    def connect(self):
        self.close()
        self.s = socket.create_connection((self.ip, self.port), timeout=2.0)
        self.s.settimeout(self.timeout)

    def close(self):
        if self.s is not None:
            try:
                self.s.close()
            except OSError:
                pass
            self.s = None

    def _txn(self, pdu):
        self.tid = (self.tid + 1) & 0xFFFF
        self.s.sendall(struct.pack(">HHH", self.tid, 0, len(pdu) + 1) + bytes([0x01]) + pdu)

    def _recv(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.s.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("socket closed")
            buf += chunk
        return buf

    def read_input(self, addr, qty):
        self._txn(bytes([0x04]) + struct.pack(">HH", addr, qty))
        head = self._recv(9)                    # MBAP(7) + func(1) + bytecount(1)
        if head[7] != 0x04:
            raise RuntimeError(f"modbus exception/func 0x{head[7]:02X}")
        data = self._recv(head[8])
        return list(struct.unpack(">" + "h" * (head[8] // 2), data))  # signed int16

    def write_single(self, addr, value):
        self._txn(bytes([0x06]) + struct.pack(">Hh", addr, int(value)))
        if self._recv(12)[7] != 0x06:
            raise RuntimeError("write_single failed")

    def write_multi(self, addr, values):
        body = b"".join(struct.pack(">h", int(v)) for v in values)
        self._txn(bytes([0x10]) + struct.pack(">HH", addr, len(values)) + bytes([len(body)]) + body)
        if self._recv(12)[7] != 0x10:
            raise RuntimeError("write_multi failed")


class DG5FModbusDriver(Node):
    def __init__(self, side, ip, port, rate, slew, hold):
        super().__init__(f"dg5f_modbus_driver_{side}")
        self.side = side
        prefix = "lj" if side == "left" else "rj"
        self.names = [f"{prefix}_dg_{f}_{j}" for f in range(1, 6) for j in range(1, 5)]
        self.name_to_idx = {n: i for i, n in enumerate(self.names)}
        self.slew = slew                        # max rad change per tick (per joint)
        # The repaired DG5F's Modbus position servo RE-PLANS on every new target, so a continuous
        # high-rate target stream stalls it (it never finishes a move — the joint only jumps when the
        # target finally holds still). So we HOLD each written target for hold_ticks, letting the servo
        # reach it, then step to the latest command. This is what makes the staged grip curl track.
        self.hold_ticks = max(1, round(hold * rate))
        self._hold = 0

        self.mb = Modbus(ip, port)
        self._connect_or_die()

        self.cur = [0.0] * MOTOR_NUM            # latest measured position (rad)
        self.cmd = None                         # desired target (rad) — None = passive
        self.written = None                     # last commanded setpoint (rad), for slewing
        self.enabled = False

        self.pub = self.create_publisher(JointState, f"/dg5f_{side}/joint_states", 10)
        self.sub = self.create_subscription(
            MultiDOFCommand, f"/dg5f_{side}/lj_dg_pospid/reference", self.on_command, 10)
        self.timer = self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(
            f"DG5F Modbus driver up: {self.mb.ip}:{self.mb.port} side={side} @ {rate}Hz "
            f"(READ-ONLY until first command on /dg5f_{side}/lj_dg_pospid/reference)")

    def _connect_or_die(self):
        candidates = [self.mb.ip] if self.mb.ip else IP_CANDIDATES
        for cand in candidates:
            try:
                self.mb.ip = cand
                self.mb.connect()
                self.get_logger().info(f"Modbus connected: {cand}:{self.mb.port}")
                return
            except OSError as e:
                self.get_logger().warn(f"connect {cand} failed: {e}")
        self.get_logger().error("DG5F not reachable on any candidate IP")
        raise SystemExit(1)

    def on_command(self, msg: MultiDOFCommand):
        # Absolute joint targets (rad), ordered by dof_names. Hold unspecified joints where they are.
        target = list(self.written) if self.written is not None else list(self.cur)
        for name, value in zip(msg.dof_names, msg.values):
            idx = self.name_to_idx.get(name)
            if idx is not None:
                target[idx] = float(value)
        self.cmd = target

    def tick(self):
        try:
            regs = self.mb.read_input(REG_CUR_POS, MOTOR_NUM)
        except (OSError, RuntimeError) as e:
            self.get_logger().warn(f"read failed ({e}); reconnecting")
            self.enabled = False
            try:
                self.mb.connect()
            except OSError:
                pass
            return

        self.cur = [r * RAD_SCALE for r in regs]
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = self.names
        js.position = self.cur
        self.pub.publish(js)

        if self.cmd is None:                    # passive until first command — no motion
            return
        if self.written is None:
            self.written = list(self.cur)       # start slewing from where we are (no jump)

        try:
            if not self.enabled:
                self.mb.write_single(REG_STARTSTOP, 1)   # enable / hold
                self.enabled = True
                self.get_logger().info("hand ENABLED (position control active)")
            # Step-and-hold: advance the written target to the latest command only every hold_ticks
            # (holding it in between) so the servo can actually reach each step; slew-limit the step so
            # a big jump is still bounded. Writing the SAME held target every tick is fine (no re-plan).
            self._hold += 1
            if self._hold >= self.hold_ticks:
                self._hold = 0
                step = self.slew * self.hold_ticks           # max advance per hold window
                for i in range(MOTOR_NUM):
                    delta = max(-step, min(step, self.cmd[i] - self.written[i]))
                    self.written[i] = min(POS_MAX, max(POS_MIN, self.written[i] + delta))
            self.mb.write_multi(REG_TARGET_POS, [round(w / RAD_SCALE) for w in self.written])
        except (OSError, RuntimeError) as e:
            self.get_logger().warn(f"write failed ({e}); reconnecting")
            self.enabled = False
            try:
                self.mb.connect()
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["left", "right"], default="left")
    ap.add_argument("--ip", default=None, help="hand IP (default: auto-pick .4.73 / .186.73)")
    ap.add_argument("--port", type=int, default=502)
    ap.add_argument("--rate", type=float, default=100.0, help="read/publish/control Hz")
    # slew is only a coarse cap on a single-tick jump; keep it well ABOVE the GUI's intended motion
    # (the staged grip curl reaches ~4 rad/s) so the driver faithfully tracks the commanded sequence
    # instead of blurring it. The hand's own position servo is the real speed limit.
    ap.add_argument("--slew", type=float, default=0.15, help="max rad change per tick per joint (100Hz -> 15 rad/s)")
    # hold: how long to keep each written target before stepping to the latest command. The DG5F's
    # Modbus servo needs a held target to complete a move; too short -> it stalls (staged curl fails).
    ap.add_argument("--hold", type=float, default=0.25, help="seconds to hold each target step (servo settle)")
    args = ap.parse_args()

    rclpy.init()
    node = DG5FModbusDriver(args.side, args.ip, args.port, args.rate, args.slew, args.hold)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.mb.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
