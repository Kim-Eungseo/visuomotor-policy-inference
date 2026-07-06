#!/usr/bin/env python3
"""End-effector (Cartesian) keyboard/button teleop GUI for the RIGHT HDR35 arm + Inspire hand.

You jog the END-EFFECTOR in base-frame X/Y/Z + roll/pitch/yaw; ikpy solves IK (seeded with the
current pose, so jogs stay local — no singularity flips) and a background thread streams the
joint solution to the arm (OpenStream, velocity-clamped) + the hand (/inspire/<side>/target).

Runs in a teleop container (PyQt5 + rclpy + hdr_stream + ikpy + host network + X11).

Controls (also on-screen buttons)
  EEF move:  X +W/-S   Y +A/-D   Z +R/-F          EEF rotate:  R +U/-J   P +I/-K   Y +O/-L
  Hand:      [ open    ] close    Space toggle     Speed: +/-   Resync target: Backspace   Quit: Esc
"""
from __future__ import annotations

import argparse
import math
import sys
import json
import os
import shutil
import threading
import time

import numpy as np

sys.path.insert(0, "/workspace/src/Robot_/src/hdr_stream")
from hdr_stream.utils.net import NetClient          # noqa: E402
from hdr_stream.utils.api import OpenStreamAPI       # noqa: E402
from hdr_stream.utils.parser import NDJSONParser     # noqa: E402
from hdr_stream.utils.dispatcher import Dispatcher   # noqa: E402

import rclpy                                          # noqa: E402
from rclpy.node import Node                           # noqa: E402
from std_msgs.msg import Float64MultiArray            # noqa: E402
from control_msgs.msg import MultiDOFCommand          # noqa: E402  (DG5F left hand)
from sensor_msgs.msg import JointState                # noqa: E402  (hand state feedback)
from sensor_msgs.msg import Image                     # noqa: E402  (camera RGB display)

from PyQt5 import QtCore, QtGui, QtWidgets            # noqa: E402

SEND_DT = 0.005          # 200 Hz arm/hand stream
LOOK_AHEAD = 0.01        # arm trajectory execution lag — low for responsive keyboard teleop
ARM_IP = {"right": "192.168.4.151", "left": "192.168.4.152"}
GET_JOINTS = "/project/robot/joints/joint_states"
ARM_MAX_DPS = 7.5        # per-joint arm follow clamp (deg/s) — safety / motion speed
IK_REJECT_DEG = 12.0     # reject an IK step that jumps any joint more than this (singularity guard)
GRIP_RATE = 0.5          # hand open/close speed: grip fraction per second (rate-limited for all inputs)
HAND_OPEN = np.array([0.9, 0.9, 0.9, 0.9, 0.6, 0.6])   # inspire (right) normalized: high = OPEN
# DG5F (left): 20 joints rad, canonical sequential order lj_dg_<finger>_<joint>. Each finger
# curls in its OWN direction (finger1's _3/_4 go negative, the thumb differs, etc.), so a uniform
# curl vector is wrong — the thumb ends up reversed. These OPEN/GRIP poses are taken straight from
# the recorded left episode; grip fraction g in [0,1] interpolates OPEN -> GRIP, correct per finger.
DG5F_NAMES = [f"lj_dg_{f}_{j}" for f in range(1, 6) for j in range(1, 5)]
# finger 1 = thumb (lj_dg_1_*). The THUMB (all 4 joints, incl. its inward rotation lj_dg_1_2)
# and each finger's _1 joint are HELD FIXED at the init/open pose. Only the flexion joints
# _2/_3/_4 of the 4 non-thumb fingers (12 joints) curl with the grip fraction g in [0,1].
DG5F_OPEN_POSE = np.array([-0.3, 1.3, 1.207, 0.5, -0.349, 0.0, 0.519, 0.414, -0.384, 0.317,
                           0.224, 0.178, -0.384, 0.641, 0.0, 0.0, 0.064, -0.419, 0.721, 0.0])
_DG5F_GRIP_FULL = np.array([-0.482, 1.104, -1.152, -1.599, -0.349, 1.245, 2.666, 1.92, -0.384, 1.576,
                            2.747, 1.92, -0.384, 1.632, 2.241, 1.855, 0.301, -0.419, 1.801, 1.551])
DG5F_ACTIVE = [(f - 1) * 4 + (j - 1) for f in range(2, 6) for j in range(2, 5)]  # 12 flexion joints
# grip target: fixed joints stay at open, active joints reach the recorded grip.
DG5F_GRIP_POSE = DG5F_OPEN_POSE.copy()
for _i in DG5F_ACTIVE:
    DG5F_GRIP_POSE[_i] = _DG5F_GRIP_FULL[_i]

# STAGED curl per finger: joint _1 fixed; then curl _4 (tip) -> _3 (middle) -> _2 (base) in order.
# Each joint interpolates open->grip within its window of the grip fraction g in [0,1].
DG5F_PHASE = {}
for _f in range(2, 6):
    DG5F_PHASE[(_f - 1) * 4 + 3] = (0.0, 1 / 3)      # _4 tip   (finishes first)
    DG5F_PHASE[(_f - 1) * 4 + 2] = (1 / 3, 2 / 3)    # _3 middle
    DG5F_PHASE[(_f - 1) * 4 + 1] = (2 / 3, 1.0)      # _2 base  (finishes last)


def dg5f_staged_pose(g):
    g = float(np.clip(g, 0.0, 1.0))
    out = DG5F_OPEN_POSE.copy()
    for i in DG5F_ACTIVE:
        s, e = DG5F_PHASE[i]
        local = 0.0 if g <= s else (1.0 if g >= e else (g - s) / (e - s))
        out[i] = DG5F_OPEN_POSE[i] + local * (DG5F_GRIP_POSE[i] - DG5F_OPEN_POSE[i])
    return out

QSS = """
QWidget { background:#1e1e2e; color:#cdd6f4; font-family:'DejaVu Sans'; font-size:12px; }
QGroupBox { border:1px solid #45475a; border-radius:8px; margin-top:12px; padding:10px; font-weight:bold; }
QGroupBox::title { subcontrol-origin:margin; left:12px; padding:0 4px; color:#89b4fa; }
QPushButton { background:#313244; border:1px solid #45475a; border-radius:6px; padding:6px 8px; min-width:34px; }
QPushButton:hover { background:#585b70; }
QPushButton:pressed { background:#89b4fa; color:#1e1e2e; font-weight:bold; }
QLabel#v { color:#a6e3a1; font-family:monospace; font-size:15px; }
QLabel#hint { color:#7f849c; }
QLabel#stat { font-weight:bold; }
QSlider::groove:horizontal { height:10px; background:#45475a; border-radius:5px; }
QSlider::handle:horizontal { background:#89b4fa; width:20px; border-radius:10px; margin:-6px 0; }
QSlider::sub-page:horizontal { background:#f38ba8; border-radius:5px; }
"""


def rot_about(axis, ang):
    a = np.asarray(axis, float); a = a / (np.linalg.norm(a) or 1.0)
    x, y, z = a; c, s, C = math.cos(ang), math.sin(ang), 1 - math.cos(ang)
    return np.array([[c + x*x*C, x*y*C - z*s, x*z*C + y*s],
                     [y*x*C + z*s, c + y*y*C, y*z*C - x*s],
                     [z*x*C - y*s, z*y*C + x*s, c + z*z*C]])


def mat_to_rpy(R):
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        return np.array([math.atan2(R[2, 1], R[2, 2]), math.atan2(-R[2, 0], sy),
                         math.atan2(R[1, 0], R[0, 0])])
    return np.array([math.atan2(-R[1, 2], R[1, 1]), math.atan2(-R[2, 0], sy), 0.0])


def axis_angle(R):           # rotation matrix -> (unit axis, angle rad) of the rotation
    ang = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0)))
    if ang < 1e-6:
        return np.array([1.0, 0.0, 0.0]), 0.0
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(ax)
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0]), ang
    return ax / n, ang


# Cartesian straight-line motion speeds (for Go Home / sequence). Slow for safety.
CART_LIN_SPEED = 0.03        # m/s   EEF translation along the straight path
CART_ANG_SPEED = 0.30        # rad/s EEF rotation
CART_STEP = 0.02             # s     waypoint interval (50 Hz)
CART_JUMP_DEG = 20.0         # abort if IK jumps a joint more than this between waypoints (singular)


def rpy_to_mat(r, p, y):     # radians -> 3x3, inverse of mat_to_rpy (R = Rz(y) Ry(p) Rx(r))
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# default HOME pose (EEF, base frame): xyz in metres, rpy in degrees. Persisted to disk and
# reloaded on launch; edits are saved immediately.
DEFAULT_HOME = {"x": -0.600, "y": -1.100, "z": 1.600, "roll": 160.0, "pitch": -21.0, "yaw": 102.0}


def home_config_path(side):
    # writable state dir mounted into the container (see teleop_gui.sh); falls back to /tmp.
    d = os.environ.get("VPI_STATE_DIR", "/state")
    if not os.path.isdir(d):
        d = "/tmp"
    return os.path.join(d, f"teleop_home_{side}.json")


def load_home(side, fallback=None):
    fallback = fallback or DEFAULT_HOME
    p = home_config_path(side)
    try:
        with open(p) as f:
            d = json.load(f)
        return {k: float(d.get(k, fallback[k])) for k in DEFAULT_HOME}
    except Exception:
        return dict(fallback)


def save_home(side, vals):
    try:
        with open(home_config_path(side), "w") as f:
            json.dump(vals, f, indent=2)
    except Exception as e:
        print(f"[home] save failed: {e}", flush=True)


# ---- DG5F hand OPEN/GRIP pose config (per-joint, persisted) ----
DG5F_LIMITS = {
    "lj_dg_1_1": (-0.8901, 0.3840), "lj_dg_1_2": (0.0, 3.1416), "lj_dg_1_3": (-1.5708, 1.5708), "lj_dg_1_4": (-1.5708, 1.5708),
    "lj_dg_2_1": (-0.6109, 0.4189), "lj_dg_2_2": (0.0, 2.0071), "lj_dg_2_3": (-1.5708, 1.5708), "lj_dg_2_4": (-1.5708, 1.5708),
    "lj_dg_3_1": (-0.6109, 0.6109), "lj_dg_3_2": (0.0, 1.9548), "lj_dg_3_3": (-1.5708, 1.5708), "lj_dg_3_4": (-1.5708, 1.5708),
    "lj_dg_4_1": (-0.4189, 0.6109), "lj_dg_4_2": (0.0, 1.9024), "lj_dg_4_3": (-1.5708, 1.5708), "lj_dg_4_4": (-1.5708, 1.5708),
    "lj_dg_5_1": (-1.0472, 0.0175), "lj_dg_5_2": (-0.6109, 0.4189), "lj_dg_5_3": (-1.5708, 1.5708), "lj_dg_5_4": (-1.5708, 1.5708),
}


def grip_config_path(side):
    d = os.environ.get("VPI_STATE_DIR", "/state")
    if not os.path.isdir(d):
        d = "/tmp"
    return os.path.join(d, f"grip_config_{side}.json")


def load_grip_config(side):
    """Return {'open':[20], 'grip':[20]} from disk, or None."""
    try:
        with open(grip_config_path(side)) as f:
            d = json.load(f)
        o, g = d.get("open"), d.get("grip")
        if o and g and len(o) == 20 and len(g) == 20:
            return {"open": [float(x) for x in o], "grip": [float(x) for x in g]}
    except Exception:
        pass
    return None


def save_grip_config(side, open_pose, grip_pose):
    try:
        with open(grip_config_path(side), "w") as f:
            json.dump({"open": [float(x) for x in open_pose], "grip": [float(x) for x in grip_pose]}, f, indent=2)
        return True
    except Exception as e:
        print(f"[grip] save failed: {e}", flush=True)
        return False


def apply_grip_config(cfg):
    """Update the module-level OPEN/GRIP poses in place (dg5f_staged_pose reads these globals)."""
    if not cfg:
        return
    DG5F_OPEN_POSE[:] = np.asarray(cfg["open"], float)
    DG5F_GRIP_POSE[:] = np.asarray(cfg["grip"], float)


class IKArm:
    """ikpy wrapper for the HDR35 (Base -> j1..j6 -> flange -> tool0)."""
    def __init__(self, urdf):
        import ikpy.chain
        mask = [False, True, True, True, True, True, True, False, False]
        self.chain = ikpy.chain.Chain.from_urdf_file(urdf, active_links_mask=mask)
        self.n = len(self.chain.links)          # 9

    def _full(self, q6):
        f = np.zeros(self.n); f[1:7] = q6; return f

    def fk(self, q6):
        return self.chain.forward_kinematics(self._full(q6))

    def ik(self, pos, R, seed6):
        sol = self.chain.inverse_kinematics(pos, R, orientation_mode="all",
                                            initial_position=self._full(seed6))
        return np.asarray(sol[1:7], float)


def step_toward(cur, tgt, max_delta):
    # JOINT-SYNCHRONIZED: scale all joints by one factor so the largest-moving joint travels at
    # max_delta and the rest move proportionally — every joint starts and arrives together.
    cur = np.asarray(cur, float); tgt = np.asarray(tgt, float)
    d = tgt - cur
    dmax = float(np.abs(d).max())
    if dmax <= max_delta or dmax == 0.0:
        return tgt.copy()
    return cur + d * (max_delta / dmax)


class ArmClient:
    def __init__(self, ip):
        self.ip = ip
        self.reconnecting = False
        self.last_state_t = 0.0
        self._recon_lock = threading.Lock()
        self._wire()

    def _wire(self):
        self.net = NetClient(self.ip, 49000); self.api = OpenStreamAPI(self.net)
        self.parser = NDJSONParser(); self.disp = Dispatcher()
        self.ok = threading.Event(); self.q = None; self._t0 = None; self.err = None
        self.disp.on_type["handshake_ack"] = lambda m: self.ok.set() if m.get("ok") else None
        self.disp.on_type["data"] = self._d
        self.disp.on_error = self._on_err

    def _on_err(self, e):
        self.err = f"{e.get('error')}: {e.get('message')}"
        print("[ARM ERR]", self.err)

    def _d(self, m):
        r = m.get("result")
        if isinstance(r, dict) and r.get("_type") == "JObject" and r.get("position"):
            self.q = [float(x) for x in r["position"]]
            self.last_state_t = time.time()

    def connect(self, t=5.0):
        self.net.connect()
        self.net.start_recv_loop(lambda b: self.parser.feed(b, self.disp.dispatch))
        self.api.handshake(major=1)
        if not self.ok.wait(t):
            raise RuntimeError("arm handshake timeout")
        self.api.monitor(url=GET_JOINTS, period_ms=4, args={}); self.api.joint_traject_init()
        self.last_state_t = time.time()

    def healthy(self):
        # a mode switch / remote-mode transition on the robot drops the monitor stream; if we
        # stop receiving joint state for a while, the OpenStream session needs a fresh handshake.
        return (time.time() - self.last_state_t) < 2.0

    def reconnect(self):
        """Full re-handshake (new socket) — needed after the robot switches to remote mode.
        Serialized by a lock so multiple presses can't spawn overlapping sockets / recv loops."""
        if not self._recon_lock.acquire(blocking=False):
            return False                     # a reconnect is already running — ignore extra presses
        self.reconnecting = True
        try:
            try: self.api.stop(target="control")     # flush the robot's trajectory buffer (old points)
            except Exception: pass
            try: self.net.close()                    # stops the old recv loop
            except Exception: pass
            time.sleep(0.4)                          # let the old recv thread / socket fully tear down
            self._wire()
            self.connect()
            print("[arm] reconnected")
            return True
        except Exception as e:
            print("[arm] reconnect failed:", e)
            return False
        finally:
            self.reconnecting = False
            self._recon_lock.release()

    def wait_state(self, t=5.0):
        t0 = time.time()
        while self.q is None:
            if time.time() - t0 > t:
                raise RuntimeError("no arm state")
            time.sleep(0.02)
        return list(self.q)

    def insert(self, deg):
        now = time.perf_counter()
        if self._t0 is None:
            self._t0 = now
        self.api.joint_traject_insert_point({"interval": SEND_DT, "time_from_start": now - self._t0,
                                             "look_ahead_time": LOOK_AHEAD,
                                             "point": [float(x) for x in deg]})

    def stop(self):
        try: self.api.stop(target="control")
        except Exception: pass

    def re_init(self):                       # restart the trajectory stream after a STOP
        try:
            self.api.joint_traject_init(); self._t0 = None
        except Exception: pass

    def close(self):
        try: self.net.close()
        except Exception: pass


class HandPub(Node):
    """Side-aware hand publisher. publish_grip(g) takes a grip fraction g in [0,1]
    (0 = open, 1 = closed) and maps it to the right hand's command:
      right = Inspire  -> Float64MultiArray on /inspire/right/target  (6 normalized)
      left  = DG5F     -> MultiDOFCommand   on /dg5f_left/lj_dg_pospid/reference (20 rad)"""
    # DG5F: the abduction joints report state with FLIPPED sign vs command, so to command a HOLD
    # at the current actual position we must negate those joints' state values.
    DG5F_FLIP = [0, 4, 8, 12, 17]     # lj_dg_1_1, 2_1, 3_1, 4_1, 5_2 (canonical indices)

    def __init__(self, side):
        super().__init__("teleop_gui_hand")
        self.side = side
        self.actual = None            # latest actual hand joints (DG5F canonical order)
        self.front = None; self.wrist = None   # latest camera RGB frames (numpy HxWx3)
        self.front_seq = 0                      # increments on each new Zivid frame (camera liveness)
        self._js_sub = None
        if side == "right":
            self.pub = self.create_publisher(Float64MultiArray, "/inspire/right/target", 1)
        else:
            self.pub = self.create_publisher(MultiDOFCommand, "/dg5f_left/lj_dg_pospid/reference", 1)
            self._resub_hand()
        # camera images from the vision node (front = Zivid, wrist = D405)
        self._cam_subs = []
        self._resub_cameras()

    def _resub_hand(self):
        """(Re)create the DG5F joint_states subscription — like the cameras, after the hand driver
        restarts the old sub can stay stuck on the dead publisher's shared-memory segment."""
        if self.side != "left":
            return
        try: self.destroy_subscription(self._js_sub)
        except Exception: pass
        self._js_sub = self.create_subscription(JointState, "/dg5f_left/joint_states", self._js, 10)

    def _resub_cameras(self):
        """(Re)create the camera subscriptions — after the vision node restarts, the old sub can
        stay stuck on the dead publisher's shared-memory segments; re-subscribing re-matches it."""
        for s in getattr(self, "_cam_subs", []):
            try: self.destroy_subscription(s)
            except Exception: pass
        img_qos = rclpy.qos.QoSProfile(depth=1, reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT)
        self._cam_subs = [
            self.create_subscription(Image, f"/system_{self.side}/zivid_rgb", self._front, img_qos),
            self.create_subscription(Image, f"/system_{self.side}/d405_rgb", self._wrist, img_qos)]

    @staticmethod
    def _decode(msg: Image):
        h, w = msg.height, msg.width
        a = np.frombuffer(msg.data, dtype=np.uint8)
        step = msg.step or w * 3
        a = a.reshape(h, step)[:, : w * 3].reshape(h, w, 3)
        if msg.encoding == "bgr8":
            a = a[:, :, ::-1]
        return np.ascontiguousarray(a)

    def _front(self, msg): self.front = self._decode(msg); self.front_seq += 1   # camera liveness
    def _wrist(self, msg): self.wrist = self._decode(msg)

    def _js(self, msg: JointState):
        idx = {n: i for i, n in enumerate(msg.name)}
        v = np.zeros(20)
        for k, n in enumerate(DG5F_NAMES):
            if n in idx:
                v[k] = msg.position[idx[n]]
        self.actual = v

    def hold_command(self):
        """DG5F command that holds the current actual pose (sign-corrected), or None."""
        if self.side != "left" or self.actual is None:
            return None
        cmd = self.actual.copy()
        for i in self.DG5F_FLIP:
            cmd[i] = -cmd[i]          # state sign is flipped vs command on these joints
        return cmd

    def publish_grip(self, g):
        g = float(np.clip(g, 0.0, 1.0))
        if self.side == "right":
            self.pub.publish(Float64MultiArray(data=[float(x) for x in HAND_OPEN * (1.0 - g)]))
        else:
            vals = dg5f_staged_pose(g).tolist()   # thumb rotation -> 4 fingers -> thumb flexion
            self.pub.publish(MultiDOFCommand(dof_names=DG5F_NAMES, values=vals, values_dot=[0.0] * 20))

    def publish_dg5f(self, vals):
        self.pub.publish(MultiDOFCommand(dof_names=DG5F_NAMES,
                                         values=[float(x) for x in vals], values_dot=[0.0] * 20))


class ControlThread(threading.Thread):
    """Streams the arm (velocity-clamped joints) + hand at SEND_DT from shared targets."""
    def __init__(self, arm, hand_pub, q0_deg, no_hand):
        super().__init__(daemon=True)
        self.arm, self.hand_pub, self.no_hand = arm, hand_pub, no_hand
        self.lock = threading.Lock()
        self.arm_cmd = np.asarray(q0_deg, float)
        self.arm_tgt = self.arm_cmd.copy()
        self.grip = 0.0                     # currently-published grip (ramped toward grip_target)
        self.grip_target = 0.0              # desired grip; grip ramps to it at GRIP_RATE
        self.running = True
        self.estopped = False               # E-STOP: freeze arm + hand
        self.hand_hold = None               # DG5F pose to hold while E-STOPped (freeze in place)
        self.hand_manual = None             # grip-config dialog: publish this 20-joint pose directly
        self.last_hand_pose = None          # last DG5F pose published (= where the hand actually is)
        self.arm_resynced = False           # set after an auto-reconnect (Window re-syncs EEF)
        self._arm_syncing = False           # hold the stream across a reconnect+resync (no stale jump)
        self._tick = 0
        self.max_dps = ARM_MAX_DPS          # per-joint speed cap (settable; sequence lowers it)

    def estop(self):
        with self.lock:
            self.estopped = True
            self.arm_tgt = self.arm_cmd.copy()      # freeze arm target at current
            self.max_dps = ARM_MAX_DPS              # drop any sequence speed-up (else resume jumps)
            self.grip_target = self.grip            # freeze the grip ramp too
            # freeze the hand at the LAST published command — that's exactly where it is now, so
            # re-sending it holds it in place (no actual-readback sign issues).
            self.hand_hold = None if self.last_hand_pose is None else np.asarray(self.last_hand_pose, float).copy()
        if self.arm is not None:
            self.arm.stop()                          # OpenStream hard STOP

    def resume(self):
        # re-sync command AND target to the ACTUAL robot position so nothing jumps on resume,
        # and force the safe speed cap (a sequence may have left it high).
        if self.arm is not None:
            if self.arm.q is not None:
                with self.lock:
                    self.arm_cmd = np.asarray(self.arm.q[:6], float)
                    self.arm_tgt = self.arm_cmd.copy()
            self.arm.re_init()                        # restart the trajectory stream
        with self.lock:
            self.max_dps = ARM_MAX_DPS
            self.hand_hold = None
            self.estopped = False

    def set_arm_tgt(self, deg):
        with self.lock:
            if self.estopped: return          # E-STOP: ignore any target change (no queuing)
            self.arm_tgt = np.asarray(deg, float).copy()

    def set_grip(self, g):
        with self.lock:
            if self.estopped: return          # E-STOP: freeze the hand (ignore grip changes)
            self.grip_target = float(np.clip(g, 0.0, 1.0))   # grip ramps toward this at GRIP_RATE

    def set_hand_manual(self, pose):        # grip-config dialog override (None = normal grip control)
        with self.lock:
            self.hand_manual = None if pose is None else np.asarray(pose, float).copy()

    def _reconnect_arm(self):
        if self.arm is None or self.arm.reconnecting:
            return
        # Hold the stream OFF across the WHOLE reconnect + resync. reconnect() clears its own
        # `reconnecting` flag as soon as the socket is back, which would otherwise let run() stream the
        # STALE arm_cmd (an absolute joint target) before we snap it to the fresh actual pose -> a jump.
        self._arm_syncing = True
        try:
            if self.arm.reconnect():
                self.arm.stop()                               # flush any trajectory points still queued
                q = self.arm.wait_state(2.0)                  # _wire() reset q=None -> waits for FRESH state
                with self.lock:
                    self.arm_cmd = np.asarray(q[:6], float)   # snap command+target to where the arm IS now
                    self.arm_tgt = self.arm_cmd.copy()
                    self.max_dps = ARM_MAX_DPS                # drop any sequence speed-up
                    self.arm_resynced = True                  # Window clears jog + re-syncs its EEF target
        except Exception:
            pass
        finally:
            self._arm_syncing = False

    def reconnect_arm(self):                # manual trigger (button)
        threading.Thread(target=self._reconnect_arm, daemon=True).start()

    def snap(self):
        with self.lock: return self.arm_cmd.copy(), self.arm_tgt.copy(), self.grip

    def run(self):
        while self.running:
            clamp = self.max_dps * SEND_DT
            gstep = GRIP_RATE * SEND_DT
            with self.lock:
                tgt = self.arm_tgt.copy(); es = self.estopped; hold = self.hand_hold
                manual = None if self.hand_manual is None else self.hand_manual.copy()
                # rate-limit the grip: ramp toward grip_target at GRIP_RATE (all inputs obey this)
                gt = self.grip_target
                if abs(gt - self.grip) <= gstep: self.grip = gt
                else: self.grip += gstep if gt > self.grip else -gstep
                grip = self.grip
            if (self.arm is not None and not es and not self.arm.reconnecting
                    and not self._arm_syncing):             # E-STOP / reconnect / resync: pause the stream
                cmd = step_toward(self.arm_cmd, tgt, clamp)
                try: self.arm.insert(cmd)
                except Exception as e: print("[arm insert err]", e)
                with self.lock: self.arm_cmd = cmd
            # auto-reconnect: if the arm stops reporting state (robot switched to remote mode etc.),
            # re-handshake in the background so control resumes without relaunching the GUI.
            self._tick += 1
            if (self.arm is not None and self._tick % 200 == 0 and not es
                    and not self.arm.reconnecting and not self.arm.healthy()):
                threading.Thread(target=self._reconnect_arm, daemon=True).start()
            if not self.no_hand and self.hand_pub is not None:
                if self.hand_pub.side == "left":
                    # pick the pose to publish, and remember it as the last command (= holds in place)
                    if es and hold is not None:      pose = hold        # E-STOP: freeze at last command
                    elif manual is not None:         pose = manual      # grip-config dialog
                    else:                            pose = dg5f_staged_pose(grip)   # normal grip
                    self.hand_pub.publish_dg5f(pose)
                    with self.lock: self.last_hand_pose = pose
                else:
                    self.hand_pub.publish_grip(grip)      # inspire (right)
            time.sleep(SEND_DT)


def jog_button(label, on_press, on_release):
    b = QtWidgets.QPushButton(label)
    b.pressed.connect(on_press); b.released.connect(on_release)
    b.setFocusPolicy(QtCore.Qt.NoFocus)   # never steal keyboard focus from the main window
    return b


class EpisodeRecorder:
    """Record teleop episodes as a RAW folder straight from the GUI — real PNG images + a per-episode
    parquet of state/action. A separate converter (convert_raw_to_lerobot.py, run in the inference
    image that HAS lerobot) turns raw/ into a proper LeRobot v3.0 dataset via the real API. We do NOT
    hand-write v3.0 here: its packed parquet + embedded PNG + meta/episodes pointers + stats aggregation
    are too fragile to reproduce in this (lerobot-less) container and would silently fail to load.

    state[26] = arm6(rad)+hand20(rad), action[26] = commanded arm6+hand20. Layout:
        <root>/raw/episode_000000/front/000000.png  wrist/000000.png  ...  frames.parquet  meta.json
    """
    IMG_W, IMG_H = 640, 480              # both cams resized to the policy's 640x480

    def __init__(self, root, fps=2, task="teleop"):
        self.root, self.fps, self.task = root, fps, task
        self.recording = False
        self.raw = os.path.join(root, "raw")
        os.makedirs(self.raw, exist_ok=True)
        # next episode index = number of raw/episode_* dirs already present (append across launches)
        self.ep_index = len([d for d in os.listdir(self.raw) if d.startswith("episode_")])

    def start(self):
        if self.recording:
            return
        self.buf = []; self.frame_i = 0
        self.ep_dir = os.path.join(self.raw, f"episode_{self.ep_index:06d}")
        os.makedirs(os.path.join(self.ep_dir, "front"), exist_ok=True)
        os.makedirs(os.path.join(self.ep_dir, "wrist"), exist_ok=True)
        self.recording = True

    def add(self, state, action, front, wrist):
        if not self.recording:
            return
        import cv2
        i = self.frame_i
        if front is not None:      # store as real PNG (RGB->BGR for cv2.imwrite), resized to 640x480
            cv2.imwrite(os.path.join(self.ep_dir, "front", f"{i:06d}.png"),
                        cv2.resize(front, (self.IMG_W, self.IMG_H))[:, :, ::-1])
        if wrist is not None:
            cv2.imwrite(os.path.join(self.ep_dir, "wrist", f"{i:06d}.png"),
                        cv2.resize(wrist, (self.IMG_W, self.IMG_H))[:, :, ::-1])
        self.buf.append({"frame_index": i, "timestamp": float(i / self.fps),   # contiguous (idle removed)
                         "observation.state": [float(x) for x in state],
                         "action": [float(x) for x in action], "task": self.task})
        self.frame_i += 1

    def stop(self):
        if not self.recording:
            return 0
        self.recording = False
        n = len(self.buf)
        if n == 0:                                  # nothing kept — drop the empty episode dir
            shutil.rmtree(self.ep_dir, ignore_errors=True)
            return 0
        import pandas as pd
        pd.DataFrame(self.buf).to_parquet(os.path.join(self.ep_dir, "frames.parquet"), index=False)
        json.dump({"fps": self.fps, "num_frames": n, "task": self.task, "image_hw": [self.IMG_H, self.IMG_W]},
                  open(os.path.join(self.ep_dir, "meta.json"), "w"), indent=2)
        self.ep_index += 1
        return n


def docker_restart(name, timeout=8):
    """Restart another container via the Docker unix socket (mounted into this container).
    Used by the Reconnect button to re-init the vision node's cameras. No docker CLI needed."""
    import http.client
    import socket as _sock

    class _UConn(http.client.HTTPConnection):
        def connect(self):
            s = _sock.socket(_sock.AF_UNIX, _sock.SOCK_STREAM); s.settimeout(timeout)
            s.connect("/var/run/docker.sock"); self.sock = s
    try:
        c = _UConn("localhost"); c.request("POST", f"/containers/{name}/restart?t=5")
        r = c.getresponse(); r.read(); c.close()
        print(f"[reconnect] restart {name}: HTTP {r.status}", flush=True)
        return r.status in (204, 200)
    except Exception as e:
        print(f"[reconnect] restart {name} failed: {e}", flush=True)
        return False


def np_to_qpixmap(rgb, w, h):
    """numpy RGB uint8 (H,W,3) -> QPixmap fit to (w,h). Downsamples big frames with cheap numpy
    slicing FIRST (a 1944x1200 Zivid frame scaled smoothly would block the Qt event loop and
    starve the jog loop), then a fast scale."""
    hh, ww = rgb.shape[:2]
    step = max(1, min(ww // max(w, 1), hh // max(h, 1)))
    if step > 1:
        rgb = rgb[::step, ::step]
    rgb = np.ascontiguousarray(rgb)
    hh, ww = rgb.shape[:2]
    qimg = QtGui.QImage(rgb.data, ww, hh, ww * 3, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg).scaled(w, h, QtCore.Qt.KeepAspectRatio,
                                                QtCore.Qt.FastTransformation)


class GripConfigDialog(QtWidgets.QDialog):
    """Popup: adjust all 20 DG5F joints manually; save the current pose as the OPEN or GRIP pose
    (persisted to grip_config_<side>.json) and load them back. While open, it drives the hand
    directly (per-joint); on close, normal grip control resumes."""
    FING = ["Thumb (1)", "Index (2)", "Middle (3)", "Ring (4)", "Pinky (5)"]

    def __init__(self, parent, ctrl, side):
        super().__init__(parent)
        self.ctrl, self.side = ctrl, side
        self.hand = ctrl.hand_pub
        self.setWindowTitle("Grip config — DG5F per-joint (OPEN / GRIP pose)")
        self.setStyleSheet(QSS)
        self.rows = []
        init = self.hand.hold_command() if self.hand is not None else None   # current actual (sign-corr)
        if init is None:
            init = DG5F_OPEN_POSE.copy()
        self._build(init)
        self.ctrl.set_hand_manual(init)                 # take over the hand
        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self._refresh); self.timer.start(120)

    def _build(self, init):
        root = QtWidgets.QVBoxLayout(self)
        bar = QtWidgets.QHBoxLayout()
        for label, fn in [("Save current as OPEN", self._save_open), ("Save current as GRIP", self._save_grip),
                          ("Load OPEN", self._load_open), ("Load GRIP", self._load_grip),
                          ("Sync to actual", self._sync)]:
            b = QtWidgets.QPushButton(label); b.setFocusPolicy(QtCore.Qt.NoFocus)
            b.clicked.connect(fn); bar.addWidget(b)
        root.addLayout(bar)
        self.status = QtWidgets.QLabel("adjust joints; hand follows live"); self.status.setObjectName("v")
        root.addWidget(self.status)
        cols = QtWidgets.QHBoxLayout(); root.addLayout(cols)
        for f in range(5):
            gb = QtWidgets.QGroupBox(self.FING[f]); gl = QtWidgets.QGridLayout(gb)
            gl.addWidget(QtWidgets.QLabel("j"), 0, 0); gl.addWidget(QtWidgets.QLabel("cmd"), 0, 2)
            gl.addWidget(QtWidgets.QLabel("act"), 0, 3)
            for j in range(4):
                idx = f * 4 + j; lo, hi = DG5F_LIMITS[DG5F_NAMES[idx]]
                gl.addWidget(QtWidgets.QLabel(f"_{j+1}"), j + 1, 0)
                s = QtWidgets.QSlider(QtCore.Qt.Horizontal); s.setFocusPolicy(QtCore.Qt.NoFocus)
                s.setMinimum(int(lo * 1000)); s.setMaximum(int(hi * 1000))
                s.setValue(int(float(init[idx]) * 1000)); s.setMinimumWidth(130)
                s.valueChanged.connect(self._changed)
                cl = QtWidgets.QLabel(f"{init[idx]:+.2f}"); cl.setObjectName("cmd")
                al = QtWidgets.QLabel(" ? "); al.setObjectName("v")
                gl.addWidget(s, j + 1, 1); gl.addWidget(cl, j + 1, 2); gl.addWidget(al, j + 1, 3)
                self.rows.append((idx, s, cl, al))
            cols.addWidget(gb)
        cb = QtWidgets.QPushButton("Close  (resume grip control)"); cb.setFocusPolicy(QtCore.Qt.NoFocus)
        cb.clicked.connect(self.accept); root.addWidget(cb)

    def _pose(self):
        p = np.zeros(20)
        for idx, s, cl, al in self.rows:
            p[idx] = s.value() / 1000.0
        return p

    def _changed(self):
        self.ctrl.set_hand_manual(self._pose())

    def _set_sliders(self, pose):
        for idx, s, cl, al in self.rows:
            s.blockSignals(True); s.setValue(int(float(pose[idx]) * 1000)); s.blockSignals(False)
        self.ctrl.set_hand_manual(self._pose())

    def _save_open(self):
        DG5F_OPEN_POSE[:] = self._pose()
        ok = save_grip_config(self.side, DG5F_OPEN_POSE, DG5F_GRIP_POSE)
        self.status.setText("saved current pose as OPEN" + ("" if ok else " (save failed!)"))

    def _save_grip(self):
        DG5F_GRIP_POSE[:] = self._pose()
        ok = save_grip_config(self.side, DG5F_OPEN_POSE, DG5F_GRIP_POSE)
        self.status.setText("saved current pose as GRIP" + ("" if ok else " (save failed!)"))

    def _load_open(self):
        self._set_sliders(DG5F_OPEN_POSE); self.status.setText("loaded OPEN pose")

    def _load_grip(self):
        self._set_sliders(DG5F_GRIP_POSE); self.status.setText("loaded GRIP pose")

    def _sync(self):
        h = self.hand.hold_command() if self.hand is not None else None
        if h is not None:
            self._set_sliders(h); self.status.setText("synced sliders to actual")

    def _refresh(self):
        act = self.hand.actual if self.hand is not None else None
        for idx, s, cl, al in self.rows:
            cl.setText(f"{s.value()/1000.0:+.2f}")
            al.setText(" ? " if act is None else f"{act[idx]:+.2f}")

    def _release(self):
        self.timer.stop(); self.ctrl.set_hand_manual(None)   # resume normal grip control

    def accept(self): self._release(); super().accept()
    def reject(self): self._release(); super().reject()
    def closeEvent(self, ev): self._release(); ev.accept()


class TeleopWindow(QtWidgets.QMainWindow):
    def __init__(self, ctrl, ik, side, has_arm, has_hand):
        super().__init__()
        self.ctrl, self.ik, self.has_arm, self.has_hand = ctrl, ik, has_arm, has_hand
        self.side = side
        self.trans_rate = 0.06         # m/s  EEF translation jog
        self.rot_rate = 0.40           # rad/s EEF rotation jog
        self.jog_lin = np.zeros(3); self.jog_ang = np.zeros(3)
        self.warn = ""
        self.seq_running = False
        self.estopped = False
        # EEF target pose, seeded from the current joints
        if has_arm and ik is not None:
            q6 = np.radians(ctrl.arm_cmd)
            self.eef = ik.fk(q6)
        else:
            self.eef = np.eye(4)
        self.setWindowTitle(f"EEF Teleop — {side.upper()}")
        self.setStyleSheet(QSS)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self._build_ui(side)
        self._last = time.perf_counter()
        self.tick = QtCore.QTimer(self); self.tick.timeout.connect(self._loop); self.tick.start(16)  # ~60Hz
        # LeRobot episode recorder (writes state[26]+action[26]+front/wrist video, per episode)
        ds_root = os.environ.get("VPI_DATASET_DIR", "/state/lerobot_dataset")
        # 2 Hz recording: action = the command being streamed at that instant, image = latest camera
        # frame. Low rate (the robot smooths between waypoints itself), one frame per command sample.
        rec_fps = int(os.environ.get("VPI_RECORD_FPS", "2"))
        self.recorder = EpisodeRecorder(ds_root, fps=rec_fps)
        self.rec_timer = QtCore.QTimer(self); self.rec_timer.timeout.connect(self._record_tick)
        self.rec_timer.start(int(1000 / rec_fps))     # 2 Hz capture (500 ms)

    # ---- key map: key -> (kind, index, sign) ; kind 'L' linear / 'A' angular
    KEYS = {
        QtCore.Qt.Key_W: ('L', 0, -1), QtCore.Qt.Key_S: ('L', 0, +1),
        QtCore.Qt.Key_A: ('L', 1, -1), QtCore.Qt.Key_D: ('L', 1, +1),
        QtCore.Qt.Key_R: ('L', 2, +1), QtCore.Qt.Key_F: ('L', 2, -1),
        QtCore.Qt.Key_U: ('A', 0, +1), QtCore.Qt.Key_J: ('A', 0, -1),
        QtCore.Qt.Key_I: ('A', 1, +1), QtCore.Qt.Key_K: ('A', 1, -1),
        QtCore.Qt.Key_O: ('A', 2, +1), QtCore.Qt.Key_L: ('A', 2, -1),
    }

    def _set_jog(self, kind, idx, val):
        (self.jog_lin if kind == 'L' else self.jog_ang)[idx] = val

    def _build_ui(self, side):
        cw = QtWidgets.QWidget(); root = QtWidgets.QVBoxLayout(cw)
        # E-STOP: big button (also bound to ESC) — halts arm + hand immediately.
        self.estop_btn = QtWidgets.QPushButton("■  E-STOP  (Esc)")
        self.estop_btn.setFocusPolicy(QtCore.Qt.NoFocus); self.estop_btn.setMinimumHeight(46)
        self.estop_btn.setStyleSheet("background:#f38ba8; color:#1e1e2e; font-weight:bold; font-size:16px;")
        self.estop_btn.clicked.connect(self._toggle_estop)
        topbar = QtWidgets.QHBoxLayout(); topbar.addWidget(self.estop_btn, 4)
        self.reconn_btn = QtWidgets.QPushButton("Reconnect"); self.reconn_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.reconn_btn.setMinimumHeight(46); self.reconn_btn.clicked.connect(self._reconnect_all)
        topbar.addWidget(self.reconn_btn, 1)
        self.rec_btn = QtWidgets.QPushButton("● REC episode"); self.rec_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.rec_btn.setMinimumHeight(46); self.rec_btn.clicked.connect(self._toggle_record)
        self.rec_btn.setStyleSheet("background:#313244; color:#cdd6f4; font-weight:bold;")
        topbar.addWidget(self.rec_btn, 1)
        self.rec_lbl = QtWidgets.QLabel(""); self.rec_lbl.setObjectName("v")
        root.addLayout(topbar); root.addWidget(self.rec_lbl)
        self.status = QtWidgets.QLabel(); self.status.setObjectName("stat"); root.addWidget(self.status)

        # EEF position
        gp = QtWidgets.QGroupBox("End-Effector  position (m, base frame)")
        gpl = QtWidgets.QGridLayout(gp)
        self.lbl_pos = QtWidgets.QLabel(); self.lbl_pos.setObjectName("v")
        gpl.addWidget(self.lbl_pos, 0, 0, 1, 6)
        for col, (ax, name) in enumerate([(0, "X"), (1, "Y"), (2, "Z")]):
            gpl.addWidget(jog_button(f"-{name}", lambda a=ax: self._set_jog('L', a, -1), lambda a=ax: self._set_jog('L', a, 0)), 1, col*2)
            gpl.addWidget(jog_button(f"+{name}", lambda a=ax: self._set_jog('L', a, +1), lambda a=ax: self._set_jog('L', a, 0)), 1, col*2+1)
        root.addWidget(gp)

        # EEF orientation
        go = QtWidgets.QGroupBox("End-Effector  orientation (deg, RPY)")
        gol = QtWidgets.QGridLayout(go)
        self.lbl_rpy = QtWidgets.QLabel(); self.lbl_rpy.setObjectName("v")
        gol.addWidget(self.lbl_rpy, 0, 0, 1, 6)
        for col, (ax, name) in enumerate([(0, "R"), (1, "P"), (2, "Y")]):
            gol.addWidget(jog_button(f"-{name}", lambda a=ax: self._set_jog('A', a, -1), lambda a=ax: self._set_jog('A', a, 0)), 1, col*2)
            gol.addWidget(jog_button(f"+{name}", lambda a=ax: self._set_jog('A', a, +1), lambda a=ax: self._set_jog('A', a, 0)), 1, col*2+1)
        root.addWidget(go)

        # Hand
        gh = QtWidgets.QGroupBox("Hand")
        ghl = QtWidgets.QHBoxLayout(gh)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.slider.setRange(0, 100)
        self.slider.valueChanged.connect(lambda v: self.ctrl.set_grip(v / 100.0))
        b_open = QtWidgets.QPushButton("Open"); b_open.clicked.connect(lambda: self._set_grip_ui(0))
        b_close = QtWidgets.QPushButton("Close"); b_close.clicked.connect(lambda: self._set_grip_ui(100))
        b_tog = QtWidgets.QPushButton("Toggle"); b_tog.clicked.connect(lambda: self._set_grip_ui(0 if self.slider.value() > 50 else 100))
        self.lbl_hand = QtWidgets.QLabel("open"); self.lbl_hand.setObjectName("v")
        b_cfg = QtWidgets.QPushButton("Grip config…"); b_cfg.clicked.connect(self._open_grip_config)
        for wdg in (self.slider, b_open, b_close, b_tog, b_cfg):   # keep keyboard focus on the window
            wdg.setFocusPolicy(QtCore.Qt.NoFocus)
        b_cfg.setEnabled(side == "left")     # per-joint config is DG5F (left) only
        ghl.addWidget(QtWidgets.QLabel("open")); ghl.addWidget(self.slider); ghl.addWidget(QtWidgets.QLabel("closed"))
        ghl.addWidget(b_open); ghl.addWidget(b_close); ghl.addWidget(b_tog); ghl.addWidget(b_cfg)
        ghl.addWidget(self.lbl_hand)
        gh.setEnabled(self.has_hand); root.addWidget(gh)

        # HOME pose: editable X/Y/Z (m) + R/P/Y (deg), persisted to disk. "Go Home" drives there.
        gm = QtWidgets.QGroupBox("Home pose  (X/Y/Z metres, R/P/Y degrees — saved on edit)")
        gml = QtWidgets.QGridLayout(gm)
        # default home: left = the specified pose; right = the current EEF pose (safe no-op start).
        if self.side == "left":
            fb = dict(DEFAULT_HOME)
        else:
            p = self.eef[:3, 3]; rpy = np.degrees(mat_to_rpy(self.eef[:3, :3]))
            fb = {"x": p[0], "y": p[1], "z": p[2], "roll": rpy[0], "pitch": rpy[1], "yaw": rpy[2]}
        hv = load_home(self.side, fb)
        self.home_spins = {}
        specs = [("x", "X", -2.0, 2.0, 0.01, 3), ("y", "Y", -2.0, 2.0, 0.01, 3),
                 ("z", "Z", -2.0, 2.0, 0.01, 3), ("roll", "R", -180.0, 180.0, 1.0, 1),
                 ("pitch", "P", -180.0, 180.0, 1.0, 1), ("yaw", "Y", -180.0, 180.0, 1.0, 1)]
        for c, (key, lab, lo, hi, step, dec) in enumerate(specs):
            gml.addWidget(QtWidgets.QLabel(lab), 0, c)
            sp = QtWidgets.QDoubleSpinBox(); sp.setRange(lo, hi); sp.setSingleStep(step)
            sp.setDecimals(dec); sp.setValue(hv[key]); sp.setFocusPolicy(QtCore.Qt.ClickFocus)
            sp.setMinimumWidth(80); sp.valueChanged.connect(self._save_home)
            gml.addWidget(sp, 1, c); self.home_spins[key] = sp
        self.home_btn = QtWidgets.QPushButton("Go Home"); self.home_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.home_btn.clicked.connect(self._go_home)
        gml.addWidget(self.home_btn, 2, 0, 1, 3)
        self.home_status = QtWidgets.QLabel("saved"); self.home_status.setObjectName("v")
        gml.addWidget(self.home_status, 2, 3, 1, 3)
        gm.setEnabled(self.has_arm); root.addWidget(gm)

        # Motion sequence: 3 relative moves (dx,dy,dz in metres, base frame), run in order.
        gs = QtWidgets.QGroupBox("Motion sequence  (Δpos in metres, base frame — run in order)")
        gsl = QtWidgets.QGridLayout(gs)
        for c, t in enumerate(["", "dx", "dy", "dz"]):
            gsl.addWidget(QtWidgets.QLabel(t), 0, c)
        self.seq_spins = []          # [(sx,sy,sz), ...] 3 rows
        for r in range(3):
            gsl.addWidget(QtWidgets.QLabel(f"step {r+1}"), r + 1, 0)
            row = []
            for c in range(3):
                sp = QtWidgets.QDoubleSpinBox(); sp.setRange(-0.5, 0.5); sp.setSingleStep(0.01)
                sp.setDecimals(3); sp.setValue(0.0)
                sp.setFocusPolicy(QtCore.Qt.ClickFocus)   # click to type a number; jog keys work otherwise
                sp.setMinimumWidth(90); gsl.addWidget(sp, r + 1, c + 1); row.append(sp)
            self.seq_spins.append(row)
        self.seq_btn = QtWidgets.QPushButton("Start sequence"); self.seq_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.seq_btn.clicked.connect(self._start_sequence)
        self.seq_status = QtWidgets.QLabel("idle"); self.seq_status.setObjectName("v")
        gsl.addWidget(self.seq_btn, 4, 0, 1, 2); gsl.addWidget(self.seq_status, 4, 2, 1, 2)
        gs.setEnabled(self.has_arm); root.addWidget(gs)

        # Cameras: front (Zivid) + wrist (D405) from the vision node
        gc = QtWidgets.QGroupBox("Cameras  (front = Zivid, wrist = D405)")
        gcl = QtWidgets.QHBoxLayout(gc)
        self.cam_front = QtWidgets.QLabel("front: waiting…"); self.cam_wrist = QtWidgets.QLabel("wrist: waiting…")
        for c in (self.cam_front, self.cam_wrist):
            c.setMinimumSize(300, 225); c.setAlignment(QtCore.Qt.AlignCenter)
            c.setStyleSheet("background:#11111b; border:1px solid #45475a;")
        gcl.addWidget(self.cam_front); gcl.addWidget(self.cam_wrist)
        root.addWidget(gc)
        self.cam_tick = QtCore.QTimer(self); self.cam_tick.timeout.connect(self._update_cameras); self.cam_tick.start(100)

        # joints + speed + help
        self.lbl_joints = QtWidgets.QLabel(); self.lbl_joints.setObjectName("v"); root.addWidget(self.lbl_joints)
        self.lbl_speed = QtWidgets.QLabel(); root.addWidget(self.lbl_speed)
        h = QtWidgets.QLabel(
            "move  X:W/S  Y:A/D  Z:R/F     rotate  R:U/J  P:I/K  Y:O/L     "
            "hand [ ] Space     speed +/-     resync Backspace     E-STOP Esc     (close window to quit)")
        h.setObjectName("hint"); root.addWidget(h)
        self.setCentralWidget(cw)

    def _update_cameras(self):
        hp = self.ctrl.hand_pub
        if hp is None:
            return
        if hp.front is not None:
            self.cam_front.setPixmap(np_to_qpixmap(hp.front, self.cam_front.width(), self.cam_front.height()))
        if hp.wrist is not None:
            self.cam_wrist.setPixmap(np_to_qpixmap(hp.wrist, self.cam_wrist.width(), self.cam_wrist.height()))

    def _reconnect_all(self):
        # Reconnect the arm (OpenStream re-handshake), restart the vision node (re-init cameras),
        # AND restart the hand driver (recovers a DG5F freeze: port open but joint_states stalled).
        if self.has_arm:
            self.ctrl.reconnect_arm()

        def do_vision():
            self.ctrl.hand_pub.front = self.ctrl.hand_pub.wrist = None    # show "waiting" until back
            docker_restart(f"teleop_vision_{self.side}")
            for _ in range(20):                       # wait for cameras to republish, then re-match
                time.sleep(2)
                self.ctrl.hand_pub._resub_cameras()
                if self.ctrl.hand_pub.front is not None:
                    break
        threading.Thread(target=do_vision, daemon=True).start()

        def do_hand():
            hp = self.ctrl.hand_pub
            hp.actual = None                          # show stale until joint_states resumes
            docker_restart(f"teleop_hand_{self.side}")
            for _ in range(15):                       # wait for joint_states, re-sub off the dead shm
                time.sleep(2)
                hp._resub_hand()
                if hp.actual is not None or self.side != "left":
                    break
        if self.has_hand:
            threading.Thread(target=do_hand, daemon=True).start()

    _rec_last_action = None

    def _toggle_record(self):
        if not self.recorder.recording:
            self._rec_last_action = None
            self.recorder.start()
            self.rec_btn.setText("■ STOP  (recording)")
            self.rec_btn.setStyleSheet("background:#f38ba8; color:#1e1e2e; font-weight:bold;")
        else:
            n = self.recorder.stop()
            self.rec_btn.setText("● REC episode")
            self.rec_btn.setStyleSheet("background:#313244; color:#cdd6f4; font-weight:bold;")
            self.rec_lbl.setText(f"saved episode {self.recorder.ep_index - 1} ({n} frames)  ->  {self.recorder.root}")

    def _record_tick(self):
        if not self.recorder.recording:
            return
        hp = self.ctrl.hand_pub
        # 2 Hz (timer fires every 500 ms). Each tick = (state, action, latest camera image). We reuse
        # the latest image (camera is ~7 Hz, faster than 2 Hz so it's always fresh) and sample the
        # command being streamed as the action. Raw PNG + parquet -> convert to LeRobot later.
        if hp is None or hp.front is None:
            return
        arm_q = self.ctrl.arm.q if (self.ctrl.arm is not None) else None
        hand_a = hp.actual
        hand_c = self.ctrl.last_hand_pose
        if arm_q is None or hand_a is None or hand_c is None:
            return                               # wait until all state is available
        state = np.concatenate([np.radians(arm_q[:6]), np.asarray(hand_a, float)])
        # action = the command being STREAMED to the robot right now (arm_cmd + last hand pose),
        # sampled at 2 Hz. Consecutive rows form the low-rate waypoint trajectory (robot smooths between).
        action = np.concatenate([np.radians(self.ctrl.arm_cmd[:6]), np.asarray(hand_c, float)])
        # Keep a frame only while actively teleoping (input held OR the command moved since last frame);
        # idle waits are skipped and kept frames are re-timestamped contiguously at the record fps.
        active = (self.jog_lin.any() or self.jog_ang.any() or self._hand_jog != 0 or self.seq_running
                  or abs(self.ctrl.grip - self.ctrl.grip_target) > 1e-4)
        if not active and self._rec_last_action is not None:
            active = np.max(np.abs(action - self._rec_last_action)) > 5e-4   # command moved (button/seq)
        if not active:
            self.rec_lbl.setText(f"● rec ep {self.recorder.ep_index}: {self.recorder.frame_i} frames"
                                 f"  (paused — idle)")
            return
        self.recorder.add(state, action, hp.front, hp.wrist)
        self._rec_last_action = action
        self.rec_lbl.setText(f"● rec ep {self.recorder.ep_index}: {self.recorder.frame_i} frames "
                             f"@ {self.recorder.fps}Hz  (raw PNG)")

    def _set_grip_ui(self, v):
        self.slider.setValue(v); self.ctrl.set_grip(v / 100.0)

    def _open_grip_config(self):
        if getattr(self, "_grip_dlg", None) is not None and self._grip_dlg.isVisible():
            self._grip_dlg.raise_(); self._grip_dlg.activateWindow(); return
        self._grip_dlg = GripConfigDialog(self, self.ctrl, self.side)
        self._grip_dlg.resize(1120, 380); self._grip_dlg.show()      # modeless: arm stays live

    def keyPressEvent(self, e):
        if e.isAutoRepeat(): return
        k = e.key()
        if k == QtCore.Qt.Key_Escape: self._estop(); return       # Esc = E-STOP (not quit)
        if self.estopped: return                                  # ignore all other keys while E-STOPped
        if k == QtCore.Qt.Key_Space: self._set_grip_ui(0 if self.slider.value() > 50 else 100); return
        if k == QtCore.Qt.Key_BracketLeft: self._set_jog('H', 0, -1); return
        if k == QtCore.Qt.Key_BracketRight: self._set_jog('H', 0, +1); return
        if k == QtCore.Qt.Key_Backspace and self.has_arm:
            self.eef = self.ik.fk(np.radians(self.ctrl.arm_cmd)); self.warn = ""; return
        if k in (QtCore.Qt.Key_Plus, QtCore.Qt.Key_Equal):
            self.trans_rate = min(self.trans_rate*1.5, 0.15); self.rot_rate = min(self.rot_rate*1.5, 1.0); return
        if k == QtCore.Qt.Key_Minus:
            self.trans_rate = max(self.trans_rate/1.5, 0.005); self.rot_rate = max(self.rot_rate/1.5, 0.03); return
        if k in self.KEYS:
            kind, idx, sign = self.KEYS[k]; self._set_jog(kind, idx, sign)

    def _set_jog(self, kind, idx, val):
        if self.estopped and val != 0: return   # E-STOP: ignore new jog input (buttons + keys)
        if kind == 'L': self.jog_lin[idx] = val
        elif kind == 'A': self.jog_ang[idx] = val
        elif kind == 'H':                 # hand via [ ]
            self._hand_jog = val

    def keyReleaseEvent(self, e):
        if e.isAutoRepeat(): return
        k = e.key()
        if k in (QtCore.Qt.Key_BracketLeft, QtCore.Qt.Key_BracketRight):
            self._hand_jog = 0
            self.ctrl.set_grip(self.ctrl.snap()[2])   # freeze grip where it is on release
            return
        if k in self.KEYS:
            kind, idx, _ = self.KEYS[k]; self._set_jog(kind, idx, 0)

    _hand_jog = 0

    def _loop(self):
        now = time.perf_counter(); dt = now - self._last; self._last = now
        if self.has_arm and self.ctrl.arm_resynced:     # after an auto-reconnect: re-sync EEF target
            try: self.eef = self.ik.fk(np.radians(self.ctrl.arm_cmd))
            except Exception: pass
            self.jog_lin[:] = 0; self.jog_ang[:] = 0
            self.ctrl.arm_resynced = False
        if self.estopped:                    # E-STOP: ignore all jog/hand input
            self._refresh(); return
        # hand jog from [ ]: hold to open/close; the ControlThread ramps the grip at GRIP_RATE
        if self._hand_jog and self.has_hand:
            self.ctrl.set_grip(1.0 if self._hand_jog > 0 else 0.0)
        # EEF jog -> IK -> joint target (disabled while a sequence runs)
        if self.has_arm and not self.seq_running and (self.jog_lin.any() or self.jog_ang.any()):
            tgt = self.eef.copy()
            tgt[:3, 3] += self.jog_lin * self.trans_rate * dt
            if self.jog_ang.any():
                for i in range(3):
                    if self.jog_ang[i]:
                        axis = np.eye(3)[i]
                        tgt[:3, :3] = rot_about(axis, self.jog_ang[i]*self.rot_rate*dt) @ tgt[:3, :3]
            q_now = np.radians(self.ctrl.arm_cmd)
            try:
                sol = self.ik.ik(tgt[:3, 3], tgt[:3, :3], q_now)
                jump = np.max(np.abs(np.degrees(sol - q_now)))
                if jump <= IK_REJECT_DEG:
                    self.eef = tgt; self.warn = ""
                    self.ctrl.set_arm_tgt(np.degrees(sol))
                else:
                    self.warn = f"IK jump {jump:.0f}deg rejected (singular?)"
            except Exception as ex:
                self.warn = f"IK fail: {ex}"
        self._refresh()

    def _toggle_estop(self):
        self._resume() if self.estopped else self._estop()

    def _estop(self):
        self.estopped = True
        self.jog_lin[:] = 0; self.jog_ang[:] = 0; self._hand_jog = 0
        self.ctrl.estop()                       # freeze arm (STOP) + hold hand
        self.warn = "E-STOP engaged — press RESUME or the button to clear"
        self.estop_btn.setText("▶  RESUME  (clear E-STOP)")
        self.estop_btn.setStyleSheet("background:#f9e2af; color:#1e1e2e; font-weight:bold; font-size:16px;")

    def _resume(self):
        # clear ALL pending input first so nothing moves the instant we un-freeze
        self.jog_lin[:] = 0; self.jog_ang[:] = 0; self._hand_jog = 0
        self.ctrl.resume()                      # re-sync to actual + restart stream
        if self.has_arm and self.ctrl.arm is not None and self.ctrl.arm.q is not None:
            try: self.eef = self.ik.fk(np.radians(self.ctrl.arm.q[:6]))
            except Exception: pass
        if self.has_hand:                       # re-sync the slider to the (frozen) actual grip
            self.slider.blockSignals(True); self.slider.setValue(int(self.ctrl.grip * 100)); self.slider.blockSignals(False)
        self._last = time.perf_counter()        # avoid a dt spike on the first loop after resume
        self.estopped = False; self.warn = ""
        self.estop_btn.setText("■  E-STOP  (Esc)")
        self.estop_btn.setStyleSheet("background:#f38ba8; color:#1e1e2e; font-weight:bold; font-size:16px;")

    def _home_vals(self):
        return {k: self.home_spins[k].value() for k in DEFAULT_HOME}

    def _save_home(self):
        save_home(self.side, self._home_vals())
        self.home_status.setText("saved")

    def _go_home(self):
        if self.seq_running or not self.has_arm or self.estopped:
            return
        v = self._home_vals()
        pose = np.eye(4)
        pose[:3, 3] = [v["x"], v["y"], v["z"]]
        pose[:3, :3] = rpy_to_mat(math.radians(v["roll"]), math.radians(v["pitch"]), math.radians(v["yaw"]))
        self.seq_running = True
        self.home_btn.setEnabled(False); self.seq_btn.setEnabled(False)
        threading.Thread(target=self._run_home, args=(pose,), daemon=True).start()

    def _cartesian_move(self, target, status=None):
        """Move the EEF in a STRAIGHT Cartesian line to `target` (4x4): position lerp + rotation
        slerp, IK each waypoint, stream. Speed is set by the interpolation, not the joint clamp."""
        arm = self.ctrl.arm
        if arm is None or arm.q is None:
            return False
        start = self.ik.fk(np.radians(arm.q[:6]))
        p0, p1 = start[:3, 3], target[:3, 3]
        dist = float(np.linalg.norm(p1 - p0))
        axis, ang = axis_angle(target[:3, :3] @ start[:3, :3].T)
        dur = max(dist / CART_LIN_SPEED, abs(ang) / CART_ANG_SPEED, 0.1)
        n = max(int(dur / CART_STEP), 1)
        self.ctrl.max_dps = 200.0            # don't clamp — speed comes from the waypoint pacing
        seed = np.radians(self.ctrl.arm_cmd)
        for i in range(1, n + 1):
            if self.estopped:                    # E-STOP aborts the motion
                if status: status("E-STOP")
                return False
            s = i / n
            pose = np.eye(4)
            pose[:3, 3] = p0 + s * (p1 - p0)
            pose[:3, :3] = rot_about(axis, s * ang) @ start[:3, :3]
            try:
                sol = self.ik.ik(pose[:3, 3], pose[:3, :3], seed)
            except Exception as ex:
                if status: status(f"IK fail ({ex}) — abort")
                return False
            if np.max(np.abs(np.degrees(sol - seed))) > CART_JUMP_DEG:
                if status: status("IK jump (singular?) — abort")
                return False
            seed = sol
            self.eef = pose
            self.ctrl.set_arm_tgt(np.degrees(sol))
            time.sleep(CART_STEP)
        return True

    def _run_home(self, pose):
        try:
            self.home_status.setText("going home (straight line) ...")
            ok = self._cartesian_move(pose, self.home_status.setText)
            self.home_status.setText("at home" if ok else self.home_status.text())
        finally:
            self.ctrl.max_dps = ARM_MAX_DPS
            self.seq_running = False
            self.home_btn.setEnabled(True); self.seq_btn.setEnabled(True)

    def _start_sequence(self):
        if self.seq_running or not self.has_arm or self.estopped:
            return
        steps = [np.array([r[0].value(), r[1].value(), r[2].value()], float) for r in self.seq_spins]
        self.seq_running = True
        self.seq_btn.setEnabled(False)
        threading.Thread(target=self._run_sequence, args=(steps,), daemon=True).start()

    def _run_sequence(self, steps):
        try:
            for i, d in enumerate(steps):
                if np.allclose(d, 0.0):
                    continue
                self.seq_status.setText(f"step {i+1}: Δ{np.round(d,3).tolist()} (straight) ...")
                tgt = self.eef.copy(); tgt[:3, 3] += d      # relative Δpos, same orientation
                ok = self._cartesian_move(tgt, self.seq_status.setText)
                if not ok:
                    break
                time.sleep(0.3)     # brief settle between steps
            else:
                self.seq_status.setText("sequence done")
        finally:
            self.ctrl.max_dps = ARM_MAX_DPS         # restore jog speed
            self.seq_running = False
            self.seq_btn.setEnabled(True)

    def _refresh(self):
        arm_ok = self.has_arm and self.ctrl.arm is not None
        if arm_ok:                          # disable the button while a reconnect is in progress
            self.reconn_btn.setEnabled(not self.ctrl.arm.reconnecting)
        if arm_ok and self.ctrl.arm.reconnecting:
            sa = "<span style='color:#f9e2af'>ARM reconnecting…</span>"
        elif arm_ok and not self.ctrl.arm.healthy():
            sa = "<span style='color:#f9e2af'>ARM no signal (auto-reconnecting)</span>"
        elif arm_ok:
            sa = "<span style='color:#a6e3a1'>ARM ✓</span>"
        else:
            sa = "<span style='color:#f38ba8'>ARM ✗</span>"
        sh = "<span style='color:#a6e3a1'>HAND ✓</span>" if self.has_hand else "<span style='color:#f38ba8'>HAND ✗</span>"
        w = f"   <span style='color:#f9e2af'>{self.warn}</span>" if self.warn else ""
        ae = f"   <span style='color:#f38ba8'>{self.ctrl.arm.err}</span>" if (arm_ok and self.ctrl.arm.err) else ""
        self.status.setText(sa + "    " + sh + w + ae)
        # show the CURRENT (actual) EEF pose = FK of the actual robot joints, not the target.
        cur = None
        if arm_ok and self.ctrl.arm.q is not None:
            try:
                cur = self.ik.fk(np.radians(self.ctrl.arm.q[:6]))
            except Exception:
                cur = None
        show = cur if cur is not None else self.eef
        p = show[:3, 3]; rpy = np.degrees(mat_to_rpy(show[:3, :3]))
        self.lbl_pos.setText(f"X {p[0]:+.3f}   Y {p[1]:+.3f}   Z {p[2]:+.3f}")
        self.lbl_rpy.setText(f"R {rpy[0]:+6.1f}   P {rpy[1]:+6.1f}   Y {rpy[2]:+6.1f}")
        g = self.ctrl.snap()[2]                       # actual (ramped) grip 0..1
        if self.has_hand and not self.slider.isSliderDown():   # slider follows the real grip (unless dragging)
            self.slider.blockSignals(True); self.slider.setValue(int(g * 100)); self.slider.blockSignals(False)
        self.lbl_hand.setText(f"{'CLOSED' if g > 0.5 else 'open'}  {g*100:.0f}%")
        if arm_ok and self.ctrl.arm.q:
            self.lbl_joints.setText("joints(deg)  " + "  ".join(f"J{i+1} {self.ctrl.arm.q[i]:+.1f}" for i in range(6)))
        self.lbl_speed.setText(f"jog speed:  {self.trans_rate*100:.1f} cm/s   {math.degrees(self.rot_rate):.0f} deg/s")

    def closeEvent(self, ev):
        self.ctrl.running = False; time.sleep(0.05)
        if self.ctrl.arm is not None:
            self.ctrl.arm.stop(); self.ctrl.arm.close()
        ev.accept()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="right", choices=["right", "left"])
    ap.add_argument("--arm-ip", default=None)
    ap.add_argument("--urdf", default="/tmp/hdr35_20.urdf")
    ap.add_argument("--no-arm", action="store_true")
    ap.add_argument("--no-hand", action="store_true")
    args = ap.parse_args()

    ik = None; arm = None; q0 = np.zeros(6)
    if not args.no_arm:
        try:
            ik = IKArm(args.urdf)
            ip = args.arm_ip or ARM_IP[args.side]
            arm = ArmClient(ip); print(f"[teleop] connecting arm {ip} ...")
            arm.connect()
            q0 = np.asarray(arm.wait_state()[:6], float)
            print(f"[teleop] arm at(deg)={np.round(q0,1).tolist()}")
        except Exception as e:
            print(f"[teleop] ARM/IK unavailable ({e}); GUI without arm.")
            try: arm.close()
            except Exception: pass
            arm = None; ik = None

    apply_grip_config(load_grip_config(args.side))   # restore saved OPEN/GRIP poses if present

    rclpy.init()
    hand_pub = HandPub(args.side)     # always (carries the camera + hand-state subscriptions too)
    threading.Thread(target=lambda: rclpy.spin(hand_pub), daemon=True).start()   # deliver callbacks
    ctrl = ControlThread(arm, hand_pub, q0, args.no_hand); ctrl.start()

    app = QtWidgets.QApplication(sys.argv)
    win = TeleopWindow(ctrl, ik, args.side, has_arm=arm is not None, has_hand=not args.no_hand)
    win.resize(620, 440); win.show()
    win.raise_(); win.activateWindow(); win.setFocus()   # grab keyboard focus on launch
    code = app.exec_()

    ctrl.running = False
    if hand_pub is not None: hand_pub.destroy_node()
    rclpy.shutdown(); sys.exit(code)


if __name__ == "__main__":
    main()
