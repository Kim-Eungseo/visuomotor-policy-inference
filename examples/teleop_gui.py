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
LOOK_AHEAD = 0.002       # arm trajectory blend/look-ahead — smaller = less command->motion delay
ARM_IP = {"right": "192.168.4.151", "left": "192.168.4.152"}
GET_JOINTS = "/project/robot/joints/joint_states"
ARM_MAX_DPS = 7.5        # per-joint arm follow clamp (deg/s) — safety / motion speed
IK_REJECT_DEG = 12.0     # reject an IK step that jumps any joint more than this (singularity guard)
GRIP_RATE = float(os.environ.get("VPI_GRIP_RATE", "0.5"))  # hand open/close speed: grip fraction/sec.
# (The staged one-shot curl is fixed at the DRIVER level — dg5f_modbus_driver.py step-and-holds each
# target so the DG5F's Modbus servo can track it — so this can stay at the responsive 0.5.)
VINN_URL = os.environ.get("VPI_INFER_URL", "http://localhost:8000")   # inference server (/predict)
SEG_URL = os.environ.get("VPI_SEG_URL", "http://localhost:8100")      # RF-DETR seg server (GPU, /segment)
DIFF_CHUNK = int(os.environ.get("VPI_DIFF_CHUNK", "8"))               # diffusion action-chunk length (model n_action_steps)
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

# Neutral START pose for policy rollouts ("Init hand"). SEPARATE from the GUI open/close (grip)
# poses AND from the training data. Based on OPEN, but the thumb's idx-2 joint (lj_dg_1_3) is
# REVERSED (opposite sign) so the thumb points down at the neutral start. Edit here to retune.
DG5F_NEUTRAL_POSE = DG5F_OPEN_POSE.copy()
DG5F_NEUTRAL_POSE[2] = -DG5F_OPEN_POSE[2]        # thumb lj_dg_1_3 (idx 2): flip vs open -> point down

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


def home_hand_path(side):
    d = os.environ.get("VPI_STATE_DIR", "/state")
    if not os.path.isdir(d):
        d = "/tmp"
    return os.path.join(d, f"teleop_home_hand_{side}.json")


def save_home_hand(side, pose):
    try:
        with open(home_hand_path(side), "w") as f:
            json.dump([float(x) for x in pose], f)
    except Exception as e:
        print(f"[home] hand save failed: {e}", flush=True)


def load_home_hand(side):
    try:
        with open(home_hand_path(side)) as f:
            return np.asarray(json.load(f), float)
    except Exception:
        return None


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
            self.hand_manual = None           # manual grip reclaims the hand (release any policy/dialog override)
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
        # next episode index = max existing index + 1 (robust to deleted episodes / gaps, so a new
        # recording never collides with a kept higher-numbered one)
        idxs = [int(d[8:]) for d in os.listdir(self.raw) if d.startswith("episode_") and d[8:].isdigit()]
        self.ep_index = (max(idxs) + 1) if idxs else 0

    def start(self):
        if self.recording:
            return
        self.buf = []; self.frame_i = 0
        self.ep_dir = os.path.join(self.raw, f"episode_{self.ep_index:06d}")
        os.makedirs(os.path.join(self.ep_dir, "front"), exist_ok=True)
        os.makedirs(os.path.join(self.ep_dir, "wrist"), exist_ok=True)
        self.recording = True

    def add(self, fields, front, wrist):
        """fields = dict of {feature_key: 1-D array}. Stored one column per key in frames.parquet,
        alongside frame_index / timestamp / task. Images are written as PNG files."""
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
        row = {"frame_index": i, "timestamp": float(i / self.fps), "task": self.task}  # idle removed
        for k, v in fields.items():
            row[k] = [float(x) for x in np.asarray(v, float).ravel()]
        self.buf.append(row)
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

    def discard(self):
        """Throw away the in-progress episode: delete its PNGs, write no parquet, keep ep_index (so the
        next recording reuses this same episode number)."""
        if not self.recording:
            return 0
        self.recording = False
        n = self.frame_i
        shutil.rmtree(self.ep_dir, ignore_errors=True)
        return n


def jpeg_b64(rgb):
    """RGB uint8 (H,W,3) -> base64 JPEG string, matching the inference client (server decodes to RGB)."""
    import io
    import base64
    from PIL import Image as PILImage
    b = io.BytesIO(); PILImage.fromarray(rgb).save(b, "JPEG", quality=90)
    return base64.b64encode(b.getvalue()).decode()


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


def docker_run_convert(host_repo, host_state, repo_id="rec/teleop", seg_url=SEG_URL, progress=None):
    """Create + start a vpi-image container that (in)crementally builds the LeRobot dataset from raw/
    (+ front_masked), STREAMING its stdout so `progress(line)` fires live. Runs via the Docker unix
    socket (the GUI container has no lerobot). Returns (ok, message). Call from a background thread."""
    import json as _json
    import http.client
    import socket as _sock

    def _conn(timeout=30):
        class _U(http.client.HTTPConnection):
            def connect(self):
                s = _sock.socket(_sock.AF_UNIX, _sock.SOCK_STREAM); s.settimeout(timeout)
                s.connect("/var/run/docker.sock"); self.sock = s
        return _U("localhost")

    ds = f"{host_state}/lerobot_dataset"
    cfg = {
        "Image": "visuomotor-policy-inference:latest",
        "Tty": True,                                    # raw (un-multiplexed) log stream
        "Cmd": ["python3", "-u", "/convert.py", "--raw", "/raw", "--out", "/outp/lerobot",
                "--repo-id", repo_id],
        "Env": ["HF_HUB_OFFLINE=1", "HF_DATASETS_OFFLINE=1", f"VPI_SEG_URL={seg_url}"],
        "HostConfig": {
            "Binds": [f"{ds}/raw:/raw:ro", f"{ds}:/outp",
                      f"{host_repo}/examples/convert_raw_to_lerobot.py:/convert.py:ro"],
            "NetworkMode": "host",
        },
    }
    KEEP = ("[progress]", "creating", "appending", "detected", "DONE", "nothing", "WARNING", "Traceback")
    last = ""
    try:
        c = _conn(); c.request("POST", "/containers/create",
                               body=_json.dumps(cfg), headers={"Content-Type": "application/json"})
        r = c.getresponse(); created = _json.loads(r.read()); c.close()
        cid = created.get("Id")
        if not cid:
            return False, f"create failed: {created.get('message', created)}"
        c = _conn(); c.request("POST", f"/containers/{cid}/start"); r = c.getresponse(); r.read(); c.close()
        # stream logs (follow) -> emit progress lines until the container exits
        try:
            lc = _conn(timeout=900)
            lc.request("GET", f"/containers/{cid}/logs?follow=1&stdout=1&stderr=1&tail=0")
            lr = lc.getresponse(); buf = b""
            while True:
                chunk = lr.read(512)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    s = line.decode("utf-8", "replace").strip()
                    if s and any(s.startswith(k) or k in s for k in KEEP):
                        last = s.replace("[progress] ", "")
                        if progress:
                            progress(last)
            lc.close()
        except Exception:
            pass
        c = _conn(); c.request("GET", f"/containers/{cid}/json"); r = c.getresponse()
        code = _json.loads(r.read()).get("State", {}).get("ExitCode", -1); c.close()
        try:
            c = _conn(); c.request("DELETE", f"/containers/{cid}?force=1"); c.getresponse().read(); c.close()
        except Exception:
            pass
        return code == 0, (f"📦 LeRobot: {last}" if code == 0 else f"convert failed (exit {code}): {last}")
    except Exception as e:
        return False, f"convert error: {e}"


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


class SegWorker(QtCore.QThread):
    """Polls the GPU RF-DETR seg server (SEG_URL /segment) with the latest front frame and emits the
    returned RGB overlay. The heavy DETR runs on the GPU server (~100-300 ms), not in this container."""
    result = QtCore.pyqtSignal(object)     # np.ndarray overlay, or ("status"|"error", message)

    def __init__(self, get_front, url):
        super().__init__()
        self._get_front = get_front
        self._url = url
        self._run = True

    def stop(self):
        self._run = False

    def run(self):
        import io
        import base64
        import requests
        from PIL import Image as PILImage
        # wait until the server is up (it takes ~1-2 min to install rfdetr + load the model on GPU)
        for _ in range(60):
            if not self._run:
                return
            try:
                h = requests.get(f"{self._url}/health", timeout=3).json()
                if h.get("ok"):
                    self.result.emit(("status", f"running ({h.get('device','?')})")); break
            except Exception:
                pass
            self.result.emit(("status", "waiting for seg server… (./seg_server.sh)"))
            self.msleep(2000)
        while self._run:
            front = self._get_front()
            if front is None:
                self.msleep(200); continue
            try:
                r = requests.post(f"{self._url}/segment", json={"image": jpeg_b64(front)}, timeout=30)
                r.raise_for_status()
                d = r.json()
                if d.get("overlay"):
                    ov = np.asarray(PILImage.open(io.BytesIO(base64.b64decode(d["overlay"]))).convert("RGB"))
                    self.result.emit(ov)
                    counts = d.get("counts", {})
                    ctag = "  ".join(f"{k}:{v}" for k, v in counts.items()) or "no masks"
                    self.result.emit(("status", f"{ctag} · {d.get('ms', 0):.0f} ms (GPU)"))
                else:
                    self.result.emit(("error", d.get("error", "no overlay"))); self.msleep(1000)
            except Exception as e:
                self.result.emit(("error", str(e))); self.msleep(1500)
            self.msleep(50)


class PostProcWorker(QtCore.QThread):
    """Async post-processing: after an episode is recorded, run every frame through the seg server's
    /segment_masked and OVERWRITE the raw PNG with the hand+hook-only version (background black). Runs
    in its own thread so it never blocks GUI teleop. Frames with no detection are left unchanged."""
    progress = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(str)

    def __init__(self, ep_dir, url, cams=("front",)):    # front only by default; wrist kept original
        super().__init__()
        self.ep_dir = ep_dir
        self.url = url
        self.cams = cams
        self._run = True

    def stop(self):
        self._run = False

    def run(self):
        import io
        import glob
        import base64
        import requests
        from PIL import Image as PILImage
        name = os.path.basename(self.ep_dir)
        try:                                            # seg server up? if not, skip (leave frames raw)
            if not requests.get(f"{self.url}/health", timeout=5).json().get("ok"):
                self.done.emit(f"{name}: seg server not ready — frames left unmasked"); return
        except Exception:
            self.done.emit(f"{name}: seg server unreachable — frames left unmasked"); return
        total = nodet = 0
        for cam in self.cams:
            files = sorted(glob.glob(os.path.join(self.ep_dir, cam, "*.png")))
            for k, fp in enumerate(files):
                if not self._run:
                    return
                try:
                    rgb = PILImage.open(fp).convert("RGB")
                    b = io.BytesIO(); rgb.save(b, "JPEG", quality=95)
                    d = requests.post(f"{self.url}/segment_masked",
                                      json={"image": base64.b64encode(b.getvalue()).decode()}, timeout=30).json()
                    if d.get("masked"):
                        PILImage.open(io.BytesIO(base64.b64decode(d["masked"]))).convert("RGB").save(fp)
                        total += 1
                        if not d.get("detected", True):
                            nodet += 1
                except Exception as e:
                    self.progress.emit(f"postproc err {os.path.basename(fp)}: {e}")
                if k % 8 == 0:
                    self.progress.emit(f"masking {name}/{cam}: {k + 1}/{len(files)}")
        self.done.emit(f"✓ masked {name}: {total} frames ({nodet} no-detect) — run ./to_lerobot.sh to package")


class ConvertWorker(QtCore.QThread):
    """Async: build the LeRobot dataset (raw -> videos incl. front_masked) via a vpi container over the
    Docker socket. Background thread so the 'Save' button never blocks GUI teleop. Streams live progress."""
    progress = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(bool, str)

    def __init__(self, host_repo, host_state, repo_id="rec/teleop"):
        super().__init__()
        self.host_repo = host_repo
        self.host_state = host_state
        self.repo_id = repo_id

    def run(self):
        ok, msg = docker_run_convert(self.host_repo, self.host_state, self.repo_id,
                                     progress=self.progress.emit)
        self.done.emit(ok, msg)


class ReplayDialog(QtWidgets.QDialog):
    """Popup: list the recorded episodes under raw/, pick one to replay, or delete one (with a
    confirm prompt). Selecting + Replay (or double-click) closes with `selected` = that episode's dir."""
    def __init__(self, parent, raw_dir):
        super().__init__(parent)
        self.setWindowTitle("Replay episode")
        self.setStyleSheet(parent.styleSheet())
        self.raw_dir = raw_dir
        self.selected = None
        v = QtWidgets.QVBoxLayout(self)
        self.count_lbl = QtWidgets.QLabel()
        self.count_lbl.setStyleSheet("font-weight:bold;")
        v.addWidget(self.count_lbl)
        v.addWidget(QtWidgets.QLabel(f"in: {raw_dir}"))
        self.listw = QtWidgets.QListWidget()
        v.addWidget(self.listw)
        self.empty_lbl = QtWidgets.QLabel("(no finished episodes yet — record one first)")
        v.addWidget(self.empty_lbl)
        self._reload()
        bar = QtWidgets.QHBoxLayout()
        rb = QtWidgets.QPushButton("▶ Replay"); rb.clicked.connect(self._go)
        db = QtWidgets.QPushButton("🗑 Delete"); db.clicked.connect(self._delete)
        db.setStyleSheet("background:#f38ba8; color:#1e1e2e; font-weight:bold;")
        cb = QtWidgets.QPushButton("Cancel"); cb.clicked.connect(self.reject)
        bar.addWidget(rb); bar.addWidget(db); bar.addStretch(1); bar.addWidget(cb)
        v.addLayout(bar)
        self.listw.itemDoubleClicked.connect(lambda _: self._go())

    def _reload(self):
        self.listw.clear()
        eps = sorted(d for d in os.listdir(self.raw_dir)) if os.path.isdir(self.raw_dir) else []
        for name in eps:
            ep = os.path.join(self.raw_dir, name)
            if not (name.startswith("episode_") and os.path.exists(os.path.join(ep, "frames.parquet"))):
                continue
            n = "?"
            try:
                n = json.load(open(os.path.join(ep, "meta.json"))).get("num_frames", "?")
            except Exception:
                pass
            it = QtWidgets.QListWidgetItem(f"{name}    ({n} frames)")
            it.setData(QtCore.Qt.UserRole, ep)
            self.listw.addItem(it)
        n_ep = self.listw.count()
        self.count_lbl.setText(f"📼 recorded episodes: {n_ep}")
        self.empty_lbl.setVisible(n_ep == 0)
        if n_ep:
            self.listw.setCurrentRow(n_ep - 1)                   # default: the latest episode

    def _delete(self):
        it = self.listw.currentItem()
        if it is None:
            return
        ep = it.data(QtCore.Qt.UserRole); name = os.path.basename(ep)
        r = QtWidgets.QMessageBox.question(
            self, "Delete episode?",
            f"Permanently delete {name}?\n\n{ep}\n\nThis cannot be undone.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if r != QtWidgets.QMessageBox.Yes:
            return
        try:
            shutil.rmtree(ep)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Delete failed", str(e)); return
        self._reload()

    def _go(self):
        it = self.listw.currentItem()
        if it is not None:
            self.selected = it.data(QtCore.Qt.UserRole)
            self.accept()


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
        # App-level shortcuts: these fire no matter which child widget holds keyboard focus
        # (keyPressEvent alone is swallowed when e.g. the grip slider is focused). This is what
        # makes '.', ',', 'h' reliable.
        for seq, fn in ((".", self._toggle_record), (",", self._discard_record),
                        ("H", self._go_home)):
            sc = QtWidgets.QShortcut(QtGui.QKeySequence(seq), self)
            sc.setContext(QtCore.Qt.ApplicationShortcut)
            sc.activated.connect(fn)
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
        # --- top bar (full width): E-STOP (also ESC) + Reconnect + REC + Replay ---
        self.estop_btn = QtWidgets.QPushButton("■  E-STOP  (Esc)")
        self.estop_btn.setFocusPolicy(QtCore.Qt.NoFocus); self.estop_btn.setMinimumHeight(46)
        self.estop_btn.setStyleSheet("background:#f38ba8; color:#1e1e2e; font-weight:bold; font-size:16px;")
        self.estop_btn.clicked.connect(self._toggle_estop)
        topbar = QtWidgets.QHBoxLayout(); topbar.addWidget(self.estop_btn, 4)
        self.reconn_btn = QtWidgets.QPushButton("Reconnect"); self.reconn_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.reconn_btn.setMinimumHeight(46); self.reconn_btn.clicked.connect(self._reconnect_all)
        topbar.addWidget(self.reconn_btn, 1)
        # rec + discard share ONE topbar slot: idle -> only REC (full width); recording -> split 50/50.
        rec_box = QtWidgets.QHBoxLayout(); rec_box.setContentsMargins(0, 0, 0, 0); rec_box.setSpacing(4)
        self.rec_btn = QtWidgets.QPushButton("● REC episode  (.)"); self.rec_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.rec_btn.setMinimumHeight(46); self.rec_btn.clicked.connect(self._toggle_record)
        self.rec_btn.setStyleSheet("background:#313244; color:#cdd6f4; font-weight:bold;")
        rec_box.addWidget(self.rec_btn, 1)
        self.discard_btn = QtWidgets.QPushButton("✖ Discard  (,)"); self.discard_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.discard_btn.setMinimumHeight(46); self.discard_btn.clicked.connect(self._discard_record)
        self.discard_btn.setStyleSheet("background:#f38ba8; color:#1e1e2e; font-weight:bold;")
        self.discard_btn.setVisible(False)               # hidden until recording starts
        rec_box.addWidget(self.discard_btn, 1)
        topbar.addLayout(rec_box, 1)
        self.replay_btn = QtWidgets.QPushButton("▶ Replay"); self.replay_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.replay_btn.setMinimumHeight(46); self.replay_btn.clicked.connect(self._open_replay)
        self.replay_btn.setStyleSheet("background:#313244; color:#cdd6f4; font-weight:bold;")
        self.replay_btn.setEnabled(self.has_arm)
        topbar.addWidget(self.replay_btn, 1)
        self.save_btn = QtWidgets.QPushButton("📦 Save → LeRobot"); self.save_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.save_btn.setMinimumHeight(46); self.save_btn.clicked.connect(self._save_lerobot)
        self.save_btn.setStyleSheet("background:#313244; color:#cdd6f4; font-weight:bold;")
        topbar.addWidget(self.save_btn, 1)
        self._convert_worker = None
        self.rec_lbl = QtWidgets.QLabel(""); self.rec_lbl.setObjectName("v")
        root.addLayout(topbar); root.addWidget(self.rec_lbl)
        self.status = QtWidgets.QLabel(); self.status.setObjectName("stat"); root.addWidget(self.status)

        # --- two columns: LEFT = controls, RIGHT = cameras / joints / segmentation (keeps it less tall) ---
        cols = QtWidgets.QHBoxLayout(); root.addLayout(cols, 1)
        left = QtWidgets.QVBoxLayout(); right = QtWidgets.QVBoxLayout()
        cols.addLayout(left, 1); cols.addLayout(right, 1)

        # EEF position (LEFT)
        gp = QtWidgets.QGroupBox("End-Effector  position (m, base frame)")
        gpl = QtWidgets.QGridLayout(gp)
        self.lbl_pos = QtWidgets.QLabel(); self.lbl_pos.setObjectName("v")
        gpl.addWidget(self.lbl_pos, 0, 0, 1, 6)
        for col, (ax, name) in enumerate([(0, "X"), (1, "Y"), (2, "Z")]):
            gpl.addWidget(jog_button(f"-{name}", lambda a=ax: self._set_jog('L', a, -1), lambda a=ax: self._set_jog('L', a, 0)), 1, col*2)
            gpl.addWidget(jog_button(f"+{name}", lambda a=ax: self._set_jog('L', a, +1), lambda a=ax: self._set_jog('L', a, 0)), 1, col*2+1)
        left.addWidget(gp)

        # EEF orientation (LEFT)
        go = QtWidgets.QGroupBox("End-Effector  orientation (deg, RPY)")
        gol = QtWidgets.QGridLayout(go)
        self.lbl_rpy = QtWidgets.QLabel(); self.lbl_rpy.setObjectName("v")
        gol.addWidget(self.lbl_rpy, 0, 0, 1, 6)
        for col, (ax, name) in enumerate([(0, "R"), (1, "P"), (2, "Y")]):
            gol.addWidget(jog_button(f"-{name}", lambda a=ax: self._set_jog('A', a, -1), lambda a=ax: self._set_jog('A', a, 0)), 1, col*2)
            gol.addWidget(jog_button(f"+{name}", lambda a=ax: self._set_jog('A', a, +1), lambda a=ax: self._set_jog('A', a, 0)), 1, col*2+1)
        left.addWidget(go)

        # Hand (LEFT)
        gh = QtWidgets.QGroupBox("Hand")
        ghv = QtWidgets.QVBoxLayout(gh)
        # row 1: open [====== slider ======] closed   + status
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.slider.setRange(0, 100)
        self.slider.valueChanged.connect(lambda v: self.ctrl.set_grip(v / 100.0))
        self.slider.setFocusPolicy(QtCore.Qt.NoFocus)
        self.lbl_hand = QtWidgets.QLabel("open"); self.lbl_hand.setObjectName("v")
        self.lbl_hand.setMinimumWidth(90); self.lbl_hand.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        row1 = QtWidgets.QHBoxLayout()
        row1.addWidget(QtWidgets.QLabel("open")); row1.addWidget(self.slider, 1)
        row1.addWidget(QtWidgets.QLabel("closed")); row1.addWidget(self.lbl_hand)
        ghv.addLayout(row1)
        # row 2: Open / Close / Toggle / Grip config buttons
        b_open = QtWidgets.QPushButton("Open"); b_open.clicked.connect(lambda: self._set_grip_ui(0))
        b_close = QtWidgets.QPushButton("Close"); b_close.clicked.connect(lambda: self._set_grip_ui(100))
        b_tog = QtWidgets.QPushButton("Toggle"); b_tog.clicked.connect(lambda: self._set_grip_ui(0 if self.slider.value() > 50 else 100))
        b_cfg = QtWidgets.QPushButton("Grip config…"); b_cfg.clicked.connect(self._open_grip_config)
        for wdg in (b_open, b_close, b_tog, b_cfg):   # keep keyboard focus on the window
            wdg.setFocusPolicy(QtCore.Qt.NoFocus)
        b_cfg.setEnabled(side == "left")     # per-joint config is DG5F (left) only
        row2 = QtWidgets.QHBoxLayout()
        for b in (b_open, b_close, b_tog, b_cfg):
            row2.addWidget(b)
        ghv.addLayout(row2)
        gh.setEnabled(self.has_hand); left.addWidget(gh)

        # HOME pose (LEFT): captured from the current pose, returned to with "Go Home" (no coord editing).
        if self.side == "left":
            fb = dict(DEFAULT_HOME)
        else:
            p = self.eef[:3, 3]; rpy = np.degrees(mat_to_rpy(self.eef[:3, :3]))
            fb = {"x": p[0], "y": p[1], "z": p[2], "roll": rpy[0], "pitch": rpy[1], "yaw": rpy[2]}
        self.home_vals = load_home(self.side, fb)        # {x,y,z,roll,pitch,yaw}; set by _set_home_current
        self.home_hand = load_home_hand(self.side)       # 20-joint DG5F home pose (None until captured)
        gm = QtWidgets.QGroupBox("Home")
        gml = QtWidgets.QHBoxLayout(gm)
        self.home_btn = QtWidgets.QPushButton("Go Home"); self.home_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.home_btn.clicked.connect(self._go_home)
        self.set_home_btn = QtWidgets.QPushButton("⟲ Set Home ← current")
        self.set_home_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.set_home_btn.clicked.connect(self._set_home_current)
        self.home_status = QtWidgets.QLabel(""); self.home_status.setObjectName("v")
        gml.addWidget(self.home_btn); gml.addWidget(self.set_home_btn); gml.addWidget(self.home_status, 1)
        gm.setEnabled(self.has_arm); left.addWidget(gm)

        # Policy controls (VINN + Diffusion + Init hand) live in a SEPARATE MODAL popup so policy mode
        # and teleop can't be mixed. Mixing is dangerous: e.g. Step -> Home -> Step made the arm rush to
        # the last step pose; or teleoping to an OOD pose then running the policy. The popup blocks the
        # main window while open; closing it exits policy mode and forgets the step state.
        self.policy_btn = QtWidgets.QPushButton("🧠 Policy  (VINN / Diffusion)…")
        self.policy_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.policy_btn.clicked.connect(self._open_policy_dialog)
        self.policy_btn.setEnabled(self.has_arm or self.has_hand)
        left.addWidget(self.policy_btn)
        self.policy_dialog = None
        # Camera input for the policies: RAW by default (the kNN banks were verified to be built from
        # RAW frames — masked input mismatches them). The popup toggle flips this for VINN + diffusion.
        self.mask_input = os.environ.get("VPI_MASK_INPUT", "0").lower() in ("1", "true", "yes")
        # Grip-boost wrapper: curl the 4 fingers a bit MORE than the policy commands so the gripper
        # closes HARDER on the object (our Modbus servo grips to a position, not a force, so extra
        # curl = more grip force). Scales with how much the policy is already gripping (0 when open).
        # Fully removable: toggle off in the popup or VPI_GRIP_BOOST=0 -> the raw policy hand passes through.
        self.grip_boost = os.environ.get("VPI_GRIP_BOOST", "1").lower() in ("1", "true", "yes")
        self.grip_boost_amt = float(os.environ.get("VPI_GRIP_BOOST_AMT", "0.3"))   # extra curl (rad) at full grip
        # DEFAULT diffusion checkpoint (override with VPI_DIFFUSION_MODEL); "Select model" swaps it.
        self.diff_model_path = os.environ.get("VPI_DIFFUSION_MODEL", "/hf/diffusion_left_teleop/diff_hd_teleop_mask_50k")
        self.diff_loaded_path = None            # which model the server currently has loaded
        self.diff_stepping = False              # re-entry guard (one inference at a time)
        self.diff_step_n = 0

        left.addStretch(1)

        # Cameras (RIGHT): front (Zivid) + wrist (D405)
        gc = QtWidgets.QGroupBox("Cameras  (front = Zivid, wrist = D405)")
        gcl = QtWidgets.QHBoxLayout(gc)
        self.cam_front = QtWidgets.QLabel("front: waiting…"); self.cam_wrist = QtWidgets.QLabel("wrist: waiting…")
        for c in (self.cam_front, self.cam_wrist):
            c.setMinimumSize(280, 210); c.setAlignment(QtCore.Qt.AlignCenter)
            c.setStyleSheet("background:#11111b; border:1px solid #45475a;")
        gcl.addWidget(self.cam_front); gcl.addWidget(self.cam_wrist)
        right.addWidget(gc)
        self.cam_tick = QtCore.QTimer(self); self.cam_tick.timeout.connect(self._update_cameras); self.cam_tick.start(100)

        # Joint values (RIGHT): arm 6 joints (deg) + DG5F hand 20 joints (rad)
        gj = QtWidgets.QGroupBox("Joint values")
        gjl = QtWidgets.QVBoxLayout(gj)
        self.lbl_arm_joints = QtWidgets.QLabel("arm: waiting…"); self.lbl_arm_joints.setObjectName("v")
        gjl.addWidget(self.lbl_arm_joints)
        self.hand_joint_lbls = []
        if self.has_hand:
            hgl = QtWidgets.QGridLayout()
            for j in range(4):
                hgl.addWidget(QtWidgets.QLabel(f"_{j+1}"), 0, j + 1)
            for f, fname in enumerate(["thumb", "index", "mid", "ring", "pinky"]):
                hgl.addWidget(QtWidgets.QLabel(fname), f + 1, 0)
                for j in range(4):
                    lb = QtWidgets.QLabel("—"); lb.setObjectName("v"); lb.setMinimumWidth(50)
                    hgl.addWidget(lb, f + 1, j + 1); self.hand_joint_lbls.append(lb)
            gjl.addLayout(hgl)
        right.addWidget(gj)

        # Segmentation (RIGHT): RF-DETR run on the front frame in a background thread (~3 s/frame, CPU)
        gseg = QtWidgets.QGroupBox("Segmentation  (RF-DETR, live · CPU ~3 s/frame)")
        gsl = QtWidgets.QVBoxLayout(gseg)
        segbar = QtWidgets.QHBoxLayout()
        self.seg_btn = QtWidgets.QPushButton("▶ Start segmentation"); self.seg_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.seg_btn.clicked.connect(self._toggle_seg)
        self.seg_status = QtWidgets.QLabel("off"); self.seg_status.setObjectName("v")
        segbar.addWidget(self.seg_btn); segbar.addWidget(self.seg_status, 1)
        gsl.addLayout(segbar)
        self.seg_view = QtWidgets.QLabel("segmentation off")
        self.seg_view.setMinimumSize(320, 240); self.seg_view.setAlignment(QtCore.Qt.AlignCenter)
        self.seg_view.setStyleSheet("background:#11111b; border:1px solid #45475a;")
        gsl.addWidget(self.seg_view)
        right.addWidget(gseg)
        self.seg_worker = None
        right.addStretch(1)

        # --- bottom (full width): speed + key help ---
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

    def _toggle_seg(self):
        if self.seg_worker is not None:                  # -> stop
            self.seg_worker.stop(); self.seg_worker = None
            self.seg_btn.setText("▶ Start segmentation")
            self.seg_status.setText("off")
            self.seg_view.setPixmap(QtGui.QPixmap()); self.seg_view.setText("segmentation off")
            return
        # polls the GPU seg server (SEG_URL /segment); start it first with ./seg_server.sh
        self.seg_worker = SegWorker(lambda: (self.ctrl.hand_pub.front if self.ctrl.hand_pub is not None else None),
                                    SEG_URL)
        self.seg_worker.result.connect(self._on_seg_result)
        self.seg_worker.start()
        self.seg_btn.setText("■ Stop segmentation"); self.seg_status.setText("starting…")

    def _on_seg_result(self, res):
        if isinstance(res, tuple):                       # ("status"|"error", message)
            kind, msg = res
            self.seg_status.setText(("⚠ " if kind == "error" else "") + msg)
            return
        self.seg_view.setText("")                        # res is an RGB overlay ndarray
        self.seg_view.setPixmap(np_to_qpixmap(res, self.seg_view.width(), self.seg_view.height()))

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
    _postproc_workers = []

    def _toggle_record(self):
        if not self.recorder.recording:                  # start recording
            self._rec_last_action = None
            self.recorder.start()
            self.rec_btn.setText("✔ Save  (.)")
            self.rec_btn.setStyleSheet("background:#a6e3a1; color:#1e1e2e; font-weight:bold;")
            self.discard_btn.setVisible(True)            # REC slot now splits 50/50 → Save | Discard
        else:                                            # SAVE (keep) the episode
            n = self.recorder.stop()
            self._end_record_ui()
            self.rec_lbl.setText(f"✔ saved episode {self.recorder.ep_index - 1} ({n} frames) — "
                                 f"📦 Save → LeRobot when done")

    def _discard_record(self):
        if not self.recorder.recording:
            return
        n = self.recorder.discard()                      # delete PNGs, keep ep_index for reuse
        self._end_record_ui()
        self.rec_lbl.setText(f"✖ discarded episode ({n} frames) — not saved")

    def _end_record_ui(self):
        self.rec_btn.setText("● REC episode  (.)")
        self.rec_btn.setStyleSheet("background:#313244; color:#cdd6f4; font-weight:bold;")
        self.discard_btn.setVisible(False)               # back to REC-only, full width

    def _save_lerobot(self):
        """Async: build/update the LeRobot dataset from raw/ (incremental append + front_masked, and a
        full rebuild if an episode was deleted). Runs in a background thread — never blocks teleop."""
        if self._convert_worker is not None and self._convert_worker.isRunning():
            self.rec_lbl.setText("conversion already running…"); return
        host_repo = os.environ.get("VPI_HOST_REPO")
        host_state = os.environ.get("VPI_HOST_STATE")
        if not host_repo or not host_state:
            self.rec_lbl.setText("Save unavailable: relaunch via ./teleop_gui.sh (needs host paths)"); return
        self.save_btn.setEnabled(False)
        self.rec_lbl.setText("📦 converting raw → LeRobot (incremental, + front_masked) …")
        self._convert_worker = ConvertWorker(host_repo, host_state)
        self._convert_worker.progress.connect(lambda s: self.rec_lbl.setText(f"📦 {s}"))
        self._convert_worker.done.connect(self._on_convert_done)
        self._convert_worker.start()

    def _on_convert_done(self, ok, msg):
        self.save_btn.setEnabled(True)
        self.rec_lbl.setText((msg if ok else "⚠ " + msg))

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
        # --- STATE = actual robot / GUI state ---
        sj = np.radians(np.asarray(arm_q[:6], float))              # state.joint (6, rad, absolute)
        se = self._pose6(sj)                                       # state.eef   (6, FK: xyz + rpy)
        sh = np.asarray(hand_a, float)                             # state.hand  (20, rad, actual)
        sg = float(self.ctrl.grip)                                 # state.grip  (1, current grip 0..1)
        # --- ACTION = what we're commanding right now ---
        aj = np.radians(np.asarray(self.ctrl.arm_cmd[:6], float))  # action.joint (6, rad, streamed cmd)
        ae = self._pose6(aj)                                       # action.eef   (6, FK of the command)
        ah = np.asarray(hand_c, float)                             # action.hand  (20, rad, last cmd pose)
        ag = float(self.ctrl.grip_target)                          # action.grip  (1, commanded grip)
        # action.key = the held teleop input directions (WASD etc.): linear xyz, angular rpy, hand jog
        ak = [float(self.jog_lin[0]), float(self.jog_lin[1]), float(self.jog_lin[2]),
              float(self.jog_ang[0]), float(self.jog_ang[1]), float(self.jog_ang[2]), float(self._hand_jog)]
        # LeRobot convention: a SINGLE observation.state + action vector (arm joints 6 + hand 20 = 26).
        # eef / grip / key are kept as extra features (allowed), not nested under observation.state.
        fields = {
            "observation.state": np.concatenate([sj, sh]),   # [26] arm joints(6,rad) + hand(20,rad), actual
            "action":            np.concatenate([aj, ah]),   # [26] arm joints(6,rad) + hand(20,rad), command
            "observation.eef":   se,                         # [6] EEF of the actual joints: x,y,z,roll,pitch,yaw
            "observation.grip":  [sg],                       # [1] grip fraction now (0..1)
            "action.eef":        ae,                         # [6] EEF of the commanded joints
            "action.grip":       [ag],                       # [1] grip fraction commanded
            "action.key":        ak,                         # [7] held teleop input: lin xyz, ang rpy, hand jog
        }
        # Keep a frame only while actively teleoping (input held OR the command moved since last frame);
        # idle waits are skipped and kept frames are re-timestamped contiguously at the record fps.
        act_vec = np.concatenate([aj, ah, [ag]])                  # combined command, for the idle gate
        active = (self.jog_lin.any() or self.jog_ang.any() or self._hand_jog != 0 or self.seq_running
                  or abs(self.ctrl.grip - self.ctrl.grip_target) > 1e-4)
        if not active and self._rec_last_action is not None:
            active = np.max(np.abs(act_vec - self._rec_last_action)) > 5e-4   # command moved (button/seq)
        if not active:
            self.rec_lbl.setText(f"● rec ep {self.recorder.ep_index}: {self.recorder.frame_i} frames"
                                 f"  (paused — idle)")
            return
        self.recorder.add(fields, hp.front, hp.wrist)
        self._rec_last_action = act_vec
        self.rec_lbl.setText(f"● rec ep {self.recorder.ep_index}: {self.recorder.frame_i} frames "
                             f"@ {self.recorder.fps}Hz  (raw PNG)")

    def _pose6(self, joints_rad):
        """FK the 6 arm joints (rad) -> EEF as [x, y, z, roll, pitch, yaw] (m, rad)."""
        try:
            T = self.ik.fk(np.asarray(joints_rad, float))
            p = T[:3, 3]; rpy = mat_to_rpy(T[:3, :3])
            return [float(p[0]), float(p[1]), float(p[2]), float(rpy[0]), float(rpy[1]), float(rpy[2])]
        except Exception:
            return [0.0] * 6

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
        # NOTE: '.', ',', 'h' are bound as app-level QShortcuts (see __init__) so they work
        # regardless of keyboard focus — do NOT also handle them here (would double-fire).
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
        # Clamp dt: this timer is 16 ms, but if the Qt event loop stalls (camera render, a button
        # click, GC) the next tick's raw dt spikes -> jog step = rate*dt would jump the EEF in one
        # tick (esp. orientation). Cap it so a stall costs a little motion, never a jump.
        now = time.perf_counter(); dt = min(now - self._last, 0.05); self._last = now
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
        return dict(self.home_vals)

    def _go_home(self):
        if self.seq_running or not self.has_arm or self.estopped:
            return
        v = self._home_vals()
        pose = np.eye(4)
        pose[:3, 3] = [v["x"], v["y"], v["z"]]
        pose[:3, :3] = rpy_to_mat(math.radians(v["roll"]), math.radians(v["pitch"]), math.radians(v["yaw"]))
        self.seq_running = True
        self.home_btn.setEnabled(False)
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

    def _set_home_current(self):
        """Capture the robot's CURRENT actual pose as Home: arm actual -> FK -> EEF (saved),
        hand actual (sign-corrected command) -> home hand pose (saved)."""
        if not self.has_arm or self.ctrl.arm is None or self.ctrl.arm.q is None:
            self.home_status.setText("no arm state — can't capture"); return
        r = QtWidgets.QMessageBox.question(
            self, "Set Home?",
            "Overwrite the saved Home with the robot's CURRENT pose (arm + hand)?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if r != QtWidgets.QMessageBox.Yes:
            return
        try:
            T = self.ik.fk(np.radians(self.ctrl.arm.q[:6]))
        except Exception as e:
            self.home_status.setText(f"FK fail: {e}"); return
        p = T[:3, 3]; rpy = np.degrees(mat_to_rpy(T[:3, :3]))
        self.home_vals = {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]),
                          "roll": float(rpy[0]), "pitch": float(rpy[1]), "yaw": float(rpy[2])}
        save_home(self.side, self.home_vals)
        msg = "Home ← current arm"
        if self.side == "left" and self.ctrl.hand_pub is not None:
            hc = self.ctrl.hand_pub.hold_command()          # sign-corrected actual = a valid command
            if hc is not None:
                self.home_hand = np.asarray(hc, float)
                save_home_hand(self.side, self.home_hand)
                msg += " + hand"
        self.home_status.setText(msg)

    def _hand_back_to_grip(self, pose):
        """Drop the manual hand override without a snap: project `pose` onto OPEN->GRIP for a grip
        fraction, set it, then clear the override so normal grip control resumes."""
        if self.side != "left":
            return
        try:
            num = den = 0.0
            for idx in DG5F_ACTIVE:
                o, g = DG5F_OPEN_POSE[idx], DG5F_GRIP_POSE[idx]
                if abs(g - o) > 1e-3:
                    num += (pose[idx] - o) * (g - o); den += (g - o) ** 2
            if den > 0:
                gest = float(np.clip(num / den, 0.0, 1.0))
                with self.ctrl.lock:
                    self.ctrl.grip = gest; self.ctrl.grip_target = gest
        except Exception:
            pass
        self.ctrl.set_hand_manual(None)

    def _ramp_hand_to(self, target):
        """Slowly move the DG5F hand to `target` (20-joint) at ~1 rad/s, then hand back to grip."""
        if self.side != "left":
            return
        lh = self.ctrl.last_hand_pose
        cur = (np.asarray(lh, float).copy() if lh is not None else np.asarray(target, float).copy())
        step = 1.0 * 0.05                       # 1 rad/s at 20 Hz
        try:
            for _ in range(400):
                if self.estopped:
                    return
                cur = cur + np.clip(target - cur, -step, step)
                self.ctrl.set_hand_manual(cur)
                if np.max(np.abs(target - cur)) < 1e-3:
                    break
                time.sleep(0.05)
        finally:
            self._hand_back_to_grip(np.asarray(target, float))   # always drop the manual override

    def _run_home(self, pose):
        hand_thread = None
        if self.side == "left" and getattr(self, "home_hand", None) is not None:
            hand_thread = threading.Thread(target=self._ramp_hand_to, args=(self.home_hand.copy(),),
                                           daemon=True)
            hand_thread.start()                 # hand ramps to home in parallel with the arm move
        try:
            self.home_status.setText("going home (straight line) ...")
            ok = self._cartesian_move(pose, self.home_status.setText)
            self.home_status.setText("at home" if ok else self.home_status.text())
        finally:
            if hand_thread is not None:
                hand_thread.join(timeout=6.0)
            self.ctrl.max_dps = ARM_MAX_DPS
            self.seq_running = False
            self.home_btn.setEnabled(True)

    # ---------- replay a recorded episode ----------
    def _open_replay(self):
        if self.seq_running or self.recorder.recording or not self.has_arm or self.estopped:
            return
        dlg = ReplayDialog(self, self.recorder.raw)
        if dlg.exec_() == QtWidgets.QDialog.Accepted and dlg.selected:
            self._start_replay(dlg.selected)

    def _start_replay(self, ep_dir):
        if self.seq_running or self.estopped or not self.has_arm:
            return
        import pandas as pd
        try:
            df = pd.read_parquet(os.path.join(ep_dir, "frames.parquet"))
            actions = np.stack(df["action"].values).astype(float)      # (N,26) = arm joints(6) + hand(20), rad
            fps = float(json.load(open(os.path.join(ep_dir, "meta.json"))).get("fps", 2))
        except Exception as e:
            self.warn = f"replay load failed: {e}"; return
        if len(actions) < 2:
            self.warn = "replay: episode too short"; return
        self.seq_running = True
        self.home_btn.setEnabled(False)
        self.rec_btn.setEnabled(False); self.replay_btn.setEnabled(False)
        threading.Thread(target=self._run_replay, args=(actions, fps, os.path.basename(ep_dir)),
                         daemon=True).start()

    def _run_replay(self, actions, fps, name):
        SMOOTH = 20.0            # resample the low-rate waypoints to 20 Hz so the arm moves continuously
        try:
            n = len(actions)
            arm_deg = np.degrees(actions[:, :6])            # (n,6)
            hand = actions[:, 6:26]                         # (n,20) DG5F command pose (left)
            tin = np.arange(n) / fps
            tout = np.arange(0.0, tin[-1] + 1e-9, 1.0 / SMOOTH)
            arm_i = np.stack([np.interp(tout, tin, arm_deg[:, j]) for j in range(6)], axis=1)
            hand_i = np.stack([np.interp(tout, tin, hand[:, k]) for k in range(20)], axis=1)

            # 1) slow, accel-safe approach to the first recorded pose — BOTH arm and hand ramp gradually.
            #    The arm streams toward the target at max_dps (velocity-clamped). The hand has NO such
            #    clamp (manual poses publish directly), so we ramp it here at HAND_RATE toward hand_i[0].
            self.rec_lbl.setText(f"replay {name}: approaching start pose ...")
            self.ctrl.set_arm_tgt(arm_i[0])
            hand_cur = None
            if self.side == "left":
                lh = self.ctrl.last_hand_pose
                hand_cur = (np.asarray(lh, float).copy() if lh is not None else hand_i[0].copy())
            HAND_RATE = 1.0                                 # rad/s hand approach speed
            hstep = HAND_RATE * 0.1                         # per 0.1 s loop iteration
            for _ in range(200):                            # up to ~20 s to reach the start
                if self.estopped:
                    self.rec_lbl.setText("replay aborted (E-STOP)"); return
                hand_ok = True
                if self.side == "left" and hand_cur is not None:
                    hand_cur = hand_cur + np.clip(hand_i[0] - hand_cur, -hstep, hstep)   # ramp the hand
                    self.ctrl.set_hand_manual(hand_cur)
                    hand_ok = np.max(np.abs(hand_i[0] - hand_cur)) < 1e-3
                q = self.ctrl.arm.q if self.ctrl.arm is not None else None
                arm_ok = q is not None and np.max(np.abs(np.asarray(q[:6], float) - arm_i[0])) < 2.0
                if arm_ok and hand_ok:
                    break
                time.sleep(0.1)

            # 2) play the trajectory at the smoothed rate (reproduces the recorded timing)
            m = len(tout); dt = 1.0 / SMOOTH
            for i in range(1, m):
                if self.estopped:
                    self.rec_lbl.setText("replay aborted (E-STOP)"); return
                self.ctrl.set_arm_tgt(arm_i[i])
                if self.side == "left":
                    self.ctrl.set_hand_manual(hand_i[i])
                if i % int(SMOOTH) == 0:
                    self.rec_lbl.setText(f"replay {name}: {tout[i]:.1f}/{tout[-1]:.1f}s")
                time.sleep(dt)
            self.rec_lbl.setText(f"replay {name}: done ({n} frames, {tout[-1]:.1f}s)")
        finally:
            if self.side == "left":                 # hand back to grip control at the final pose (no snap)
                self._hand_back_to_grip(np.asarray(actions[-1, 6:26], float))
            self.ctrl.max_dps = ARM_MAX_DPS
            self.seq_running = False
            self.home_btn.setEnabled(True)
            self.rec_btn.setEnabled(True); self.replay_btn.setEnabled(self.has_arm)

    # ---------- one-shot VINN policy inference ----------
    def _vinn_infer(self):
        if self.seq_running or self.recorder.recording or not self.has_arm or self.estopped:
            return
        self.seq_running = True
        self.vinn_btn.setEnabled(False); self.home_btn.setEnabled(False)
        threading.Thread(target=self._run_vinn, daemon=True).start()

    def _seg_masked(self, img, requests):
        """Return the hand+hook-only masked RGB (black background) via the seg server (/segment_masked),
        so the image MATCHES the mask kNN bank. Falls back to the raw image on any failure."""
        import base64
        from PIL import Image as PILImage
        try:
            d = requests.post(f"{SEG_URL}/segment_masked", json={"image": jpeg_b64(img)}, timeout=15).json()
            if d.get("masked"):
                return np.asarray(PILImage.open(io.BytesIO(base64.b64decode(d["masked"]))).convert("RGB"))
        except Exception:
            pass
        return img

    def _grip_boost(self, hand):
        """WRAPPER: curl the 4 fingers' flexion joints more than the policy commanded so the gripper
        grips harder. Scales with how much each joint is already gripping (0 when open -> no change
        to an open hand). Returns the raw hand unchanged when the wrapper is toggled off."""
        if not self.grip_boost:
            return hand
        h = np.asarray(hand, float).copy()
        for i in DG5F_ACTIVE:                       # fingers 2-5, flexion joints _2/_3/_4
            span = DG5F_GRIP_POSE[i] - DG5F_OPEN_POSE[i]   # curl direction + range for this joint
            if abs(span) > 1e-3:
                frac = float(np.clip((h[i] - DG5F_OPEN_POSE[i]) / span, 0.0, 1.0))  # 0=open, 1=grip
                h[i] += self.grip_boost_amt * frac * np.sign(span)
        return h

    def _run_vinn(self):
        try:
            hp = self.ctrl.hand_pub
            front = hp.front if hp is not None else None
            wrist = hp.wrist if hp is not None else None
            if front is None or wrist is None:
                self.vinn_status.setText("VINN: no camera image"); return
            import requests
            # VINN needs the kNN policy on the shared inference server; a diffusion Step may have
            # swapped it out (server holds ONE policy at a time). Load kNN if it isn't already active.
            try:
                info = requests.get(f"{VINN_URL}/info", timeout=5).json()
            except Exception:
                info = {}
            if info.get("knn") is None:
                self.vinn_status.setText("VINN: loading kNN policy on server …")
                requests.post(f"{VINN_URL}/load", json={"policy": "knn", "model_id": "knn"},
                              timeout=180).raise_for_status()
                self.diff_loaded_path = None            # diffusion got swapped out -> reload on next Step
            if self.mask_input:                        # RAW by default (banks are raw); toggle in popup
                self.vinn_status.setText("VINN: masking cams …")
                front, wrist = self._seg_masked(front, requests), self._seg_masked(wrist, requests)
            payload = {"front_rgb": jpeg_b64(front), "wrist_rgb": jpeg_b64(wrist), "state": []}
            self.vinn_status.setText("VINN: inferring …")
            r = requests.post(f"{VINN_URL}/predict", json=payload, timeout=30)
            r.raise_for_status()
            action = np.asarray(r.json()["action"], dtype=float).reshape(-1)
            if action.shape[0] < 26:
                self.vinn_status.setText(f"VINN: bad action dim {action.shape}"); return
            arm_deg = np.degrees(action[:6]); hand = self._grip_boost(action[6:26])   # arm6(rad)+hand20(rad), grip-boosted
            self.vinn_status.setText(f"VINN → arm {np.round(arm_deg, 1).tolist()}")
            # apply: arm target (velocity-clamped, gradual) + hand slow ramp (then back to grip)
            self.ctrl.set_arm_tgt(arm_deg)
            if self.side == "left":
                self._ramp_hand_to(hand)
            for _ in range(150):                      # wait until the arm reaches (keeps jog blocked)
                if self.estopped:
                    break
                q = self.ctrl.arm.q if self.ctrl.arm is not None else None
                if q is not None and np.max(np.abs(np.asarray(q[:6], float) - arm_deg)) < 2.0:
                    break
                time.sleep(0.1)
            self.vinn_status.setText(f"VINN done → arm {np.round(arm_deg, 1).tolist()}  "
                                     f"hand[:4] {np.round(hand[:4], 2).tolist()}")
        except Exception as e:
            self.vinn_status.setText(f"VINN fail: {e}")
        finally:
            self.ctrl.max_dps = ARM_MAX_DPS
            self.seq_running = False
            self.vinn_btn.setEnabled(True); self.home_btn.setEnabled(True)

    # ---------- policy popup: isolate VINN/Diffusion from teleop (modal) ----------
    def _build_policy_dialog(self):
        """Build the modal policy popup (VINN + Diffusion + Init hand + E-STOP). Widgets are assigned
        to the same self.* names the policy methods already use, and the dialog is reused across opens."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Policy — VINN / Diffusion")
        dlg.setModal(True)                       # block the main window while open
        dlg.setMinimumWidth(480)
        v = QtWidgets.QVBoxLayout(dlg)
        v.addWidget(QtWidgets.QLabel(
            "Policy mode — the main window is blocked. Closing this exits policy mode and resets the\n"
            "step, so the arm won't rush to the last step pose next time."))

        # RAW/MASK input toggle — applies to BOTH VINN and diffusion. Default RAW (banks are raw-built).
        self.mask_btn = QtWidgets.QPushButton()
        self.mask_btn.setCheckable(True); self.mask_btn.setChecked(self.mask_input)
        self.mask_btn.setFocusPolicy(QtCore.Qt.NoFocus)

        def _upd_mask(_=None):
            self.mask_input = self.mask_btn.isChecked()
            self.mask_btn.setText(f"Camera input:  {'MASK (hand+hook, seg)' if self.mask_input else 'RAW'}")
            self.mask_btn.setStyleSheet("background:#f9e2af; color:#1e1e2e; font-weight:bold;"
                                        if self.mask_input else "")
        self.mask_btn.toggled.connect(_upd_mask); _upd_mask()
        v.addWidget(self.mask_btn)

        # Grip-boost toggle — curl the fingers harder than the policy commands (removable wrapper).
        self.grip_btn = QtWidgets.QPushButton()
        self.grip_btn.setCheckable(True); self.grip_btn.setChecked(self.grip_boost)
        self.grip_btn.setFocusPolicy(QtCore.Qt.NoFocus)

        def _upd_grip(_=None):
            self.grip_boost = self.grip_btn.isChecked()
            self.grip_btn.setText("Grip boost:  " + (f"ON (+{self.grip_boost_amt:.2f} rad)" if self.grip_boost else "OFF"))
            self.grip_btn.setStyleSheet("background:#a6e3a1; color:#1e1e2e; font-weight:bold;" if self.grip_boost else "")
        self.grip_btn.toggled.connect(_upd_grip); _upd_grip()
        v.addWidget(self.grip_btn)

        gv = QtWidgets.QGroupBox("VINN policy  (kNN inference -> robot)")
        gvl = QtWidgets.QHBoxLayout(gv)
        self.vinn_btn = QtWidgets.QPushButton("🧠 Run VINN once"); self.vinn_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.vinn_btn.clicked.connect(self._vinn_infer)
        self.vinn_status = QtWidgets.QLabel(""); self.vinn_status.setObjectName("v")
        gvl.addWidget(self.vinn_btn); gvl.addWidget(self.vinn_status, 1)
        gv.setEnabled(self.has_arm); v.addWidget(gv)

        gd = QtWidgets.QGroupBox("Diffusion policy  (select model -> step-wise inference -> robot)")
        gdl = QtWidgets.QVBoxLayout(gd)
        drow = QtWidgets.QHBoxLayout()
        self.diff_model_btn = QtWidgets.QPushButton("📁 Select model"); self.diff_model_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.diff_model_btn.clicked.connect(self._diff_pick_model)
        self.diff_run_btn = QtWidgets.QPushButton("▶ Step"); self.diff_run_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.diff_run_btn.clicked.connect(self._diff_step)
        drow.addWidget(self.diff_model_btn); drow.addWidget(self.diff_run_btn)
        self.diff_status = QtWidgets.QLabel(); self.diff_status.setObjectName("v")
        gdl.addLayout(drow); gdl.addWidget(self.diff_status)
        gd.setEnabled(self.has_arm); v.addWidget(gd)

        if self.side == "left":
            self.inithand_btn = QtWidgets.QPushButton("✋ Init hand (neutral)")
            self.inithand_btn.setFocusPolicy(QtCore.Qt.NoFocus)
            self.inithand_btn.clicked.connect(self._init_hand)
            self.inithand_btn.setEnabled(self.has_hand)
            v.addWidget(self.inithand_btn)

        # E-STOP inside the popup — the main E-STOP button is unreachable while this is modal.
        estop = QtWidgets.QPushButton("■  E-STOP")
        estop.setFocusPolicy(QtCore.Qt.NoFocus); estop.setMinimumHeight(42)
        estop.setStyleSheet("background:#f38ba8; color:#1e1e2e; font-weight:bold;")
        estop.clicked.connect(self._estop)       # sets window.estopped + ctrl.estop() (breaks the chunk loop)
        v.addWidget(estop)

        close = QtWidgets.QPushButton("Close (exit policy mode)")
        close.setFocusPolicy(QtCore.Qt.NoFocus); close.clicked.connect(dlg.accept)
        v.addWidget(close)
        self.policy_dialog = dlg

    def _open_policy_dialog(self):
        """Open the modal policy popup. Enter a FRESH policy session (forget prior step + reset the
        server episode), then block on it. On close, exit policy mode cleanly."""
        if self.recorder.recording or self.seq_running:
            return
        if self.policy_dialog is None:
            self._build_policy_dialog()
        self.diff_stepping = False
        self.diff_step_n = 0
        self.diff_status.setText(f"model: {os.path.basename(self.diff_model_path)}  (click Step to load + run)")
        self._reset_policy_server()
        self.policy_dialog.exec_()               # MODAL: blocks the main window until closed
        self._exit_policy_mode()

    def _exit_policy_mode(self):
        """Leaving the popup: release the policy's hold so teleop resumes cleanly and the arm does NOT
        rush to the last step pose. Freeze the arm where it is; hand back to normal grip control."""
        try:
            if self.ctrl is not None:
                self.ctrl.set_hand_manual(None)                  # hand -> normal grip control
                cmd, _, _ = self.ctrl.snap()
                self.ctrl.set_arm_tgt(cmd)                       # freeze arm target at current command
        except Exception:
            pass
        self.diff_stepping = False
        self.diff_step_n = 0
        self._reset_policy_server()

    def _reset_policy_server(self):
        """Best-effort: clear the inference server's action queue / episode (fresh rollout)."""
        def _do():
            try:
                import requests
                requests.post(f"{VINN_URL}/reset", timeout=5)
            except Exception:
                pass
        threading.Thread(target=_do, daemon=True).start()

    def _init_hand(self):
        """Move the DG5F to a neutral (open) start pose so a diffusion rollout begins from a known
        hand configuration. Drives the hand directly via the manual channel, overriding any GUI
        grip / staged / home hand setting. Hand-only (no arm needed)."""
        if self.side != "left" or self.ctrl is None or self.estopped:
            return
        neutral = DG5F_NEUTRAL_POSE.copy()                   # separate neutral pose (thumb idx2 flipped down)
        self.ctrl.set_grip(0.0)                              # reset grip state (release staged/manual)
        self.ctrl.set_hand_manual(neutral)                  # then firmly hold the neutral pose
        self.diff_status.setText("hand → neutral init pose (open)")

    # ---------- diffusion policy: model select + step-wise closed loop ----------
    def _diff_pick_model(self):
        """Pick a different diffusion checkpoint folder (loaded lazily on the next Run)."""
        start = os.path.dirname(self.diff_model_path) if self.diff_model_path else "/hf"
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select diffusion checkpoint folder", start)
        if not d:
            return
        self.diff_model_path = d
        self.diff_loaded_path = None                  # force a reload on the next Run
        self.diff_status.setText(f"model: {os.path.basename(d)}  (selected — press Run to load)")

    def _diff_step(self):
        """One diffusion inference -> one robot step, per click (NOT a loop)."""
        if self.diff_stepping or self.diff_model_path is None or not self.has_arm \
                or self.estopped or self.seq_running or self.recorder.recording:
            return
        self.diff_stepping = True
        self.diff_run_btn.setEnabled(False)
        threading.Thread(target=self._run_diffusion_step, daemon=True).start()

    def _diff_obs(self):
        """Build (front, wrist, state26, eef6, grip1) from the live robot, or None."""
        hp = self.ctrl.hand_pub
        front = hp.front if hp is not None else None
        wrist = hp.wrist if hp is not None else None
        q = self.ctrl.arm.q if self.ctrl.arm is not None else None
        if front is None or wrist is None or q is None:
            return None
        arm_rad = np.radians(np.asarray(q[:6], float))
        hand = np.asarray(getattr(hp, "actual", None), float).reshape(-1) if getattr(hp, "actual", None) is not None else np.zeros(20)
        hand = (hand[:20] if hand.shape[0] >= 20 else np.concatenate([hand, np.zeros(20 - hand.shape[0])]))
        state = np.concatenate([arm_rad, hand]).astype(float)      # 26 = arm6 + hand20 (rad)
        # eef (6) = FK xyz + rpy(rad); grip (1) = current grip 0..1.
        # NOTE: the eef/grip encoding must match what the checkpoint was TRAINED on — if the
        # policy behaves oddly, this is the first place to reconcile (order / units / rotation rep).
        try:
            T = self.ik.fk(arm_rad)
            eef = np.concatenate([T[:3, 3], mat_to_rpy(T[:3, :3])])
        except Exception:
            eef = np.zeros(6)
        grip = [float(self.ctrl.snap()[2])]
        return front, wrist, state.tolist(), eef.tolist(), grip

    def _run_diffusion_step(self):
        import requests
        try:
            # (re)load the selected/default model on the server if needed; reset the episode on load.
            if self.diff_loaded_path != self.diff_model_path:
                self.diff_status.setText(f"loading {os.path.basename(self.diff_model_path)} …")
                r = requests.post(f"{VINN_URL}/load",
                                  json={"model_id": self.diff_model_path, "policy": "diffusion", "step_mode": False},
                                  timeout=300)
                r.raise_for_status()
                self.diff_loaded_path = self.diff_model_path
                requests.post(f"{VINN_URL}/reset", timeout=10)
                self.diff_step_n = 0
            obs = self._diff_obs()
            if obs is None:
                self.diff_status.setText("diffusion: no camera/arm signal"); return
            front, wrist, state, eef, grip = obs
            if self.mask_input:                        # RAW by default; toggle in popup
                self.diff_status.setText("diffusion: masking cams …")
                front, wrist = self._seg_masked(front, requests), self._seg_masked(wrist, requests)
            payload = {"front_rgb": jpeg_b64(front), "wrist_rgb": jpeg_b64(wrist),
                       "state": state, "eef": eef, "grip": grip}
            # ACTION CHUNK: the server is loaded step_mode=False, so the model queues an
            # n_action_steps chunk; each /predict serves the NEXT queued action. Execute the whole
            # chunk from this one observation (a short open-loop rollout per click, not one step).
            for i in range(DIFF_CHUNK):
                if self.estopped:
                    break
                a = np.asarray(requests.post(f"{VINN_URL}/predict", json=payload, timeout=30).json()["action"],
                               float).reshape(-1)
                if a.shape[0] < 26:
                    self.diff_status.setText(f"diffusion: bad action dim {a.shape}"); return
                arm_deg = np.degrees(a[:6]); hand = self._grip_boost(a[6:26])   # grip-boosted (removable wrapper)
                self.ctrl.set_arm_tgt(arm_deg)
                if self.side == "left":
                    self.ctrl.set_hand_manual(hand)    # follow ONLY the model (overrides GUI hand settings)
                self.diff_step_n += 1
                self.diff_status.setText(f"chunk {i + 1}/{DIFF_CHUNK}  step {self.diff_step_n}  "
                                         f"arm {np.round(arm_deg, 1).tolist()}")
                # let the velocity-clamped arm track this sub-target before the next chunk action
                for _ in range(30):
                    if self.estopped:
                        break
                    q = self.ctrl.arm.q if self.ctrl.arm is not None else None
                    if q is not None and np.max(np.abs(np.asarray(q[:6], float) - arm_deg)) < 2.0:
                        break
                    time.sleep(0.05)
        except Exception as e:
            self.diff_status.setText(f"diffusion fail: {e}")
        finally:
            self.diff_stepping = False
            self.diff_run_btn.setEnabled(True)

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
        if arm_ok and self.ctrl.arm.q is not None:
            self.lbl_arm_joints.setText("arm(deg)  " + "  ".join(f"J{i+1} {self.ctrl.arm.q[i]:+.1f}" for i in range(6)))
        if self.has_hand and self.hand_joint_lbls:      # DG5F 20 joints (rad), canonical order
            ha = self.ctrl.hand_pub.actual if self.ctrl.hand_pub is not None else None
            if ha is not None and len(ha) >= len(self.hand_joint_lbls):
                for k, lb in enumerate(self.hand_joint_lbls):
                    lb.setText(f"{float(ha[k]):+.2f}")
        self.lbl_speed.setText(f"jog speed:  {self.trans_rate*100:.1f} cm/s   {math.degrees(self.rot_rate):.0f} deg/s")

    def closeEvent(self, ev):
        if getattr(self, "seg_worker", None) is not None:
            self.seg_worker.stop(); self.seg_worker.wait(2000)
        for w in self._postproc_workers:               # let masking finish quietly; don't hang the close
            w.stop(); w.wait(2000)
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
    win.resize(1200, 800); win.show()      # two-column layout needs the extra width (left controls | right cams/joints/seg)
    win.raise_(); win.activateWindow(); win.setFocus()   # grab keyboard focus on launch
    code = app.exec_()

    ctrl.running = False
    if hand_pub is not None: hand_pub.destroy_node()
    rclpy.shutdown(); sys.exit(code)


if __name__ == "__main__":
    main()
