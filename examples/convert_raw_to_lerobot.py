#!/usr/bin/env python3
"""Convert the GUI's RAW teleop recording into a proper LeRobot v3.0 dataset using the real lerobot
API (run INSIDE visuomotor-policy-inference:latest, which ships lerobot 0.5.1 — the GUI container has
no lerobot). Images are stored in image mode (use_videos=False -> PNG embedded in the parquet, no mp4).

Raw layout produced by teleop_gui.py's EpisodeRecorder:
    <raw>/episode_000000/front/000000.png  wrist/000000.png  ...  frames.parquet  meta.json
    frames.parquet columns: frame_index, timestamp, observation.state[26], action[26], task

Usage (normally invoked by ../to_lerobot.sh):
    python3 convert_raw_to_lerobot.py --raw /raw --out /out [--repo-id rec/teleop] [--task NAME] [--fps N]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")        # never phone the Hub for a local-only dataset
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
import pandas as pd
from PIL import Image                               # vpi image has PIL, not cv2

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="raw dir containing episode_* folders")
    ap.add_argument("--out", required=True, help="output LeRobot dataset dir (must be empty/absent)")
    ap.add_argument("--repo-id", default="rec/teleop")
    ap.add_argument("--task", default=None, help="override the task string (else per-frame from parquet)")
    ap.add_argument("--fps", type=int, default=None, help="override fps (else from each episode's meta.json)")
    args = ap.parse_args()

    eps = sorted(glob.glob(os.path.join(args.raw, "episode_*")))
    eps = [e for e in eps if os.path.exists(os.path.join(e, "frames.parquet"))]
    if not eps:
        raise SystemExit(f"no finished episodes (with frames.parquet) under {args.raw}")

    meta0 = json.load(open(os.path.join(eps[0], "meta.json")))
    fps = args.fps or int(meta0["fps"])
    H, W = meta0.get("image_hw", [480, 640])

    if os.path.isdir(args.out) and os.listdir(args.out):
        raise SystemExit(f"--out '{args.out}' exists and is non-empty. Delete it first (it is regenerable "
                         f"from raw/).")

    feats = {
        "observation.state": {"dtype": "float32", "shape": (26,), "names": None},
        "action": {"dtype": "float32", "shape": (26,), "names": None},
        "observation.images.front": {"dtype": "image", "shape": (H, W, 3),
                                     "names": ["height", "width", "channel"]},
        "observation.images.wrist": {"dtype": "image", "shape": (H, W, 3),
                                     "names": ["height", "width", "channel"]},
    }
    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=fps, root=args.out, features=feats,
                               use_videos=False)

    total = 0
    for ep in eps:
        df = pd.read_parquet(os.path.join(ep, "frames.parquet"))
        for _, row in df.iterrows():
            i = int(row["frame_index"])
            fp = os.path.join(ep, "front", f"{i:06d}.png")
            wp = os.path.join(ep, "wrist", f"{i:06d}.png")
            if not (os.path.exists(fp) and os.path.exists(wp)):
                raise SystemExit(f"missing image for {os.path.basename(ep)} frame {i}: {fp} / {wp}")
            ds.add_frame({
                "observation.state": np.asarray(row["observation.state"], dtype=np.float32),
                "action": np.asarray(row["action"], dtype=np.float32),
                "observation.images.front": np.asarray(Image.open(fp).convert("RGB")),   # HWC RGB uint8
                "observation.images.wrist": np.asarray(Image.open(wp).convert("RGB")),
                "task": args.task or str(row["task"]),
            })
        ds.save_episode()
        total += len(df)
        print(f"  + {os.path.basename(ep)}: {len(df)} frames")

    ds.finalize()      # writes meta/episodes + stats; REQUIRED for the dataset to load
    print(f"DONE: {len(eps)} episodes, {total} frames @ {fps} Hz -> {args.out}")


if __name__ == "__main__":
    main()
