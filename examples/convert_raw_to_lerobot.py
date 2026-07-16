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
import base64
import glob
import io
import json
import os
import shutil

os.environ.setdefault("HF_HUB_OFFLINE", "1")        # never phone the Hub for a local-only dataset
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
import pandas as pd
import requests
from PIL import Image                               # vpi image has PIL, not cv2

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def seg_mask(rgb, url):
    """Send an RGB frame to the seg server and return the hand+hook-only masked RGB (rest black).
    On any failure, returns the original frame (so conversion never breaks on a seg hiccup)."""
    try:
        b = io.BytesIO(); Image.fromarray(rgb).save(b, "JPEG", quality=95)
        d = requests.post(f"{url}/segment_masked",
                          json={"image": base64.b64encode(b.getvalue()).decode()}, timeout=30).json()
        if d.get("masked"):
            return np.asarray(Image.open(io.BytesIO(base64.b64decode(d["masked"]))).convert("RGB"))
    except Exception:
        pass
    return rgb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="raw dir containing episode_* folders")
    ap.add_argument("--out", required=True, help="output LeRobot dataset dir (must be empty/absent)")
    ap.add_argument("--repo-id", default="rec/teleop")
    ap.add_argument("--task", default=None, help="override the task string (else per-frame from parquet)")
    ap.add_argument("--fps", type=int, default=None, help="override fps (else from each episode's meta.json)")
    ap.add_argument("--no-video", action="store_true",
                    help="store images embedded in the parquet instead of encoding mp4 videos (default: video)")
    ap.add_argument("--seg-url", default=os.environ.get("VPI_SEG_URL", "http://localhost:8100"),
                    help="seg server for the extra observation.images.front_masked feature")
    ap.add_argument("--no-mask", action="store_true", help="skip the front_masked feature (seg off)")
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild from scratch instead of appending only new episodes (default: incremental)")
    args = ap.parse_args()
    use_videos = not args.no_video
    do_mask = not args.no_mask and bool(args.seg_url)
    if do_mask:
        try:
            do_mask = bool(requests.get(f"{args.seg_url}/health", timeout=5).json().get("ok"))
        except Exception:
            do_mask = False
        if not do_mask:
            print("WARNING: seg server not reachable -> skipping observation.images.front_masked "
                  "(start it with ./seg_server.sh)")

    eps = sorted(glob.glob(os.path.join(args.raw, "episode_*")))
    eps = [e for e in eps if os.path.exists(os.path.join(e, "frames.parquet"))]
    if not eps:
        raise SystemExit(f"no finished episodes (with frames.parquet) under {args.raw}")

    meta0 = json.load(open(os.path.join(eps[0], "meta.json")))
    fps = args.fps or int(meta0["fps"])
    H, W = meta0.get("image_hw", [480, 640])

    # INCREMENTAL: a manifest tracks which raw episodes are already in the LeRobot dataset, so we only
    # convert NEW ones (append) instead of rebuilding everything. `--rebuild` forces a full rebuild.
    manifest_path = os.path.join(args.out, ".converted_raw.json")
    have_ds = (os.path.isdir(args.out)
               and os.path.exists(os.path.join(args.out, "meta", "info.json")))
    if args.rebuild:
        have_ds = False
    converted = set()
    if have_ds and os.path.exists(manifest_path):
        converted = set(json.load(open(manifest_path)))
        # A previously-converted episode that is GONE from raw was deleted. LeRobot v3 packs episodes
        # into shared parquet/mp4 files, so cleanly removing one mid-dataset means repacking + renumbering
        # every episode index/offset and recomputing stats — i.e. a full rebuild of the remaining episodes.
        deleted = converted - {os.path.basename(e) for e in eps}
        if deleted:
            print(f"detected deleted episode(s) {sorted(deleted)} -> full rebuild "
                  f"(repack + renumber + fix meta)")
            shutil.rmtree(args.out, ignore_errors=True)
            have_ds = False
            converted = set()
    new_eps = [e for e in eps if os.path.basename(e) not in converted]
    if have_ds and not new_eps:
        print(f"nothing new — all {len(eps)} raw episode(s) already in {args.out}")
        return

    df0 = pd.read_parquet(os.path.join(eps[0], "frames.parquet"))
    SKIP = {"frame_index", "timestamp", "task"}
    vec_cols = [c for c in df0.columns if c not in SKIP]

    if have_ds:                                        # APPEND to the existing dataset
        ds = LeRobotDataset(repo_id=args.repo_id, root=args.out)
        print(f"appending {len(new_eps)} new episode(s) to {ds.num_episodes} existing "
              f"(mask={'on' if do_mask else 'off'})")
    else:                                              # CREATE a fresh dataset
        feats = {c: {"dtype": "float32", "shape": (len(df0.iloc[0][c]),), "names": None} for c in vec_cols}
        img_dtype = "video" if use_videos else "image"
        for k in ("observation.images.front", "observation.images.wrist"):
            feats[k] = {"dtype": img_dtype, "shape": (H, W, 3), "names": ["height", "width", "channel"]}
        if do_mask:                                    # extra: front with only hand+hook kept (raw kept too)
            feats["observation.images.front_masked"] = {"dtype": img_dtype, "shape": (H, W, 3),
                                                        "names": ["height", "width", "channel"]}
        print(f"creating dataset ({'video' if use_videos else 'image'} mode, mask={'on' if do_mask else 'off'}):",
              {c: feats[c]["shape"] for c in feats})
        ds = LeRobotDataset.create(repo_id=args.repo_id, fps=fps, root=args.out, features=feats,
                                   use_videos=use_videos)
        converted = set()                              # fresh dataset -> nothing converted yet
        new_eps = eps

    add_masked = do_mask and ("observation.images.front_masked" in ds.meta.info["features"])
    total = 0
    ne = len(new_eps)
    for ei, ep in enumerate(new_eps):
        name = os.path.basename(ep)
        df = pd.read_parquet(os.path.join(ep, "frames.parquet"))
        nfr = len(df)
        for j, (_, row) in enumerate(df.iterrows()):
            i = int(row["frame_index"])
            fp = os.path.join(ep, "front", f"{i:06d}.png")
            wp = os.path.join(ep, "wrist", f"{i:06d}.png")
            if not (os.path.exists(fp) and os.path.exists(wp)):
                raise SystemExit(f"missing image for {name} frame {i}: {fp} / {wp}")
            frame = {c: np.asarray(row[c], dtype=np.float32) for c in vec_cols}
            front = np.asarray(Image.open(fp).convert("RGB"))   # HWC RGB uint8 (raw, unmodified)
            frame["observation.images.front"] = front
            frame["observation.images.wrist"] = np.asarray(Image.open(wp).convert("RGB"))
            if add_masked:                              # hand+hook-only version of the FRONT frame
                frame["observation.images.front_masked"] = seg_mask(front, args.seg_url)
            frame["task"] = args.task or str(row["task"])
            ds.add_frame(frame)
            if j % 5 == 0 or j == nfr - 1:              # live progress (streamed to the GUI)
                print(f"[progress] ep {ei + 1}/{ne} {name}: frame {j + 1}/{nfr}"
                      f"{' +mask' if add_masked else ''}", flush=True)
        print(f"[progress] encoding video for {name} …", flush=True)
        ds.save_episode()
        converted.add(name)
        total += nfr
        print(f"  + {os.path.basename(ep)}: {len(df)} frames")

    ds.finalize()      # writes meta/episodes + stats; REQUIRED for the dataset to load
    json.dump(sorted(converted), open(manifest_path, "w"))
    print(f"DONE: +{len(new_eps)} episode(s), {total} frames -> {args.out} (total {ds.num_episodes})")


if __name__ == "__main__":
    main()
