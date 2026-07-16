#!/usr/bin/env python3
"""Mask raw teleop episodes to HAND+HOOK only via the seg server (/segment_masked), overwriting the
raw PNGs (background zeroed). Same operation as the GUI's async PostProcWorker, but standalone — for
retroactive / manual masking of episodes recorded before auto-masking existed.

  python3 mask_raw_episodes.py --raw <raw_dir> [--url http://localhost:8100] [--episodes episode_000001 ...]
"""
from __future__ import annotations

import argparse
import base64
import glob
import io
import os

import requests
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="raw dir containing episode_* folders")
    ap.add_argument("--url", default=os.environ.get("VPI_SEG_URL", "http://localhost:8100"))
    ap.add_argument("--episodes", nargs="*", default=None, help="episode names (default: all)")
    ap.add_argument("--cams", nargs="*", default=["front"])   # front only; wrist kept original
    args = ap.parse_args()

    if not requests.get(f"{args.url}/health", timeout=5).json().get("ok"):
        raise SystemExit("seg server not ready — start it with ./seg_server.sh")

    eps = args.episodes or [os.path.basename(d)
                            for d in sorted(glob.glob(os.path.join(args.raw, "episode_*")))]
    for ep in eps:
        d = os.path.join(args.raw, ep)
        for cam in args.cams:
            files = sorted(glob.glob(os.path.join(d, cam, "*.png")))
            nodet = 0
            for fp in files:
                rgb = Image.open(fp).convert("RGB")
                b = io.BytesIO(); rgb.save(b, "JPEG", quality=95)
                r = requests.post(f"{args.url}/segment_masked",
                                  json={"image": base64.b64encode(b.getvalue()).decode()}, timeout=30).json()
                if r.get("masked"):
                    Image.open(io.BytesIO(base64.b64decode(r["masked"]))).convert("RGB").save(fp)
                    if not r.get("detected", True):
                        nodet += 1
            print(f"  {ep}/{cam}: {len(files)} frames masked ({nodet} no-detect)", flush=True)
    print("done")


if __name__ == "__main__":
    main()
