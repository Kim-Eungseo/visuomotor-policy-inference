#!/usr/bin/env python3
"""Prune over-represented low-motion (grasp / hold) frames from a VINN kNN bank.

The kNN policy collapses into an idle-loop after grasping because the grasp-HOLD region has many
near-identical low-action frames — they dominate the top-k vote and keep retrieving "stay put".
This rebalances the bank by KEYFRAME selection: within each episode, keep a frame only when the
cumulative action change since the last kept frame exceeds TAU (rad), or every G frames as a floor
(so long holds still contribute a few frames, not hundreds). Motion frames are kept fully; holds
are thinned. Episode ids are preserved (provenance unchanged); episode endpoints are always kept.

  python examples/prune_bank.py IN.npz OUT.npz            # defaults TAU=0.02 G=8
  VPI_PRUNE_TAU=0.03 VPI_PRUNE_G=10 python examples/prune_bank.py IN.npz OUT.npz

Bank format: .npz with emb (N,D) L2-norm, act (N,26), epi (N,). Frames must be in dataset order
(episodes contiguous, timesteps ascending) — as produced by knn_build_bank.py.
"""
import os
import sys

import numpy as np

TAU = float(os.environ.get("VPI_PRUNE_TAU", "0.02"))   # keep when cumulative |Δaction| (rad) >= TAU
G = int(os.environ.get("VPI_PRUNE_G", "8"))            # ...or every G frames (hold floor)


def prune(inp: str, out: str) -> None:
    z = np.load(inp)
    emb, act, epi = z["emb"], z["act"], z["epi"].reshape(-1)
    uepi = np.unique(epi)
    if not np.all(np.diff(epi) >= 0):
        print("WARN: epi not non-decreasing — frames may not be in dataset order", file=sys.stderr)
    # per-frame within-episode action delta (L2 over all joints)
    d = np.zeros(len(act), np.float32)
    for e in uepi:
        m = np.where(epi == e)[0]
        d[m[1:]] = np.linalg.norm(np.diff(act[m], axis=0), axis=1)
    keep = np.zeros(len(act), bool)
    for e in uepi:
        m = np.where(epi == e)[0]
        keep[m[0]] = True; keep[m[-1]] = True        # always keep episode endpoints
        acc = 0.0; gap = 0
        for j in range(1, len(m)):
            acc += d[m[j]]; gap += 1
            if acc >= TAU or gap >= G:
                keep[m[j]] = True; acc = 0.0; gap = 0
    tmp = out + ".tmp.npz"
    np.savez(tmp, emb=emb[keep], act=act[keep], epi=epi[keep].astype(np.int64))
    os.replace(tmp, out)
    moving = d > 0.02
    print(f"[prune] TAU={TAU} G={G}: kept {keep.sum()}/{len(keep)} ({keep.mean()*100:.1f}%)  "
          f"| moving(>0.02) retained {keep[moving].mean()*100:.1f}%  "
          f"| episodes {len(uepi)} -> {len(np.unique(epi[keep]))}  -> {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(1)
    prune(sys.argv[1], sys.argv[2])
