"""Build the idle-filtered serving bank: drop near-zero-action frames (Linf(consecutive-action
delta) < tau, default 3e-3 rad) plus the 32 all-zero-arm teleop-glitch frames.

Sources: the full mask12 bank (left_vinn/banks/left_infonce_mask12.npz, dataset-ordered) and the
per-frame motion stats (analysis/zero_action/motion.npz, same row order — asserted). Episode-last
frames (no forward delta) are KEPT. Offline A/B: bank 88k->59k (-33%), holdout k=32 L1
0.0633->0.0632; caveat: late-episode (60-90%) coverage thins ~2x — validate grasp/settle on-robot.
NOTE for serving: knn_runner builds future chunks from consecutive bank rows, so chunks
fast-forward across deleted holds (boundary jumps are tiny by construction; Ruckig absorbs them).

  conda run -n lr python build_left_filtered_bank.py [--tau 3e-3]
"""
import argparse

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--tau", type=float, default=3e-3)
ap.add_argument("--bank", default="left_vinn/banks/left_infonce_mask12.npz")
ap.add_argument("--motion", default="analysis/zero_action/motion.npz")
ap.add_argument("--out", default="left_vinn/banks/left_infonce_mask12_filt3e3.npz")
args = ap.parse_args()

z = np.load(args.bank)
m = np.load(args.motion)
emb, act, epi = z["emb"].astype(np.float32), z["act"].astype(np.float32), z["epi"].astype(np.int64)
assert np.array_equal(epi, m["epi"]), "bank/motion row order mismatch"
fi, linf = m["fi"].astype(np.int64), m["linf_delta"]

valid = ~np.isnan(linf)
idle = valid & (linf < args.tau)                      # episode-last frames (NaN) are kept
zero_arm = np.abs(act[:, :6]).max(1) == 0             # teleop-not-engaged glitch frames
keep = ~idle & ~zero_arm
print(f"total {len(epi)} | idle<{args.tau}: {idle.sum()} | zero-arm: {zero_arm.sum()} "
      f"(overlap {int((idle & zero_arm).sum())}) | kept {keep.sum()} ({keep.mean():.1%})")

np.savez(args.out, emb=emb[keep], act=act[keep], epi=epi[keep], fi=fi[keep],
         linf_delta=linf[keep].astype(np.float32), filter_tau=np.float32(args.tau))
kept_eps = len(np.unique(epi[keep]))
print(f"saved {args.out}: emb{emb[keep].shape} act{act[keep].shape} eps={kept_eps}")
assert kept_eps == len(np.unique(epi)), "an entire episode vanished — check tau"
