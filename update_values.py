"""Turn per-episode selection logs + episode outcome labels into the per-entry value sidecar
(alpha/beta Beta posteriors aligned to bank rows) that knn_runner's top1/thompson modes consume.

Stateless: recomputes from prior + ALL labeled episodes each run (idempotent).
  labels.csv lines: <episode log filename>,<score in [0,1]>   # 1=clean success, 0=fail;
  fractional scores allowed (graded reward, see VINN_RL_PLAN.md rubric). Episodes aborted for
  reasons unrelated to the policy (cable snag, external bump): just OMIT the line -> ignored.
Credit (asymmetric): the SUCCESS mass R is credited uniformly to every selected entry (a success
validates the whole followed path), but the FAILURE mass (1-R) is recency-weighted from the end
of the log — 0.5^((T_end - t_lastsel) / (fps*fail_halflife)) — so the choices right before the
failure take the blame and the early approach is spared. STOP the episode as soon as it fails:
the log end IS the failure timestamp. Same-demo neighbors within +-prop_w raw frames get the
same (scaled) update x prop_c. --fail_halflife 0 restores uniform blame.

  conda run -n lr python update_values.py --logs rl_logs --labels rl_logs/labels.csv \
      --bank left_vinn/banks/left_infonce_mask12_filt3e3.npz --out values_left_filt3e3.npz
"""
import argparse
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--logs", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--bank", required=True, help="bank npz (for epi / frame alignment)")
ap.add_argument("--out", required=True)
ap.add_argument("--prior_a", type=float, default=3.0)
ap.add_argument("--prior_b", type=float, default=1.0)
ap.add_argument("--prop_w", type=int, default=15, help="temporal propagation window, raw frames")
ap.add_argument("--prop_c", type=float, default=0.3, help="propagation credit scale")
ap.add_argument("--fail_halflife", type=float, default=3.0, help="failure-blame half-life, seconds (0=uniform)")
ap.add_argument("--fps", type=float, default=30.0, help="control tick rate of the logs")
args = ap.parse_args()

z = np.load(args.bank)
epi = np.asarray(z["epi"], np.int64)
if "fi" in z:
    fi = np.asarray(z["fi"], np.int64)
else:                                              # reconstruct frame index by cumcount per episode
    fi = np.concatenate([np.arange(c) for c in np.bincount(epi)])
M = len(epi)
alpha = np.full(M, args.prior_a)
beta = np.full(M, args.prior_b)

n_ep = n_sel = 0
for line in open(args.labels):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    fname, r = line.rsplit(",", 1)
    R = float(r)
    assert 0.0 <= R <= 1.0, f"score out of [0,1]: {line}"
    path = os.path.join(args.logs, fname.strip())
    if not os.path.exists(path):
        print(f"  WARN missing log {fname}, skipped")
        continue
    last_t: dict[int, int] = {}                    # entry -> last tick it was selected
    t_end = 0
    for rec in map(json.loads, open(path)):
        t_end = max(t_end, int(rec.get("t", 0)))
        if rec.get("sel") is not None:
            last_t[int(rec["sel"])] = int(rec.get("t", 0))
    if not last_t:
        print(f"  WARN {fname}: no selections (softmin/fallback only), skipped")
        continue
    n_ep += 1; n_sel += len(last_t)
    for i, tl in last_t.items():
        rw = 1.0 if args.fail_halflife <= 0 else 0.5 ** ((t_end - tl) / (args.fps * args.fail_halflife))
        alpha[i] += R                              # success mass: uniform over the whole path
        beta[i] += (1 - R) * rw                    # failure mass: recency-weighted from log end
        near = np.flatnonzero((epi == epi[i]) & (np.abs(fi - fi[i]) <= args.prop_w) & (np.arange(M) != i))
        alpha[near] += args.prop_c * R
        beta[near] += args.prop_c * (1 - R) * rw

np.savez(args.out, alpha=alpha.astype(np.float32), beta=beta.astype(np.float32),
         prior_a=np.float32(args.prior_a), prior_b=np.float32(args.prior_b))
v = alpha / (alpha + beta)
touched = int(((alpha != args.prior_a) | (beta != args.prior_b)).sum())
print(f"episodes={n_ep} unique-selections={n_sel} | entries touched={touched}/{M} "
      f"| v: min={v.min():.3f} mean={v.mean():.3f} max={v.max():.3f} -> {args.out}")
