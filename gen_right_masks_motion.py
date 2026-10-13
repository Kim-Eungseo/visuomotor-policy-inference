"""Model-free foreground masks for the RIGHT dataset via static-camera background subtraction.

The left masks came from RF-DETR (DG5F-trained); it hallucinates on the INSPIRE right hand and
GroundingDINO+SAM also fails on this dark industrial scene (see scratchpad right_seg_*.png).
The front camera is STATIC, so per-episode temporal-median background subtraction segments the
moving foreground (arm + hand + grasped strut + cable) robustly with no model at all.

Output npz matches masks_left_stride3.npz exactly (masks packbits (N,480,80), epi, fi, thresh,
stride) so left_infonce.py --use_mask --mask_npz works unchanged. Semantic difference vs left:
foreground includes the ARM (not just hand+hook) — for bg-swap that is fine (dynamics preserved).

  conda run -n lr python gen_right_masks_motion.py
"""
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = "/home/ngseo/remove_hook/lerobot/data/lerobot/hyundai_uiwang_right"
OUT = "/home/ngseo/visuomotor-policy-inference/masks_right_stride3.npz"
STRIDE, THRESH, BG_SAMPLES = 3, 0.10, 60

ds = LeRobotDataset("Ngseo/hyundai-uiwang-right", root=ROOT, revision="main")
fkey = next(k for k in ds.meta.camera_keys if "front" in k)
print(f"{len(ds)} frames | stride {STRIDE} -> ~{len(ds)//STRIDE} masks", flush=True)

M, P, I = [], [], []
t0 = time.time()


def flush(e, buf, fis):
    if not buf:
        return
    x = torch.stack(buf)                                            # (n,3,480,640) uint8
    sub = x[torch.linspace(0, len(x) - 1, steps=min(BG_SAMPLES, len(x))).long()]
    bg = sub.float().div(255).median(0).values                      # static background estimate
    for s in range(0, len(x), 32):
        d = (x[s:s + 32].float().div(255) - bg).abs().amax(1, keepdim=True)   # (b,1,480,640)
        m = (d > THRESH).float()
        m = F.max_pool2d(m, 9, 1, 4)                                # dilate
        m = F.avg_pool2d(m, 9, 1, 4) > 0.45                         # despeckle
        for k in range(m.shape[0]):
            M.append(np.packbits(m[k, 0].numpy(), axis=-1))
    P.extend([e] * len(buf)); I.extend(fis)
    el = time.time() - t0
    print(f"  ep{e}: {len(buf)} masks (total {len(M)}) | {len(M)/el:.1f} fps", flush=True)


dl = DataLoader(ds, batch_size=64, num_workers=6, shuffle=False)
cur_e, buf, fis = None, [], []
for b in dl:
    epi = np.asarray(b["episode_index"]).reshape(-1)
    fi = np.asarray(b["frame_index"]).reshape(-1)
    keep = fi % STRIDE == 0
    for j in np.flatnonzero(keep):
        e = int(epi[j])
        if cur_e is not None and e != cur_e:
            flush(cur_e, buf, fis); buf, fis = [], []
        cur_e = e
        buf.append((b[fkey][j] * 255).round().clamp(0, 255).to(torch.uint8))
        fis.append(int(fi[j]))
flush(cur_e, buf, fis)

np.savez(OUT, masks=np.stack(M), epi=np.array(P, np.int64), fi=np.array(I, np.int64),
         thresh=THRESH, stride=STRIDE)
print(f"saved {OUT}: {len(M)} masks ({np.stack(M).nbytes/1e9:.2f}GB packed)", flush=True)
