"""Generate RF-DETR hand+hook union masks for the MAIN left dataset (stride-3 frames, ALL episodes).

Runs `Ngseo/hyundai-uiwang-hook-segmentation` (2-class RF-DETR Seg2XLarge) on every 3rd frame's
front_rgb and stores the union(hand, hook) mask bit-packed at full 480x640:
  out npz: masks (N, 480, 80) uint8 packbits, epi (N,), fi (N,), thresh
~15 fps on the RTX PRO 4500 -> ~41 min for 36.9k frames. DataLoader workers decode in parallel,
GPU seg is the bottleneck. Covers ALL episodes so both holdout-A/B and full-data BYOL can use it.
"""
import time

import numpy as np
import torch
from PIL import Image
from rfdetr import RFDETRSeg2XLarge
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset

WEIGHT = "/home/ngseo/.cache/huggingface/hub/models--Ngseo--hyundai-uiwang-hook-segmentation/snapshots/ea720f7741c09dca11577ba70f4ba1d098eb62b9/best_2cls_ver1.pth"
ROOT = "/home/ngseo/remove_hook/lerobot/data/lerobot/hyundai_uiwang_left"
OUT = "/home/ngseo/visuomotor-policy-inference/masks_left_stride3.npz"
STRIDE, THRESH = 3, 0.5

ds = LeRobotDataset("Ngseo/hyundai-uiwang-left", root=ROOT, revision="main")
fkey = next(k for k in ds.meta.camera_keys if "front" in k)
model = RFDETRSeg2XLarge(num_classes=2, pretrain_weights=WEIGHT)
print(f"{len(ds)} frames | stride {STRIDE} -> ~{len(ds)//STRIDE} masks | key {fkey}", flush=True)

dl = DataLoader(ds, batch_size=32, num_workers=6, shuffle=False)
M, P, I = [], [], []
t0 = time.time()
seen = 0
for b in dl:
    epi = np.asarray(b["episode_index"]).reshape(-1)
    fi = np.asarray(b["frame_index"]).reshape(-1)
    keep = np.where(fi % STRIDE == 0)[0]
    seen += len(epi)
    for j in keep:
        hwc = (b[fkey][j].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        with torch.autocast("cuda", dtype=torch.float16), torch.no_grad():
            det = model.predict(Image.fromarray(hwc), threshold=THRESH)
        union = np.zeros(hwc.shape[:2], bool)
        if det.mask is not None:
            for m in det.mask:
                union |= m.astype(bool)
        M.append(np.packbits(union, axis=-1))          # (480, 80) uint8
        P.append(int(epi[j])); I.append(int(fi[j]))
    if len(M) % 900 < 11 and len(M):
        el = time.time() - t0
        rate = len(M) / el
        eta = (len(ds) // STRIDE - len(M)) / max(rate, 1e-6) / 60
        print(f"  masks {len(M)} | {rate:.1f} fps | ETA {eta:.0f} min", flush=True)

np.savez(OUT, masks=np.stack(M), epi=np.array(P, np.int64), fi=np.array(I, np.int64),
         thresh=THRESH, stride=STRIDE)
print(f"saved {OUT}: {len(M)} masks ({np.stack(M).nbytes/1e9:.2f}GB packed)", flush=True)
