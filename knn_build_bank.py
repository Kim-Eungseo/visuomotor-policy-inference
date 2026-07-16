"""VINN-style kNN policy — step 1: build the DINOv3 embedding memory bank (LEFT dataset).

Encodes every demo frame's (front_rgb, wrist_rgb) with a frozen DINOv3-ViTB16 into a
concatenated, L2-normalized CLS embedding, and stores (embedding -> action, episode) so a
non-parametric kNN policy can retrieve neighbor actions at inference (no head training).
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel

from lerobot.datasets.lerobot_dataset import LeRobotDataset

DINO = os.environ.get("VPI_KNN_DINO", "facebook/dinov3-vitb16-pretrain-lvd1689m")
REPO = os.environ.get("VPI_KNN_REPO", "local/hyundai_uiwang_left")
ROOT = os.environ.get("VPI_KNN_ROOT", "/home/ngseo/remove_hook/lerobot/data/lerobot/hyundai_uiwang_left")
OUT = os.environ.get("VPI_KNN_OUT", "/home/ngseo/visuomotor-policy-inference/knn_bank_left.npz")
FKEY = os.environ.get("VPI_KNN_FKEY", "observation.images.front_rgb")   # our dataset: observation.images.front
WKEY = os.environ.get("VPI_KNN_WKEY", "observation.images.wrist_rgb")   #              observation.images.wrist
CAMS = [c.strip() for c in os.environ.get("VPI_KNN_CAMS", "front,wrist").split(",") if c.strip()]
CAM_KEY = {"front": FKEY, "wrist": WKEY}

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained(DINO).to(dev).eval()
for p in model.parameters():
    p.requires_grad = False
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)


@torch.no_grad()
def encode(x):  # x: (B,3,H,W) float [0,1] -> (B,768) L2-normalized CLS
    x = F.interpolate(x.to(dev, non_blocking=True), size=(224, 224), mode="bilinear", align_corners=False)
    x = (x.clamp(0.0, 1.0) - MEAN) / STD
    cls = model(pixel_values=x).last_hidden_state[:, 0]   # CLS token (index 0, before registers)
    return F.normalize(cls, dim=-1)


def main():
    ds = LeRobotDataset(REPO, root=ROOT)
    N = len(ds)
    print(f"dataset {REPO}: {N} frames | encoder {DINO} on {dev}")
    dl = DataLoader(ds, batch_size=128, num_workers=int(os.environ.get("VPI_KNN_WORKERS","2")), shuffle=False, pin_memory=True)

    embs, acts, epis = [], [], []
    done = 0
    for b in dl:
        e = F.normalize(torch.cat([encode(b[CAM_KEY[c]]) for c in CAMS], dim=-1), dim=-1)  # (B, 768*len(CAMS))
        embs.append(e.cpu().numpy())
        acts.append(np.asarray(b["action"], dtype=np.float32))
        epis.append(np.asarray(b["episode_index"]).reshape(-1))
        done += e.shape[0]
        if done % 5120 < 128:
            print(f"  encoded {done}/{N}")
    emb = np.concatenate(embs).astype(np.float32)
    act = np.concatenate(acts).astype(np.float32)
    epi = np.concatenate(epis).astype(np.int64)
    np.savez(OUT, emb=emb, act=act, epi=epi)
    print(f"saved {OUT}: emb{emb.shape} act{act.shape} episodes={len(np.unique(epi))}")


if __name__ == "__main__":
    main()
