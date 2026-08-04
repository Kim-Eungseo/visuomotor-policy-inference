"""Export BYOL-finetuned DINOv3-ViTS16 encoders + build VINN kNN banks over the full teleop repo.

For each (checkpoint) it:
  1. loads stock dinov3-vits16, applies the BYOL backbone weights, and `save_pretrained`s a self-
     contained HF model dir  ->  point VPI_KNN_DINO at it (app/knn_runner.py loads it directly).
  2. encodes EVERY frame's (front, wrist) CLS -> concat -> L2-norm and saves a bank .npz
     (emb (N,768), act (N,26), epi (N,)) matching the runner's encode path exactly.

  conda run -n lr python build_vinn_artifacts.py
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel

from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO = "Ngseo/hyundai-uiwang-left-teleop"
DINO = "facebook/dinov3-vits16-pretrain-lvd1689m"
FKEY, WKEY = "observation.images.front", "observation.images.wrist"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# (checkpoint, encoder export dir, bank output) — 100ep e30 = best (still improving, less noisy holdout)
CONFIGS = [
    ("byol100_mask_e30.pt",   "vinn/encoders/vits16_byol_mask",   "vinn/banks/teleop_mask.npz"),
    ("byol100_nomask_e30.pt", "vinn/encoders/vits16_byol_nomask", "vinn/banks/teleop_nomask.npz"),
]

MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)


@torch.no_grad()
def enc(model, x):  # (B,3,H,W) [0,1] -> (B,384) L2-norm CLS  (same as app/knn_runner._embed)
    x = F.interpolate(x.to(DEV), size=(224, 224), mode="bilinear", align_corners=False)
    x = (x.clamp(0, 1) - MEAN) / STD
    return F.normalize(model(pixel_values=x).last_hidden_state[:, 0], dim=-1)


def build(ckpt, enc_out, bank_out):
    print(f"\n=== {ckpt} -> {enc_out} + {bank_out} ===")
    model = AutoModel.from_pretrained(DINO).to(DEV).eval()
    sd = torch.load(ckpt, map_location=DEV)
    sd = sd.get("backbone", sd)
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"  loaded backbone: missing={len(miss)} unexpected={len(unexp)}")
    for p in model.parameters():
        p.requires_grad = False

    os.makedirs(enc_out, exist_ok=True)
    model.save_pretrained(enc_out)
    print(f"  encoder saved -> {enc_out} ({', '.join(sorted(os.listdir(enc_out)))})")

    ds = LeRobotDataset(REPO, root=os.environ.get("VPI_TELEOP_ROOT") or None, revision="main")
    dl = DataLoader(ds, batch_size=64, num_workers=6, shuffle=False)
    embs, acts, epis = [], [], []
    for b in dl:
        e = F.normalize(torch.cat([enc(model, b[FKEY]), enc(model, b[WKEY])], dim=-1), dim=-1)
        embs.append(e.cpu().numpy())
        acts.append(np.asarray(b["action"], np.float32))
        epis.append(np.asarray(b["episode_index"]).reshape(-1))
    emb = np.concatenate(embs).astype(np.float32)
    act = np.concatenate(acts).astype(np.float32)
    epi = np.concatenate(epis).astype(np.int64)
    os.makedirs(os.path.dirname(bank_out), exist_ok=True)
    np.savez(bank_out, emb=emb, act=act, epi=epi)
    print(f"  bank saved -> {bank_out}: emb{emb.shape} act{act.shape} episodes={len(np.unique(epi))}")


if __name__ == "__main__":
    for c in CONFIGS:
        build(*c)
    print("\nALL DONE")
