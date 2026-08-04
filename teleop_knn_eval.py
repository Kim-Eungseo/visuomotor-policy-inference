"""kNN (VINN) L1 eval on the 1026-frame teleop dataset, held-out by episode.

Encodes every frame's (front, wrist) CLS with DINOv3-ViTS16 (frozen, or a BYOL-finetuned
checkpoint via --ckpt), builds a memory bank from the TRAIN episodes, and measures single-step
kNN L1 on the HELD-OUT episodes. Same encode path as app/knn_runner.py so numbers are comparable.

  conda run -n lr python teleop_knn_eval.py                 # frozen baseline
  conda run -n lr python teleop_knn_eval.py --ckpt out.pt   # finetuned encoder
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModel

from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO = "Ngseo/hyundai-uiwang-left-teleop"
DINO = "facebook/dinov3-vits16-pretrain-lvd1689m"
FKEY, WKEY = "observation.images.front", "observation.images.wrist"


def load_action_split():
    """Return index where hand/finger joints start in the 26-dim action (via info.json names)."""
    try:
        info = json.load(open(hf_hub_download(REPO, "meta/info.json", repo_type="dataset")))
        names = info["features"]["action"].get("names")
        names = names["motors"] if isinstance(names, dict) else names
        if names:
            for i, n in enumerate(names):
                if any(t in str(n).lower() for t in ("finger", "hand", "dg5", "thumb", "index", "grip")):
                    return i, names
    except Exception:
        pass
    return 7, None  # fallback: first 7 = arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="BYOL-finetuned backbone state_dict (.pt)")
    ap.add_argument("--holdout", type=int, default=4, help="# of trailing episodes held out for eval")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(DINO).to(dev).eval()
    tag = "frozen"
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location=dev)
        sd = sd.get("backbone", sd)  # accept {"backbone": ...} or raw state_dict
        missing, unexpected = model.load_state_dict(sd, strict=False)
        tag = f"finetuned({args.ckpt.split('/')[-1]})"
        print(f"loaded ckpt: missing={len(missing)} unexpected={len(unexpected)}")
    for p in model.parameters():
        p.requires_grad = False

    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    @torch.no_grad()
    def enc(x):  # x: (B,3,H,W) [0,1] -> (B,384) L2-normalized CLS
        x = F.interpolate(x.to(dev), size=(224, 224), mode="bilinear", align_corners=False)
        x = (x.clamp(0, 1) - mean) / std
        return F.normalize(model(pixel_values=x).last_hidden_state[:, 0], dim=-1)

    ds = LeRobotDataset(REPO, root=os.environ.get("VPI_TELEOP_ROOT") or None, revision="main")
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=64, num_workers=6, shuffle=False)

    embs, acts, epis = [], [], []
    for b in dl:
        e = F.normalize(torch.cat([enc(b[FKEY]), enc(b[WKEY])], dim=-1), dim=-1)
        embs.append(e.cpu())
        acts.append(torch.as_tensor(np.asarray(b["action"], np.float32)))
        epis.append(torch.as_tensor(np.asarray(b["episode_index"]).reshape(-1)))
    emb = torch.cat(embs)                       # (N, 768)
    act = torch.cat(acts).numpy()               # (N, 26)
    epi = torch.cat(epis).numpy()               # (N,)

    n_ep = int(epi.max()) + 1
    ho = set(range(n_ep - args.holdout, n_ep))
    tr = np.array([e not in ho for e in epi])
    qe = ~tr
    bank_e, bank_a = emb[tr].to(dev), act[tr]
    q_e, q_a = emb[qe].to(dev), act[qe]
    hi, names = load_action_split()
    print(f"encoder={tag} | bank={tr.sum()} frames ({n_ep-args.holdout} eps) | "
          f"query={qe.sum()} frames ({args.holdout} eps) | arm=[:{hi}] hand=[{hi}:]")

    sim = q_e @ bank_e.T                          # (Q, B) cosine
    for k in (1, 8, 16, 32):
        vals, idx = torch.topk(sim, min(k, sim.shape[1]), dim=1)
        w = torch.softmax(vals / 0.05, dim=1).cpu().numpy()      # (Q,k) LWR weights
        nb = bank_a[idx.cpu().numpy()]                            # (Q,k,26)
        pred = (w[:, :, None] * nb).sum(1)                        # (Q,26)
        l1 = np.abs(pred - q_a)
        print(f"  k={k:>2}: L1 overall={l1.mean():.4f} | arm={l1[:, :hi].mean():.4f} | hand={l1[:, hi:].mean():.4f}")


if __name__ == "__main__":
    main()
