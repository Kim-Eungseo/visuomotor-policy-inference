"""BYOL fine-tune of DINOv3-ViTS16 on the 1026-frame teleop set (background-invariance probe).

Positives per frame:
  - front  : two independent augmentations           (general invariance)
  - wrist  : two independent augmentations           (wrist cam, no mask available)
  - (front, front_masked) : SHARED random crop box, independent photometric  [--use_mask]
      -> the pair differs mainly by background presence => teaches background-invariance.

Trains only the last `--unfreeze` transformer blocks (+ final norm) + BYOL projector/predictor
on the TRAIN episodes (trailing `--holdout` episodes are never seen). Saves the online backbone
state_dict so teleop_knn_eval.py --ckpt can measure held-out kNN L1 vs the frozen encoder.

  conda run -n lr python teleop_byol.py --use_mask 1 --out byol_teleop_mask.pt
  conda run -n lr python teleop_byol.py --use_mask 0 --out byol_teleop_nomask.pt   # ablation
"""
import argparse
import copy
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO = "Ngseo/hyundai-uiwang-left-teleop"
DINO = "facebook/dinov3-vits16-pretrain-lvd1689m"
FKEY, WKEY, MKEY = "observation.images.front", "observation.images.wrist", "observation.images.front_masked"

_photo = T.Compose([
    T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
    T.RandomGrayscale(p=0.2),
    T.RandomApply([T.GaussianBlur(9, (0.1, 2.0))], p=0.5),
])
_norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def _rrc(H, W, scale=(0.5, 1.0), ratio=(3 / 4, 4 / 3)):
    area = H * W
    for _ in range(10):
        ta = random.uniform(*scale) * area
        ar = math.exp(random.uniform(math.log(ratio[0]), math.log(ratio[1])))
        w = int(round(math.sqrt(ta * ar)))
        h = int(round(math.sqrt(ta / ar)))
        if 0 < w <= W and 0 < h <= H:
            return random.randint(0, H - h), random.randint(0, W - w), h, w
    return 0, 0, H, W


def _one(u8):  # uint8 CHW -> normalized (3,224,224), one random augmentation
    x = u8.float() / 255.0
    i, j, h, w = _rrc(x.shape[1], x.shape[2])
    return _norm(_photo(TF.resized_crop(x, i, j, h, w, [224, 224], antialias=True)))


def _shared(a8, b8):  # same crop box on both, independent photometric
    a, b = a8.float() / 255.0, b8.float() / 255.0
    i, j, h, w = _rrc(a.shape[1], a.shape[2])
    a = _norm(_photo(TF.resized_crop(a, i, j, h, w, [224, 224], antialias=True)))
    b = _norm(_photo(TF.resized_crop(b, i, j, h, w, [224, 224], antialias=True)))
    return a, b


class Pairs(Dataset):
    def __init__(self, front, wrist, masked, use_mask):
        self.front, self.wrist, self.masked, self.use_mask = front, wrist, masked, use_mask

    def __len__(self):
        return self.front.shape[0]

    def __getitem__(self, i):
        out = {"f1": _one(self.front[i]), "f2": _one(self.front[i]),
               "w1": _one(self.wrist[i]), "w2": _one(self.wrist[i])}
        if self.use_mask:
            out["fm"], out["mm"] = _shared(self.front[i], self.masked[i])
        return out


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Linear(h, o))


def unfreeze_last(model, n):
    import re
    pat = re.compile(r"\.(?:layer|layers|blocks)\.(\d+)\.")
    idxs = {int(m.group(1)) for nm, _ in model.named_parameters() if (m := pat.search(nm))}
    keep = set(range(max(idxs) - n + 1, max(idxs) + 1)) if idxs else set()
    n_tr = 0
    for nm, p in model.named_parameters():
        m = pat.search(nm)
        p.requires_grad = (m is not None and int(m.group(1)) in keep) or (m is None and "norm" in nm.lower())
        n_tr += p.requires_grad
    print(f"backbone blocks={max(idxs)+1 if idxs else '?'} | unfreeze last {n} -> {n_tr} trainable param-tensors")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use_mask", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--unfreeze", type=int, default=2)
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--save_every", type=int, default=10, help="checkpoint every N epochs (sweep for early-stop)")
    ap.add_argument("--out", default="byol_teleop_mask.pt")
    args = ap.parse_args()
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- preload train-episode frames into RAM (uint8 CHW) ----
    ds = LeRobotDataset(REPO, root=os.environ.get("VPI_TELEOP_ROOT") or None, revision="main")
    dl0 = DataLoader(ds, batch_size=64, num_workers=6, shuffle=False)
    F_, W_, M_, E_ = [], [], [], []
    for b in dl0:
        E_.append(np.asarray(b["episode_index"]).reshape(-1))
        F_.append((b[FKEY] * 255).round().to(torch.uint8))
        W_.append((b[WKEY] * 255).round().to(torch.uint8))
        M_.append((b[MKEY] * 255).round().to(torch.uint8))
    epi = np.concatenate(E_)
    n_ep = int(epi.max()) + 1
    tr = epi < (n_ep - args.holdout)
    front = torch.cat(F_)[tr]; wrist = torch.cat(W_)[tr]; masked = torch.cat(M_)[tr]
    print(f"preloaded {front.shape[0]} train frames ({n_ep-args.holdout} eps) | use_mask={args.use_mask}")

    dl = DataLoader(Pairs(front, wrist, masked, bool(args.use_mask)), batch_size=args.bs,
                    shuffle=True, num_workers=6, drop_last=True, persistent_workers=True)

    # ---- online / target ----
    online_bb = unfreeze_last(AutoModel.from_pretrained(DINO).to(dev), args.unfreeze)
    dim = online_bb.config.hidden_size
    projector = mlp(dim, 1024, 256).to(dev)
    predictor = mlp(256, 1024, 256).to(dev)
    tgt_bb = copy.deepcopy(online_bb).eval().requires_grad_(False)
    proj_t = copy.deepcopy(projector).eval().requires_grad_(False)

    params = [p for p in online_bb.parameters() if p.requires_grad] + \
             list(projector.parameters()) + list(predictor.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    def emb(bb, x):
        return bb(pixel_values=x).last_hidden_state[:, 0]

    def dist(p, z):
        return (2 - 2 * (F.normalize(p, dim=-1) * F.normalize(z, dim=-1)).sum(-1))

    def byol(v1, v2):
        p1 = predictor(projector(emb(online_bb, v1)))
        p2 = predictor(projector(emb(online_bb, v2)))
        with torch.no_grad():
            z1 = proj_t(emb(tgt_bb, v1)); z2 = proj_t(emb(tgt_bb, v2))
        return (dist(p1, z2) + dist(p2, z1)).mean() / 2

    @torch.no_grad()
    def ema_update(o, t, m):
        for po, pt in zip(o.parameters(), t.parameters()):
            pt.data.mul_(m).add_(po.data, alpha=1 - m)
        for bo, bt in zip(o.buffers(), t.buffers()):
            bt.data.copy_(bo.data)

    steps = args.epochs * len(dl)
    step = 0
    for ep in range(args.epochs):
        online_bb.train(); projector.train(); predictor.train()
        tot = 0.0
        for b in dl:
            b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            loss = byol(b["f1"], b["f2"]) + byol(b["w1"], b["w2"])
            if args.use_mask:
                loss = loss + byol(b["fm"], b["mm"])
            opt.zero_grad(); loss.backward(); opt.step()
            m = 1 - (1 - args.ema) * (math.cos(math.pi * step / steps) + 1) / 2  # ema -> 1.0
            ema_update(online_bb, tgt_bb, m); ema_update(projector, proj_t, m)
            tot += loss.item(); step += 1
        print(f"epoch {ep+1:>2}/{args.epochs}  loss={tot/len(dl):.4f}  ema={m:.4f}", flush=True)
        if (ep + 1) % args.save_every == 0 or (ep + 1) == args.epochs:
            p = args.out.replace(".pt", f"_e{ep+1}.pt")
            torch.save({"backbone": online_bb.state_dict()}, p)
            print(f"  saved -> {p}", flush=True)


if __name__ == "__main__":
    main()
