"""BYOL fine-tune of DINOv3-ViTS16 on the MAIN left dataset (131 ep / 110k frames, 30fps).

Differences vs teleop_byol.py:
  - keys are front_rgb / wrist_rgb, no mask view -> plain 2-aug BYOL (the winner on teleop anyway).
  - 110k full-res frames don't fit in RAM: exploit 30fps temporal redundancy by keeping every 3rd
    frame (within-episode) and preloading at 240x320 uint8 (~14GB). RandomResizedCrop is
    relative-scale, so augmenting from 240x320 ~ equals augmenting from 480x640.
  - trailing --holdout episodes are excluded from SSL (clean kNN eval afterwards).

  conda run -n lr python left_byol.py --epochs 6 --save_every 2 --out byol_left.pt
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

REPO = "Ngseo/hyundai-uiwang-left"
ROOT = os.environ.get("VPI_LEFT_ROOT", "/home/ngseo/remove_hook/lerobot/data/lerobot/hyundai_uiwang_left")
DINO = "facebook/dinov3-vits16-pretrain-lvd1689m"

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


def _shared(a8, b8):  # same crop box on both views, independent photometric (mask positive)
    a, b = a8.float() / 255.0, b8.float() / 255.0
    i, j, h, w = _rrc(a.shape[1], a.shape[2])
    a = _norm(_photo(TF.resized_crop(a, i, j, h, w, [224, 224], antialias=True)))
    b = _norm(_photo(TF.resized_crop(b, i, j, h, w, [224, 224], antialias=True)))
    return a, b


class Pairs(Dataset):
    def __init__(self, front, wrist, mask=None):
        self.front, self.wrist, self.mask = front, wrist, mask

    def __len__(self):
        return self.front.shape[0]

    def __getitem__(self, i):
        out = {"f1": _one(self.front[i]), "f2": _one(self.front[i]),
               "w1": _one(self.wrist[i]), "w2": _one(self.wrist[i])}
        if self.mask is not None:
            masked = self.front[i] * self.mask[i]          # uint8 * {0,1} -> background zeroed
            out["fm"], out["mm"] = _shared(self.front[i], masked)
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
    print(f"backbone blocks={max(idxs)+1 if idxs else '?'} | unfreeze last {n} -> {n_tr} trainable param-tensors",
          flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--unfreeze", type=int, default=2)
    ap.add_argument("--holdout", type=int, default=26, help="# trailing episodes excluded from SSL")
    ap.add_argument("--stride", type=int, default=3, help="keep every Nth frame (30fps redundancy)")
    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--save_every", type=int, default=2)
    ap.add_argument("--use_mask", type=int, default=0, help="add (front, front*mask) shared-crop positive")
    ap.add_argument("--mask_npz", default="masks_left_stride3.npz")
    ap.add_argument("--out", default="byol_left.pt")
    args = ap.parse_args()
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- preload: every `stride`-th frame of train episodes, downscaled to 240x320 uint8 ----
    ds = LeRobotDataset(REPO, root=ROOT, revision="main")
    fkey = next(k for k in ds.meta.camera_keys if "front" in k)
    wkey = next(k for k in ds.meta.camera_keys if "wrist" in k)
    n_ep = ds.num_episodes
    lim = n_ep - args.holdout
    print(f"{REPO}: {len(ds)} frames / {n_ep} eps | train eps < {lim} | stride {args.stride} | keys {fkey},{wkey}",
          flush=True)

    mlut = None
    if args.use_mask:
        z = np.load(args.mask_npz)
        mrow = {(int(e), int(f)): r for r, (e, f) in enumerate(zip(z["epi"], z["fi"]))}
        mpacked = z["masks"]                                               # (N,480,80) packbits
        print(f"mask npz: {mpacked.shape[0]} masks loaded", flush=True)

    dl0 = DataLoader(ds, batch_size=128, num_workers=6, shuffle=False)
    F_, W_, M_ = [], [], []
    done = 0
    for nb, b in enumerate(dl0):
        epi = np.asarray(b["episode_index"]).reshape(-1)
        fi = np.asarray(b["frame_index"]).reshape(-1)
        keepn = (epi < lim) & (fi % args.stride == 0)
        keep = torch.from_numpy(keepn)
        done += len(epi)
        if keep.any():
            for key, buf in ((fkey, F_), (wkey, W_)):
                x = b[key][keep].to(dev)                                   # (k,3,480,640) float [0,1]
                x = F.interpolate(x, size=(240, 320), mode="bilinear", align_corners=False)
                buf.append((x * 255).round().clamp(0, 255).to(torch.uint8).cpu())
            if args.use_mask:
                rows = [mrow[(int(e), int(f))] for e, f in zip(epi[keepn], fi[keepn])]
                mb = np.unpackbits(mpacked[rows], axis=-1)[:, :, :640]     # (k,480,640) {0,1}
                mt = torch.from_numpy(mb).float().unsqueeze(1).to(dev)
                mt = F.max_pool2d(mt, 2)                                   # (k,1,240,320) keep thin hooks
                M_.append((mt > 0).to(torch.uint8).cpu())
        if nb % 60 == 0:
            print(f"  preload {done}/{len(ds)} (kept {sum(t.shape[0] for t in F_)})", flush=True)
    front, wrist = torch.cat(F_), torch.cat(W_)
    mask = torch.cat(M_) if args.use_mask else None                        # (M,1,240,320) uint8 {0,1}
    del F_, W_, M_
    print(f"preloaded {front.shape[0]} train frames at 240x320 "
          f"(~{front.numel() * 2 / 1e9:.1f}GB RAM) | use_mask={args.use_mask}", flush=True)

    dl = DataLoader(Pairs(front, wrist, mask), batch_size=args.bs, shuffle=True,
                    num_workers=6, drop_last=True, persistent_workers=True)

    # ---- BYOL (same recipe as teleop) ----
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
        for nb, b in enumerate(dl):
            b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            loss = byol(b["f1"], b["f2"]) + byol(b["w1"], b["w2"])
            if "fm" in b:
                loss = loss + byol(b["fm"], b["mm"])
            opt.zero_grad(); loss.backward(); opt.step()
            m = 1 - (1 - args.ema) * (math.cos(math.pi * step / steps) + 1) / 2
            ema_update(online_bb, tgt_bb, m); ema_update(projector, proj_t, m)
            tot += loss.item(); step += 1
            if nb % 100 == 0:
                print(f"  ep{ep+1} step {nb}/{len(dl)} loss={loss.item():.4f}", flush=True)
        print(f"epoch {ep+1:>2}/{args.epochs}  loss={tot/len(dl):.4f}  ema={m:.4f}", flush=True)
        if (ep + 1) % args.save_every == 0 or (ep + 1) == args.epochs:
            p = args.out.replace(".pt", f"_e{ep+1}.pt")
            torch.save({"backbone": online_bb.state_dict()}, p)
            print(f"  saved -> {p}", flush=True)


if __name__ == "__main__":
    main()
