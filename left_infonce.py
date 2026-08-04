"""InfoNCE (NT-Xent) fine-tune of DINOv3-ViTS16 on the MAIN left dataset (131 ep / 110k frames, 30fps).

Rationale vs left_byol.py: VINN's downstream task IS retrieval — InfoNCE optimizes exactly the
frame-discrimination (alignment + uniformity) that kNN needs, where BYOL only pulls positives.
  - positive: with prob --tcn_p the second view is a TCN-style temporally-mined neighbor
    (same episode, |dt| <= --tcn_win raw frames), else a second augmentation of the same frame.
    Both views always get independent RandomResizedCrop + photometric augs (same stack as BYOL).
  - false-negative guard: same-episode frames within --neg_margin raw frames of the anchor are
    MASKED OUT of the negatives (at 30fps they are near-duplicates with near-identical actions;
    repelling them is exactly wrong for VINN).
  - per-camera loss (front NT-Xent + wrist NT-Xent), no cross-camera positive: the serving bank
    concatenates front+wrist CLS, forcing them together would collapse complementary info.
  - --use_mask 1: background-swap AUGMENTATION (not BYOL's invariance positive, which lost both
    A/Bs): with prob --bg_p a front view's background (1-mask region, hand+hook mask) is replaced
    by the front frame of a random OTHER episode. Widens the data distribution instead of forcing
    invariance through the loss.
  - --edge_p: peripheral-clutter augmentation — with prob edge_p a front view gets 1-3 random
    crops from other frames pasted into the border band (outer 25%; center untouched). Views are
    perturbed independently, so positives pair (cluttered, clean) -> robustness to unexpected
    objects at the image periphery.
Preload/holdout/stride identical to left_byol.py; checkpoints stay {"backbone": ...} so
left_knn_eval.py --ckpt works unchanged.

  conda run -n lr python left_infonce.py --epochs 6 --save_every 2 --out infonce_left.pt
"""
import argparse
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


class Pairs(Dataset):
    """View 1 = aug(frame i); view 2 = aug(frame j) where j = i or a same-episode temporal
    neighbor (TCN mining). Same j for front & wrist. Returns epi + both raw frame times so the
    loss can mask same-episode near-in-time false negatives."""

    def __init__(self, front, wrist, epi, fi, tcn_p, tcn_win, stride, mask=None, bg_p=0.5,
                 edge_p=0.0):
        self.front, self.wrist = front, wrist
        self.epi, self.fi = epi, fi
        self.mask, self.bg_p = mask, bg_p                      # (M,1,240,320) uint8 {0,1} or None
        self.edge_p = edge_p
        self.p, self.K = tcn_p, max(1, tcn_win // stride)      # window in preloaded-row units
        starts = np.flatnonzero(np.diff(epi)) + 1              # rows are episode-contiguous, time-sorted
        starts = np.concatenate([[0], starts])
        ends = np.concatenate([starts[1:], [len(epi)]])
        seg = np.searchsorted(starts, np.arange(len(epi)), "right") - 1
        self.lo, self.hi = starts[seg], ends[seg]

    def __len__(self):
        return self.front.shape[0]

    def _edge(self, x):  # paste 1-3 random-frame crops into the border band; center stays clean
        x = x.clone()                                          # never mutate the shared preload
        _, H, W = x.shape
        bh, bw = H // 4, W // 4                                # band thickness
        for _ in range(random.randint(1, 3)):
            src = self.front[random.randrange(len(self.epi))]
            side = random.randrange(4)
            if side < 2:                                       # top / bottom band
                ph, pw = random.randint(H // 8, bh), random.randint(W // 8, W // 2)
                i0 = random.randint(0, bh - ph) + (0 if side == 0 else H - bh)
                j0 = random.randint(0, W - pw)
            else:                                              # left / right band
                ph, pw = random.randint(H // 8, H // 2), random.randint(W // 8, bw)
                i0 = random.randint(0, H - ph)
                j0 = random.randint(0, bw - pw) + (0 if side == 2 else W - bw)
            si, sj = random.randint(0, H - ph), random.randint(0, W - pw)
            x[:, i0:i0 + ph, j0:j0 + pw] = src[:, si:si + ph, sj:sj + pw]
        return x

    def _front(self, i):  # front view; with prob bg_p swap background for another episode's frame
        x = self.front[i]
        if self.mask is not None and random.random() < self.bg_p:
            for _ in range(4):
                r = random.randrange(len(self.epi))
                if self.epi[r] != self.epi[i]:
                    x = x * self.mask[i] + self.front[r] * (1 - self.mask[i])
                    break
        if random.random() < self.edge_p:
            x = self._edge(x)
        return _one(x)

    def __getitem__(self, i):
        j = i
        if random.random() < self.p:
            j = random.randint(max(self.lo[i], i - self.K), min(self.hi[i] - 1, i + self.K))
        return {"f1": self._front(i), "f2": self._front(j),
                "w1": _one(self.wrist[i]), "w2": _one(self.wrist[j]),
                "epi": int(self.epi[i]), "t1": int(self.fi[i]), "t2": int(self.fi[j])}


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


def info_nce(z1, z2, epi, t1, t2, tau, margin):
    """Symmetric NT-Xent over 2B views. Positive of row i is its paired view; negatives are all
    other views EXCEPT same-episode frames with |dt| < margin (false negatives at 30fps)."""
    B = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2]), dim=-1)
    sim = z @ z.T / tau
    ee, tt = torch.cat([epi, epi]), torch.cat([t1, t2])
    fneg = (ee[:, None] == ee[None, :]) & ((tt[:, None] - tt[None, :]).abs() < margin)  # incl. self
    ar = torch.arange(2 * B, device=z.device)
    pos = torch.cat([ar[B:], ar[:B]])
    fneg[ar, pos] = False                                       # keep the positive logit
    sim = sim.masked_fill(fneg, float("-inf"))
    return F.cross_entropy(sim, pos), (sim.argmax(1) == pos).float().mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=128, help="InfoNCE wants negatives; no EMA target so this fits")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--unfreeze", type=int, default=2)
    ap.add_argument("--holdout", type=int, default=26, help="# trailing episodes excluded from SSL")
    ap.add_argument("--stride", type=int, default=3, help="keep every Nth frame (30fps redundancy)")
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--tcn_p", type=float, default=0.5, help="prob view2 is a temporal neighbor")
    ap.add_argument("--tcn_win", type=int, default=15, help="positive window, raw frames (15 = 0.5s)")
    ap.add_argument("--neg_margin", type=int, default=45, help="same-episode frames closer than this are not negatives")
    ap.add_argument("--use_mask", type=int, default=0, help="background-swap augmentation on front views")
    ap.add_argument("--bg_p", type=float, default=0.5, help="prob a front view gets its background swapped")
    ap.add_argument("--edge_p", type=float, default=0.0, help="prob a front view gets border-band clutter patches")
    ap.add_argument("--mask_npz", default="masks_left_stride3.npz")
    ap.add_argument("--save_every", type=int, default=2)
    ap.add_argument("--out", default="infonce_left.pt")
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

    mrow = mpacked = None
    if args.use_mask:
        z = np.load(args.mask_npz)
        mrow = {(int(e), int(f)): r for r, (e, f) in enumerate(zip(z["epi"], z["fi"]))}
        mpacked = z["masks"]                                               # (N,480,80) packbits
        print(f"mask npz: {mpacked.shape[0]} masks loaded", flush=True)

    dl0 = DataLoader(ds, batch_size=128, num_workers=6, shuffle=False)
    F_, W_, E_, T_, M_ = [], [], [], [], []
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
            E_.append(epi[keepn]); T_.append(fi[keepn])
            if args.use_mask:
                rows = [mrow[(int(e), int(f))] for e, f in zip(epi[keepn], fi[keepn])]
                mb = np.unpackbits(mpacked[rows], axis=-1)[:, :, :640]     # (k,480,640) {0,1}
                mt = torch.from_numpy(mb).float().unsqueeze(1).to(dev)
                mt = F.max_pool2d(mt, 2)                                   # (k,1,240,320) keep thin hooks
                M_.append((mt > 0).to(torch.uint8).cpu())
        if nb % 60 == 0:
            print(f"  preload {done}/{len(ds)} (kept {sum(t.shape[0] for t in F_)})", flush=True)
    front, wrist = torch.cat(F_), torch.cat(W_)
    epi_r, fi_r = np.concatenate(E_), np.concatenate(T_)
    mask = torch.cat(M_) if args.use_mask else None
    del F_, W_, E_, T_, M_
    print(f"preloaded {front.shape[0]} train frames at 240x320 (~{front.numel() * 2 / 1e9:.1f}GB RAM) "
          f"| use_mask={args.use_mask}", flush=True)

    dl = DataLoader(Pairs(front, wrist, epi_r, fi_r, args.tcn_p, args.tcn_win, args.stride,
                          mask=mask, bg_p=args.bg_p, edge_p=args.edge_p),
                    batch_size=args.bs, shuffle=True, num_workers=6, drop_last=True,
                    persistent_workers=True)

    # ---- SimCLR-style: backbone + projector, loss on projector output, bank uses raw CLS ----
    bb = unfreeze_last(AutoModel.from_pretrained(DINO).to(dev), args.unfreeze)
    projector = mlp(bb.config.hidden_size, 1024, 256).to(dev)
    params = [p for p in bb.parameters() if p.requires_grad] + list(projector.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    def z(x):
        return projector(bb(pixel_values=x).last_hidden_state[:, 0])

    for ep in range(args.epochs):
        bb.train(); projector.train()
        tot = tacc = 0.0
        for nb, b in enumerate(dl):
            b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            lf, af = info_nce(z(b["f1"]), z(b["f2"]), b["epi"], b["t1"], b["t2"], args.tau, args.neg_margin)
            lw, aw = info_nce(z(b["w1"]), z(b["w2"]), b["epi"], b["t1"], b["t2"], args.tau, args.neg_margin)
            loss = lf + lw
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); tacc += (af.item() + aw.item()) / 2
            if nb % 100 == 0:
                print(f"  ep{ep+1} step {nb}/{len(dl)} loss={loss.item():.4f} "
                      f"pos-acc f={af.item():.3f} w={aw.item():.3f}", flush=True)
        print(f"epoch {ep+1:>2}/{args.epochs}  loss={tot/len(dl):.4f}  pos-acc={tacc/len(dl):.3f}", flush=True)
        if (ep + 1) % args.save_every == 0 or (ep + 1) == args.epochs:
            p = args.out.replace(".pt", f"_e{ep+1}.pt")
            torch.save({"backbone": bb.state_dict()}, p)
            print(f"  saved -> {p}", flush=True)


if __name__ == "__main__":
    main()
