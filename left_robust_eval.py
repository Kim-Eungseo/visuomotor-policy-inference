"""Perturbed-holdout kNN eval: does mask/edge augmentation buy robustness the clean holdout hides?

Applies deterministic synthetic perturbations to the FRONT holdout query images only (bank and
wrist stay clean), then measures k=32 kNN L1 per encoder. Motivated by the real-world finding
that mask-trained encoders are much better on-robot despite tying on clean holdout L1.

Conditions: clean | clutter (3 other-episode patches in the outer 30% band) | bgshift (global
photometric gain) | clutter+bgshift | bgswap (only with --bgswap_mask: replace everything
OUTSIDE the foreground mask with another episode's frame — the direct probe of bg-swap training).
Same perturbation recipe per query across encoders (seeded).

Queries are re-encoded from 240x320 (bank embeddings come from the /tmp caches, encoded at full
res) so 'clean' here is the internal baseline — compare across a row, not to published numbers.

  conda run -n lr python left_robust_eval.py                     # left trio, as published
  conda run -n lr python left_robust_eval.py --side right --holdout 24 --qstride 15 \
      --bgswap_mask masks_right_stride3.npz \
      --encoders edge_e2:infonce_right_e2.pt:/tmp/emb_right_infonce_right_e2.npz,maskedge_e2:infonce_right_mask_e2.pt:/tmp/emb_right_infonce_right_mask_e2.npz
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel

from lerobot.datasets.lerobot_dataset import LeRobotDataset

DINO = "facebook/dinov3-vits16-pretrain-lvd1689m"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
K = 32
DEFAULT_ENCODERS = ",".join([
    "nomask_e2:infonce_notcn_left_e2.pt:/tmp/emb_left_infonce_notcn_left_e2.npz",
    "mask_e6:infonce_mask_left_e6.pt:/tmp/emb_left_infonce_mask_left_e6.npz",
    "mask12_e6:infonce_mask12_left_e6.pt:/tmp/emb_left_infonce_mask12_left_e6.npz",
])
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)


def load_holdout(side, holdout, qstride, mask_npz):
    repo = f"Ngseo/hyundai-uiwang-{side}"
    root = os.environ.get(f"VPI_{side.upper()}_ROOT",
                          f"/home/ngseo/remove_hook/lerobot/data/lerobot/hyundai_uiwang_{side}")
    ds = LeRobotDataset(repo, root=root, revision="main")
    lim = ds.num_episodes - holdout
    fkey = next(k for k in ds.meta.camera_keys if "front" in k)
    wkey = next(k for k in ds.meta.camera_keys if "wrist" in k)
    lut = mpacked = None
    if mask_npz:
        z = np.load(mask_npz)
        lut = {(int(e), int(f)): r for r, (e, f) in enumerate(zip(z["epi"], z["fi"]))}
        mpacked = z["masks"]
    dl = DataLoader(ds, batch_size=128, num_workers=6, shuffle=False)
    F_, W_, E_, I_, M_ = [], [], [], [], []
    for b in dl:
        epi = np.asarray(b["episode_index"]).reshape(-1)
        fi = np.asarray(b["frame_index"]).reshape(-1)
        keepn = (epi >= lim) & (fi % qstride == 0)
        if keepn.any():
            keep = torch.from_numpy(keepn)
            for key, buf in ((fkey, F_), (wkey, W_)):
                x = b[key][keep].to(DEV)
                x = F.interpolate(x, size=(240, 320), mode="bilinear", align_corners=False)
                buf.append((x * 255).round().clamp(0, 255).to(torch.uint8).cpu())
            E_.append(epi[keepn]); I_.append(fi[keepn])
            if lut is not None:
                rows = [lut[(int(e), int(f))] for e, f in zip(epi[keepn], fi[keepn])]
                mb = np.unpackbits(mpacked[rows], axis=-1)[:, :, :640]
                mt = torch.from_numpy(mb).float().unsqueeze(1)
                M_.append((F.max_pool2d(mt, 2) > 0).to(torch.uint8))
    maskq = torch.cat(M_) if lut is not None else None
    return torch.cat(F_), torch.cat(W_), np.concatenate(E_), np.concatenate(I_), lim, maskq


def make_recipes(Q, epi):
    """Per-query deterministic perturbation params, identical for every encoder."""
    recipes = []
    H, W, band = 240, 320, 72          # outer 30% band
    for qi in range(Q):
        rs = np.random.RandomState(1000 + qi)
        patches = []
        for _ in range(3):
            for _try in range(8):
                src = rs.randint(Q)
                if epi[src] != epi[qi]:
                    break
            side = rs.randint(4)
            if side < 2:
                ph, pw = rs.randint(30, band + 1), rs.randint(40, 161)
                i0 = rs.randint(0, band - ph + 1) + (0 if side == 0 else H - band)
                j0 = rs.randint(0, W - pw + 1)
            else:
                ph, pw = rs.randint(30, 121), rs.randint(40, band + 1)
                i0 = rs.randint(0, H - ph + 1)
                j0 = rs.randint(0, band - pw + 1) + (0 if side == 2 else W - band)
            si, sj = rs.randint(0, H - ph + 1), rs.randint(0, W - pw + 1)
            patches.append((src, i0, j0, si, sj, ph, pw))
        gain = rs.uniform(0.7, 1.3, 3) * rs.uniform(0.6, 1.4)   # per-channel x global illumination
        for _try in range(8):
            bg_src = rs.randint(Q)
            if epi[bg_src] != epi[qi]:
                break
        recipes.append((patches, gain.astype(np.float32), bg_src))
    return recipes


def perturb(front, recipes, clutter, shift):
    if not clutter and not shift:
        return front
    out = front.clone()
    for qi, (patches, gain, _) in enumerate(recipes):
        x = out[qi]
        if clutter:
            for src, i0, j0, si, sj, ph, pw in patches:
                x[:, i0:i0 + ph, j0:j0 + pw] = front[src][:, si:si + ph, sj:sj + pw]
        if shift:
            g = torch.tensor(gain).view(3, 1, 1)
            out[qi] = (x.float() * g).round().clamp(0, 255).to(torch.uint8)
    return out


def bgswap(front, maskq, recipes):
    out = front.clone()
    for qi, (_, _, bg_src) in enumerate(recipes):
        m = maskq[qi]
        out[qi] = front[qi] * m + front[bg_src] * (1 - m)
    return out


@torch.no_grad()
def encode(model, imgs):
    E = []
    for s in range(0, imgs.shape[0], 128):
        x = imgs[s:s + 128].to(DEV).float() / 255.0
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - MEAN) / STD
        E.append(F.normalize(model(pixel_values=x).last_hidden_state[:, 0], dim=-1))
    return torch.cat(E)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--holdout", type=int, default=26)
    ap.add_argument("--qstride", type=int, default=5, help="use 15 with --bgswap_mask (stride-3 masks)")
    ap.add_argument("--encoders", default=DEFAULT_ENCODERS, help="comma list of tag:ckpt:cache")
    ap.add_argument("--bgswap_mask", default="", help="fg-mask npz -> adds the bgswap condition")
    args = ap.parse_args()

    front, wrist, epi_q, fi_q, lim, maskq = load_holdout(args.side, args.holdout, args.qstride,
                                                         args.bgswap_mask)
    Q = front.shape[0]
    print(f"holdout queries: {Q} ({args.holdout} eps, stride {args.qstride}, side {args.side})", flush=True)
    recipes = make_recipes(Q, epi_q)
    conds = [("clean", 0, 0), ("clutter", 1, 0), ("bgshift", 0, 1), ("clutter+shift", 1, 1)]
    fronts = {name: perturb(front, recipes, c, s) for name, c, s in conds}
    if maskq is not None:
        fronts["bgswap"] = bgswap(front, maskq, recipes)
        conds.append(("bgswap", 0, 0))

    print(f"{'encoder':<14}" + "".join(f"{n:>15}" for n, _, _ in conds) + f"   (k={K} L1, front-only perturb)")
    for spec in args.encoders.split(","):
        tag, ckpt, cache = spec.split(":")
        z = np.load(cache)
        epi_b, fi_b = z["epi"], z["fi"]
        bm = epi_b < lim
        qm = (epi_b >= lim) & (fi_b % args.qstride == 0)
        assert np.array_equal(epi_b[qm], epi_q) and np.array_equal(fi_b[qm], fi_q), "query order mismatch"
        bank_e = torch.tensor(z["emb"][bm].astype(np.float32), device=DEV)
        bank_a = torch.tensor(z["act"][bm].astype(np.float32))
        q_a = torch.tensor(z["act"][qm].astype(np.float32))

        model = AutoModel.from_pretrained(DINO).to(DEV).eval()
        sd = torch.load(ckpt, map_location=DEV)
        model.load_state_dict(sd.get("backbone", sd), strict=False)
        for p in model.parameters():
            p.requires_grad = False
        we = encode(model, wrist)                            # wrist stays clean; encode once
        row = f"{tag:<14}"
        for name, _, _ in conds:
            fe = encode(model, fronts[name])
            q = F.normalize(torch.cat([fe, we], dim=-1), dim=-1)
            preds = []
            for s in range(0, Q, 2048):
                vals, idx = torch.topk(q[s:s + 2048] @ bank_e.T, K, dim=1)
                w = torch.softmax(vals / 0.05, dim=1).cpu()
                preds.append((w[:, :, None] * bank_a[idx.cpu()]).sum(1))
            l1 = (torch.cat(preds) - q_a).abs().mean().item()
            row += f"{l1:>15.4f}"
        print(row, flush=True)
        del model, bank_e; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
