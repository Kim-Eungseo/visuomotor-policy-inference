"""Mask-latent concat retrieval A/B: does appending a hand+hook mask-pooled DINOv3 patch vector
to the retrieval embedding improve kNN ranking?

Representation: normalize(concat(front_CLS_n, wrist_CLS_n, alpha * maskpool_n)) where maskpool =
mean of the front patch tokens whose 14x14 grid cell overlaps the RF-DETR hand+hook mask
(masks_left_stride3.npz, ALL episodes, stride-3 frames). alpha=0 reproduces the serving baseline.
No extra encoder: patch tokens come from the same DINOv3 forward that yields the CLS.

Bank = train-episode stride-3 rows; query = holdout rows with fi%15==0 (stride-3 x stride-5).
Absolute L1 is NOT comparable to left_knn_eval (denser 88k bank there) — read across alphas.

  conda run -n lr python left_maskcat_eval.py --ckpt infonce_mask12_left_e6.pt
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel

from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO = "Ngseo/hyundai-uiwang-left"
ROOT = "/home/ngseo/remove_hook/lerobot/data/lerobot/hyundai_uiwang_left"
DINO = "facebook/dinov3-vits16-pretrain-lvd1689m"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
GRID = 14                                                   # 224/16 patch grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="infonce_mask12_left_e6.pt")
    ap.add_argument("--mask_npz", default="masks_left_stride3.npz")
    ap.add_argument("--holdout", type=int, default=26)
    ap.add_argument("--alphas", default="0,0.25,0.5,1.0,2.0")
    args = ap.parse_args()
    tag = os.path.basename(args.ckpt).replace(".pt", "")
    cache = f"/tmp/maskcat_{tag}.npz"

    if not os.path.exists(cache):
        z = np.load(args.mask_npz)
        mrow = {(int(e), int(f)): r for r, (e, f) in enumerate(zip(z["epi"], z["fi"]))}
        mpacked = z["masks"]
        model = AutoModel.from_pretrained(DINO).to(DEV).eval()
        sd = torch.load(args.ckpt, map_location=DEV)
        model.load_state_dict(sd.get("backbone", sd), strict=False)
        for p in model.parameters():
            p.requires_grad = False

        @torch.no_grad()
        def enc(x, want_patches=False):
            x = F.interpolate(x.to(DEV), size=(224, 224), mode="bilinear", align_corners=False)
            x = (x.clamp(0, 1) - MEAN) / STD
            h = model(pixel_values=x).last_hidden_state          # (B, 1+reg+196, 384)
            cls = F.normalize(h[:, 0], dim=-1)
            return (cls, h[:, -GRID * GRID:]) if want_patches else (cls, None)

        ds = LeRobotDataset(REPO, root=ROOT, revision="main")
        fkey = next(k for k in ds.meta.camera_keys if "front" in k)
        wkey = next(k for k in ds.meta.camera_keys if "wrist" in k)
        dl = DataLoader(ds, batch_size=64, num_workers=6, shuffle=False)
        CF, CW, MP, A, P, I = [], [], [], [], [], []
        done = 0
        for nb, b in enumerate(dl):
            epi = np.asarray(b["episode_index"]).reshape(-1)
            fi = np.asarray(b["frame_index"]).reshape(-1)
            keepn = fi % 3 == 0
            done += len(epi)
            if not keepn.any():
                continue
            keep = torch.from_numpy(keepn)
            cf, patches = enc(b[fkey][keep], want_patches=True)
            cw, _ = enc(b[wkey][keep])
            rows = [mrow[(int(e), int(f))] for e, f in zip(epi[keepn], fi[keepn])]
            mb = np.unpackbits(mpacked[rows], axis=-1)[:, :, :640]           # (k,480,640)
            mt = torch.from_numpy(mb).float().unsqueeze(1).to(DEV)
            cell = F.adaptive_max_pool2d(mt, (GRID, GRID)).flatten(1) > 0    # (k,196) any overlap
            wsum = cell.float().unsqueeze(-1)                                # (k,196,1)
            pool = (patches * wsum).sum(1) / wsum.sum(1).clamp_min(1)        # mean over masked cells
            pool = torch.where(cell.any(1, keepdim=True), pool, torch.zeros_like(pool))
            CF.append(cf.cpu().numpy().astype(np.float16))
            CW.append(cw.cpu().numpy().astype(np.float16))
            MP.append(F.normalize(pool, dim=-1).cpu().numpy().astype(np.float16))
            A.append(np.asarray(b["action"], np.float32)[keepn])
            P.append(epi[keepn]); I.append(fi[keepn])
            if nb % 200 == 0:
                print(f"  encoded {done} (kept {sum(x.shape[0] for x in CF)})", flush=True)
        np.savez(cache, cf=np.concatenate(CF), cw=np.concatenate(CW), mp=np.concatenate(MP),
                 act=np.concatenate(A), epi=np.concatenate(P), fi=np.concatenate(I))
        print(f"cached -> {cache}", flush=True)
        del model; torch.cuda.empty_cache()

    z = np.load(cache)
    cf, cw, mp = (torch.tensor(z[k].astype(np.float32)) for k in ("cf", "cw", "mp"))
    act, epi, fi = torch.tensor(z["act"]), z["epi"], z["fi"]
    lim = int(epi.max()) + 1 - args.holdout
    bm = epi < lim
    qm = (epi >= lim) & (fi % 15 == 0)
    empty = (mp.abs().sum(1) == 0)
    print(f"rows={len(epi)} bank={bm.sum()} query={qm.sum()} | empty masks: {empty.float().mean():.1%}")

    print(f"{'alpha':<8}{'k=8':>9}{'k=32':>9}   (kNN L1, stride-3 bank)")
    for a in [float(s) for s in args.alphas.split(",")]:
        e = F.normalize(torch.cat([cf, cw, a * mp], dim=-1), dim=-1)
        be, qe = e[bm].to(DEV), e[qm].to(DEV)
        ba, qa = act[bm], act[qm]
        line = f"{a:<8}"
        for k in (8, 32):
            preds = []
            for s in range(0, qe.shape[0], 2048):
                vals, idx = torch.topk(qe[s:s + 2048] @ be.T, k, dim=1)
                w = torch.softmax(vals / 0.05, dim=1).cpu()
                preds.append((w[:, :, None] * ba[idx.cpu()]).sum(1))
            line += f"{(torch.cat(preds) - qa).abs().mean().item():>9.4f}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
