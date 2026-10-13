"""Held-out kNN L1 on the MAIN left dataset (110k): frozen DINOv3 vs BYOL checkpoints.

Encodes every frame (front+wrist CLS concat, L2-norm — same as the VINN runner), caches the
embeddings to an npz (reusable as a serving bank), then measures kNN L1 with:
  bank  = all frames of train episodes (ep < n_ep - holdout)
  query = every 5th frame of the holdout episodes (30fps -> plenty)

  conda run -n lr python left_knn_eval.py                     # frozen
  conda run -n lr python left_knn_eval.py --ckpt byol_left_e6.pt
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
ROOT = os.environ.get("VPI_LEFT_ROOT", "/home/ngseo/remove_hook/lerobot/data/lerobot/hyundai_uiwang_left")
DINO = "facebook/dinov3-vits16-pretrain-lvd1689m"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--holdout", type=int, default=26)
    ap.add_argument("--qstride", type=int, default=5)
    ap.add_argument("--cache", default=None, help="npz cache path (default derived from ckpt name)")
    ap.add_argument("--crop_bottom", type=float, default=0.0, help="front crop, must match training")
    ap.add_argument("--crop_side", type=float, default=0.0, help="front crop, must match training")
    ap.add_argument("--arch", default="dino", choices=["dino", "resnet50"],
                    help="frozen-eval encoder; ckpts carry their own arch")
    ap.add_argument("--side", default="left", choices=["left", "right"])
    args = ap.parse_args()

    tag = (("frozen" if args.arch == "dino" else f"frozen_{args.arch}")
           if not args.ckpt else os.path.basename(args.ckpt).replace(".pt", ""))
    if args.crop_bottom or args.crop_side:
        tag += f"_cb{args.crop_bottom}_cs{args.crop_side}"
    cache = args.cache or f"/tmp/emb_{args.side}_{tag}.npz"

    if os.path.exists(cache):
        z = np.load(cache)
        emb, act, epi, fi = z["emb"].astype(np.float32), z["act"], z["epi"], z["fi"]
        print(f"[{tag}] cache hit {cache}: emb{emb.shape}", flush=True)
    else:
        sd, arch = None, args.arch
        if args.ckpt:
            raw = torch.load(args.ckpt, map_location=DEV)
            arch = raw.get("arch", args.arch) if isinstance(raw, dict) else args.arch
            sd = raw.get("backbone", raw)
        if arch == "resnet50":
            import torchvision
            model = torchvision.models.resnet50(
                weights=None if sd else torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
            model.fc = torch.nn.Identity()
            model = model.to(DEV).eval()
        else:
            model = AutoModel.from_pretrained(DINO).to(DEV).eval()
        if sd is not None:
            miss, unexp = model.load_state_dict(sd, strict=False)
            print(f"[{tag}] loaded ckpt arch={arch} (missing={len(miss)} unexpected={len(unexp)})", flush=True)
        for p in model.parameters():
            p.requires_grad = False

        @torch.no_grad()
        def enc(x, crop=False):
            x = x.to(DEV)
            if crop and (args.crop_bottom or args.crop_side):   # front-only task-region crop
                H, W = x.shape[-2:]
                x = x[..., :int(H * (1 - args.crop_bottom)),
                      int(W * args.crop_side):W - int(W * args.crop_side)]
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            x = (x.clamp(0, 1) - MEAN) / STD
            h = model(x) if arch == "resnet50" else model(pixel_values=x).last_hidden_state[:, 0]
            return F.normalize(h, dim=-1)

        repo = f"Ngseo/hyundai-uiwang-{args.side}"
        root = os.environ.get(f"VPI_{args.side.upper()}_ROOT",
                              f"/home/ngseo/remove_hook/lerobot/data/lerobot/hyundai_uiwang_{args.side}")
        ds = LeRobotDataset(repo, root=root, revision="main")
        fkey = next(k for k in ds.meta.camera_keys if "front" in k)
        wkey = next(k for k in ds.meta.camera_keys if "wrist" in k)
        dl = DataLoader(ds, batch_size=128, num_workers=6, shuffle=False)
        E, A, P, I = [], [], [], []
        done = 0
        for nb, b in enumerate(dl):
            e = F.normalize(torch.cat([enc(b[fkey], crop=True), enc(b[wkey])], dim=-1), dim=-1)
            E.append(e.cpu().numpy().astype(np.float16))
            A.append(np.asarray(b["action"], np.float32))
            P.append(np.asarray(b["episode_index"]).reshape(-1))
            I.append(np.asarray(b["frame_index"]).reshape(-1))
            done += e.shape[0]
            if nb % 100 == 0:
                print(f"  [{tag}] encoded {done}", flush=True)
        emb = np.concatenate(E).astype(np.float32)
        act = np.concatenate(A); epi = np.concatenate(P); fi = np.concatenate(I)
        np.savez(cache, emb=emb.astype(np.float16), act=act, epi=epi, fi=fi)
        print(f"[{tag}] encoded {emb.shape[0]} frames -> {cache}", flush=True)
        del model; torch.cuda.empty_cache()
        act = act.astype(np.float32)

    n_ep = int(epi.max()) + 1
    lim = n_ep - args.holdout
    bm = epi < lim
    qm = (epi >= lim) & (fi % args.qstride == 0)
    bank_e = torch.tensor(emb[bm], device=DEV)
    q_e = torch.tensor(emb[qm], device=DEV)
    bank_a, q_a = act[bm], act[qm]
    print(f"[{tag}] bank={bm.sum()} ({lim} eps) | query={qm.sum()} ({args.holdout} eps, stride {args.qstride})",
          flush=True)

    # chunked cosine kNN (query rows in blocks to bound GPU memory)
    for k in (1, 8, 32):
        preds = []
        for s in range(0, q_e.shape[0], 2048):
            sim = q_e[s:s + 2048] @ bank_e.T
            vals, idx = torch.topk(sim, k, dim=1)
            w = torch.softmax(vals / 0.05, dim=1).cpu().numpy()
            preds.append((w[:, :, None] * bank_a[idx.cpu().numpy()]).sum(1))
        pred = np.concatenate(preds)
        l1 = np.abs(pred - q_a)
        print(f"  [{tag}] k={k:>2}: L1 overall={l1.mean():.4f} | arm={l1[:, :7].mean():.4f} | "
              f"hand={l1[:, 7:].mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
