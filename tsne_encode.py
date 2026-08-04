"""Step A: frozen-DINOv3 encode a dataset's frames -> npz (emb, epi, fi). NO sklearn/matplotlib import
(avoids the OpenMP+DataLoader-fork deadlock). Mirrors the proven build_vinn_artifacts encode path."""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel

from lerobot.datasets.lerobot_dataset import LeRobotDataset

DINO = "facebook/dinov3-vits16-pretrain-lvd1689m"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)


def pick_keys(cam_keys):
    return (next(k for k in cam_keys if "front" in k and "mask" not in k),
            next(k for k in cam_keys if "wrist" in k))


@torch.no_grad()
def enc(model, x):
    x = F.interpolate(x.to(DEV), size=(224, 224), mode="bilinear", align_corners=False)
    x = (x.clamp(0, 1) - MEAN) / STD
    return F.normalize(model(pixel_values=x).last_hidden_state[:, 0], dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ds = LeRobotDataset(args.repo, root=args.root, revision="main")
    fkey, wkey = pick_keys(ds.meta.camera_keys)
    print(f"{args.repo}: {len(ds)} frames, {ds.num_episodes} eps | keys: {fkey}, {wkey}", flush=True)

    model = AutoModel.from_pretrained(DINO).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad = False

    dl = DataLoader(ds, batch_size=128, num_workers=6, shuffle=False)
    embs, epis, fidx, done = [], [], [], 0
    for nb, b in enumerate(dl):
        e = F.normalize(torch.cat([enc(model, b[fkey]), enc(model, b[wkey])], dim=-1), dim=-1)
        embs.append(e.cpu().numpy().astype(np.float16))
        epis.append(np.asarray(b["episode_index"]).reshape(-1))
        fidx.append(np.asarray(b["frame_index"]).reshape(-1))
        done += e.shape[0]
        if nb % 20 == 0:
            print(f"  encoded {done}/{len(ds)}", flush=True)
    np.savez(args.out,
             emb=np.concatenate(embs).astype(np.float16),
             epi=np.concatenate(epis).astype(np.int64),
             fi=np.concatenate(fidx).astype(np.int64),
             n_ep=ds.num_episodes)
    print(f"saved -> {args.out}  ({done} frames)", flush=True)


if __name__ == "__main__":
    main()
